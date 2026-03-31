"""
Remote model interface for OpenAI, HuggingFace, Groq, and Google APIs.

Provides a unified ``batch_query`` interface over multiple cloud providers.
For OpenAI models, requests are submitted via the Batch API (50 % cost
reduction) and polled until completion.  All other providers fall back to
sequential single-request calls.

Supported ``config.source`` values:
  - ``"openai"``      — native OpenAI (uses Batch API)
  - ``"huggingface"`` — HuggingFace Inference API
  - ``"groq"``        — Groq cloud (OpenAI-compatible endpoint)
  - ``"google"``      — Google Generative Language (OpenAI-compatible endpoint)
  - ``"anthropic"``   — Anthropic Messages API
"""


import os
import json
import time
import tempfile
from openai import OpenAI
from anthropic import Anthropic
from huggingface_hub import InferenceClient

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    force=True,
)

logger = logging.getLogger(__name__)

class RemoteModel():
    """
    Unified interface to remote language model APIs.

    Instantiates the appropriate client based on ``config.source`` and exposes
    a ``batch_query`` method compatible with the rest of the pipeline.

    Attributes:
        config: Model config DotMap (``name``, ``source``, ``gen_kwargs``).
        name (str): Model identifier forwarded to API calls.
        max_retries (int): Retry limit for the OpenAI client.
        expo_factor (int): Back-off factor (not used directly; for reference).
        batch_size (int): Always 1 for non-batch providers.
        client: Provider-specific completions client.
        openai_client: Raw ``openai.OpenAI`` instance (only for OpenAI source,
            needed to access the Batch and Files APIs).
    """

    def __init__(self, config, seed=42) -> None:
        """
        Initialize the remote model client.

        Args:
            config: Model config DotMap with at minimum:
                - ``source``: one of ``"openai"``, ``"huggingface"``, ``"groq"``,
                  ``"google"``, or ``"anthropic"``.
                - ``name``: model identifier (e.g. ``"gpt-4o"``, ``"meta-llama/..."``).
                - ``gen_kwargs``: generation parameters (temperature, max_tokens, etc.).
            seed (int): Random seed (unused in practice; kept for API compatibility).
        """
        self.config = config
        self.seed = seed
        self.name = self.config.name 
        self.max_retries, self.expo_factor = 15, 2
        self.batch_size = 1

        base_url = None 
         
        if self.config.source == "huggingface":
            self.client = InferenceClient().chat.completions
        elif self.config.source == "anthropic":
            # defaults to os.environ.get("ANTHROPIC_API_KEY")
            self.client = Anthropic().messages
        elif self.config.source == "groq":
            self.client = OpenAI(
                base_url="https://api.groq.com/openai/v1",
                api_key=os.environ.get("GROQ_API_KEY")
            ).chat.completions

        elif self.config.source == "google":
            self.client = OpenAI(
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
                api_key=os.environ.get("GOOGLE_API_KEY")
            ).chat.completions
        else:
            self.openai_client = OpenAI(
                # api_key defaults to os.environ.get("OPENAI_API_KEY")
                max_retries=self.max_retries,
                timeout=300.0, # 5 minutes timeout
            )
            self.client = self.openai_client.chat.completions

    
    def query(self, messages, gen_kwargs):
        """
        Send a single chat request and return the response text.

        Note: ``gen_kwargs`` argument is overridden by ``config.gen_kwargs`` if
        the latter is set — the parameter is kept for API compatibility.

        Args:
            messages (list[dict]): Conversation as a flat list of
                ``{"role": ..., "content": ...}`` dicts.
            gen_kwargs (dict): Ignored; generation kwargs come from config.

        Returns:
            str: The model's response text.

        Raises:
            ValueError: If a batched message list (list of lists) is passed.
        """
        gen_kwargs = {}
        if self.config.gen_kwargs:
            print("Using model gen kwargs")
            gen_kwargs = self.config.gen_kwargs.toDict()

        if type(messages[0]) == list:
            msg = """
            You passed in argument as multiple list of messages 
            but not suported yet, only generating for one"
            """
            raise ValueError(msg)
                    
        completions = self.client.create(
                    model=self.config.name,
                    messages=messages,
                    **gen_kwargs,
        )

        if self.config.source == "anthropic":
            r = completions.content[0].text
        else:
            r = completions.choices[0].message.content
        
        return r
    

    def batch_query(self, batch, gen_kwargs, cfm=False):
        """
        Processes multiple requests. Uses OpenAI Batch API for OpenAI models
        (50% cost reduction), falls back to sequential queries for other providers.
        
        Args:
            batch: List of message lists to process
            gen_kwargs: Generation parameters (temperature, max_tokens, etc.)
            cfm: unused legacy parameter
            
        Returns:
            List of response texts corresponding to each request in the batch
        """
        # Only use Batch API for native OpenAI models
        if self.config.source not in ("huggingface", "anthropic", "groq", "google"):
            return self._openai_batch_query(batch, gen_kwargs)
        
        # Fallback to sequential for other providers
        return [self.query(m, gen_kwargs) for m in batch]

    def _openai_batch_query(self, batch, gen_kwargs, poll_interval=10, timeout=86400):
        """
        Implements OpenAI Batch API (https://platform.openai.com/docs/guides/batch).
        
        Workflow:
          1. Write a JSONL file with one request per line
          2. Upload it with purpose="batch"
          3. Create a batch job targeting /v1/chat/completions
          4. Poll until completed (or failed/expired)
          5. Download the output file, parse results, return in input order
        
        Args:
            batch: List of message lists
            gen_kwargs: Generation kwargs (temperature, max_tokens, etc.)
            poll_interval: Seconds between status checks (default 30)
            timeout: Max seconds to wait before raising TimeoutError (default 24h)
            
        Returns:
            List of response texts in the same order as the input batch
        """
        
        # Resolve gen_kwargs from config if needed (mirrors self.query logic)
        resolved_kwargs = {}
        if not gen_kwargs and self.config.gen_kwargs:
            resolved_kwargs = self.config.gen_kwargs.toDict()
            logger.info(f"Using model gen kwargs {gen_kwargs}")
        elif gen_kwargs:
            logger.info(f"Using set argument config {gen_kwargs}")
        else:
             raise ValueError("Unknown generation config to use")

        # 1. Build JSONL content
        jsonl_lines = []
        for idx, messages in enumerate(batch):

            # Remove trailing assistant prefill if present (OpenAI doesn't support prefill)
            if messages and messages[-1].get("role") == "assistant":
                logger.debug(f"Request {idx}: removing trailing assistant prefill turn")
                messages = messages[:-1]
                
            request = {
                "custom_id": f"request-{idx}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": self.config.name,
                    "messages": messages,
                    **resolved_kwargs,
                }
            }
            jsonl_lines.append(json.dumps(request))
        
        jsonl_content = "\n".join(jsonl_lines)

        # 2. Upload the JSONL file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False
        ) as f:
            f.write(jsonl_content)
            tmp_path = f.name

        try:
            with open(tmp_path, "rb") as f:
                uploaded_file = self.openai_client.files.create(
                    file=f,
                    purpose="batch"
                )
        finally:
            os.unlink(tmp_path)

        # 3. Create the batch
        batch_job = self.openai_client.batches.create(
            input_file_id=uploaded_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )
        
        logger.info(f"[Batch API] Created batch {batch_job.id} with {len(batch)} requests")

        # 4. Poll until done
        elapsed = 0
        while True:
            batch_status = self.openai_client.batches.retrieve(batch_job.id)
            status = batch_status.status
            
            completed = batch_status.request_counts.completed
            failed = batch_status.request_counts.failed
            total = batch_status.request_counts.total
            logger.info(
                f"[Batch API] Status: {status} "
                f"({completed}/{total} completed, {failed} failed)"
            )

            if status == "completed":
                break
            elif status in ("failed", "expired", "cancelled"):
                # Try to surface errors if available
                errors = getattr(batch_status, "errors", None)
                raise RuntimeError(
                    f"Batch {batch_job.id} ended with status '{status}'. "
                    f"Errors: {errors}"
                )
            
            time.sleep(poll_interval)
            elapsed += poll_interval
            if elapsed >= timeout:
                # Attempt cancellation before raising
                try:
                    self.openai_client.batches.cancel(batch_job.id)
                except Exception:
                    pass
                raise TimeoutError(
                    f"Batch {batch_job.id} did not complete within {timeout}s"
                )

        # 5. Download and parse results
        output_file_id = batch_status.output_file_id
        if not output_file_id:
            raise RuntimeError(
                f"Batch {batch_job.id} completed but has no output_file_id"
            )
        
        result_content = self.openai_client.files.content(output_file_id).text

        # Parse results into a dict keyed by custom_id
        results_map = {}
        for line in result_content.strip().split("\n"):
            if not line:
                continue
            result = json.loads(line)
            custom_id = result["custom_id"]
            
            if result.get("error"):
                print(
                    f"[Batch API] Warning: {custom_id} failed: "
                    f"{result['error']}"
                )
                results_map[custom_id] = None
            else:
                body = result["response"]["body"]
                text = body["choices"][0]["message"]["content"]
                results_map[custom_id] = text

        # Return in input order
        ordered_results = []
        for idx in range(len(batch)):
            cid = f"request-{idx}"
            ordered_results.append(results_map.get(cid))
        
        n_failed = sum(1 for r in ordered_results if r is None)
        if n_failed:
            print(f"[Batch API] Warning: {n_failed}/{len(batch)} requests failed")

        return ordered_results