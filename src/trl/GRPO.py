"""
Group Relative Policy Optimization (GRPO) trainer for student code simulation.

Trains the model using execution-based rewards: the model generates code,
the code is executed against unit tests, and the reward signal drives policy
updates.  The reward function assigns:
  - ``2.0`` for exact code match (same normalized AST as student submission).
  - ``1.0`` for grade match (same unit-test score, different code).
  - ``0.0`` for no match.
  - ``-0.5`` for syntax errors.
  - ``-1.0`` for missing fenced code block.
  - A soft length penalty (weighted 0.5) is added to all scores to discourage
    pathologically long completions.
  - All scores are clipped to a minimum of ``-0.75`` for stability.

Context window management:
  - Prompts are left-truncated to ``max_prompt_length`` tokens (preamble
    preserved), identical to the SFT/DPO strategy.
"""

import re
import time 
import io
import random
import tokenize
import pandas as pd 
import Levenshtein
from codebleu import calc_codebleu

from copy import deepcopy
from typing import Optional, Tuple, List, Dict, Any
from operator import add
from datasets import Dataset
from src.utils.core import claim_memory
from vllm import SamplingParams
from trl.rewards import get_soft_overlong_punishment
from trl import GRPOTrainer, GRPOConfig
from src.trl.TRL import TRL 
from src.utils.normalization import robust_normalize

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", force=True)
logger = logging.getLogger(__name__)


# =============================================================================
# Context Window Truncation (SAME AS SFT/DPO)
# =============================================================================

def chat_len_tokens(tokenizer, window: List[Dict[str, Any]]) -> int:
    """Calculate token count for a conversation window."""
    ids = tokenizer.apply_chat_template(
        window, 
        tokenize=True, 
        add_generation_prompt=False
    )
    return int(len(ids))


def left_truncate_by_assistant_turns_to_fit(
    messages_prefix: List[Dict[str, Any]],
    tokenizer,
    max_length: int,
) -> Optional[Tuple[int, List[Dict[str, Any]]]]:
    """
    Keep the *largest* number of recent assistant turns by truncating from the left.
    
    CRITICAL: Always preserves system prompt and first user message (problem description).
    Only truncates intermediate assistant-user exchange pairs.
    
    Returns (start_idx, truncated_prefix) or None if even minimal context doesn't fit.
    """
    if chat_len_tokens(tokenizer, messages_prefix) <= max_length:
        return 0, messages_prefix

    # Identify essential preamble (system + first user message)
    preamble_end = 0
    
    # Find system message if present
    if messages_prefix and messages_prefix[0].get("role") == "system":
        preamble_end = 1
    
    # Find first user message (problem description)
    for i in range(preamble_end, len(messages_prefix)):
        if messages_prefix[i].get("role") == "user":
            preamble_end = i + 1
            break
    
    # Split into preamble (must keep) and suffix (can truncate)
    preamble = messages_prefix[:preamble_end]
    suffix = messages_prefix[preamble_end:]
    
    # Check if preamble alone exceeds limit
    preamble_tokens = chat_len_tokens(tokenizer, preamble)
    if preamble_tokens > max_length:
        return None
    
    # Find assistant positions in the suffix
    a_pos = [i for i, m in enumerate(suffix) if m.get("role") == "assistant"]
    
    if not a_pos:
        # No assistant turns in suffix, just return preamble
        if preamble_tokens <= max_length:
            return 0, preamble
        return None
    
    A = len(a_pos)
    
    # Try progressively smaller suffix windows
    for k in range(A, -1, -1):
        if k == 0:
            combined = preamble
        else:
            suffix_start = a_pos[A - k]
            combined = preamble + suffix[suffix_start:]
        
        if chat_len_tokens(tokenizer, combined) <= max_length:
            actual_start = 0 if k == 0 else (preamble_end + suffix_start)
            return actual_start, combined
    
    # Fallback to just preamble
    if chat_len_tokens(tokenizer, preamble) <= max_length:
        return preamble_end, preamble
    
    return None


# =============================================================================
# GRPO Dataset Processing (WITH TRUNCATION)
# =============================================================================

def process_balanced_predictions(
    df, 
    extract_grade, 
    tokenizer,
    max_prompt_length: int,
    k=-1
):
    """
    Process student trajectories into training examples with TRUNCATION.
    
    CRITICAL CHANGE: Instead of creating full context and filtering later,
    we truncate context here to preserve preamble and fit max_prompt_length.
    
    For each position in a trajectory, extracts:
    - prompt: TRUNCATED conversation history (with preamble preserved)
    - student_code: the immediate next submission
    - previous_code: the current submission
    - remaining_trajectory: all submissions after the immediate next
    - current_grade: grade of the immediate next submission
    - remaining_grades: grades of all future submissions
    """

    code_block_full_pattern = re.compile(
        r"(```(?:[a-zA-Z0-9_+-]*)\s*\n.*?\n?```)",
        re.DOTALL,
    )

    match_python = re.compile(
        rf"(?:<actual_submission>\s*\n(?P<actual>.*?)\n</actual_submission>[\s\n]*)?" 
        rf"(?:```diff\s*\n(?P<thought>.*?)\n```[\s\n]*)?"
        rf"```python\s*\n"
        rf"(?P<code>.*?)"
        rf"\n?```",
        re.DOTALL
    )

    dataframe = []
    skipped_truncation = 0
    truncated_count = 0
    
    for message, problem_id in zip(df.messages, df.problem_id):
        assistant_idxs = list(range(4, len(message), 2))
        if k == -1: 
            sample_k = len(assistant_idxs)
        else:
            sample_k = k
        random.shuffle(assistant_idxs)

        count = 0
        for i in assistant_idxs:
            if count >= sample_k: 
                break 

            # Extract current submission
            match = match_python.search(message[i]["content"])
            if not match:
                continue
            student_code = match.group("code").strip()

            # Extract previous submission
            match = match_python.search(message[i - 2]["content"])
            if not match:
                continue
            previous_code = match.group("code").strip()

            # Extract grade for current submission (from feedback that follows it)
            if i + 1 >= len(message):
                continue
            current_grade = extract_grade(message[i + 1]["content"])

            # Extract remaining trajectory with grades
            remaining_trajectory = []
            remaining_grades = []
            for j in range(i + 2, len(message), 2):
                future_match = match_python.search(message[j]["content"])
                if not future_match:
                    continue
                future_code = future_match.group("code").strip()
                
                # Extract grade for this future submission
                if j + 1 < len(message):
                    future_grade = extract_grade(message[j + 1]["content"])
                    remaining_trajectory.append(future_code)
                    remaining_grades.append(future_grade)

            # Build context and clean up code blocks
            context_slice = deepcopy(message[:i])
            for j in range(2, i, 2):
                prev_raw = context_slice[j]["content"]
                m_code = code_block_full_pattern.search(prev_raw)
                context_slice[j]["content"] = (
                    m_code.group(1).strip() if m_code else prev_raw.strip()
                )

            # ============================================================
            # NEW: Apply truncation WITH PREAMBLE PRESERVATION
            # ============================================================
            # Reserve space for generation (~500 tokens typical for student code)
            prompt_budget = max_prompt_length
            
            trunc_result = left_truncate_by_assistant_turns_to_fit(
                context_slice, 
                tokenizer, 
                prompt_budget
            )
            
            if trunc_result is None:
                # Context too long even with truncation (very rare)
                print("Skipped truncation", skipped_truncation)
                skipped_truncation += 1
                continue
            
            start_idx, context_truncated = trunc_result
            
            # Track if truncation happened
            if start_idx > 0:
                truncated_count += 1
            
            # Verify preamble preserved
            if context_truncated:
                first_role = context_truncated[0]["role"]
                assert first_role in ["system", "user"], \
                    f"Expected system/user as first role, got {first_role}"
            # ============================================================

            dataframe.append({
                "problem_id": problem_id,
                "prompt": context_truncated,  # Use truncated version
                "student_code": student_code,
                "previous_code": previous_code,
                "remaining_trajectory": remaining_trajectory,
                "current_grade": current_grade,
                "remaining_grades": remaining_grades,
            })

            count += 1 

    logger.info(f"GRPO dataset processing complete:")
    logger.info(f"  Total examples: {len(dataframe)}")
    logger.info(f"  Truncated: {truncated_count}")
    logger.info(f"  Skipped (too long): {skipped_truncation}")
    
    return pd.DataFrame(dataframe)


class GRPO(TRL):
    """
    GRPO (Group Relative Preference Optimization) trainer for learning student code generation behavior.
    
    This trainer teaches a model to generate code like real students do, using a trajectory-aware
    reward signal that rewards generations matching any point in the student's remaining trajectory.
    
    Extends TRL with Unsloth-compatible GRPOTrainer and GRPOConfig integration.
    """

    def __init__(self, config, test_run, lazy_load=False) -> None:
        """
        Initialize the GRPO student imitation trainer.

        Args:
            config: Experiment configuration (file path or dict).
            test_run (bool): If True, runs a lightweight test to validate setup.
            lazy_load (bool): If True, defers dataset and model loading until needed.
        """
        super().__init__(config, test_run, lazy_load=lazy_load)

        # Configure trainer components
        self.Trainer = GRPOTrainer
        self.TrainerArgs = GRPOConfig
        # Prefix to enforce reasoning blocks in model output
        self.prefill = "<think>\n\n</think>\n" #"```" if self.config.task.force_thinking else "```"
        

    def prepare_dataset(self, df):
        """
        Build the GRPO training dataset from student trajectories.

        Processing steps:
          1. Pre-sample the trajectory dataframe if it exceeds
             ``config.task.max_dataset_size``.
          2. Call ``process_balanced_predictions`` to extract (prompt,
             student_code, previous_code, current_grade) tuples with
             preamble-preserving truncation.
          3. Apply the tokenizer chat template to each prompt string
             (sequentially — no multiprocessing to avoid pickling issues).
          4. Optionally append a ``<think>`` prefill if
             ``config.task.force_thinking`` is True.
          5. Skip examples that still exceed ``max_prompt_length`` after
             truncation (rare).

        Args:
            df (pd.DataFrame): Preprocessed trajectory dataframe.

        Returns:
            datasets.Dataset: Each example has ``prompt`` (rendered string),
            ``student_code``, ``previous_code``, ``current_grade``, and
            ``problem_id`` fields.

        Raises:
            ValueError: If no examples survive processing.
        """
        claim_memory()

        tokenizer = self.agent.tokenizer
        max_prompt_length = self.config.task.args.max_prompt_length
        max_size = self.config.task.max_dataset_size
        k = self.config.task.k

        logger.info(f"Preparing GRPO dataset with max_prompt_length={max_prompt_length}")
        
        # ============================================================================
        # Step 1: Pre-sample DataFrame
        # ============================================================================
        if len(self.dataframe) > max_size:
            logger.info(f"Pre-sampling from {len(self.dataframe)} to {max_size}")
            self.dataframe = self.dataframe.sample(n=max_size, random_state=42).reset_index(drop=True)
            claim_memory()
        
        # ============================================================================
        # Step 2: Process balanced predictions
        # ============================================================================
        dataframe = process_balanced_predictions(
            self.dataframe, 
            self.ds_handler.extract_grade,
            tokenizer=tokenizer,
            max_prompt_length=max_prompt_length,
            k=k,
        )
        
        logger.info(f"Processed {len(dataframe)} examples")
        
        if len(dataframe) > max_size:
            logger.info(f"Sub-sampling dataset from {len(dataframe)} to {max_size}")
            dataframe = dataframe.sample(n=max_size, random_state=42).reset_index(drop=True)
        
        claim_memory()
        
        # ============================================================================
        # Step 3: Process examples WITHOUT dataset.map()
        # ============================================================================
        logger.info(f"Processing {len(dataframe)} examples sequentially...")
        
        processed_examples = []
        
        for idx in range(len(dataframe)):
            if idx % 100 == 0:
                logger.info(f"Processed {idx}/{len(dataframe)}")
                if idx > 0:
                    claim_memory()
            
            row = dataframe.iloc[idx]
            
            # Apply chat template
            try:
                prompt = tokenizer.apply_chat_template(
                    row["prompt"], 
                    tokenize=False, 
                    add_generation_prompt=True 
                )
                
                # Add thinking prefix if needed
                if self.config.task.force_thinking:
                    prompt = prompt + self.prefill
                
                # Verify token count
                token_count = len(tokenizer(prompt).input_ids)
                if token_count > max_prompt_length:
                    logger.warning(f"Example {idx} exceeds max_length: {token_count} > {max_prompt_length} - skipping")
                    continue
                
                processed_examples.append({
                    "prompt": prompt,
                    "student_code":  row["student_code"], 
                    "previous_code": row["previous_code"],
                    "current_grade": row["current_grade"],
                    "problem_id": row["problem_id"],
                })
                
            except Exception as e:
                logger.warning(f"Error processing example {idx}: {e}")
                continue
        
        logger.info(f"Successfully processed {len(processed_examples)}/{len(dataframe)} examples")
        
        # Free memory
        del dataframe
        claim_memory()
        
        # ============================================================================
        # Step 4: Convert to dataset (now all strings are already formatted)
        # ============================================================================
        if len(processed_examples) == 0:
            raise ValueError("No examples survived processing!")
        
        dataset = Dataset.from_list(processed_examples)
        del processed_examples
        claim_memory()
        
        logger.info(f"Final dataset size: {len(dataset)} examples")
        
        # Quick verification
        logger.info("\nVerifying sample:")
        for i in range(min(3, len(dataset))):
            prompt = dataset[i]["prompt"]
            token_count = len(tokenizer(prompt).input_ids)
            
            has_problem = any(indicator in prompt.lower() for indicator in [
                "write", "function", "implement", "create", "def ", "class "
            ])
            
            status = "✓" if has_problem else "⚠️"
            logger.info(f"  {status} Example {i}: {token_count} tokens")
        
        return dataset


    def train(self, dataset, train_args, **other_trainer_args):
        """
        Run GRPO training with an execution-based reward function.

        Defines the reward function inline and passes it as ``reward_funcs``
        to the parent ``TRL.train`` method.  The reward evaluates generated
        code by:
          1. Extracting the fenced Python block from the completion.
          2. Checking for syntax errors (compile).
          3. Running the code against the exercise's unit tests via
             ``ds_handler.execute``.
          4. Comparing the result to the student's actual next submission and
             its unit-test grade.

        Also configures vLLM sampling parameters for GRPO rollout generation.

        Args:
            dataset: HuggingFace Dataset produced by ``prepare_dataset``.
            train_args (dict): Training arguments from ``prepare_training``.
            **other_trainer_args: Forwarded to the parent train method.
        """
        gamma = 0.8  # Keep for potential experiments
        grade_reward_weight = 1.0 # Weight for grade matching component

        # ------------------------------------------------------------
        # Regexes
        # ------------------------------------------------------------
        code_only = re.compile(
            r"```python\s*\n(?P<code>.*?)\n?```",
            re.DOTALL | re.IGNORECASE,
        )

        # ------------------------------------------------------------
        # Length penalty
        # ------------------------------------------------------------
        reward_max_len = int(0.95 * self.config.task.args.max_completion_length)
        cache_len = int(0.10 * reward_max_len)
        length_reward = get_soft_overlong_punishment(
            max_completion_len=reward_max_len,
            soft_punish_cache=cache_len,
        )

        # ------------------------------------------------------------
        # Reward function
        # ------------------------------------------------------------
        def reward(prompts, completions, completion_ids, **kwargs):
            scores = []

            for i, completion in enumerate(completions):

                response = completion
                if self.config.task.force_thinking:
                    response = self.prefill + response

                # ----------------------------------------------------
                # Require fenced python code
                # ----------------------------------------------------
                m_code = code_only.search(response)
                if not m_code:
                    scores.append(-1.0)
                    continue

                generated_code = m_code.group("code").strip()
                student_code = kwargs["student_code"][i]
                current_grade = kwargs["current_grade"][i]
                problem_id = kwargs["problem_id"][i]
                previous_code = kwargs["previous_code"][i]

                # ----------------------------------------------------
                # Syntax check
                # ----------------------------------------------------
                try:
                    compile(generated_code, "<string>", "exec")
                except SyntaxError:
                    scores.append(-0.5)
                    continue

                # ----------------------------------------------------
                # Execute generated code to get actual grade
                # ----------------------------------------------------
                try:
                    _, _, real_grade = self.ds_handler.execute(generated_code, problem_id)
                except Exception as e:
                    logger.warning(f"Code execution failed: {e}")
                    scores.append(0.0)
                    continue

                # ----------------------------------------------------
                # Simple reward: exact match > grade match > nothing
                # ----------------------------------------------------
                
                # Normalize both codes for comparison

                generated_normalized = robust_normalize(generated_code)
                student_normalized = robust_normalize(student_code)
                prior_normalized = robust_normalize(previous_code)
                
                # Check exact match first
                if generated_normalized == student_normalized:
                    score = 2.0
                    logger.info(f"✓✓ EXACT MATCH: grade={real_grade:.0f}/{current_grade:.0f}, reward=2.0")
                
                # Check grade match
                elif abs(real_grade - current_grade) < 1e-6:  # Exact grade match
                    score = 1.0
                    logger.info(f"✓ GRADE MATCH: grade={real_grade:.0f}/{current_grade:.0f}, reward=1.0")
                # No match
                else:
                    score = 0.0
                    logger.info(f"✗ NO MATCH: grade={real_grade:.0f}/{current_grade:.0f}, reward=0.0")

                scores.append(score)
                
            # --------------------------------------------------------
            # Length regularization
            # --------------------------------------------------------
            scaled_len = list(map(lambda x: 0.5 * x, length_reward(completion_ids)))
            scores = list(map(add, scores, scaled_len))

            # Stability clipping
            scores = [max(s, -0.75) for s in scores]

            claim_memory()
            
            return scores

        from vllm import SamplingParams
        vllm_sampling_params = SamplingParams(
            min_p = 0.1,
            top_p = 1.0,
            top_k = -1,
            temperature = 1.0,
            seed = 3407,
            stop = [self.agent.tokenizer.eos_token],
            include_stop_str_in_output = True,
        )
        train_args["vllm_sampling_params"] = vllm_sampling_params
        args = {"reward_funcs": reward}
        return super().train(dataset, train_args, **args)






def python_tokenize(code: str):
    """
    Tokenize Python code into a list of meaningful Python tokens,
    ignoring whitespace and comments.
    """
    tokens = []
    skip = {
        tokenize.ENCODING,
        tokenize.ENDMARKER,
        tokenize.NEWLINE,
        tokenize.NL,
        tokenize.COMMENT,
    }

    try:
        for tok in tokenize.generate_tokens(io.StringIO(code).readline):
            if tok.type not in skip:
                tokens.append(tok.string)

        return tokens
    except:
        return []


def normalized_token_code_similarity(code1: str, code2: str) -> float:
    """
    Compute normalized Levenshtein distance between tokenized Python programs.
    """
    if not code1 and not code2:
        return 1.0
    if not code1 or not code2:
        return 0.0
    
    tok1 = python_tokenize(code1)
    tok2 = python_tokenize(code2)

    dist = Levenshtein.distance(tok1, tok2)
    max_len = max(len(tok1), len(tok2), 1)

    return 1 - (dist / max_len)


def compute_codebleu(reference, prediction):
    """Compute CodeBLEU score between reference and prediction."""
    if not reference or not prediction: 
        return 0.0
    try:
        results = calc_codebleu(
            [reference], [prediction], 
            lang="python", 
            weights=(0.25, 0.25, 0.25, 0.25), 
            tokenizer=None
        )
        return results['codebleu']
    except:
        return 0.0