"""
Supervised Fine-Tuning (SFT) trainer for student code simulation.

Trains the model to predict each student's next code submission given all prior
conversation turns.  Each trajectory is *expanded*: a trajectory with N student
turns produces N individual training examples (one per turn to predict).

Context window management:
  - Each training example is left-truncated to ``max_length`` tokens.
  - Truncation always preserves the system prompt and the first user turn
    (problem description).  Only intermediate exchange turns are dropped.
  - Loss is computed on assistant turns only (``train_on_responses_only``).

Dataset split: 80 % train / 20 % validation, split by student ID so no student
appears in both splits.

Note: ``split_by_student`` is defined twice in this module (module-level and
inside the class section) — only the first definition is used.
"""

from __future__ import annotations

import re
from copy import deepcopy
from typing import Any, Dict, List, Tuple, Optional
from src.trl.TRL import TRL 
from trl import SFTConfig, SFTTrainer

import pandas as pd 
from random import sample
from datasets import Dataset, DatasetDict

# Keep FULL fenced blocks (including ```...```)
_CODE_FENCE_WITH_FENCES_RE = re.compile(
    r"```[a-zA-Z0-9_+-]*\s*\n.*?\n?```",
    re.DOTALL,
)

def extract_last_fenced_block_with_fences(text: str) -> Optional[str]:
    last = None
    for m in _CODE_FENCE_WITH_FENCES_RE.finditer(text or ""):
        last = m.group(0)
    return last


def remove_actual_submission_tags_inplace(messages: List[Dict[str, Any]]) -> None:
    pat = re.compile(r"<actual_submission>.*?</actual_submission>\s*\n?\s*", re.DOTALL)
    for msg in messages:
        if msg.get("role") == "assistant":
            msg["content"] = pat.sub("", msg.get("content", ""))


def chat_len_tokens(tokenizer, window: List[Dict[str, Any]]) -> int:
    ids = tokenizer.apply_chat_template(window, tokenize=True, add_generation_prompt=False)
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
    
    # Check if preamble alone exceeds limit (very rare but possible)
    preamble_tokens = chat_len_tokens(tokenizer, preamble)
    if preamble_tokens > max_length:
        # Even essential context too long - return None
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
            # Just preamble, no suffix
            combined = preamble
        else:
            # Preamble + last k assistant turns from suffix
            suffix_start = a_pos[A - k]  # Start at this assistant turn in suffix
            combined = preamble + suffix[suffix_start:]
        
        if chat_len_tokens(tokenizer, combined) <= max_length:
            # Return actual start index in original messages_prefix
            actual_start = 0 if k == 0 else (preamble_end + suffix_start)
            return actual_start, combined
    
    # Even preamble + 0 suffix doesn't fit (shouldn't happen if we checked above)
    if chat_len_tokens(tokenizer, preamble) <= max_length:
        return preamble_end, preamble
    
    return None


def expand_trajectory_to_sft_rows(
    messages: List[Dict[str, Any]],
    tokenizer,
    max_length: int,
    add_think_prefix: bool = False,
    think_prefix: str = "<think>\n\n</think>\n\n",
    skip_first_n_assistants: int = 0,
) -> Dict[str, List[Any]]:
    """
    Produce expanded SFT rows from a single trajectory.
    
    Each row predicts one full assistant message (student code submission).
    Prompts are left-truncated by assistant turns to fit max_length,
    ALWAYS preserving system prompt and problem description.
    
    Returns dict with aligned lists (all same length):
      - text: rendered chat strings for TRL SFTTrainer
      - messages: list of message lists for each training example
    """
    msgs = deepcopy(messages)
    remove_actual_submission_tags_inplace(msgs)

    out_text: List[str] = []
    out_messages: List[List[Dict[str, Any]]] = []
    out_start_index: List[int] = []
    out_assistant_index: List[int] = []
    out_prompt_tokens: List[int] = []

    # Find assistant turns with code blocks (student submissions)
    assistant_positions = [
        i for i, m in enumerate(msgs)
        if m.get("role") == "assistant"
        and i > 2
    ]

    for j, msg_idx in enumerate(assistant_positions):
        if j < skip_first_n_assistants:
            continue

        assistant_msg = deepcopy(msgs[msg_idx])
        content = assistant_msg.get("content", "")

        # Only include turns with code blocks
        last_block = extract_last_fenced_block_with_fences(content)
        if last_block is None:
            continue

        if add_think_prefix:
            assistant_msg["content"] = think_prefix + assistant_msg["content"]

        prefix = msgs[:msg_idx]  # Prompt excludes target assistant message

        trunc = left_truncate_by_assistant_turns_to_fit(prefix, tokenizer, max_length)
        if trunc is None:
            continue
        start_i, prompt_msgs = trunc

        # Verify preamble preservation (sanity check during development)
        if prompt_msgs:
            first_role = prompt_msgs[0]["role"]
            assert first_role in ["system", "user"], \
                f"First message should be system/user, got {first_role}"

        # Build training example: [prompt] + [target assistant msg]
        full_msgs = prompt_msgs + [assistant_msg]

        # Render as text for TRL
        text = tokenizer.apply_chat_template(
            full_msgs,
            tokenize=False,
            add_generation_prompt=False,
        )

        out_text.append(text)
        out_messages.append(full_msgs)
        out_start_index.append(start_i)
        out_assistant_index.append(msg_idx)
        out_prompt_tokens.append(chat_len_tokens(tokenizer, prompt_msgs))

    return {
        "text": out_text,
        "messages": out_messages,
        # Uncomment if needed for debugging:
        # "start_index": out_start_index,
        # "assistant_index": out_assistant_index,
        # "prompt_tokens": out_prompt_tokens,
    }


def remove_actual_submission_tags(messages):
    """
    Remove <actual_submission> tags from all assistant messages.
    
    These tags contain ground truth student code for evaluation purposes
    but were NOT present during training. They must be stripped before
    sending messages to the model to avoid distribution mismatch.
    """
    cleaned = deepcopy(messages)
    
    pattern = re.compile(
        r'<actual_submission>.*?</actual_submission>\s*\n?\s*',
        re.DOTALL
    )
    
    for msg in cleaned:
        if msg.get('role') == 'assistant':
            msg['content'] = pattern.sub('', msg['content'])
        #if msg.get('role') == 'user':
            #msg['content'] = pattern.sub('', msg['content'])

    return cleaned


def split_by_student(df, perc=25):
    """Split by student ID instead of problem ID"""
    user_ids = list(df["user_id"].unique())
    k = int(len(user_ids) * perc / 100)
    val_students = sample(user_ids, k)
    
    df.loc[df["user_id"].isin(val_students), "split"] = "val"
    df.loc[~df["user_id"].isin(val_students), "split"] = "train"
    
    train = df[df.split == "train"].reset_index(drop=True)
    val = df[df.split == "val"].reset_index(drop=True)
    
    return train, val



class SFT(TRL):

    def __init__(self, config, test_run, lazy_load=False) -> None:
        super().__init__(config, test_run, lazy_load=lazy_load)
        
        self.Trainer = SFTTrainer
        self.TrainerArgs = SFTConfig
        self.other_args = {"dataset_text_field": "text"}
        self.response_only = True
        self.response_part = self.response_part + "<think>"


    def prepare_dataset(self, dataframe):
        """
        Expand trajectories into SFT training examples and return a DatasetDict.

        For each trajectory, produces one training example per student
        submission turn.  Each example is:
          - Left-truncated to ``config.task.args.max_length`` tokens while
            preserving the system prompt and problem description.
          - Rendered as a single string via the tokenizer's chat template.
          - Validated for preamble preservation (first 3 examples logged).

        Args:
            dataframe (pd.DataFrame): Preprocessed trajectory dataframe with a
                ``messages`` column and a ``user_id`` column.

        Returns:
            DatasetDict: ``{"train": Dataset, "test": Dataset}`` where each
            example has ``text`` (rendered chat string) and ``messages`` fields.
        """
        max_len = self.config.task.args.max_length
        #max_len_model = self.config.model.from_pretrained_kwargs.max_seq_length
        #max_len = min(max_len, max_len_model)
        tok = self.agent.tokenizer

        print(f"\nPreparing SFT dataset with max_length={max_len}")

        # Clean messages
        self.dataframe["messages"] = self.dataframe.messages.apply(remove_actual_submission_tags)

        # Split by student
        train, val = split_by_student(self.dataframe, perc=20)
        train_ds = Dataset.from_pandas(train, preserve_index=False).shuffle(seed=42)
        val_ds = Dataset.from_pandas(val, preserve_index=False).shuffle(seed=42)

        def _expand(batch):
            """
            Process all trajectories in the batch and collect all expanded rows.
            
            Input batch is a dict of lists where each list element is one trajectory.
            Output must be a dict of lists where all lists have the same total length.
            """
            all_texts = []
            all_messages = []
            
            # Iterate over all trajectories in this batch
            for messages in batch["messages"]:
                expanded = expand_trajectory_to_sft_rows(
                    messages=messages,
                    tokenizer=tok,
                    max_length=max_len,
                    add_think_prefix=not ("<think>" in tok.chat_template),
                    skip_first_n_assistants=0,
                )
                # Collect all rows from this trajectory
                all_texts.extend(expanded["text"])
                all_messages.extend(expanded["messages"])
            
            # Return aligned lists
            return {
                "text": all_texts,
                "messages": all_messages,
            }

        # Remove columns that won't align after expansion
        remove_cols = [c for c in train_ds.column_names if c not in ["messages"]]

        print("Expanding trajectories into training examples...")
        train_exp = train_ds.map(
            _expand,
            batched=True,
            batch_size=10,  # Process multiple trajectories at once
            remove_columns=remove_cols,
            desc="Expanding train trajectories",
        )

        val_exp = val_ds.map(
            _expand,
            batched=True,
            batch_size=10,
            remove_columns=remove_cols,
            desc="Expanding val trajectories",
        )

        dataset = DatasetDict({
            "train": train_exp.shuffle(seed=42), 
            "test": val_exp.shuffle(seed=42)
        })

        # Sanity checks
        print(f"\nDataset statistics:")
        print(f"  Train: {len(dataset['train'])} examples")
        print(f"  Val: {len(dataset['test'])} examples")
        
        # Verify preamble preservation
        print("\nVerifying preamble preservation (first 3 examples)...")
        for i in range(min(3, len(dataset["train"]))):
            text = dataset["train"][i]["text"]
            # Check that problem description is present
            has_problem = ("def " in text or "write" in text.lower() or 
                        "function" in text.lower() or "class" in text.lower())
            if not has_problem:
                print(f"⚠️  Warning: Example {i} might be missing problem description")
                print(f"First 200 chars: {text[:200]}")
            else:
                print(f"✓ Example {i} has problem context")
        
        print("\nFirst example (full):")
        print("=" * 70)
        print(dataset["train"][0]["text"])
        print("=" * 70)

        return dataset



def split_by_student(df, perc=25):
    """Split by student ID instead of problem ID"""
    user_ids = list(df["user_id"].unique())
    k = int(len(user_ids) * perc / 100)
    val_students = sample(user_ids, k)
    
    df.loc[df["user_id"].isin(val_students), "split"] = "val"
    df.loc[~df["user_id"].isin(val_students), "split"] = "train"
    
    train = df[df.split == "train"].reset_index(drop=True)
    val = df[df.split == "val"].reset_index(drop=True)
    
    return train, val



