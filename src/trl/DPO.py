"""
Direct Preference Optimization (DPO) trainer for student code simulation.

Builds preference pairs from the temporal ordering of student submissions:
earlier submissions are treated as *chosen* and later ones as *rejected*.
The intuition is that a model trained to simulate students should produce
code that looks like an *earlier* step in the learning trajectory, not a
more polished future submission.

Three sampling strategies (selected via ``config.task.sampling``):
  - ``"next"``   — each chosen is paired with the submission two steps later
                   (``a_t`` chosen vs ``a_{t+2}`` rejected).
  - ``"random"`` — each chosen is paired with a randomly sampled future
                   submission at least 3 steps ahead.
  - ``"grade"``  — chosen and rejected differ in unit-test grade, sampling
                   the first future submission with a different grade.

Context window management:
  - Each full chosen/rejected conversation is built first, then independently
    left-truncated to ``max_length`` tokens while preserving the preamble.
  - Dataset split: 80 % train / 20 % val by student ID.
"""

import re
import random
from random import sample
import pandas as pd 
from tqdm import tqdm 
from copy import deepcopy
from typing import Optional, Tuple, List, Dict, Any
from datasets import Dataset, DatasetDict
from trl import DPOTrainer, DPOConfig


from src.trl.TRL import TRL
from src.utils.normalization import robust_normalize

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", force=True)
logger = logging.getLogger(__name__)

# ============================================================================
# DPO Class
# ============================================================================

class DPO(TRL):
    """
    DPO trainer with context window management for student simulation.
    """

    def __init__(self, config, test_run, lazy_load=False) -> None:
        """
        Initialize the DPO student imitation trainer.

        Args:
            config: Experiment configuration (file path or dict).
            test_run (bool): If True, runs a lightweight test to validate setup.
            lazy_load (bool): If True, defers dataset and model loading until needed.
        """
        super().__init__(config, test_run, lazy_load=lazy_load)

        # Configure trainer components
        self.Trainer = DPOTrainer
        self.TrainerArgs = DPOConfig
        self.other_args = {}        

    def prepare_dataset(self, df):
        """
        Prepare preference dataset with context window truncation.
        """
        
        print("Preparing dataset for training")
        tokenizer = self.agent.tokenizer
        max_len = self.config.task.args.max_length
        #max_len_model = self.agent.config.from_pretrained_kwargs.max_seq_length
        #max_len = min(max_lemax_lenn_task, max_len_model)
        
        print(f"Max context length: {max_len} tokens")
        
        # Create preference pairs with appropriate sampling strategy
        max_pairs_per_traj = self.config.task.max_pairs_per_traj
        
        if self.config.task.sampling == "next":
            dataframe = process_dpo_next_preferences(
                self.dataframe,
                tokenizer=tokenizer,
                max_length=max_len,
                extract_grade_fn=self.ds_handler.extract_grade,
                max_pairs_per_traj=max_pairs_per_traj
            )
        elif self.config.task.sampling == "random":
            dataframe = process_dpo_temporal_preferences(
                self.dataframe,
                tokenizer=tokenizer,
                max_length=max_len,
                extract_grade_fn=self.ds_handler.extract_grade,
                max_pairs_per_traj=max_pairs_per_traj
            )
        elif self.config.task.sampling == "grade":
            dataframe = process_dpo_grades(
                self.dataframe,
                tokenizer=tokenizer,
                max_length=max_len,
                extract_grade_fn=self.ds_handler.extract_grade,
                max_pairs_per_traj=max_pairs_per_traj
            )
        else:
            raise ValueError(f"Unknown sampling strategy: {self.config.task.sampling}")
        
        dataset = Dataset.from_pandas(dataframe)
        print(f"Initial dataset size: {len(dataset)}")

        def tokenize(example):
            """
            Convert raw prompts to chat format.
            
            Note: Truncation already applied during preference pair creation,
            so this just renders the messages as text.
            """
            example["prompt"] = tokenizer.apply_chat_template(
                example["prompt"], 
                tokenize=False, 
                enable_thinking=False,
                add_generation_prompt=False 
            )
            if "<think>" not in tokenizer.chat_template:
                example["chosen"]["content"] = "<think>\n\n</think>\n\n" + example["chosen"]
                example["rejected"]["content"] = "<think>\n\n</think>\n\n" + example["rejected"]

            example["chosen"] = tokenizer.apply_chat_template(
                example["chosen"], 
                tokenize=False, 
                add_generation_prompt=False 
            )
            example["rejected"] = tokenizer.apply_chat_template(
                example["rejected"], 
                tokenize=False,
                add_generation_prompt=False 
            )
            return example 
        
        dataset = dataset.map(tokenize)
        
        # Filter out pairs that still exceed max_length after truncation
        # (shouldn't happen often, but safety check)
        filter = lambda ex: len(tokenizer(ex["chosen"]).input_ids) < max_len
        dataset = dataset.filter(filter)
        filter = lambda ex: len(tokenizer(ex["rejected"]).input_ids) < max_len
        dataset = dataset.filter(filter)

        print(f"Dataset after length filtering: {len(dataset)}")
        
        # Show examples
        for i in range(min(2, len(dataset))):
            print(f"\n{'='*60}")
            print(f"Example {i}")
            print(f"{'='*60}")
            print("Chosen):")
            print(dataset[i]["chosen"])#[:500] + "..." if len(dataset[i]["chosen"]) > 500 else dataset[i]["chosen"])
            print("\nRejected:")
            print(dataset[i]["rejected"])#[:500] + "..." if len(dataset[i]["rejected"]) > 500 else dataset[i]["rejected"])

        # Split by student
        train, val = split_by_student(dataset.to_pandas(), 20)
        dataset = DatasetDict({
            "train": Dataset.from_pandas(train).shuffle(seed=42),
            "test": Dataset.from_pandas(val).shuffle(seed=42)
        })

        print(f"\nFinal splits:")
        print(f"  Train: {len(dataset['train'])} pairs")
        print(f"  Val: {len(dataset['test'])} pairs")

        return dataset



# =============================================================================
# Context Window Truncation (FINAL CORRECTED VERSION)
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
# Preference Pair Generation (FINAL CORRECTED)
# =============================================================================

def process_dpo_temporal_preferences(
    df,
    tokenizer,
    max_length: int,
    extract_grade_fn=None,
    max_pairs_per_traj=-1
):
    """
    Build DPO preference pairs using the random temporal sampling strategy.

    For each trajectory, samples anchor positions at random.  For each anchor
    ``a_t`` (chosen), samples a rejected submission ``a_{t+k}`` (k >= 3) from
    future positions that has a different normalized code.

    The full chosen and rejected conversations are built independently and then
    each is truncated to ``max_length`` tokens (preamble preserved).

    Args:
        df (pd.DataFrame): Trajectory dataframe with ``messages``, ``problem_id``,
            and ``user_id`` columns.
        tokenizer: HuggingFace tokenizer (used for token counting).
        max_length (int): Maximum token budget for chosen/rejected sequences.
        extract_grade_fn: Unused for this strategy; kept for API uniformity.
        max_pairs_per_traj (int): Cap on pairs per trajectory (-1 = unlimited).

    Returns:
        pd.DataFrame: Preference pairs with columns ``prompt``, ``chosen``,
        ``rejected``, ``user_id``, ``problem_id``.
    """

    think_code_inner_pattern = re.compile(
        r"(?:<think>\s*(.*?)\s*</think>\s*)?"
        r"```(?:[a-zA-Z0-9_+-]*)\s*\n(.*?)\n?```",
        re.DOTALL,
    )

    code_block_full_pattern = re.compile(
        r"(```(?:[a-zA-Z0-9_+-]*)\s*\n.*?\n?```)",
        re.DOTALL,
    )

    rows = []

    logger.info(f"Processing DPO temporal preferences, dataset size: {len(df)}")

    pairs = list(zip(df.messages, df.problem_id, df.user_id))
    for messages, problem_id, user_id in tqdm(pairs, desc="Creating preference pairs"):

        assistant_idxs = list(range(4, len(messages), 2))
        if len(assistant_idxs) < 2:
            continue

        # Pre-extract code for all assistant messages
        codes = {}
        for idx in assistant_idxs:
            m = think_code_inner_pattern.search(messages[idx]["content"])
            if m is not None:
                codes[idx] = m.group(2).strip()

        # Valid anchor positions: need at least one future step
        anchor_positions = assistant_idxs[:-1]
        random.shuffle(anchor_positions)

        # Track which indices have been used as rejected
        rejected_indices = set()

        pair_count = 0

        for i in anchor_positions:
            if max_pairs_per_traj != -1 and pair_count >= max_pairs_per_traj:
                break

            idx_pos = assistant_idxs.index(i)

            # -------- chosen (a_{t+1})
            if i not in codes:
                continue
            chosen_code = codes[i]

            # -------- Build context up to time t
            context_slice = deepcopy(messages[:i])
            
            # Strip code from context to keep only fenced blocks
            for j in range(2, i, 2):
                prev_raw = context_slice[j]["content"]
                m_code = code_block_full_pattern.search(prev_raw)
                context_slice[j]["content"] = (
                    m_code.group(1).strip() if m_code else prev_raw.strip()
                )

            # -------- rejected (a_{t+k}), randomly sample from future
            future_candidates = []
            for future_idx_pos in range(idx_pos + 3, len(assistant_idxs)):
                candidate_idx = assistant_idxs[future_idx_pos]
                
                if candidate_idx in codes and candidate_idx not in rejected_indices:
                    # Must be different from chosen
                    if robust_normalize(codes[candidate_idx]) != \
                       robust_normalize(chosen_code):
                        future_candidates.append(candidate_idx)

            if not future_candidates:
                continue

            rejected_idx = random.choice(future_candidates)
            rejected_code = codes[rejected_idx]

            # ============================================================
            # NEW APPROACH: Build full conversations FIRST
            # ============================================================
            chosen_full = deepcopy(context_slice)
            chosen_full.append({
                "role": "assistant",
                "content": f"```python\n{chosen_code}\n```",
            })

            rejected_full = deepcopy(context_slice)
            rejected_full.append({
                "role": "assistant",
                "content": f"```python\n{rejected_code}\n```",
            })

            # ============================================================
            # Truncate EACH conversation independently to fit max_length
            # ============================================================
            chosen_trunc = left_truncate_by_assistant_turns_to_fit(
                chosen_full, tokenizer, max_length
            )
            rejected_trunc = left_truncate_by_assistant_turns_to_fit(
                rejected_full, tokenizer, max_length
            )
            
            if chosen_trunc is None or rejected_trunc is None:
                continue  # One or both don't fit even when truncated
            
            _, chosen_conv = chosen_trunc
            _, rejected_conv = rejected_trunc

            # Verify preamble preserved in both
            if chosen_conv:
                assert chosen_conv[0]["role"] in ["system", "user"], \
                    f"Chosen: Expected system/user first, got {chosen_conv[0]['role']}"
            if rejected_conv:
                assert rejected_conv[0]["role"] in ["system", "user"], \
                    f"Rejected: Expected system/user first, got {rejected_conv[0]['role']}"
            # ============================================================

            # Extract just the prompt (everything except final assistant turn)
            # This is what goes in the "prompt" field for DPO
            prompt = chosen_conv[:-1]  # Same as rejected_conv[:-1] up to truncation

            rows.append({
                "user_id": user_id,
                "problem_id": problem_id,
                "prompt": prompt,
                "chosen": chosen_conv,
                "rejected": rejected_conv,
            })

            rejected_indices.add(rejected_idx)
            pair_count += 1

    logger.info(f"Created {len(rows)} preference pairs")
    return pd.DataFrame(rows)


def process_dpo_next_preferences(
    df,
    tokenizer,
    max_length: int,
    extract_grade_fn=None,
    max_pairs_per_traj=-1
):
    """
    Build DPO preference pairs using the next-step sampling strategy.

    For each anchor ``a_t`` (chosen), the rejected is always ``a_{t+2}`` (the
    submission two steps later).  Pairs where chosen and rejected have the same
    normalized code are skipped.

    Args:
        df (pd.DataFrame): Trajectory dataframe.
        tokenizer: HuggingFace tokenizer.
        max_length (int): Maximum token budget.
        extract_grade_fn: Unused; kept for API uniformity.
        max_pairs_per_traj (int): Cap on pairs per trajectory (-1 = unlimited).

    Returns:
        pd.DataFrame: Preference pairs with columns ``prompt``, ``chosen``,
        ``rejected``, ``user_id``, ``problem_id``.
    """
    
    think_code_inner_pattern = re.compile(
        r"(?:<think>\s*(.*?)\s*</think>\s*)?"
        r"```(?:[a-zA-Z0-9_+-]*)\s*\n(.*?)\n?```",
        re.DOTALL,
    )

    code_block_full_pattern = re.compile(
        r"(```(?:[a-zA-Z0-9_+-]*)\s*\n.*?\n?```)",
        re.DOTALL,
    )

    rows = []
    logger.info(f"Processing DPO next-step preferences, dataset size: {len(df)}")

    pairs = list(zip(df.messages, df.problem_id, df.user_id))
    for messages, problem_id, user_id in tqdm(pairs, desc="Creating preference pairs"):

        assistant_idxs = list(range(4, len(messages), 2))
        if len(assistant_idxs) < 3:
            continue

        codes = {}
        for idx in assistant_idxs:
            m = think_code_inner_pattern.search(messages[idx]["content"])
            if m is not None:
                codes[idx] = m.group(2).strip()

        anchor_positions = assistant_idxs[:-2]
        random.shuffle(anchor_positions)

        pair_count = 0

        for i in anchor_positions:
            if max_pairs_per_traj != -1 and pair_count >= max_pairs_per_traj:
                break

            idx_pos = assistant_idxs.index(i)

            if i not in codes:
                continue
            chosen_code = codes[i]

            # Build context
            context_slice = deepcopy(messages[:i])
            
            for j in range(2, i, 2):
                prev_raw = context_slice[j]["content"]
                m_code = code_block_full_pattern.search(prev_raw)
                context_slice[j]["content"] = (
                    m_code.group(1).strip() if m_code else prev_raw.strip()
                )

            # rejected (a_{t+2})
            next_idx_pos = idx_pos + 2
            
            if next_idx_pos >= len(assistant_idxs):
                continue
                
            rejected_idx = assistant_idxs[next_idx_pos]
            
            if rejected_idx not in codes:
                continue
                
            rejected_code = codes[rejected_idx]

            if robust_normalize(rejected_code) == robust_normalize(chosen_code):
                continue

            # Build full conversations FIRST
            chosen_full = deepcopy(context_slice)
            chosen_full.append({"role": "assistant", "content": f"```python\n{chosen_code}\n```"})

            rejected_full = deepcopy(context_slice)
            rejected_full.append({"role": "assistant", "content": f"```python\n{rejected_code}\n```"})

            # Truncate independently
            chosen_trunc = left_truncate_by_assistant_turns_to_fit(chosen_full, tokenizer, max_length)
            rejected_trunc = left_truncate_by_assistant_turns_to_fit(rejected_full, tokenizer, max_length)
            
            if chosen_trunc is None or rejected_trunc is None:
                continue
            
            _, chosen_conv = chosen_trunc
            _, rejected_conv = rejected_trunc

            # Verify preamble
            if chosen_conv:
                assert chosen_conv[0]["role"] in ["system", "user"]
            if rejected_conv:
                assert rejected_conv[0]["role"] in ["system", "user"]

            prompt = chosen_conv[:-1]

            rows.append({
                "user_id": user_id,
                "problem_id": problem_id,
                "prompt": prompt,
                "chosen": chosen_conv,
                "rejected": rejected_conv,
            })
            
            pair_count += 1

    logger.info(f"Created {len(rows)} preference pairs")
    return pd.DataFrame(rows)


def process_dpo_grades(
    df,
    tokenizer,
    max_length: int,
    extract_grade_fn,
    max_pairs_per_traj=-1
):
    """
    Build DPO preference pairs using grade-based sampling.

    For each anchor ``a_t`` (chosen), the rejected is the first future
    submission that has a *different* unit-test grade.  This directly encodes
    the preference that submissions with lower grades should be avoided in
    favour of those with higher grades — or vice versa depending on grade order.

    Args:
        df (pd.DataFrame): Trajectory dataframe.
        tokenizer: HuggingFace tokenizer.
        max_length (int): Maximum token budget.
        extract_grade_fn (callable): ``FalconCode.extract_grade`` — parses
            a grade float from the grader feedback string following each
            assistant turn.
        max_pairs_per_traj (int): Cap on pairs per trajectory (-1 = unlimited).

    Returns:
        pd.DataFrame: Preference pairs with columns ``prompt``, ``chosen``,
        ``rejected``, ``user_id``, ``problem_id``.
    """
    
    think_code_inner_pattern = re.compile(
        r"(?:<think>\s*(.*?)\s*</think>\s*)?"
        r"```(?:[a-zA-Z0-9_+-]*)\s*\n(.*?)\n?```",
        re.DOTALL,
    )

    code_block_full_pattern = re.compile(
        r"(```(?:[a-zA-Z0-9_+-]*)\s*\n.*?\n?```)",
        re.DOTALL,
    )

    rows = []
    logger.info(f"Processing DPO grade-based preferences, dataset size: {len(df)}")

    pairs = list(zip(df.messages, df.problem_id, df.user_id))
    for messages, problem_id, user_id in tqdm(pairs, desc="Creating preference pairs"):

        assistant_idxs = list(range(4, len(messages), 2))
        if len(assistant_idxs) < 2:
            continue

        # Pre-compute grades
        grades = []
        for idx in assistant_idxs:
            m = think_code_inner_pattern.search(messages[idx]["content"])
            if m is None:
                grades.append(None)
                continue
            
            feedback_idx = idx + 1
            if feedback_idx < len(messages) and messages[feedback_idx]["role"] == "user":
                feedback_content = messages[feedback_idx]["content"]
                grade = extract_grade_fn(feedback_content)
                grades.append(grade)
            else:
                grades.append(None)

        # logger.info(f"GRADES OBTAINED for sampling {grades}")

        anchor_positions = assistant_idxs[:-1]
        random.shuffle(anchor_positions)

        rejected_indices = set()
        pair_count = 0

        for i in anchor_positions:
            if max_pairs_per_traj != -1 and pair_count >= max_pairs_per_traj:
                break

            idx_pos = assistant_idxs.index(i)

            m_next = think_code_inner_pattern.search(messages[i]["content"])
            if m_next is None:
                continue
            chosen_code = m_next.group(2).strip()

            # Build context
            context_slice = deepcopy(messages[:i])
            
            for j in range(2, i, 2):
                prev_raw = context_slice[j]["content"]
                m_code = code_block_full_pattern.search(prev_raw)
                context_slice[j]["content"] = (
                    m_code.group(1).strip() if m_code else prev_raw.strip()
                )

            chosen_grade = grades[idx_pos]

            if chosen_grade is None:
                continue

            rejected_idx = None
            for future_idx_pos in range(idx_pos + 1, len(assistant_idxs)):
                candidate_idx = assistant_idxs[future_idx_pos]
                future_grade = grades[future_idx_pos]
                
                if (future_grade is not None and 
                    future_grade != chosen_grade and 
                    candidate_idx not in rejected_indices):
                    rejected_idx = candidate_idx
                    break

            if rejected_idx is None:
                continue

            m_future = think_code_inner_pattern.search(messages[rejected_idx]["content"])
            if m_future is None:
                continue
            rejected_code = m_future.group(2).strip()

            if robust_normalize(chosen_code) == robust_normalize(rejected_code):
                continue

            # Build full conversations FIRST
            chosen_full = deepcopy(context_slice)
            chosen_full.append({"role": "assistant", "content": f"```python\n{chosen_code}\n```"})

            rejected_full = deepcopy(context_slice)
            rejected_full.append({"role": "assistant", "content": f"```python\n{rejected_code}\n```"})

            # Truncate independently
            chosen_trunc = left_truncate_by_assistant_turns_to_fit(chosen_full, tokenizer, max_length)
            rejected_trunc = left_truncate_by_assistant_turns_to_fit(rejected_full, tokenizer, max_length)
            
            if chosen_trunc is None or rejected_trunc is None:
                continue
            
            _, chosen_conv = chosen_trunc
            _, rejected_conv = rejected_trunc

            # Verify preamble
            if chosen_conv:
                assert chosen_conv[0]["role"] in ["system", "user"]
            if rejected_conv:
                assert rejected_conv[0]["role"] in ["system", "user"]

            prompt = chosen_conv[:-1]

            rows.append({
                "user_id": user_id,
                "problem_id": problem_id,
                "prompt": prompt,
                "chosen": chosen_conv,
                "rejected": rejected_conv,
            })

            rejected_indices.add(rejected_idx)
            pair_count += 1

    logger.info(f"Created {len(rows)} preference pairs")
    return pd.DataFrame(rows)


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
