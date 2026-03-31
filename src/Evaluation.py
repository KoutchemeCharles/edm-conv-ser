"""
Evaluation Module for Student Problem-Solving Trajectory Simulation
====================================================================

This module implements a comprehensive evaluation system for assessing how well
trained language models can simulate authentic student programming behavior.
The evaluation compares model-generated code trajectories against real student
trajectories collected from automated assessment systems.

Architecture Overview
---------------------
The evaluation system operates in two phases per task:

1. GENERATION MODE (was_done=False):
   - Model generates code solutions iteratively based on conversation history
   - Each generation is executed against unit tests and graded
   - Conversation history grows with model outputs and grader feedback
   - Student code at corresponding positions is extracted for comparison
   - Continues until model reaches 100% or gives up

2. RECORDING MODE (was_done=True):
   - Triggered when model reaches 100% grade or explicitly gives up (exit())
   - Model stops generating new code
   - Evaluation continues to record remaining student submissions
   - Ensures complete student trajectory capture for analysis
   - Model metrics remain frozen at final achieved values

Key Design Decisions
--------------------
- Dual Format Support: Handles both complete Python code and unified diffs
- Content-Based Diff Application: Robust to LLM line number errors
- <actual_submission> Tags: Ground truth student code embedded in messages
- Batch Processing: vLLM batching for efficient parallel evaluation
- Parallel Execution: ThreadPoolExecutor for concurrent code execution

State Variables
---------------
- is_done: Task completely finished, should be removed from active queue
- was_done: Model reached 100%, switched to recording mode
- current_message_index: Position in original student conversation
- model_generations: Growing conversation history (frozen in recording mode)

Example Timeline
----------------
    Step 1: Model: 50%, Student: 40%  [GENERATION MODE]
    Step 2: Model: 75%, Student: 60%  [GENERATION MODE]  
    Step 3: Model: 100%, Student: 80% [GENERATION MODE -> RECORDING MODE]
    Step 4: Model: 100%, Student: 85% [RECORDING MODE - model frozen]
    Step 5: Model: 100%, Student: 100% [RECORDING MODE -> COMPLETE]
    
Authors: [Anonymous for review]
Version: 2.1 (Parallelized)
"""

import os
import re
import time 
import pandas as pd
import multiprocessing as mp
from copy import deepcopy
from warnings import warn
from difflib import SequenceMatcher
from dataclasses import dataclass
from typing import Optional, List, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.Experiment import Experiment

# =============================================================================
# CONFIGURATION CONSTANTS
# =============================================================================

# Maximum number of generation steps before forced termination
MAX_ITER = 5

# Minimum similarity ratio for fuzzy line matching in diff application
# Set to 0.85 to avoid matching semantically different but syntactically similar lines
# e.g., "total = sum(nums)" vs "total = len(nums)" have ~80% similarity
FUZZY_MATCH_THRESHOLD = 0.90

# Default minimum k value (number of prior submissions for context)
DEFAULT_MIN_K = 2

# Default evaluation sample size (number of students to evaluate)
DEFAULT_EVAL_SAMPLE_SIZE = 1000

# Maximum number of parallel workers for code execution
MAX_PARALLEL_WORKERS = 8

# =============================================================================
# DIFF APPLICATION UTILITIES
# =============================================================================

def parse_unified_diff(diff_text):
    """
    Parse unified diff format into structured hunks.
    
    Unified diff format (from GNU diff) consists of:
    - File headers: --- old_file / +++ new_file (ignored here)
    - Hunk headers: @@ -old_start,old_count +new_start,new_count @@
    - Change lines: '-' for deletions, '+' for additions, ' ' for context
    
    Parameters
    ----------
    diff_text : str
        Raw unified diff text, potentially from LLM generation
        
    Returns
    -------
    list of dict
        List of hunks, each containing:
        - 'old_start': int, 1-indexed line number (used as hint only)
        - 'changes': list of (operation, line_content) tuples
          where operation is 'delete', 'add', or 'context'
          
    Notes
    -----
    Line numbers in LLM-generated diffs are frequently incorrect, so they
    are used only as hints for disambiguation, not as authoritative positions.
    
    Examples
    --------
    >>> diff = '''@@ -5,2 +5,2 @@
    ... - old_line
    ... + new_line'''
    >>> hunks = parse_unified_diff(diff)
    >>> hunks[0]['changes']
    [('delete', ' old_line'), ('add', ' new_line')]
    """
    hunks = []
    current_hunk = None
    
    lines = diff_text.split('\n')
    
    for line in lines:
        # Skip file headers (--- and +++ lines)
        if line.startswith('---') or line.startswith('+++'):
            continue
        
        # Parse hunk header: @@ -old_start,old_count +new_start,new_count @@
        hunk_match = re.match(r'@@\s+-(\d+)(?:,(\d+))?\s+\+(\d+)(?:,(\d+))?\s+@@', line)
        if hunk_match:
            # Save previous hunk before starting new one
            if current_hunk is not None:
                hunks.append(current_hunk)
            
            current_hunk = {
                'old_start': int(hunk_match.group(1)),
                'changes': []
            }
            continue
        
        # Skip lines before first hunk header
        if current_hunk is None:
            continue
        
        # Parse change lines based on prefix character
        if line.startswith('-'):
            current_hunk['changes'].append(('delete', line[1:]))
        elif line.startswith('+'):
            current_hunk['changes'].append(('add', line[1:]))
        elif line.startswith(' '):
            current_hunk['changes'].append(('context', line[1:]))
        elif line.strip():
            # Non-empty lines without standard prefix treated as context
            # This handles malformed diffs from LLMs
            current_hunk['changes'].append(('context', line))
    
    # Append final hunk
    if current_hunk is not None:
        hunks.append(current_hunk)
    
    return hunks


def find_best_single_line_match(program_lines, target_line, hint_pos):
    """
    Find the best matching line for a target using content-based matching.
    
    Uses a two-pass strategy:
    1. Exact match (after stripping whitespace) - preferred
    2. Fuzzy match with similarity threshold - fallback
    
    When multiple matches exist, prefers the one closest to hint_pos
    to maintain locality in edits.
    
    Parameters
    ----------
    program_lines : list of str
        Current program lines to search within
    target_line : str
        The line content we're looking for
    hint_pos : int
        The hunk's stated line number (0-indexed), used as tie-breaker
        
    Returns
    -------
    int or None
        Best match position (0-indexed), or None if no acceptable match found
        
    Notes
    -----
    The fuzzy threshold is set relatively high (0.85) to avoid matching
    lines that are syntactically similar but semantically different.
    """
    if not program_lines:
        return None
    
    target_stripped = target_line.strip()
    
    # Pass 1: Exact match (whitespace-normalized)
    exact_matches = []
    for i, line in enumerate(program_lines):
        if line.strip() == target_stripped:
            exact_matches.append(i)
    
    if len(exact_matches) == 1:
        return exact_matches[0]
    elif len(exact_matches) > 1:
        # Multiple exact matches - prefer closest to hint position
        return min(exact_matches, key=lambda x: abs(x - hint_pos))
    
    # Pass 2: Fuzzy match for minor content differences
    best_score = 0.0
    best_pos = None
    
    for i, line in enumerate(program_lines):
        score = SequenceMatcher(None, target_stripped, line.strip()).ratio()
        
        if score >= FUZZY_MATCH_THRESHOLD:
            # Apply small distance penalty to prefer matches near hint
            # This helps maintain edit locality
            distance_penalty = abs(i - hint_pos) * 0.01
            adjusted_score = score - distance_penalty
            
            if adjusted_score > best_score:
                best_score = adjusted_score
                best_pos = i
    
    return best_pos


def apply_hunk_robust(program_lines, hunk):
    """
    Apply a single diff hunk using content-based matching.
    
    This implementation is designed to be robust to LLM-generated diffs
    where line numbers are often incorrect but content is typically accurate.
    
    Strategy
    --------
    1. Collect all deletions and additions from the hunk
    2. For each deletion, find and remove the best matching line by content
    3. Track the position of the last successful removal
    4. Insert all additions at the last removal position
    
    The insertion-at-last-removal strategy ensures that any lines between
    deleted lines (but not themselves deleted) remain above the inserted
    content, handling incomplete diffs gracefully.
    
    Parameters
    ----------
    program_lines : list of str
        Current program as list of lines
    hunk : dict
        Parsed hunk with 'old_start' and 'changes' keys
        
    Returns
    -------
    list of str
        Modified program lines after applying the hunk
        
    Notes
    -----
    When deletions are processed sequentially, indices shift after each
    removal. The content-based matching naturally handles this since we
    search by content, not by index.
    """
    if not program_lines:
        program_lines = []
    
    result = program_lines.copy()
    old_start = hunk['old_start'] - 1  # Convert to 0-indexed
    changes = hunk['changes']
    
    # Separate operations
    deletions = [line for op, line in changes if op == 'delete']
    additions = [line for op, line in changes if op == 'add']
    
    # Handle pure insertion (no deletions)
    if not deletions:
        insert_pos = min(old_start, len(result))
        for idx, add_line in enumerate(additions):
            result.insert(insert_pos + idx, add_line)
        return result
    
    # Process deletions with content-based matching
    last_removal_pos = None
    hint_pos = old_start
    
    for del_line in deletions:
        match_pos = find_best_single_line_match(result, del_line, hint_pos)
        
        if match_pos is not None:
            last_removal_pos = match_pos
            result.pop(match_pos)
            # Update hint for next search - look near where we just removed
            # Note: indices have shifted, but content-based search handles this
            hint_pos = match_pos
    
    # Determine insertion position
    if last_removal_pos is not None:
        insert_pos = last_removal_pos
    else:
        # No deletions matched - fall back to hunk's stated position
        insert_pos = min(old_start, len(result))
    
    # Insert all additions at determined position
    for idx, add_line in enumerate(additions):
        result.insert(insert_pos + idx, add_line)
    
    return result


def apply_diff_robust(program_lines, diff_text):
    """
    Apply a unified diff to a program with robust error handling.
    
    This is the main entry point for diff application. It parses the diff
    into hunks and applies each sequentially using content-based matching.
    
    Parameters
    ----------
    program_lines : list of str
        Current program as list of lines (or empty list)
    diff_text : str
        Unified diff text to apply
        
    Returns
    -------
    list of str
        Modified program lines after applying all hunks
        
    Notes
    -----
    Individual hunk failures are logged but don't abort the entire diff.
    This resilience is important for handling imperfect LLM-generated diffs.
    """
    if not isinstance(program_lines, list):
        program_lines = []
    
    result = program_lines.copy()
    hunks = parse_unified_diff(diff_text)
    
    for i, hunk in enumerate(hunks):
        try:
            result = apply_hunk_robust(result, hunk)
        except Exception as e:
            print(f"Warning: Failed to apply hunk {i}: {e}")
            continue
    
    return result


# =============================================================================
# CONVERSATION HISTORY UTILITIES
# =============================================================================

def reconstruct_program_from_history(messages):
    """
    Reconstruct the current program state from a conversation history.
    
    This function handles the dual format where assistant messages may contain
    either complete Python programs (```python blocks) or incremental edits
    (```diff blocks). The strategy accounts for the fact that full code can
    appear at any point in the trajectory, not just at the beginning.
    
    Strategy
    --------
    1. Scan all messages to find the LAST complete program submission
    2. Apply all diffs that appear AFTER that complete program
    
    This correctly handles scenarios like:
    - Student submits full code, then diffs, then full code again, then diffs
    - The final state is: last_full_code + subsequent_diffs
    
    Parameters
    ----------
    messages : list of dict
        Conversation history with 'role' and 'content' keys
        
    Returns
    -------
    str
        The reconstructed program as a single string
        
    Examples
    --------
    >>> messages = [
    ...     {'role': 'user', 'content': 'Problem description...'},
    ...     {'role': 'assistant', 'content': '```python\\ndef f(): pass\\n```'},
    ...     {'role': 'user', 'content': 'Tests: 0/5'},
    ...     {'role': 'assistant', 'content': '```diff\\n@@ -1 +1 @@\\n-def f(): pass\\n+def f(): return 1\\n```'},
    ... ]
    >>> reconstruct_program_from_history(messages)
    'def f(): return 1'
    """
    current_program = []
    last_full_code_index = -1
    
    # Pass 1: Find the LAST complete program submission
    for i, msg in enumerate(messages):
        if msg.get('role') != 'assistant':
            continue
            
        content = msg.get('content', '')
        
        # Check for complete Python code block
        python_match = re.search(r'```python\s*\n(.*?)```', content, re.DOTALL)
        if python_match:
            program_text = python_match.group(1).strip()
            current_program = program_text.split('\n') if program_text else []
            last_full_code_index = i
    
    # If no complete program found, return empty
    if last_full_code_index == -1:
        raise ValueError("No last full program found!! Should not be possible")
        return ''
    
    # Pass 2: Apply all diffs that come AFTER the last complete program
    for i in range(last_full_code_index + 1, len(messages)):
        msg = messages[i]
        
        if msg.get('role') != 'assistant':
            continue
            
        content = msg.get('content', '')
        
        # Check for diff block
        diff_match = re.search(r'```diff\s*\n(.*?)```', content, re.DOTALL)
        if diff_match:
            diff_text = diff_match.group(1)
            try:
                current_program = apply_diff_robust(current_program, diff_text)
            except Exception as e:
                print(f"Warning: Failed to apply diff at message {i}: {e}")
                # Continue with current program state
    
    return '\n'.join(current_program)


def extract_code_from_actual_submission(message_content):
    """
    Extract ground truth student code from <actual_submission> tags.
    
    During data preparation, the actual student code is embedded in
    assistant messages within <actual_submission> tags. This allows
    the evaluation to access ground truth without reconstructing from
    diffs (which may have been modified for training).
    
    Parameters
    ----------
    message_content : str
        Content of a message that may contain <actual_submission> tags
        
    Returns
    -------
    str or None
        The code content if tags found, None otherwise
    """
    match = re.search(
        r'<actual_submission>\s*\n(.*?)\n</actual_submission>', 
        message_content, 
        re.DOTALL
    )
    if match:
        return match.group(1).strip()
    return None


def remove_actual_submission_tags(messages):
    """
    Remove <actual_submission> tags from all assistant messages.
    
    These tags contain ground truth student code for evaluation purposes
    but were NOT present during training. They must be stripped before
    sending messages to the model to avoid distribution mismatch.
    
    Parameters
    ----------
    messages : list of dict
        Conversation history with 'role' and 'content' keys
        
    Returns
    -------
    list of dict
        Deep copy of messages with <actual_submission> blocks removed
    """
    cleaned = deepcopy(messages)
    
    # Pattern matches the tags and any trailing whitespace/newlines
    pattern = re.compile(
        r'<actual_submission>.*?</actual_submission>\s*\n?\s*',
        re.DOTALL
    )
    
    for msg in cleaned:
        if msg.get('role') == 'assistant':
            msg['content'] = pattern.sub('', msg['content'])
    
    return cleaned


# =============================================================================
# PARALLEL EXECUTION UTILITIES
# =============================================================================

@dataclass
class ExecutionRequest:
    """Data needed for code execution."""
    task_index: int
    code: Optional[str]  # None if extraction failed
    problem_id: str
    agent_output: str
    is_recording_mode: bool
    extraction_error: Optional[str] = None


@dataclass  
class ExecutionResult:
    """Results from code execution."""
    task_index: int
    code: str
    unit_tests: str
    output_content: str
    grade: Optional[int]
    error: Optional[str] = None


def _execute_single(ds_handler, req: ExecutionRequest) -> ExecutionResult:
    """
    Execute a single code snippet. Designed to run in a thread.
    
    Parameters
    ----------
    ds_handler : object
        Dataset handler with execute() method
    req : ExecutionRequest
        Request containing code and metadata
        
    Returns
    -------
    ExecutionResult
        Result of execution including grade and output
    """
    # Recording mode or extraction failure - no execution needed
    if req.is_recording_mode or req.code is None:
        return ExecutionResult(
            task_index=req.task_index,
            code=req.code or "",
            unit_tests="",
            output_content="",
            grade=None,
            error=req.extraction_error
        )
    
    # start = time.time()

    try:
        unit_tests, output_content, grade = ds_handler.execute(
            req.code, req.problem_id
        )

        # elapsed = time.time() - start
    
        return ExecutionResult(
            task_index=req.task_index,
            code=req.code,
            unit_tests=unit_tests,
            output_content=output_content,
            grade=grade
        )
    except Exception as e:
        return ExecutionResult(
            task_index=req.task_index,
            code=req.code,
            unit_tests="",
            output_content="",
            grade=None,
            error=str(e)
        )


def _execute_student_code(ds_handler, code: str, problem_id: str) -> Optional[int]:
    """
    Execute student code to get grade. Designed to run in a thread.
    
    Parameters
    ----------
    ds_handler : object
        Dataset handler with execute() method
    code : str
        Student code to execute
    problem_id : str
        Problem identifier
        
    Returns
    -------
    int or None
        Grade from execution, or None on error
    """
    try:
        _, _, grade = ds_handler.execute(code, problem_id)
        return grade
    except Exception:
        return None


from tqdm import tqdm

from tqdm import tqdm

def parallel_process_outputs(tasks: List['EvaluationTask'], 
                             outputs: List[str], 
                             ds_handler,
                             max_workers: int = None) -> None:
    """
    Process receive_output for multiple tasks in parallel.
    
    Three-phase approach:
    1. Extract code from outputs (fast, sequential)
    2. Execute code in parallel using ThreadPoolExecutor
    3. Update task states with results (parallel with ThreadPoolExecutor)
    
    Parameters
    ----------
    tasks : list of EvaluationTask
        Tasks to process (mix of generation and recording mode)
    outputs : list of str
        Model outputs corresponding to each task
        (empty strings for recording mode tasks)
    ds_handler : object
        Dataset handler with execute() method
    max_workers : int, optional
        Max parallel executions. Default: min(MAX_PARALLEL_WORKERS, cpu_count * 2, len(tasks))
    """
    if not tasks:
        return
        
    if max_workers is None:
        max_workers = min(MAX_PARALLEL_WORKERS, mp.cpu_count() * 2, len(tasks))
    
    # =========================================================================
    # Phase 1: Extract code and prepare execution requests (fast, sequential)
    # =========================================================================
    execution_requests = []
    
    for i, (task, output) in tqdm(enumerate(zip(tasks, outputs)), 
                                   total=len(tasks),
                                   desc="Extracting code from outputs"):
        if task.was_done:
            # Recording mode - no execution needed
            execution_requests.append(ExecutionRequest(
                task_index=i,
                code=None,
                problem_id=task.row.problem_id,
                agent_output="",
                is_recording_mode=True
            ))
        else:
            agent_output = task.pre_thought + output
            try:
                code = task._extract_code(agent_output)
                execution_requests.append(ExecutionRequest(
                    task_index=i,
                    code=code,
                    problem_id=task.row.problem_id,
                    agent_output=agent_output,
                    is_recording_mode=False
                ))
            except Exception as e:
                # Extraction failed - will trigger recording mode
                execution_requests.append(ExecutionRequest(
                    task_index=i,
                    code=None,
                    problem_id=task.row.problem_id,
                    agent_output=agent_output,
                    is_recording_mode=False,
                    extraction_error=str(e)
                ))
    
    # =========================================================================
    # Phase 2: Execute code in parallel (slow, parallelizable)
    # =========================================================================
    results = [None] * len(tasks)
    
    # Only submit tasks that need execution (not recording mode, has code)
    tasks_needing_execution = [
        req for req in execution_requests 
        if not req.is_recording_mode and req.code is not None
    ]
    
    if tasks_needing_execution:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_req = {
                executor.submit(_execute_single, ds_handler, req): req 
                for req in tasks_needing_execution
            }
            
            for future in tqdm(as_completed(future_to_req), 
                              total=len(future_to_req),
                              desc=f"Executing code in parallel ({max_workers} workers)"):
                req = future_to_req[future]
                try:
                    result = future.result()
                    results[result.task_index] = result
                except Exception as e:
                    results[req.task_index] = ExecutionResult(
                        task_index=req.task_index,
                        code=req.code or "",
                        unit_tests="",
                        output_content="",
                        grade=None,
                        error=str(e)
                    )
    
    # Fill in results for recording mode and failed extraction tasks
    for req in tqdm(execution_requests, desc="Filling missing results"):
        if results[req.task_index] is None:
            results[req.task_index] = ExecutionResult(
                task_index=req.task_index,
                code="",
                unit_tests="",
                output_content="",
                grade=None,
                error=req.extraction_error
            )
    
    # =========================================================================
    # Phase 3: Update task states with results in parallel
    # =========================================================================
    print("Updating tasks states and receiving outputs")
    
    def _update_single_task(task, req, result, ds_handler):
        """Helper function to update a single task's state."""
        task._receive_output_with_result(req.agent_output, result, ds_handler)
        return task
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(_update_single_task, task, req, result, ds_handler): i
            for i, (task, req, result) in enumerate(zip(tasks, execution_requests, results))
        }
        
        for future in tqdm(as_completed(future_to_idx),
                          total=len(future_to_idx),
                          desc=f"Updating task states ({max_workers} workers)"):
            try:
                future.result()  # This will raise if there was an exception
            except Exception as e:
                idx = future_to_idx[future]
                print(f"Error updating task {idx}: {e}")
                # Optionally re-raise or handle the error

# =============================================================================
# EVALUATION TASK CLASS
# =============================================================================

class EvaluationTask:
    """
    Manages a single evaluation trajectory for one student's problem attempt.
    
    This class encapsulates all state and logic for evaluating how well a
    model can simulate a specific student's problem-solving process. It
    handles conversation management, code extraction, execution, and
    metrics recording.
    
    Attributes
    ----------
    row : namedtuple
        DataFrame row containing student/problem metadata and messages
    config : object
        Configuration object with task settings and generation kwargs
    agent : object
        Language model agent for generation
    ds_handler : object
        Dataset handler for code execution and grading
    pattern : re.Pattern
        Regex pattern for code extraction (kept for compatibility)
    k : int
        Starting position in trajectory (number of prior submissions)
    lang : str
        Programming language (default: 'python')
    
    State Attributes
    ----------------
    messages : list
        Original conversation with <actual_submission> tags intact
    model_generations : list
        Growing conversation history for model (tags stripped)
    grade : int
        Current model grade (0-100)
    n_iter : int
        Current iteration number
    is_done : bool
        Task completely finished
    was_done : bool
        Model reached 100%, in recording mode
    current_message_index : int
        Position in original messages for student code extraction
    information : list
        Recorded metrics for each step
    """
    
    def __init__(self, row, config, agent, ds_handler, pattern, lang="python", k=2, pre_thought=""):
        """
        Initialize an evaluation task for a specific student trajectory.
        
        Parameters
        ----------
        row : namedtuple
            DataFrame row with fields: user_id, problem_id, messages
        config : object
            Configuration with task.gen_kwargs for generation parameters
        agent : object
            Language model agent with batch_query method
        ds_handler : object
            Handler with execute() and get_summative_feedback_and_grade() methods
        pattern : re.Pattern
            Regex for code extraction (legacy, task uses internal patterns)
        lang : str, optional
            Programming language for code blocks (default: 'python')
        k : int, optional
            Starting position - model sees first k submissions as context
            
        Notes
        -----
        The k parameter determines how much context the model receives:
        - k=2: Model sees problem + 1st submission + feedback, generates 2nd
        - k=4: Model sees problem + 3 submissions + feedback, generates 4th
        """
        self.row = row
        self.config = config
        self.agent = agent
        self.ds_handler = ds_handler
        self.pattern = pattern
        self.k = k
        self.lang = lang
        self.pre_thought=pre_thought

        
        # Escape language name for regex
        lang_escaped = re.escape(self.lang)

        # Pattern for complete Python code with optional thinking block
        # Expected format after prepending pre_thought:
        # <think>\n...thinking...\n</think>\n```python\ncode\n```
        self.match_python = re.compile(
            rf"(?:<actual_submission>\s*\n(?P<actual>.*?)\n</actual_submission>[\s\n]*)?" 
            rf"(?:<think>\s*\n?(?P<think>.*?)\n?</think>[\s\n]*)?"
            rf"(?:```diff\s*\n(?P<thought>.*?)\n```[\s\n]*)?"
            rf"```python\s*\n"
            rf"(?P<code>.*?)"
            rf"\n?```",
            re.DOTALL
        )

        # Pattern for unified diff with optional thinking block
        # Expected format: <think>\n...thinking...\n</think>\n```diff\n...\n```
        self.match_diff = re.compile(
            rf"(?:<think>\s*\n?(?P<think>.*?)\n?</think>[\s\n]*)?" 
            rf"```diff\s*\n"
            rf"(?P<diff>.*?)"
            rf"\n?```",
            re.DOTALL,
        )

        self._prepare_iter()

    def _prepare_iter(self):
        """
        Initialize conversation history and state for iteration.
        
        This method:
        1. Loads the original conversation (with <actual_submission> tags)
        2. Creates model conversation with tags stripped
        3. Sets up the prefilled assistant turn
        4. Initializes all state variables
        
        Notes
        -----
        The <actual_submission> tags are stripped for model input because
        the model was not trained with these tags. They are preserved in
        self.messages for ground truth extraction.
        """
        # Load original conversation (contains <actual_submission> tags for ground truth)
        if isinstance(self.row.messages, str):
            self.messages = eval(self.row.messages)
        else:
            self.messages = deepcopy(self.row.messages)

        # Find indices of all assistant responses (positions 2, 4, 6, ...)
        self.assistant_responses_index = list(range(2, len(self.messages), 2))

        # Validate k is within bounds
        if self.k > len(self.assistant_responses_index):
            raise ValueError(
                f"k={self.k} exceeds available submissions "
                f"({len(self.assistant_responses_index)})"
            )

        # Create model conversation history up to k-th submission
        # CRITICAL: Remove <actual_submission> tags since model wasn't trained with them
        messages_for_model = remove_actual_submission_tags(self.messages)
        
        # Include messages up to and including the k-th assistant response
        cutoff_index = self.assistant_responses_index[self.k - 1] + 1
        self.model_generations = deepcopy(messages_for_model[:cutoff_index])
        
        # Prefill the last assistant turn to guide output format
        self.model_generations[-1]["content"] = self.pre_thought

        # Initialize generation parameters
        self.gen_kwargs = self.config.task.gen_kwargs.toDict()
        
        # Initialize state
        self.grade = 0
        self.n_iter = 1
        self.information = []

        # Task completion flags
        self.is_done = False      # Task completely finished
        self.was_done = False     # Model reached 100%, recording mode

        # Track position in original messages for student code extraction
        self.current_message_index = self.assistant_responses_index[self.k - 1]


    def get_next_prompt(self, tokenizer=None, max_context_length: int = None) -> Optional[List[dict]]:
        """
        Get the next prompt for model generation, with optional context truncation.
        
        [... docstring unchanged ...]
        """
        if self.is_done or self.was_done:
            return None
        
        prompt = deepcopy(self.model_generations)
        
        # =====================================================================
        # Step 1: Clean prior assistant turns - keep ONLY the last code block
        # =====================================================================
        # Pattern to match <think>...</think> blocks
        think_pattern = re.compile(r'<think>.*?</think>\s*', re.DOTALL)
        
        # Pattern to extract code from markdown code blocks
        code_block_pattern = re.compile(r'```(?:python)?\s*\n(.*?)```', re.DOTALL)
        
        # Process all assistant turns except the last one
        for i in range(len(prompt) - 1):  # Exclude last message
            if prompt[i].get('role') == 'assistant':
                original_content = prompt[i]['content']
                
                # First remove <think> blocks
                cleaned_content = think_pattern.sub('', original_content)
                
                # Then extract only the LAST code block
                code_matches = code_block_pattern.findall(cleaned_content)
                
                if code_matches:
                    # Take only the last code block
                    prompt[i]['content'] = code_matches[-1].strip()
                else:
                    # Fallback: if no code blocks, assume entire content is code
                    # (handles cases where model outputs raw code without markdown)
                    prompt[i]['content'] = cleaned_content.strip()
        
        # =====================================================================
        # Step 2: Context length truncation (unchanged)
        # =====================================================================
        if tokenizer is None or max_context_length is None:
            return prompt
        
        # Check current length
        try:
            token_count = len(tokenizer.apply_chat_template(prompt, tokenize=True))
        except Exception as e:
            print(f"Warning: Failed to tokenize prompt for length check: {e}")
            return prompt
        
        if token_count < max_context_length:
            return prompt
        
        # Need to truncate - remove early assistant+user pairs
        # Structure: [system, user(problem), assistant, user, ..., assistant(prefill)]
        # Must keep: system (0), first user (1), last assistant (prefill)
        # Removable: everything from index 2 to second-to-last, in pairs
        
        # Minimum messages: system + problem + prefill = 3
        min_messages = 3
        truncated_turns = 0
        
        while token_count > max_context_length and len(prompt) > min_messages:
            # Remove the message at index 2 (first removable turn after system+problem)
            # This should be an assistant turn (code submission)
            if len(prompt) <= min_messages:
                break
                
            removed_role = prompt[2].get('role', 'unknown')
            prompt.pop(2)
            truncated_turns += 1
            
            # After popping, if index 2 is now a user turn (feedback), remove it too
            # to maintain assistant-user pair removal
            if len(prompt) > min_messages and len(prompt) > 2:
                if prompt[2].get('role') == 'user':
                    prompt.pop(2)
                    truncated_turns += 1
            
            # Recheck length
            try:
                token_count = len(tokenizer.apply_chat_template(prompt, tokenize=True))
            except Exception:
                break
        
        if False and truncated_turns > 0:
            print(f"  Truncated {truncated_turns} messages from context "
                  f"(now {len(prompt)} messages, ~{token_count} tokens)")
                
        return prompt

    def _receive_output_with_result(self, agent_output: str, exec_result: ExecutionResult, ds_handler):
        """
        Update task state with pre-computed execution result.
        
        This method processes the result from parallel execution and updates
        all task state accordingly. It is called from parallel_process_outputs
        after code has been executed in parallel.
        
        Parameters
        ----------
        agent_output : str
            The full agent output (with prefill prepended), or empty for recording mode
        exec_result : ExecutionResult
            Pre-computed execution result from parallel processing
        ds_handler : object
            Dataset handler for feedback generation and student grade extraction
        """
        # =====================================================================
        # Step 1: Handle execution errors or recording mode
        # =====================================================================
        # print("Agent output", agent_output)
        if exec_result.error and not self.was_done:
            print(f"Execution error for {self.row.user_id}/{self.row.problem_id}: {exec_result.error}")
            self.was_done = True
            agent_output = ""
        
        if self.was_done:
            agent_output = ""
            generated_code = ""
            unit_tests = ""
            output_content = ""
            grade = None
        else:
            generated_code = exec_result.code
            unit_tests = exec_result.unit_tests
            output_content = exec_result.output_content
            grade = exec_result.grade

        # =====================================================================
        # Step 2: Extract student's ground truth code
        # =====================================================================
        if self.current_message_index < len(self.messages):
            try:
                # Primary method: Extract from <actual_submission> tags
                student_message = self.messages[self.current_message_index]["content"]
                student_code = extract_code_from_actual_submission(student_message)

                if student_code is None:
                    # Fallback: Extract from code blocks (backwards compatibility)
                    student_code = self._extract_code_from_student(student_message)
                
                student_user_turn = self.messages[self.current_message_index + 1]["content"]
                student_grade = ds_handler.extract_grade(student_user_turn)
                if student_grade is None:
                    # Execute to get grade - this is still sequential per task
                    # but happens after parallel model code execution
                    _, _, student_grade = ds_handler.execute(
                        student_code, self.row.problem_id
                    )
            except Exception as e:
                print(f"Error extracting student code at index {self.current_message_index}: {e}")
                student_code = ""
                student_grade = None
        else:
            # Student trajectory exhausted
            student_code = ""
            student_grade = None

        # Update model's grade
        self.grade = grade

        # =====================================================================
        # Step 3: Update conversation history (generation mode only)
        # =====================================================================
        if not self.was_done:
            # Update the prefilled assistant turn with actual output
            assert self.model_generations[-1]["role"] == "assistant"
            self.model_generations[-1]["content"] = agent_output

            # Generate and append grader feedback
            feedback, _ = ds_handler.get_summative_feedback_and_grade(output_content)
        
            user_turn = {"role": "user", "content": feedback}
            self.model_generations.append(user_turn)

        # =====================================================================
        # Step 4: Record metrics for this step
        # =====================================================================
        prior_code = self.messages[self.current_message_index - 2]["content"]
        prior_code = self._extract_code_from_student(prior_code)
        
        self.information.append({
            "student_id": self.row.user_id,
            "problem_id": self.row.problem_id,
            "successful": self.grade == 100,
            "@k": self.k,
            "step": self.n_iter,
            "code": generated_code,
            "full": agent_output,
            "student_code": student_code,
            "student_grade": student_grade,
            "unit_tests": unit_tests,
            "messages": deepcopy(self.model_generations),
            "prior_code": prior_code,
            "ground_truth_trajectory": deepcopy(self.messages),
            "output_content": output_content,
            "grade": grade,
            "mode": "recording" if self.was_done else "generation",
        })

        # =====================================================================
        # Step 5: Check for student trajectory exhaustion (recording mode)
        # =====================================================================

        # Advance position in original messages
        self.current_message_index += 2

        if self.current_message_index >= len(self.messages):
            self.is_done = True
            return

        # =====================================================================
        # Step 6: Add prefill for next iteration (generation mode only)
        # =====================================================================
        if not self.was_done:
            prefill = {"role": "assistant", "content": self.pre_thought}
            self.model_generations.append(prefill)

        self.n_iter += 1

        # =====================================================================
        # Step 7: Check termination conditions
        # =====================================================================
        
        # Check if model explicitly gave up
        model_gave_up = False
        if not self.was_done and generated_code:
            code_stripped = generated_code.strip()
            # Remove comments to check for exit() call
            code_no_comments = '\n'.join(
                line.split('#')[0].strip() 
                for line in code_stripped.split('\n')
            ).strip()
            if code_no_comments in ['exit()', 'exit', 'quit()', 'quit']:
                # print("Model gave up")
                model_gave_up = True
        
        # Transition to recording mode if model reached maximum grade or gave up
        if (self.grade == 100 or model_gave_up) and not self.was_done:
            self.was_done = True
            """
            if model_gave_up:
                print(f"Task {self.row.user_id}/{self.row.problem_id}: "
                      f"Model gave up at step {self.n_iter - 1}")
            else:
                print(f"Task {self.row.user_id}/{self.row.problem_id}: "
                      f"Model reached 100% at step {self.n_iter - 1}")
            """

        # Check iteration limit
        if self.n_iter > MAX_ITER:
            self.is_done = True

    def receive_output(self, agent_output):
        """
        Process model output and update task state (sequential version).
        
        This is the original sequential implementation, kept for backward
        compatibility. For better performance, use parallel_process_outputs()
        which calls _receive_output_with_result() after parallel execution.
        
        Parameters
        ----------
        agent_output : str
            Raw model output (continuation of prefilled assistant turn),
            or empty string in recording mode
        """
        # =====================================================================
        # Step 1: Prepare agent output based on mode
        # =====================================================================
        if self.was_done:
            # Recording mode - no new model output
            agent_output = ""
        else:
            # Generation mode - prepend the prefill
            agent_output = self.pre_thought + agent_output

        # =====================================================================
        # Step 2: Extract and execute model's code
        # =====================================================================
        if self.was_done:
            # Recording mode - reuse previous values
            generated_code = ""
            unit_tests = ""
            output_content = ""
            grade = None
        else:
            try:
                generated_code = self._extract_code(agent_output)
                unit_tests, output_content, grade = self.ds_handler.execute(
                    generated_code, self.row.problem_id
                )

            except Exception as e:
                print(f"Error extracting/executing model code: {e}")
                self.was_done = True
                # Recording mode - reuse previous values
                generated_code = ""
                unit_tests = ""
                output_content = ""
                grade = None

        # =====================================================================
        # Step 3: Extract student's ground truth code
        # =====================================================================
        if self.current_message_index < len(self.messages):
            try:
                # Primary method: Extract from <actual_submission> tags
                student_message = self.messages[self.current_message_index]["content"]
                student_code = extract_code_from_actual_submission(student_message)

                if student_code is None:
                    # Fallback: Extract from code blocks (backwards compatibility)
                    student_code = self._extract_code_from_student(student_message)
                
                student_user_turn = self.messages[self.current_message_index + 1]["content"]
                student_grade = self.ds_handler.extract_grade(student_user_turn)
                if student_grade is None:
                    _, _, student_grade = self.ds_handler.execute(
                        student_code, self.row.problem_id
                    )
            except Exception as e:
                print(f"Error extracting student code at index {self.current_message_index}: {e}")
                student_code = ""
                student_grade = None
        else:
            # Student trajectory exhausted
            student_code = ""
            student_grade = None

        # Update model's grade
        self.grade = grade

        # =====================================================================
        # Step 4: Update conversation history (generation mode only)
        # =====================================================================
        if not self.was_done:
            # Update the prefilled assistant turn with actual output
            assert self.model_generations[-1]["role"] == "assistant"
            self.model_generations[-1]["content"] = agent_output

            # Generate and append grader feedback
            feedback, _ = self.ds_handler.get_summative_feedback_and_grade(output_content)
        
            user_turn = {"role": "user", "content": feedback}
            self.model_generations.append(user_turn)

        # Advance position in original messages
        self.current_message_index += 2

        # =====================================================================
        # Step 5: Record metrics for this step
        # =====================================================================
        self.information.append({
            "student_id": self.row.user_id,
            "problem_id": self.row.problem_id,
            "successful": self.grade == 100,
            "@k": self.k,
            "step": self.n_iter,
            "code": generated_code,
            "full": agent_output,
            "student_code": student_code,
            "student_grade": student_grade,
            "unit_tests": unit_tests,
            "messages": deepcopy(self.model_generations),
            "output_content": output_content,
            "grade": grade,
            "mode": "recording" if self.was_done else "generation",
        })

        # =====================================================================
        # Step 6: Check for student trajectory exhaustion (recording mode)
        # =====================================================================
        if self.was_done and self.current_message_index >= len(self.messages):
            self.is_done = True
            return

        # =====================================================================
        # Step 7: Add prefill for next iteration (generation mode only)
        # =====================================================================
        if not self.was_done:
            prefill = {"role": "assistant", "content": self.pre_thought}
            self.model_generations.append(prefill)

        self.n_iter += 1

        # =====================================================================
        # Step 8: Check termination conditions
        # =====================================================================
        
        # Check if model explicitly gave up
        model_gave_up = False
        if not self.was_done and generated_code:
            code_stripped = generated_code.strip()
            # Remove comments to check for exit() call
            code_no_comments = '\n'.join(
                line.split('#')[0].strip() 
                for line in code_stripped.split('\n')
            ).strip()
            if code_no_comments in ['exit()', 'exit', 'quit()', 'quit']:
                print("Model gave up")
                model_gave_up = True
        
        # Transition to recording mode if model succeeded or gave up
        if (self.grade == 100 or model_gave_up) and not self.was_done:
            self.was_done = True
            """
            if model_gave_up:
                print(f"Task {self.row.user_id}/{self.row.problem_id}: "
                      f"Model gave up at step {self.n_iter - 1}")
            else:
                print(f"Task {self.row.user_id}/{self.row.problem_id}: "
                      f"Model reached 100% at step {self.n_iter - 1}")
            """

        # Check iteration limit
        if self.n_iter > MAX_ITER:
            self.is_done = True
            
    def _extract_code(self, assistant_turn):
        """
        Extract executable code from an assistant's message.
        
        Handles two formats:
        1. Complete Python code in ```python blocks -> return directly
        2. Unified diff in ```diff blocks -> reconstruct from history
        
        Parameters
        ----------
        assistant_turn : str
            The assistant's message content
            
        Returns
        -------
        str
            Executable Python code
            
        Raises
        ------
        ValueError
            If no valid code format is found
        """

        # Check for complete Python code
        python_match = self.match_python.search(assistant_turn + "```") # added because some models stop too early for some reason
        if python_match:
            generated_code = python_match.group("code")
            if not generated_code:
                raise ValueError('Python code block is empty')
            return generated_code.strip()
        
        # Check for diff format first (more specific)
        diff_match = self.match_diff.search(assistant_turn)
        if diff_match:
            # Create temporary history with current turn for reconstruction
            temp_history = deepcopy(self.model_generations)
            temp_history[-1]["content"] = assistant_turn
            
            try:
                generated_code = reconstruct_program_from_history(temp_history)
                if not generated_code:
                    raise ValueError("Diff reconstruction resulted in empty program")
                return generated_code
            except Exception as e:
                print(f"Error reconstructing from diff: {e}")
                print(f"Assistant turn preview: {assistant_turn[:200]}...")
                raise ValueError(f'Failed to reconstruct program from diff: {e}')
    
        #raise ValueError(
            #'Agent did not generate valid code (neither Python nor diff format)' \
            #f'{assistant_turn}'
        #)
        warn('Agent did not generate valid code (neither Python nor diff format)' \
            f'{assistant_turn}')
        return assistant_turn
    
    def _extract_code_from_student(self, student_turn):
        """
        Extract code from a student's message (fallback method).
        
        This is a fallback for cases where <actual_submission> tags are
        not present. It tries multiple extraction strategies.
        
        Parameters
        ----------
        student_turn : str
            The student message content
            
        Returns
        -------
        str
            Extracted code
            
        Raises
        ------
        ValueError
            If no code can be extracted
            
        Notes
        -----
        This method is deprecated in favor of <actual_submission> tags
        but kept for backwards compatibility with older data formats.
        """
        # Try <actual_submission> tags (shouldn't reach here if tags exist)
        code = extract_code_from_actual_submission(student_turn)
        if code:
            return code
        
        # Try Python code blocks
        python_match = self.match_python.search(student_turn)
        if python_match:
            code = python_match.group("code")
            if code:
                return code.strip()
        
        # Try generic code blocks
        generic_match = re.search(
            r'```[a-zA-Z0-9_+-]*\s*\n(.*?)\n?```', 
            student_turn, 
            re.DOTALL
        )
        if generic_match:
            return generic_match.group(1).strip()
        
        raise ValueError('Could not extract student code')
    

    def get_results(self):
        """
        Return all recorded metrics as a DataFrame.
        
        Returns
        -------
        pd.DataFrame
            One row per iteration with columns:
            - student_id, problem_id: Identifiers
            - @k: Starting position
            - step: Iteration number
            - code: Model's generated code
            - full: Complete model output
            - student_code: Ground truth student code
            - student_grade: Student's grade at this position
            - unit_tests: Test results
            - messages: Conversation history snapshot
            - output_content: Raw execution output
            - grade: Model's grade
            - mode: 'generation' or 'recording'
        """
        return pd.DataFrame(self.information)


# =============================================================================
# MAIN EVALUATION CLASS
# =============================================================================

class Evaluation(Experiment):
    """
    Orchestrates batch evaluation of multiple student trajectories.
    
    This class manages the parallel evaluation of many student problem-solving
    trajectories using vLLM batching for efficiency. It creates an EvaluationTask
    for each (student, problem, k) tuple and processes them in batches.
    
    Key Features
    ------------
    - Parallel batch processing via vLLM
    - Parallel code execution via ThreadPoolExecutor
    - Continues recording student submissions after model reaches 100%
    - Saves intermediate results after each batch
    - Handles both complete Python code and diff-based outputs
    - Configurable starting position (k) and sample size
    
    Attributes
    ----------
    config : object
        Configuration object with task settings
    test_run : bool
        Whether this is a test run
    pattern : re.Pattern
        Code extraction pattern (legacy compatibility)
    """

    def __init__(self, config, test_run, lazy_load=False, is_training=False):
        """
        Initialize evaluation with configuration.
        
        Parameters
        ----------
        config : object
            Configuration object with:
            - task.gen_kwargs: Generation parameters
            - task.min_k (optional): Minimum k value (default: 2)
            - task.eval_sample_size (optional): Max students to evaluate
        test_run : bool
            If True, run in test mode
        lazy_load : bool, optional
            Whether to lazy load data
        is_training : bool, optional
            Whether this is training mode
        """
        super().__init__(config, test_run, lazy_load, is_training)

        # Pattern for code extraction (kept for backward compatibility)
        self.pattern = re.compile(
            r"(?:<think>.*?</think>\s*)?"        # optional thinking block
            r"```(?:[a-zA-Z0-9_+-]*)\s*\n"       # opening fence with language
            r"(.*?)"                             # capture code content
            r"\n?```",                           # closing fence
            re.DOTALL,
        )

    def run(self):
        """
        Execute the main evaluation loop with parallel code execution.
        
        Algorithm
        ---------
        1. Create EvaluationTask for each (row, k) combination
        2. While tasks remain:
           a. Separate tasks into generation vs. recording mode
           b. Batch prompts from generation tasks to vLLM
           c. Execute all code in parallel using ThreadPoolExecutor
           d. Update task states with execution results
           e. Save intermediate results
           f. Remove completed tasks
        3. Save final results to CSV
        
        Results are saved to self.results_save_path.
        """
        dataframe = []  # Accumulates DataFrames from completed tasks

        # Get configuration parameters with defaults
        min_k = DEFAULT_MIN_K
        eval_sample_size = DEFAULT_EVAL_SAMPLE_SIZE

        # =====================================================================
        # Step 1: Initialize tasks
        # =====================================================================
        all_tasks = []
        sample_size = min(len(self.dataframe), eval_sample_size)
        sampled_rows = self.dataframe.sample(sample_size, random_state=42)
        sampled_rows.to_csv(self.dataframe_save_path, index=False)
        print("Saved dataframe to the save path", self.dataframe_save_path)
        
        print(f"Creating tasks for {sample_size} students with min_k={min_k}")
        
        for row in tqdm(
                sampled_rows.itertuples(index=False),
                total=len(sampled_rows),
                desc="Creating evaluation tasks",
            ):            # Parse messages
            if isinstance(row.messages, str):
                messages = eval(row.messages)
            else:
                messages = row.messages

            # Find all assistant response positions
            assistant_indices = [
                i for i, m in enumerate(messages) 
                if m['role'] == 'assistant'
            ]
            num_submissions = len(assistant_indices)
            
            # Create tasks for each valid k value
            # k must be >= min_k to ensure sufficient context
            # k must be <= num_submissions to have a target
            for k in range(min_k, num_submissions + 1):
                task = EvaluationTask(
                        row=row,
                        config=self.config,
                        agent=self.agent,
                        ds_handler=self.ds_handler,
                        pattern=self.pattern,
                        lang=self.ds_handler.lang,
                        k=k,
                        pre_thought=self.config.task.pre_thought
                    )
                all_tasks.append(task)
                try:
                    pass
                except Exception as e:
                    print(f"Failed to create task for {row.user_id}/{row.problem_id} "
                          f"k={k}: {e}")
                    continue

        print(f"Created {len(all_tasks)} total tasks")
        
        gen_kwargs = self.config.task.gen_kwargs.toDict()
        active_tasks = [t for t in all_tasks if not t.is_done]

        # =====================================================================
        # Step 2: Process batches until all tasks complete
        # =====================================================================
        batch_num = 0
        while active_tasks:
            batch_num += 1
            print(f"\n=== Batch {batch_num}: {len(active_tasks)} active tasks ===")

            # Separate tasks by mode
            tasks_needing_generation = [t for t in active_tasks if not t.was_done]
            tasks_in_recording_mode = [t for t in active_tasks if t.was_done]
            
            print(f"  Generation mode: {len(tasks_needing_generation)}")
            print(f"  Recording mode: {len(tasks_in_recording_mode)}")

            # Collect all tasks and outputs for parallel processing
            all_batch_tasks = []
            all_batch_outputs = []

            # Get tokenizer and max context length for truncation
            tokenizer = getattr(self.agent, 'tokenizer', None)
            max_context_length = gen_kwargs.get("truncate_prompt_tokens", None)
            #max_len_model = self.agent.config.from_pretrained_kwargs.max_seq_length
            #max_context_length = min(max_context_length, max_len_model)

            # Process generation tasks
            if tasks_needing_generation:
                valid_tasks = []
                prompts = []
                
                for task in tasks_needing_generation:
                    # get_next_prompt now handles truncation internally
                    prompt = task.get_next_prompt(
                        tokenizer=tokenizer,
                        max_context_length=max_context_length
                    )
                    
                    if prompt is None:
                        continue
                    
                    # After truncation, check if even minimal prompt still exceeds limit
                    # This handles edge cases where problem description alone is too long
                    if tokenizer is not None and max_context_length is not None:
                        try:
                            token_count = len(tokenizer.apply_chat_template(
                                prompt, tokenize=True
                            ))
                            if token_count >= max_context_length:
                                print(f"Task {task.row.user_id}/{task.row.problem_id}: "
                                    f"Minimal context still exceeds limit ({token_count} tokens) - marking done")
                                task.is_done = True
                                continue
                        except Exception as e:
                            print(f"Warning: Failed to check token count: {e}")
                    
                    # Only include tasks that pass the length check
                    valid_tasks.append(task)
                    prompts.append(prompt)
                
                # Batch query only for valid tasks that passed length check
                if valid_tasks:
                    outputs = self.agent.batch_query(prompts, gen_kwargs, cfm=True)
                    all_batch_tasks.extend(valid_tasks)
                    all_batch_outputs.extend(outputs)

            # Add recording mode tasks (empty outputs)
            all_batch_tasks.extend(tasks_in_recording_mode)
            all_batch_outputs.extend([""] * len(tasks_in_recording_mode))

            # ⚡ PARALLEL EXECUTION - Process all outputs concurrently
            if all_batch_tasks:
                print(f"  Processing {len(all_batch_tasks)} tasks in parallel...")
                parallel_process_outputs(
                    all_batch_tasks, 
                    all_batch_outputs, 
                    self.ds_handler,
                    max_workers=min(MAX_PARALLEL_WORKERS, mp.cpu_count() * 2, len(all_batch_tasks))
                )

            # Collect completed tasks
            for task in all_batch_tasks:
                if task.is_done:
                    dataframe.append(task.get_results())

            # Save intermediate results
            if dataframe:
                df = pd.concat(dataframe, axis=0, ignore_index=True)
                df.to_csv(self.results_save_path, index=False)
                print(f"  Saved {len(df)} rows to {self.results_save_path}")

            # Remove completed tasks
            active_tasks = [t for t in active_tasks if not t.is_done]

        # =====================================================================
        # Step 3: Final save
        # =====================================================================
        if dataframe:
            final_df = pd.concat(dataframe, axis=0, ignore_index=True)
            final_df.to_csv(self.results_save_path, index=False)
            print(f"\n=== Evaluation Complete ===")
            print(f"Total rows: {len(final_df)}")
            print(f"Saved to: {self.results_save_path}")
        else:
            print("Warning: No results collected")