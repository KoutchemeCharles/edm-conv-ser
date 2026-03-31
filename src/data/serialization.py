import difflib
import tokenize
import io
import math 
import re
from difflib import SequenceMatcher
from src.utils.normalization import robust_normalize

def tokenize_python(code_str):
    """
    Tokenize Python code using Python's built-in tokenizer.
    
    Args:
        code_str: Python source code as string
        
    Returns:
        List of token strings, or None if tokenization fails
    """
    try:
        tokens = []
        readline = io.StringIO(code_str).readline
        for tok in tokenize.generate_tokens(readline):
            # Skip whitespace and metadata tokens
            if tok.type not in (tokenize.ENCODING, tokenize.NEWLINE, 
                               tokenize.NL, tokenize.INDENT, 
                               tokenize.DEDENT, tokenize.ENDMARKER):
                tokens.append(tok.string)
        return tokens
    except (tokenize.TokenError, IndentationError):
        # Code has syntax errors - cannot tokenize
        return None


def tokenize_simple(code_str):
    """
    Simple regex-based tokenization fallback for syntactically invalid code.
    
    Args:
        code_str: Source code as string
        
    Returns:
        List of word-like tokens (always succeeds)
    """
    return re.findall(r'\w+', code_str)


def compute_token_similarity(prev_code, curr_code):
    """
    Compute similarity ratio between two code strings based on their tokens.
    Uses Python's tokenizer if both compile, otherwise falls back to simple tokenization.
    
    Args:
        prev_code: Previous code version
        curr_code: Current code version
        
    Returns:
        Float between 0 (completely different) and 1 (identical)
    """
    # Try Python tokenizer first (more accurate)
    prev_tokens = tokenize_python(prev_code)
    curr_tokens = tokenize_python(curr_code)
    
    # Fall back to simple tokenization if either fails
    if prev_tokens is None or curr_tokens is None:
        prev_tokens = tokenize_simple(prev_code)
        curr_tokens = tokenize_simple(curr_code)
    
    # Compute similarity using sequence matching
    return SequenceMatcher(None, prev_tokens, curr_tokens).ratio()


def generate_unified_diff(prev_code, curr_code, submission_number):
    """
    Generate a unified diff between two code versions.
    
    Args:
        prev_code: Previous code version
        curr_code: Current code version
        submission_number: Sequential submission number for diff headers
        
    Returns:
        Formatted diff string (without the file header lines)
    """
    full_diff = "\n".join(difflib.unified_diff(
        prev_code.splitlines(),
        curr_code.splitlines(),
        fromfile=f"submission_{submission_number}",
        tofile=f"submission_{submission_number + 1}",
        lineterm="",
        n=0,  # No unchanged context lines
    ))
    
    # Remove the first two lines (file headers: --- and +++)
    return "\n".join(full_diff.split("\n")[2:])


def count_diff_hunks(diff_text: str) -> int:
    """Count the number of hunks in a unified diff."""
    return diff_text.count("@@") // 2  # Each hunk has one @@ line (start only in our format)


def compute_diff_change_ratio(prev_code: str, diff_text: str) -> float:
    """
    Compute the ratio of changed lines to original code lines.
    
    Returns:
        Ratio of changed lines (additions + deletions) to original line count.
        Returns 1.0 if original code is empty.
    """
    prev_lines = prev_code.strip().splitlines()
    if not prev_lines:
        return 1.0
    
    changed_lines = 0
    for line in diff_text.splitlines():
        if line.startswith('+') or line.startswith('-'):
            # Exclude hunk headers (they also start with special chars but contain @@)
            if not line.startswith('+++') and not line.startswith('---'):
                changed_lines += 1
    
    return changed_lines / len(prev_lines)


def should_use_single_block_diff(prev_code: str, curr_code: str, submission_number: int = 0) -> bool:
    """
    Determine if a diff should be used based on the one_block strategy.
    
    Returns True if:
    - The diff contains exactly one contiguous hunk, AND
    - The changed lines are less than 50% of the original code
    """
    diff = generate_unified_diff(prev_code, curr_code, submission_number)
    
    num_hunks = count_diff_hunks(diff)
    if num_hunks != 1:
        return False
    
    change_ratio = compute_diff_change_ratio(prev_code, diff)
    return change_ratio < 0.5


def format_code_as_assistant_payload(
    prev_code,
    curr_code,
    lang,
    format_type,
    submission_number=0,
    changed_grade=False,
    strategy="one_block",
):
    """
    Format code submission according to the specified representation format.

    Args:
        prev_code: Previous code version (empty string for first submission)
        curr_code: Current code version
        lang: Programming language identifier (e.g., 'python')
        format_type: One of 'code' or 'dual'
        submission_number: Sequential submission number for diff headers
        strategy: Strategy for dual format selection.
                  - 'one_block': diff when single hunk and <50% change (default)
                  - 'content_diff': switch based on token similarity
                  - 'progression': switch based on grade change

    Returns:
        Formatted string to be used as assistant message content
    """
    # Embed actual submission for evaluation/reconstruction purposes
    ground_truth = (
        f"<actual_submission>\n{curr_code.strip()}\n</actual_submission>\n\n"
    )

    # --- CODE ONLY -----------------------------------------------------------
    if format_type == "code":
        return ground_truth + f"```{lang}\n{curr_code.strip()}\n```"

    # --- DUAL MODE -----------------------------------------------------------
    elif format_type == "dual":
        # First submission is always full code
        if not prev_code:
            return ground_truth + f"```{lang}\n{curr_code.strip()}\n```"

        # Strategy 1: legacy content-based switching
        if strategy == "content_diff":
            similarity = compute_token_similarity(prev_code, curr_code)
            if similarity > 0.5:
                diff = generate_unified_diff(prev_code, curr_code, submission_number)
                return ground_truth + f"```diff\n{diff}\n```"
            else:
                return ground_truth + f"```{lang}\n{curr_code.strip()}\n```"

        # Strategy 2: progression / state-based switching
        elif strategy == "progression":
            if changed_grade:
                return ground_truth + f"```{lang}\n{curr_code.strip()}\n```"
            else:
                diff = generate_unified_diff(prev_code, curr_code, submission_number)
                return ground_truth + f"```diff\n{diff}\n```"

        # Strategy 3: one_block - single hunk and <50% change
        elif strategy == "one_block":
            diff = generate_unified_diff(prev_code, curr_code, submission_number)
            num_hunks = count_diff_hunks(diff)
            change_ratio = compute_diff_change_ratio(prev_code, diff)

            if num_hunks == 1 and change_ratio < 0.5:
                return ground_truth + f"```diff\n{diff}\n```"
            else:
                return ground_truth + f"```{lang}\n{curr_code.strip()}\n```"

        else:
            raise ValueError(f"Unknown dual-format strategy: {strategy}")

    else:
        raise ValueError(f"Unknown format type: {format_type}")


def create_exit_message(lang, final_grade):
    """
    Create the final exit message for incomplete trajectories.
    
    Args:
        lang: Programming language identifier
        final_grade: Student's final grade (not 100)
        
    Returns:
        Tuple of (assistant_message, user_message)
    """
    assistant_msg = {"role": "assistant", "content": f"```{lang}\nexit()\n```"}
    user_msg = {"role": "user", "content": f"-- GRADE: {final_grade}%"}
    return assistant_msg, user_msg


def conversationalize_improvements(sub_df, messages, lang, format, feedback_level, get_all):
    """
    Convert student submission trajectory into conversational format.
    
    This function transforms a sequence of student code submissions into a dialogue
    between a student (assistant) and a learning environment (user). Each submission
    is represented according to the specified format (full code, diff, or dual), and
    is followed by feedback from the automated grading system.
    
    Args:
        sub_df: DataFrame with student submissions, containing columns:
                - code: the submitted code
                - type: submission type (should not be 'TEST')
                - grade: numerical grade (0-100, or -1 for compilation error)
                - output_content: feedback from grading system
        messages: List to append conversational turns to
        lang: Programming language identifier (e.g., 'python')
        format: Code representation format - one of:
                - 'code': always full code
                - 'dual': diffs for minor edits (<50% changed), full code for major rewrites
        feedback_level: Boolean indicating whether to include unit test feedback
        
    Returns:
        Updated messages list with conversational trajectory
    """
    # Track state across submissions
    prev_code = ""
    prev_grade = -1
    changed_grade = False

    # Process each submission in chronological order
    for step, row in sub_df.iterrows():
        code = (row.get("code") or "").strip()
        submission_type = row.get("type")
        grade = row.get("grade")
        
        # Validate grade exists
        if grade is None:
            print("Warning: None grade encountered!")
        
        # Sanity check: we should only see actual submissions, not test runs
        assert submission_type != 'TEST', f"Unexpected TEST type in submission"

        # Skip duplicate submissions (same AST representation)
        ncode = robust_normalize(code)
        nprev = robust_normalize(prev_code)
        if ncode == nprev:
            continue

        # Format code according to specified representation
        submission_number = len(messages) // 2  # Track submission count
        assistant_payload = format_code_as_assistant_payload(
            prev_code, code, lang, format, submission_number, changed_grade  # Pass the number directly
        )
        
        # Add student's code submission
        messages.append({"role": "assistant", "content": assistant_payload})
        
        summative_feedback = row.get("summative_feedback", "")
        parsed_grade = row.get("parsed_grade", "")
        output_content = row.get("output_content", None)

        if output_content is None:
            output_content, summative_feedback, parsed_grade = get_all(row)
        else:
            raise ValueError("Output content should be precomputed in dataframe for efficiency")
        
        # Validate parsed grade matches recorded grade
        if (not math.isnan(row.grade)) and (parsed_grade != row.grade):
            return []
        
        # Generate environment feedback
        if not feedback_level:
            # No detailed feedback - only indicate success/failure
            if grade == 100:
                summative_feedback = "Correct!"
            else:
                summative_feedback = "You will not be provided unit test feedback. Resubmit your program.\n"

        user_turn = f"<actual_grade>{grade}</actual_grade>\n" + summative_feedback
        
        # Add environment's feedback
        messages.append({"role": "user", "content": summative_feedback})

        changed_grade = (prev_grade != -1) and (prev_grade != grade)

        # Update state for next iteration
        prev_code = code
        prev_grade = grade

        # Stop if student reached correct solution
        if grade == 100:
            break

    return messages