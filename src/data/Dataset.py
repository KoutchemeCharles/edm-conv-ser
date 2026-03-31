"""
Abstract dataset base class and conversation dataframe builders.

Provides:
  - :class:`Dataset` — base class that concrete dataset handlers (e.g.
    :class:`~src.data.falcon.FalconCode.FalconCode`) extend.
  - :func:`build_conversations_dataframe` — sequential builder (reference
    implementation).
  - :func:`build_conversations_dataframe_parallel` — parallel builder using a
    thread pool (used by default in production).
  - :func:`remove_mad_outliers` — trajectory-length outlier filter.
"""

from tqdm import tqdm
import pandas as pd
from src.data.Student import Student
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from src.utils.core import claim_memory


class Dataset(object):
    """
    Base class for datasets. This class is intended to be extended by specific dataset implementations.
    It provides a common interface for loading and processing datasets.

    Expected columns:

    problem_id: the unique identifier of the specific programming exercise.
    user_id: the unique identifier for each student 

    """

    def __init__(self, config):
        self.config = config



def build_conversations_dataframe(
    exec_df: pd.DataFrame,
    student_config,
    get_all
):
    """
    Build a dataframe of conversationalized student trajectories (sequential).

    Groups execution logs by student, instantiates one :class:`Student` per
    student, and concatenates all resulting conversation dataframes.  This is
    the sequential reference implementation; prefer
    :func:`build_conversations_dataframe_parallel` for production use.

    Args:
        exec_df (pd.DataFrame): Preprocessed execution-level dataframe with a
            ``user`` column.
        student_config: Config DotMap forwarded to the ``Student`` constructor
            (controls code format, feedback level, etc.).
        get_all (callable): Function ``(row) -> (output_content, summative_feedback,
            grade)`` that executes a submission and returns grader output.

    Returns:
        pd.DataFrame: Concatenated trajectory dataframe with one row per
        (student, problem) pair and a ``user_id`` column added.
    """

    # Group execution logs by user (preserve original ordering)
    exec_groups = dict(tuple(exec_df.groupby("user", sort=False)))
    user_ids = sorted(exec_groups.keys())

    rows = []

    for uid in tqdm(user_ids):
        user_exec_df = exec_groups[uid]
        # Build student object
        student = Student(
            uid,
            user_exec_df,
            student_config,
            get_all
        )

        # Get conversations for this user
        user_df = student.get_conversations()

        if user_df is None or len(user_df) == 0:
            print("Skipping user dataframe")
            continue

        # Attach metadata
        user_df = user_df.copy()
        user_df["user_id"] = uid

        rows.append(user_df)

    if not rows:
        return pd.DataFrame()
    
    return pd.concat(rows, axis=0, ignore_index=True)




def _process_single_user(uid, exec_groups, student_config, get_all):
    """
    Worker function to process a single user's execution logs.
    
    Args:
        uid: User ID to process
        exec_groups: Dictionary of user execution dataframes
        student_config: Configuration for Student instantiation
        get_all: Parameter for Student
    
    Returns:
        pd.DataFrame or None: User conversations with metadata, or None if skipped
    """
    try:
        user_exec_df = exec_groups[uid]
        
        # Build student object
        student = Student(
            uid,
            user_exec_df,
            student_config,
            get_all
        )
        
        # Get conversations for this user
        user_df = student.get_conversations()
        
        if user_df is None or len(user_df) == 0:
            return None
        
        # Attach metadata
        user_df = user_df.copy()
        user_df["user_id"] = uid
        
        claim_memory()

        return user_df
        
    except Exception as e:
        print(f"Error processing user {uid}: {e}")
        return None


def build_conversations_dataframe_parallel(
    exec_df: pd.DataFrame,
    student_config,
    get_all,
    max_workers=None
):
    """
    Build a dataframe of conversationalized student trajectories (parallel).

    Equivalent to :func:`build_conversations_dataframe` but processes students
    concurrently using a ``ThreadPoolExecutor``.  Errors in individual student
    processing are caught and logged without aborting the whole run.

    Args:
        exec_df (pd.DataFrame): Preprocessed execution-level dataframe with a
            ``user`` column.
        student_config: Config DotMap forwarded to the ``Student`` constructor.
        get_all (callable): Function ``(row) -> (output_content, summative_feedback,
            grade)`` that executes a submission and returns grader output.
        max_workers (int, optional): Thread pool size.  Defaults to
            ``min(32, cpu_count + 4)`` (Python ``ThreadPoolExecutor`` default).

    Returns:
        pd.DataFrame: Concatenated trajectory dataframe with one row per
        (student, problem) pair and a ``user_id`` column added.
    """
    
    # Group execution logs by user (preserve original ordering)
    exec_groups = dict(tuple(exec_df.groupby("user", sort=False)))
    user_ids = sorted(exec_groups.keys())
    
    # Create partial function with fixed arguments
    worker_fn = partial(
        _process_single_user,
        exec_groups=exec_groups,
        student_config=student_config,
        get_all=get_all
    )
    
    rows = []
    
    # Process users in parallel with thread pool
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # executor.map preserves order and shows progress with tqdm
        results = list(tqdm(
            executor.map(worker_fn, user_ids),
            total=len(user_ids),
            desc="Processing users"
        ))
    
    # Filter out None results
    rows = [df for df in results if df is not None]
    
    if not rows:
        return pd.DataFrame()
    
    return pd.concat(rows, axis=0, ignore_index=True)


def remove_mad_outliers(df, threshold=3.5):
    """Remove outliers using Median Absolute Deviation (MAD) per problem."""
    
    df["traj_len"] = df.messages.apply(lambda m: len(m))
    
    def is_outlier(group):
        median = group["traj_len"].median()
        mad = (group["traj_len"] - median).abs().median()
        
        if mad == 0:
            # All values identical, keep all
            return pd.Series(False, index=group.index)
        
        # Modified z-score
        modified_z = 0.6745 * (group["traj_len"] - median) / mad
        
        return modified_z.abs() > threshold
    
    # Mark outliers per problem
    outlier_mask = df.groupby("problem_id", group_keys=False).apply(is_outlier)
    
    # Keep only non-outliers
    return df[~outlier_mask.values]  # Use .values to get the boolean array