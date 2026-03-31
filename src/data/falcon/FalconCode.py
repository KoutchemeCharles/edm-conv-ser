"""
FalconCode dataset handler.

Loads the ``koutch/falcon_code`` dataset from HuggingFace Hub, applies
filtering rules from the experiment config, runs the student code against
unit tests, and calls the conversationalization pipeline to produce the
training/evaluation dataframe.

The resulting dataframe has one row per (student, problem) trajectory that
satisfies the quality filters. Each row contains:
  - ``messages``: the full multi-turn conversation (list of dicts)
  - ``problem_id``: exercise identifier
  - ``user_id``: student identifier
  - ``traj_len``: number of messages in the trajectory
"""

import os
import re
import glob
import tempfile
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset, concatenate_datasets
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.data.Dataset import Dataset, build_conversations_dataframe, build_conversations_dataframe_parallel
from src.data.falcon.execution import run_execution
from src.data.falcon.execution import get_unit_test_score
from src.utils.core import claim_memory

import shutil
import tempfile

# Temporary workspace for code execution (shared across all runs in the process)
WORK_DIR = os.path.join(tempfile.gettempdir(), "student_exec")
os.makedirs(WORK_DIR, exist_ok=True)


class FalconCode(Dataset):
    """
    Dataset handler for the FalconCode programming exercise dataset.

    Wraps the ``koutch/falcon_code`` HuggingFace dataset. On construction it
    loads the requested split; ``get_split()`` triggers the full preprocessing
    pipeline (filtering, code execution, conversationalization).

    Attributes:
        config: Dataset config DotMap (split, preprocessing flags, student format).
        lang (str): Programming language — always ``"python"`` for FalconCode.
        exec_df (pd.DataFrame): Raw execution log dataframe from HuggingFace.
        dataframe (pd.DataFrame): Processed trajectory dataframe (set after
            calling ``get_split()``).
    """

    splits = ["train", "val", "test"]

    def __init__(self, config) -> None:
        """
        Load the raw FalconCode execution logs from HuggingFace Hub.

        Args:
            config: Dataset config DotMap with at minimum:
                - ``split``: one of ``"train"``, ``"val"``, ``"test"``, or ``"all"``
                - ``preprocessing``: filtering flags (``remove_exams``, etc.)
                - ``student``: student format and feedback level settings

        Raises:
            ValueError: If ``config.split`` is not a recognised value.
        """
        self.config = config
        self.lang = "python"

        self.code_block_full_pattern = re.compile(
            rf"(```{self.lang}\s*\n.*?\n?```)",
            re.DOTALL,
        )

        dd = load_dataset("koutch/falcon_code", "submitted")
        if self.config.split in self.splits:
            dataset = dd[self.config.split]
        elif self.config.split == "all":
            dataset = concatenate_datasets(dd.values())
        else:
            raise ValueError(f"Unknown split {self.config.split}")

        self.exec_df = dataset.to_pandas()

    def get_split(self):
        """
        Run the full preprocessing pipeline and return the trajectory dataframe.

        Returns:
            pd.DataFrame: Preprocessed and conversationalized trajectory dataframe.
        """
        dataframe = self.preprocess()
        return dataframe

    def preprocess(self):
        """
        Filter raw execution logs and build the conversationalized dataframe.

        Pipeline steps:
          1. Apply structural filters (``_preprocess``): remove redacted
             submissions, select one problem per group, filter to lab exercises.
          2. Execute each student's code against unit tests in parallel and
             collect grader output via ``build_conversations_dataframe_parallel``.
          3. Keep only trajectories with more than 6 messages (≥3 submissions).
          4. Remove trajectory-length outliers per problem using MAD filtering.

        Returns:
            pd.DataFrame: Trajectory dataframe stored in ``self.dataframe``.
        """
        dataframe = self.exec_df

        dataframe = self._preprocess(dataframe)
        claim_memory()

        def get_all(row):
            """Execute ``row.code`` on its test cases and return (output, feedback, grade)."""
            _, oc, grade = self.execute(row.code, row["problem_id"])
            sf, _ = self.get_summative_feedback_and_grade(oc)
            return oc, sf, grade

        dataframe = build_conversations_dataframe_parallel(dataframe, self.config.student, get_all)
        # Require at least 3 full assistant turns (6 messages: system + problem + ≥2 rounds)
        dataframe = dataframe[dataframe.messages.apply(len) > 6].reset_index(drop=True)
        # Remove unusually long trajectories (outliers inflate loss and memory usage)
        dataframe = remove_mad_outliers(dataframe)

        self.dataframe = dataframe

        if not self.config.student.feedback_level:
            print("NOTE: The handler will not be returning summative feedback")

        return self.dataframe

    def _preprocess(self, dataframe):
        """
        Apply structural filters to the raw execution log dataframe.

        Transformations applied (in order):
          1. Derive ``problem_group`` from the prefix of ``problem_id``
             (e.g. ``"lab03_task1"`` → ``"lab03"``).
          2. Rename columns to the canonical schema
             (``source_code`` → ``code``, ``score`` → ``grade``, etc.).
          3. Keep only (problem, student) pairs where the student's final
             submission reached grade ≥ 50 (partially correct solutions).
          4. Remove rows whose problem description or test case refers to a
             ``.csv`` file (data-science exercises outside scope).
          5. Remove redacted submissions and non-lab exercises.
          6. Optionally remove exam submissions (controlled by config).
          7. Select the most common ``problem_id`` variant per ``problem_group``
             so each group is represented by a single canonical exercise.

        Args:
            dataframe (pd.DataFrame): Raw execution log dataframe.

        Returns:
            pd.DataFrame: Filtered dataframe ready for conversationalization.
        """
        dataframe["problem_group"] = dataframe.problem_id.apply(
            lambda assignment: assignment.split("_")[0]
        )

        rename_columns = {
            "prompt": "problem_description",
            "source_code": "code",
            "student_id": "user",
            "score": "grade"
        }
        dataframe.rename(columns=rename_columns, inplace=True)

        def _solved(g: pd.DataFrame) -> bool:
            sub = g.sort_values("timestamp")
            last_grade = sub["grade"].iloc[-1]
            return last_grade >= 50

        groups = dataframe.groupby(["problem_id", "user"], as_index=False)
        dataframe = groups[dataframe.columns].filter(_solved)
        # Exclude exercises that require reading external CSV files
        dataframe = dataframe[[".csv" not in p for p in dataframe.problem_description]]
        dataframe = dataframe[[".csv" not in t for t in dataframe.testcase]]

        print("Number of users", dataframe.user.nunique())
        mask = (
            (~dataframe.redacted.astype(bool)) &
            (dataframe["type"] == "lab") &
            (dataframe.user.isin(sorted(dataframe.user.unique())))
        )

        if self.config.preprocessing.remove_exams:
            mask = mask & (~dataframe.exam.astype(bool))

        dataframe = dataframe.loc[mask]

        # One canonical exercise per problem group (most common problem_id wins)
        selected_exercises = dataframe.groupby("problem_group")["problem_id"].agg(
            lambda x: x.mode()[0]
        )
        print(f"Selected {len(selected_exercises)} exercises from problem groups")
        dataframe = dataframe[dataframe["problem_id"].isin(selected_exercises.values)]

        return dataframe

    def get_summative_feedback_and_grade(self, output_content):
        """
        Extract summative feedback and grade from raw grader output.

        The grader output contains detailed per-test feedback followed by a
        summative section starting with the word "Feedback". This method
        returns the full output as feedback (the split at "feedback" is kept
        as a no-op for now) and parses the ``Unit Test Returned:`` score line.

        Args:
            output_content (str): Raw string output from the grader subprocess.

        Returns:
            tuple[str, float]: ``(summative_feedback, grade)`` where ``grade``
            is a float in [0, 100].
        """
        i = output_content.lower().find("feedback")
        summative_feedback = output_content[i:] if i != -1 else output_content
        return output_content, get_unit_test_score(output_content)

    def extract_grade(self, feedback_content: str) -> int | None:
        """
        Parse the numerical grade from summative feedback.

        Used by DPO and GRPO to assign rewards based on solution quality.
        Returns ``None`` when ``config.student.feedback_level`` is False
        (i.e. when the student does not receive unit test feedback).

        Args:
            feedback_content (str): Feedback string from the grader.

        Returns:
            float or None: Grade in [0, 100], or ``None`` if feedback is
            disabled in the config.
        """
        if not self.config.student.feedback_level:
            return None
        return get_unit_test_score(feedback_content)

    def execute(self, code, problem_id):
        """
        Execute student code against all test cases for a given exercise.

        Retrieves the test case for ``problem_id`` from the raw dataframe,
        runs the code in an isolated subprocess via
        :func:`~src.data.falcon.execution.run_execution`, and returns the
        grader output.

        Args:
            code (str): Student Python program to execute.
            problem_id (str): Exercise identifier used to look up test cases.

        Returns:
            tuple: ``(None, output_content, grade)`` where:
            - ``output_content`` (str): Full grader output string.
            - ``grade`` (int): Integer grade in [0, 100].
        """
        df = self.exec_df[self.exec_df.problem_id == problem_id]
        testcases = [t for t in df.testcase.unique() if pd.notna(t) and t][0]

        info = {
            "problem_id": problem_id,
            "max_score": 100,
            "code": code,
            "testcase": testcases
        }
        result = run_execution(info, 5)
        grade = result["grade"]
        output_content = result["output"]
        unit_tests_results = None

        return unit_tests_results, output_content, int(grade)


# =============================================================================
# Standalone execution utilities (not used by FalconCode directly)
# =============================================================================

def execute_programs(dataframe):
    """
    Execute all programs in a dataframe sequentially and annotate results.

    This is a simple streaming version kept for reference. For bulk execution,
    prefer ``execute_programs_parallel`` which is significantly faster.

    Args:
        dataframe (pd.DataFrame): Must contain ``problem_id``, ``code``,
            ``testcase``, and ``max_score`` columns.

    Returns:
        pd.DataFrame: Input dataframe with three new columns added:
        ``output_content``, ``summative_feedback``, and ``parsed_grade``.
    """
    output_content = []
    parsed_grades = []

    for idx, row in tqdm(dataframe.iterrows(), total=len(dataframe), desc="Running programs"):
        result = run_execution(row, 5)
        output_content.append(result["output"])
        parsed_grades.append(int(result["grade"]))

    dataframe["output_content"] = output_content
    dataframe["summative_feedback"] = output_content
    dataframe["parsed_grade"] = parsed_grades

    return dataframe


def _run_single(problem_dict):
    """
    Execute a single program from a lightweight dict.

    Designed to be called from a ``ThreadPoolExecutor`` worker.  The dict
    must contain all fields required by ``run_execution`` plus an ``_index``
    key used to map results back to their original position.

    Args:
        problem_dict (dict): Problem data with an extra ``_index`` key.

    Returns:
        tuple: ``(index, output_str, grade_float)``
    """
    idx = problem_dict.pop("_index")
    try:
        result = run_execution(problem_dict, 5)
        return idx, result["output"], result["grade"]
    except Exception as e:
        return idx, f"Error: {e}", 0.0


NEEDED_COLS = ["problem_id", "code", "testcase", "max_score"]


def execute_programs_parallel(dataframe, max_workers=None, batch_size=500):
    """
    Execute student programs in parallel using a thread pool.

    Programs are processed in batches to bound peak memory usage.
    Results are written back to the dataframe in their original row order.

    Args:
        dataframe (pd.DataFrame): Must contain ``problem_id``, ``code``,
            ``testcase``, and ``max_score`` columns.
        max_workers (int, optional): Number of worker threads.
            Defaults to ``min(8, cpu_count // 2)``.
        batch_size (int): Number of rows per batch (default 500).

    Returns:
        pd.DataFrame: Input dataframe with three new columns added:
        ``output_content``, ``summative_feedback``, and ``parsed_grade``.
    """
    if max_workers is None:
        max_workers = min(8, max(1, os.cpu_count() // 2))

    total_rows = len(dataframe)
    num_batches = (total_rows + batch_size - 1) // batch_size

    all_outputs = [""] * total_rows
    all_grades = [0.0] * total_rows

    # Map dataframe index → position in result arrays
    idx_to_pos = {idx: pos for pos, idx in enumerate(dataframe.index)}

    print(f"Executing {total_rows} programs | {max_workers} workers | {num_batches} batches of {batch_size}")

    for batch_num in range(num_batches):
        batch_start = batch_num * batch_size
        batch_end = min(batch_start + batch_size, total_rows)
        batch_df = dataframe.iloc[batch_start:batch_end]

        rows = []
        for idx, row in batch_df.iterrows():
            d = {col: row[col] for col in NEEDED_COLS}
            d["_index"] = idx
            rows.append(d)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_run_single, r) for r in rows]

            desc = f"Batch {batch_num + 1}/{num_batches}"
            for future in tqdm(as_completed(futures), total=len(futures), desc=desc, unit="row"):
                idx, output, grade = future.result()
                pos = idx_to_pos[idx]
                all_outputs[pos] = output
                all_grades[pos] = int(grade)

    dataframe["output_content"] = all_outputs
    dataframe["summative_feedback"] = all_outputs
    dataframe["parsed_grade"] = all_grades

    return dataframe


def remove_mad_outliers(df, threshold=3.5):
    """
    Remove trajectory-length outliers per problem using Median Absolute Deviation.

    For each problem, computes a modified z-score based on trajectory length
    (number of messages).  Trajectories with a score above ``threshold`` are
    considered outliers and dropped.  This prevents pathologically long
    trajectories from dominating loss and memory consumption during training.

    The modified z-score is: ``0.6745 * |x - median| / MAD``

    Args:
        df (pd.DataFrame): DataFrame with a ``messages`` column and a
            ``problem_id`` column.
        threshold (float): Modified z-score cutoff (default 3.5, standard
            for MAD-based outlier detection).

    Returns:
        pd.DataFrame: Filtered dataframe with outlier rows removed.
    """
    df["traj_len"] = df.messages.apply(lambda m: len(m))

    def is_outlier(group):
        median = group["traj_len"].median()
        mad = (group["traj_len"] - median).abs().median()

        if mad == 0:
            # All trajectories have the same length — no outliers
            return pd.Series(False, index=group.index)

        modified_z = 0.6745 * (group["traj_len"] - median) / mad
        return modified_z.abs() > threshold

    outlier_mask = df.groupby("problem_id", group_keys=False).apply(is_outlier)
    return df[~outlier_mask.values]
