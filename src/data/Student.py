# -*- coding: utf-8 -*-
"""
Conversational serialization utilities for student programming traces.

This module exposes:
  - Standalone helpers for tokenization, LCS, diffing, and output shortening.
  - The `student` class that builds conversations per exercise without changing
    any behavioral logic from the original implementation.
"""

from __future__ import annotations
import difflib
import textwrap
from warnings import warn
import pandas as pd
from typing import List, Dict

from src.data.serialization import conversationalize_improvements

# ---------------------------------------------------------------------------
# student class (public API preserved, logic unchanged)
# ---------------------------------------------------------------------------

class Student:
    """
    Conversational serialization for a single student over one assignment/exercise group.

    Implements §4.1 data preparation:
      - System prompt contains learner background and prior-behavior statistics.
      - Assistant turns represent student code as EDIT vs REWRITE.
      - student turns represent environment outcomes (local run) or grader summaries.
      - First assistant turn is always a full code REWRITE.
      - Each assistant turn ends with <LOCAL> or <SUBMISSION> tag.
    """

    def __init__(
        self,
        student_id: str,
        exec_df: pd.DataFrame,
        config,
        get_all
    ):
        self.student_id = student_id
        self.exec_df = exec_df
        self.lang = "python"
        self.config = config 
        self.get_all = get_all

    def _get_background_text(self):
        """
        Return a background description string for the student.

        Not used in the current FalconCode pipeline (background information is
        not available in the dataset).  Subclasses may override this to inject
        student-specific context into the system prompt.
        """
        raise NotImplementedError()
    
    # ------------------------ public API ------------------------

    def get_conversations(self) -> pd.DataFrame:
        """
        Build one conversation per exercise whose final SUBMISSION grade equals 100.

        Returns:
            DataFrame with a 'messages' column, one row per exercise instance.
        """
        
        filtered = self.exec_df

        frames: List[pd.DataFrame] = []
        for eid in sorted(filtered.problem_id.unique()):
            sub_df = filtered[filtered.problem_id == eid]
            timestamp = sub_df.sort_values(by="timestamp").iloc[-1]
            if len(sub_df) < 6: 
                continue 
            series = self.conversationalize(sub_df)
            frame = series.to_frame()
            frame["problem_id"] = eid
            frame["timestamp"] = timestamp
            frames.append(frame)

        conversations_df = pd.DataFrame()
        if frames: conversations_df = pd.concat(frames, axis=0)
            
        return conversations_df


    def conversationalize(self, sub_df: pd.DataFrame) -> List[Dict[str, str]]:
        """
        Convert a single-exercise trace (one student's interaction on one assignment)
        into a conversational format using the unified diff representation.

        The resulting message sequence alternates between:
        - `assistant` turns: student's code (local runs as <think> diffs + submissions)
        - `user` turns: grader feedback or runtime output

        Behavior summary:
        -----------------
        - TEST runs are stored as incremental unified diffs inside <think> tags.
        - SUBMISSION runs contain the final full code block and grader output.
        - The first local run is diffed against an empty string (creation step).
        - Identical re-runs (same code and type) are skipped.
        - The previous submission becomes the baseline for the next sequence of local edits.
        """

        # Ensure chronological order
        sub_df = sub_df.sort_values("timestamp").reset_index(drop=True)

        # Initialize conversation with system and user turns
        messages: List[Dict[str, str]] = []
        messages.append(self.form_system_prompt())
        messages.append(self.form_first_user_turn(sub_df))
        messages = conversationalize_improvements(sub_df, messages, 
                                                    self.lang, 
                                                    self.config.format,
                                                    self.config.feedback_level,
                                                    self.get_all)

        # ---- Debug output (optional) ----
        """
        print("Messages obtained")
        for d in messages:
            print(d["role"])
            print(d["content"])
            print()
        print("----\n")
        """

        return pd.Series([messages], name="messages")


    def form_first_user_turn(self, sub_df: pd.DataFrame) -> Dict[str, str]:
        """
        Build the initial user turn that carries exercise metadata and problem text.
        """

        description = sub_df.iloc[0].get("problem_description").strip().strip('"')

        content = """
        {problem_description}
        """

        content = textwrap.dedent(content).strip()
        content = content.format(
            problem_description=description,
        ).strip()

        return {"role": "user", "content": content}
    

    def form_system_prompt(self) -> Dict[str, str]:
        """
        Generate system prompt based on the configured code representation format.

        Supported formats: ``code`` (full rewrite every turn), ``dual``
        (diff for small edits, full rewrite for large ones).

        Returns:
            Dictionary with 'role' and 'content' keys for system message.
        """
        
        # Common preamble for all formats
        preamble = textwrap.dedent("""
            You are a first-year novice student learning how to program in {language}.
            You will be given a programming problem.
            Your goal is to iteratively write code until the program passes all tests.

            After each submission you will receive grader feedback including:
            * a grade, and
            * a summary of test results.
                                   
            CRITICAL - YOU ARE A NOVICE LEARNER:
                * You are NOT an expert. You do NOT solve problems correctly on the first try.
                * You are uncertain and make common beginner mistakes: off-by-one errors, 
                  wrong variable names, incorrect logic, forgetting return values, etc.
                * You sometimes misunderstand requirements or miss edge cases.
                * You do not always correctly interpret environment feedback.
                * Fixing one bug may introduce new bugs elsewhere.
                * Your approach is exploratory trial-and-error, not systematic.
                * You never resubmit identical code twice in a row.
                * Think like a struggling student learning through mistakes, NOT like a helpful AI.
            """).strip()
        
        # Common giving up section
        giving_up = textwrap.dedent("""
            [GIVING UP]

            If you do not know how to solve the assignment and want to give up, submit:
            ```{language}
            exit()
            ```
            """).strip()
        
        # Format-specific sections
        if self.config.format == "code":
            format_section = textwrap.dedent("""
                # SUBMISSION FORMAT

                For every submission, write the complete program as a code block.

                Example:
                ```{language}
                def compute_average(nums):
                    if not nums:
                        return 0
                    return sum(nums) / len(nums)
                ```
                                             
                Structure constraints:
                - A code block must appear in every response
                - Do not output anything after the closing ```
                """).strip()

        elif self.config.format == "dual":
            format_section = textwrap.dedent("""
                # SUBMISSION FORMAT

                **First submission:**
                Submit the complete program as a code block.

                **Subsequent submissions:**
                You may submit your program either as:
                - A complete program in a ```{language} ``` code block
                - A set of edits in a ```diff``` code block:
                  - @@ -N +M @@ indicates the line number where changes occur
                  - Prefix lines to remove with -
                  - Prefix lines to add with +
                                                                        
                Example full program:
                ```{language}
                def compute_average(nums):
                    if not nums:
                        return 0
                    return sum(nums) / len(nums)
                ```
                                        
                Example diff:
                ```diff
                @@ -4 +4 @@
                -elif bruxo == 'GINA':
                +elif bruxo == 'HARRY':
                ```

                # RESPONSE FORMAT SUMMARY

                - First submission: full program
                - Subsequent submissions: diff or full program
                - Giving up: exit() as full program

                Structure constraints:
                - A code block (full program or diff) must appear in every response
                - Do not output anything after the closing ```
                """).strip()
            
        else:
            raise ValueError(f"Unknown format: {self.config.format}")

        content = preamble + format_section # + giving_up
        content = content.format(language=self.lang)

        return {"role": "system", "content": content}


