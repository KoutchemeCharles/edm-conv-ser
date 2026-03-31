"""
execution.py — Single-subprocess execution engine for student programs.

Architecture:
  Parent process → ONE subprocess per execution → exec(testcase) → exec(student code)

The autograder's run_script() uses exec() instead of spawning another subprocess,
eliminating the nested subprocess overhead of the original design.

Fixes applied:
  - Pickle-based result passing (no fragile stdout parsing)
  - PYTHONHASHSEED=0 for deterministic set/dict ordering
  - random.seed(42) injected into testcases for reproducibility
  - sys.dont_write_bytecode=True to prevent __pycache__ buildup
  - Parent-managed work_dir with guaranteed cleanup (no /tmp accumulation)
  - lru_cache on code string generation (no repeated disk reads)
  - Broad try/except in grade_fn for robustness
"""

import os
import sys
import base64
import pickle
import shutil
import tempfile
import subprocess
from functools import lru_cache
from typing import Dict


@lru_cache(maxsize=1)
def get_utility_functions_code():
    """
    Returns utility functions + merged autograder functions as a code string.
    These get embedded directly into the subprocess script.
    """
    return """
def write(filename, content):
    with open(filename, 'w') as f:
        f.write(content)

def does_compile(code):
    try:
        compile(code, '<string>', 'exec')
        return True
    except:
        return False

# --- Merged autograder functions (replaces autograder.py) ---

def print_styled(style, text, last_char='\\n'):
    print(text, end=last_char)

def get_inputs(input_list):
    result = ""
    for i in input_list:
        result += str(i) + "\\n"
    return result

def run_script(filename, input_list=[], timeout_in_seconds=10):
    \"\"\"
    Merged version of autograder.run_script().
    Uses exec() instead of subprocess.Popen() to avoid nested subprocess overhead.
    \"\"\"
    show_input = False
    show_output = False
    show_feedback = True

    if show_input and len(input_list) > 0:
        print("Inputs Provided:")
        for item in input_list:
            print(str(item))
        print()

    # Read student code from file
    with open(filename, 'r') as f:
        code = f.read()

    # Strip autograder imports from student code (they'd fail in this context)
    code = code.replace("from cs110 import autograder", "")
    code = code.replace("import autograder", "")

    # Prepare I/O streams
    input_str = get_inputs(input_list)
    stdin_stream = io.StringIO(input_str)
    stdout_stream = io.StringIO()
    stderr_stream = io.StringIO()

    try:
        exec_globals = {
            '__builtins__': __builtins__,
            '__name__': '__main__',
        }
        # Patch input() to read from provided inputs instead of real stdin
        exec_globals['input'] = lambda prompt='': stdin_stream.readline().rstrip('\\n')

        with contextlib.redirect_stdout(stdout_stream):
            with contextlib.redirect_stderr(stderr_stream):
                exec(compile(code, filename, 'exec'), exec_globals)

        out = stdout_stream.getvalue()
        err = stderr_stream.getvalue()
    except SystemExit:
        # Student code called sys.exit() or exit() — not an error
        out = stdout_stream.getvalue()
        err = stderr_stream.getvalue()
    except Exception as e:
        out = stdout_stream.getvalue()
        err = traceback.format_exc()

    if show_output:
        print("Your Program's Output:")
        if out != '':
            print(out)
        else:
            print("No Output Produced\\n")

    if err != '':
        print("Error Occurred:")
        print(err)
        print()

    if show_feedback:
        print("Feedback:")

    return (out, err)

def code_compiles(filename):
    import py_compile
    try:
        py_compile.compile(filename, doraise=True)
        return True
    except Exception:
        return False

def equals(value, expected_value, delta=0.01):
    try:
        return (float(value) >= float(expected_value) - delta and
                float(value) <= float(expected_value) + delta)
    except Exception:
        return value == expected_value

def compare_strings(student_output_list, expected_output_list, auto_trim=True,
                    check_order=True):
    num_matches = 0
    for i in range(len(student_output_list)):
        print("Line " + str(i+1) + ": ", end='')
        if i < len(expected_output_list):
            if auto_trim:
                student_output_list[i] = student_output_list[i].strip()
                expected_output_list[i] = expected_output_list[i].strip()
            if student_output_list[i] == expected_output_list[i]:
                print("CORRECT")
                num_matches += 1
            else:
                print("INCORRECT (Expected: '{}')".format(expected_output_list[i]))
        else:
            print("INCORRECT (Unexpected Line: '{}')".format(student_output_list[i]))
    print(num_matches, "out of", len(expected_output_list), "lines match")
    return num_matches

# --- exec_script: runs the testcase code ---

def exec_script(script):
    \"\"\"Execute testcase script with access to all utility/autograder functions.\"\"\"
    cwd = os.getcwd()
    if cwd not in sys.path:
        sys.path.insert(0, cwd)

    stdout_stream = io.StringIO()

    # Use module globals so testcase code can see run_script, equals, etc.
    g = dict(globals())
    g['__builtins__'] = __builtins__

    with contextlib.redirect_stdout(stdout_stream):
        with contextlib.redirect_stderr(stdout_stream):
            exec(script, g)
    return stdout_stream.getvalue()
"""


@lru_cache(maxsize=1)
def get_grade_fn_code():
    """Returns grade_fn and its helpers as a code string."""
    return """
def create_execution_string(testcase):
    # Remove autograder imports — functions are already in global scope
    testcase = testcase.replace("from cs110 import autograder", "")
    testcase = testcase.replace("import autograder", "")
    # Remove autograder. prefix since functions are global
    testcase = testcase.replace("autograder.", "")
    # Remove main guard and manual test invocation
    testcase = testcase.replace("if __name__ == '__main__':", "")
    testcase = testcase.replace("result = test_passed()", "")
    testcase = testcase.replace('print("Unit Test Returned:", result)', "")
    testcase = testcase.strip()
    # Seed random for reproducibility across runs
    testcase = "import random\\nrandom.seed(42)\\n" + testcase
    # Append test invocation
    testcase = testcase + "\\nresult = test_passed()\\n"
    testcase = testcase + 'print("Unit Test Returned:", result)'
    return testcase

def get_unit_test_score(testcase_output):
    lines = testcase_output.splitlines()
    utr = [l for l in lines if l.startswith("Unit Test Returned:")]
    if utr:
        return float(utr[0].replace("Unit Test Returned:", "").strip())
    return 0.0

def grade_fn(problem):
    if not does_compile(problem["code"]):
        return "Error: code does not compile", 0.0

    # Write student code to file so run_script can read it
    write(problem["problem_id"] + ".py", problem["code"])

    exec_string = create_execution_string(problem["testcase"])
    try:
        unit_test_output = exec_script(exec_string)
        score = get_unit_test_score(unit_test_output)
        score = (score / problem["max_score"]) * 100
        return unit_test_output, score
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}\\n{traceback.format_exc()}", 0.0
"""


def run_execution(problem: Dict, timeout: float) -> Dict:
    """
    Execute grade_fn in an isolated subprocess with timeout.
    Results are passed via pickle file (not stdout parsing).
    """
    # Convert pandas Series to dict if needed
    if hasattr(problem, 'to_dict'):
        problem_dict = problem.to_dict()
    else:
        problem_dict = dict(problem)

    # Serialize problem via pickle+base64
    problem_b64 = base64.b64encode(pickle.dumps(problem_dict)).decode()

    # Create result file and work directory (parent-managed for guaranteed cleanup)
    result_file = tempfile.mktemp(suffix='.pkl')
    work_dir = tempfile.mkdtemp()

    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        script_path = f.name

        script = f"""
import sys
sys.dont_write_bytecode = True
import os, io, contextlib, pickle, base64, traceback

# Deserialize problem
problem_data = pickle.loads(base64.b64decode('{problem_b64}'))

# --- Utility and autograder functions ---
{get_utility_functions_code()}

# --- Grade function ---
{get_grade_fn_code()}

# --- Main ---
def main():
    os.chdir('{work_dir}')
    try:
        result = grade_fn(problem_data)
        output, grade = result[0], result[1]
    except Exception as e:
        output = f"Error: {{type(e).__name__}}: {{e}}\\n{{traceback.format_exc()}}"
        grade = 0.0

    with open('{result_file}', 'wb') as f:
        pickle.dump({{'output': str(output), 'grade': float(grade)}}, f)

if __name__ == "__main__":
    main()
"""
        f.write(script)

    try:
        subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            env={**os.environ, 'PYTHONHASHSEED': '0'},
        )

        # Read result from pickle file
        with open(result_file, 'rb') as rf:
            data = pickle.load(rf)

        return {
            "output": data["output"],
            "passed": data["grade"] == 100.0,
            "grade": data["grade"],
        }

    except subprocess.TimeoutExpired:
        return {"output": "Error: execution timeout", "passed": False, "grade": 0.0}
    except (FileNotFoundError, pickle.UnpicklingError, KeyError):
        return {"output": "Error: subprocess crashed", "passed": False, "grade": 0.0}
    except Exception as e:
        return {"output": f"Error: {e}", "passed": False, "grade": 0.0}
    finally:
        # Guaranteed cleanup — no /tmp accumulation even on timeouts
        for p in (script_path, result_file):
            try:
                os.unlink(p)
            except OSError:
                pass
        try:
            shutil.rmtree(work_dir, ignore_errors=True)
        except OSError:
            pass


def get_unit_test_score(testcase_output):
    """Parse unit test score from testcase output. Available for external use."""
    lines = testcase_output.splitlines()
    utr = [l for l in lines if l.startswith("Unit Test Returned:")]
    if utr:
        return float(utr[0].replace("Unit Test Returned:", "").strip())
    return 0.0