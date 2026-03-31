#!/usr/bin/env python
# coding: utf-8
"""
Rollout Evaluation: K-Steps Ahead Performance Analysis

This script processes rollout evaluation results to generate:
1. Aggregate table showing metrics for k=1 to 5 steps ahead
2. Visualization showing performance by starting position and steps ahead

SIMPLIFIED VERSION: Focus on core metrics:
- Coverage (Cov): fraction of trajectory the model generated predictions for
- Grade Proximity (GP): 1 - grade_distance, higher is better (0-1)
- CodeBLEU (CB): code similarity metric

Data structure expected in generations.csv:
- student_id: student identifier
- problem_id: problem identifier
- @k: which submission this trajectory started from (starting position)
- step: how many steps into this trajectory (k-steps-ahead)
- code: model's generated code
- student_code: ground truth student code at this position
- grade: model's grade
- student_grade: ground truth grade
"""

# =============================================================================
# SECTION 1: IMPORTS AND SETUP
# =============================================================================

import warnings
warnings.filterwarnings('ignore')
import logging
logging.disable(logging.CRITICAL)

import os
import json
import ast
import io
import tokenize
import hashlib
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import Levenshtein
import libcst as cst
from codebleu import calc_codebleu
from tqdm.auto import tqdm


# =============================================================================
# SECTION 2: CODE NORMALIZATION UTILITIES
# =============================================================================

class NormalizeIdentifiers(ast.NodeTransformer):
    """AST transformer to normalize variable and function names."""
    
    def __init__(self):
        self.var_counter = 0
        self.func_counter = 0
        self.var_names = {}
        self.func_names = {}

    def _get_var_name(self):
        name = f"var_{self.var_counter}"
        self.var_counter += 1
        return name

    def _get_func_name(self):
        name = f"func_{self.func_counter}"
        self.func_counter += 1
        return name

    def visit_Name(self, node):
        if isinstance(node.ctx, (ast.Load, ast.Store, ast.Del)):
            if node.id not in self.var_names:
                self.var_names[node.id] = self._get_var_name()
            node.id = self.var_names[node.id]
        return self.generic_visit(node)

    def visit_arg(self, node):
        if node.arg not in self.var_names:
            self.var_names[node.arg] = self._get_var_name()
        node.arg = self.var_names[node.arg]
        return node

    def visit_FunctionDef(self, node):
        if node.name not in self.func_names:
            self.func_names[node.name] = self._get_func_name()
        node.name = self.func_names[node.name]
        self.generic_visit(node)
        return node


def normalize_code_to_ast_string(source_code):
    """Parse code to AST and normalize identifiers."""
    try:
        tree = ast.parse(source_code)
    except SyntaxError:
        return None

    normalizer = NormalizeIdentifiers()
    normalized_tree = normalizer.visit(tree)
    ast.fix_missing_locations(normalized_tree)
    normalized_ast = ast.dump(normalized_tree, annotate_fields=True, include_attributes=False)
    return normalized_ast


def normalize_with_libcst(code):
    """Attempt to normalize code using libcst (more tolerant parser)."""
    try:
        module = cst.parse_module(code)
        return module.code
    except Exception:
        return None


def code_to_hash(normalized_ast: str) -> str:
    """Convert normalized AST string to hash."""
    return hashlib.md5(normalized_ast.encode('utf-8')).hexdigest()


def robust_normalize(code):
    """Robustly normalize code for comparison."""
    if not code: 
        return ""
    
    normalized_ast = normalize_code_to_ast_string(code)
    if normalized_ast is not None:
        return code_to_hash(normalized_ast)
    
    red = normalize_with_libcst(code)
    if red is not None:
        return code_to_hash(red)

    return code_to_hash(code)


# =============================================================================
# SECTION 3: METRIC COMPUTATION FUNCTIONS
# =============================================================================

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


def does_compile(code):
    """Check if code compiles to valid Python AST."""
    try:
        ast.parse(code)
        return True
    except:
        return False


def python_tokenize(code: str):
    """Tokenize Python code into a list of meaningful Python tokens."""
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
    except:
        return []

    return tokens


def norm_edit_distance(code1: str, code2: str) -> float:
    """Compute normalized Levenshtein edit distance between tokenized Python programs."""
    if not does_compile(code1) or not does_compile(code2):
        return 1.0 
        
    tok1 = python_tokenize(code1)
    tok2 = python_tokenize(code2)

    if not tok1 or not tok2:
        return 1.0

    dist = Levenshtein.distance(tok1, tok2)
    denom = max(len(tok1), len(tok2), 1)
    
    return dist / denom


# =============================================================================
# SECTION 4: EXPERIMENT CONFIGURATION UTILITIES
# =============================================================================

def get_original_model_config(exp_config):
    """Extract the base model configuration and training pipeline from experiment config."""
    training_pipeline = []
    model_config = exp_config.get("model")
    
    # Handle None or non-dict model config
    if model_config is None or not isinstance(model_config, dict):
        return None, []
    
    while model_config is not None and isinstance(model_config, dict) and "model" in model_config and isinstance(model_config.get("model"), dict):
        task_name = model_config.get("task", {}).get("name", "unknown")
        training_pipeline.append(task_name)
        model_config = model_config["model"]
    
    return model_config, training_pipeline[::-1]


def extract_format_from_dataset_name(dataset_name):
    """Extract the format from the dataset name."""
    if not dataset_name:
        return ""
    
    parts = dataset_name.split("_")
    if len(parts) > 1:
        format_str = parts[-1].upper()
        return format_str
    
    return ""


def determine_training_method(pipeline_list):
    """Determine the training method from the pipeline list."""
    if not pipeline_list:
        return "BASE"
    
    pipeline_upper = [p.upper() for p in pipeline_list]
    return "-".join(pipeline_upper)


def extract_dataset_name_from_config(exp_config):
    """Extract dataset name from experiment configuration."""
    dataset_name = ''
    try:
        config_dict = exp_config
        if isinstance(exp_config, list) and len(exp_config) > 0:
            config_dict = exp_config[0]
        
        if isinstance(config_dict, dict):
            dataset_field = config_dict.get('dataset', None)
            
            if dataset_field is not None:
                if isinstance(dataset_field, dict):
                    dataset_name = dataset_field.get('name', '')
                elif isinstance(dataset_field, list) and len(dataset_field) > 0:
                    if isinstance(dataset_field[0], dict):
                        dataset_name = dataset_field[0].get('name', '')
    except Exception as e:
        print(f"  Warning: Could not extract dataset name: {e}")
        dataset_name = ''
    
    return dataset_name


# =============================================================================
# SECTION 5: CONFIGURATION
# =============================================================================

BASE_PATH = "/scratch/work/koutchc1/experiments/edm26/"

MODEL_MAPPING = {
    'unsloth/Meta-Llama-3.1-8B-bnb-4bit': 'Llama',
    'unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit': 'Llama',
    'unsloth/SmolLM3-3B': 'Smol', 
    'unsloth/SmolLM3-3B-unsloth-bnb-4bit': 'Smol',
    'unsloth/Qwen3-4B-Instruct-2507': 'Qwen3-4B',
    'unsloth/Qwen3-4B-instruct': 'Qwen3-4B',
    'unsloth/Qwen3-4B': 'Qwen3-4B',
    'Qwen3-4B': 'Qwen3-4B',
    'unsloth/Qwen3-8B': 'Qwen3-8B',
    'Qwen3-8B': 'Qwen3-8B',
    'Qwen': 'Qwen3-4B',  # Legacy mapping
    'gpt-5-mini': 'GPT-5-mini',
    'GPT-5-mini': 'GPT-5-mini',
}

VALID_FORMATS = {'PARA', 'CODE', 'EDIT', 'DUAL'}
VALID_TRAINING = {'BASE', 'SFT', 'SFT_ALL', 'DPO', 'DAPO', 'NCA'}

METHOD_DISPLAY_MAPPING = {
    'BASE': 'BASE',
    'SFT-PARA': 'PARA',
    'SFT-CODE': 'SFT',
    'PARA': 'PARA',
    'CODE': 'SFT',
    'SFT-DPO_0.1-CODE': 'DPO',
    'SFT-DPO_0.5-CODE': 'DPO',
    'DPO-CODE': 'DPO',
    'SFT-DAPO-CODE': 'DAPO',
    'DAPO-CODE': 'DAPO',
}

MAX_K = 5
STARTING_POSITIONS = None


# =============================================================================
# SECTION 6: K-STEPS-AHEAD METRICS COMPUTATION (WITH COVERAGE)
# =============================================================================

def compute_k_steps_metrics(df, max_k=5):
    """
    Compute metrics for k-steps-ahead predictions.
    
    CRITICAL DESIGN DECISION:
    Code-based metrics (codebleu) are ONLY computed when BOTH model generated 
    code AND ground truth exists. Otherwise set to NaN.
    
    This prevents BIAS: if model correctly stops at 100% grade, it shouldn't be 
    penalized with 0.0 scores on steps it didn't need to generate.
    
    Tracking columns:
    - has_ground_truth: 1 if student code exists at this step, 0 otherwise
    - has_model_code: 1 if model generated non-empty code (not "exit()"), 0 otherwise
    - has_valid_pair: 1 if BOTH exist (metrics are computed), 0 otherwise
    
    Coverage = has_valid_pair.sum() / has_ground_truth.sum()
    """
    results = []
    grouped = df.groupby(['student_id', 'problem_id', 'starting_k'])
    
    print(f"Processing {len(grouped)} trajectories...")
    
    for (student_id, problem_id, start_k), traj_df in tqdm(grouped, desc="Computing k-steps metrics"):
        traj_df = traj_df.sort_values('step')
        
        if 'successful' in traj_df.columns:
            successful_val = traj_df['successful'].iloc[-1]
            if isinstance(successful_val, bool):
                trajectory_solved = successful_val
            else:
                trajectory_solved = (successful_val == 100 or successful_val == True)
        else:
            trajectory_solved = None
        
        for k in range(1, max_k + 1):
            step_data = traj_df[traj_df['step'] == k]
            
            if len(step_data) == 0:
                continue
            
            if len(step_data) > 1:
                step_data = step_data.iloc[[0]]
            
            row = step_data.iloc[0]
            
            # Extract and clean codes
            pred_code_raw = row['code'] if pd.notna(row['code']) else ""
            true_code_raw = row['student_code'] if pd.notna(row['student_code']) else ""
            
            if pred_code_raw is None:
                pred_code = ""
            elif isinstance(pred_code_raw, str):
                pred_code = pred_code_raw.strip()
            else:
                pred_code = ""
            
            if true_code_raw is None:
                true_code = ""
            elif isinstance(true_code_raw, str):
                true_code = true_code_raw.strip()
            else:
                true_code = ""
            
            is_recording_mode = (
                not pred_code or 
                pred_code.lower().startswith('exit()') or
                pred_code.lower().strip() == 'exit' or
                pred_code == '...'
            )
            
            has_ground_truth = int(bool(true_code))
            has_model_code = int(not is_recording_mode)
            has_valid_pair = int(has_ground_truth and has_model_code)
            
            pred_grade = round(row['grade']) if pd.notna(row['grade']) else np.nan
            true_grade = round(row['student_grade']) if pd.notna(row['student_grade']) else np.nan
            
            if has_valid_pair:
                codebleu = compute_codebleu(true_code, pred_code)
                
                if pd.notna(pred_grade) and pd.notna(true_grade):
                    grade_difference = abs(pred_grade - true_grade) / 100.0
                    grade_proximity = 1.0 - grade_difference
                else:
                    grade_difference = np.nan
                    grade_proximity = np.nan
            else:
                codebleu = np.nan
                grade_difference = np.nan
                grade_proximity = np.nan
            
            metrics = {
                'student_id': student_id,
                'problem_id': problem_id,
                'starting_position': start_k,
                'k': k,
                'has_ground_truth': has_ground_truth,
                'has_model_code': has_model_code,
                'has_valid_pair': has_valid_pair,
                'true_grade': true_grade,
                'pred_grade': pred_grade,
                'codebleu': codebleu,
                'grade_difference': grade_difference,
                'grade_proximity': grade_proximity,
                'trajectory_solved': trajectory_solved,
            }
            
            results.append(metrics)
    
    results_df = pd.DataFrame(results)
    
    if len(results_df) > 0:
        recording_mode_rows = (results_df['has_model_code'] == 0).sum()
        total_rows = len(results_df)
        print(f"  Recording mode: {recording_mode_rows}/{total_rows} rows ({100*recording_mode_rows/total_rows:.1f}%)")
    
    return results_df



def aggregate_k_steps_metrics(metrics_df):
    """
    Aggregate k-steps metrics across all trajectories.
    
    Coverage per k = (valid_pairs at k) / (ground_truth at k)
    Metrics are computed with skipna=True, so NaN from invalid pairs excluded.
    """
    # For coverage: sum then divide
    coverage_per_k = metrics_df.groupby('k').agg({
        'has_ground_truth': 'sum',
        'has_valid_pair': 'sum',
    }).reset_index()
    coverage_per_k['coverage'] = np.where(
        coverage_per_k['has_ground_truth'] > 0,
        coverage_per_k['has_valid_pair'] / coverage_per_k['has_ground_truth'],
        np.nan
    )
    
    # For metrics: mean (NaN excluded automatically)
    metrics_per_k = metrics_df.groupby('k').agg({
        'grade_difference': 'mean',
        'grade_proximity': 'mean',
        'codebleu': 'mean',
    }).reset_index()
    
    # Merge
    agg_metrics = coverage_per_k[['k', 'coverage']].merge(metrics_per_k, on='k')
    
    for col in ['coverage', 'grade_difference', 'grade_proximity', 'codebleu']:
        agg_metrics[col] = agg_metrics[col].round(3)
    
    return agg_metrics


# =============================================================================
# SECTION 7: DATA LOADING AND PROCESSING
# =============================================================================

def process_rollout_experiment(path):
    """Process a single rollout experiment directory."""
    config_path = os.path.join(path, "experiment_configuration.json")
    with open(config_path) as fp:
        exp_config = json.load(fp)

    # Skip preprocessing experiments (no "model" key)
    if "model" not in exp_config:
        print(f"  Skipping (no model key): {os.path.basename(path)}")
        return None

    original_model_config, pipeline_list = get_original_model_config(exp_config)
    
    # Skip if model config is invalid
    if original_model_config is None or not isinstance(original_model_config, dict):
        print(f"  Skipping (invalid model config): {os.path.basename(path)}")
        return None
    
    model_name = original_model_config.get("name", "unknown")
    dataset_name = extract_dataset_name_from_config(exp_config)
    format_str = extract_format_from_dataset_name(dataset_name)
    training_method = determine_training_method(pipeline_list)
    
    if training_method == "BASE":
        method_str = "BASE"
        format_str = ""
    else:
        method_str = f"{training_method}-{format_str}"
    
    print(f"  Processing: {os.path.basename(path)}")
    print(f"    Dataset: {dataset_name}, Format: {format_str if format_str else 'N/A'}, Method: {method_str}")
    
    csv_path = os.path.join(path, "generations.csv")
    df = pd.read_csv(csv_path)
    
    if '@k' in df.columns:
        df = df.rename(columns={'@k': 'starting_k'})
    elif 'starting_k' not in df.columns:
        print(f"Warning: No '@k' or 'starting_k' column in {path}")
        return None
    
    metrics_df = compute_k_steps_metrics(df, max_k=MAX_K)
    
    metrics_df['model_name'] = MODEL_MAPPING.get(model_name, model_name)
    metrics_df['format'] = format_str
    metrics_df['training'] = training_method
    metrics_df['method'] = method_str
    metrics_df['experiment'] = os.path.basename(path)
    
    # Report coverage
    if 'has_valid_pair' in metrics_df.columns:
        total_gt = metrics_df['has_ground_truth'].sum()
        total_model = metrics_df['has_model_code'].sum()
        total_valid = metrics_df['has_valid_pair'].sum()
        coverage = total_valid / total_gt if total_gt > 0 else 0
        recording_mode_count = total_gt - total_model
        print(f"    Coverage: {coverage:.3f} ({total_valid}/{total_gt} valid pairs, {recording_mode_count} recording mode rows)")
    
    agg_metrics = aggregate_k_steps_metrics(metrics_df)
    
    return {
        'model_name': MODEL_MAPPING.get(model_name, model_name),
        'format': format_str,
        'training': training_method,
        'method': method_str,
        'experiment': os.path.basename(path),
        'detailed_metrics': metrics_df,
        'aggregated_metrics': agg_metrics,
    }


def find_rollout_experiments(base_path):
    """Find all rollout experiment directories."""
    experiment_dirs = []
    
    for root, subdirs, files in os.walk(base_path):
        if "generations.csv" in files:
            if "one_step" not in root.lower():
                experiment_dirs.append(root)
    
    print(f"Found {len(experiment_dirs)} rollout experiment directories")
    return experiment_dirs


def load_all_rollout_data(base_path):
    """Load and process all rollout experiments."""
    experiment_dirs = find_rollout_experiments(base_path)
    
    results = []
    all_detailed_metrics = []
    
    for exp_dir in tqdm(experiment_dirs, desc="Processing experiments"):
        try:
            result = process_rollout_experiment(exp_dir)
            if result is not None:
                results.append(result)
                all_detailed_metrics.append(result['detailed_metrics'])
        except Exception as e:
            print(f"Error processing {exp_dir}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if not all_detailed_metrics:
        raise ValueError("No experiments were successfully processed!")
    
    detailed_df = pd.concat(all_detailed_metrics, ignore_index=True)
    
    detailed_df['method'] = detailed_df['method'].map(
        lambda x: METHOD_DISPLAY_MAPPING.get(x, x)
    )
    
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)
    
    print("\nUnique Methods found:")
    for method in sorted(detailed_df['method'].unique()):
        count = (detailed_df['method'] == method).sum()
        print(f"  - {method}: {count} records")
    
    print("\nBy Model:")
    for model in sorted(detailed_df['model_name'].unique()):
        count = (detailed_df['model_name'] == model).sum()
        methods = detailed_df.loc[detailed_df['model_name'] == model, 'method'].unique()
        print(f"  {model}: {count} records, methods: {sorted(methods)}")
    
    # Coverage statistics
    if 'has_valid_pair' in detailed_df.columns and 'has_ground_truth' in detailed_df.columns:
        print("\nCoverage Statistics (valid_pairs / ground_truth_steps):")
        total_gt = detailed_df['has_ground_truth'].sum()
        total_valid = detailed_df['has_valid_pair'].sum()
        overall_coverage = total_valid / total_gt if total_gt > 0 else 0
        print(f"  Overall: {overall_coverage:.3f} ({total_valid}/{total_gt} valid pairs)")
        for model in sorted(detailed_df['model_name'].unique()):
            model_df = detailed_df[detailed_df['model_name'] == model]
            model_gt = model_df['has_ground_truth'].sum()
            model_valid = model_df['has_valid_pair'].sum()
            model_coverage = model_valid / model_gt if model_gt > 0 else 0
            print(f"  {model}: {model_coverage:.3f} ({model_valid}/{model_gt})")
    
    print("="*80 + "\n")
    
    aggregated_df = detailed_df.groupby(['model_name', 'format', 'training', 'method', 'k']).agg({
        'has_ground_truth': 'sum',
        'has_model_code': 'sum',
        'has_valid_pair': 'sum',
        'grade_difference': 'mean',
        'grade_proximity': 'mean',
        'codebleu': 'mean',
    }).reset_index()
    
    # Add row count per group for coverage computation
    row_counts = detailed_df.groupby(['model_name', 'format', 'training', 'method', 'k']).size().reset_index(name='n_rows')
    aggregated_df = aggregated_df.merge(row_counts, on=['model_name', 'format', 'training', 'method', 'k'])
    
    # Compute coverage properly: valid_pairs / ground_truth
    aggregated_df['coverage'] = np.where(
        aggregated_df['has_ground_truth'] > 0,
        aggregated_df['has_valid_pair'] / aggregated_df['has_ground_truth'],
        np.nan
    )
    
    for col in ['coverage', 'grade_difference', 'grade_proximity', 'codebleu']:
        aggregated_df[col] = aggregated_df[col].round(3)
    
    return results, detailed_df, aggregated_df


# =============================================================================
# SECTION 8: TABLE GENERATION (SIMPLIFIED: Cov, GD, CB only)
# =============================================================================

def create_simple_metrics_table(aggregated_df):
    """
    Create a simple table showing core metrics averaged across all k values.
    
    Metrics:
    - Cov (Coverage): valid_pairs / ground_truth_steps
    - GP (Grade Proximity): 1 - grade_distance, higher is better (0-1)
    - CB (CodeBLEU): code similarity score
    - GD (Grade Distance): normalized absolute difference (0-1), kept for reference
    """
    # For coverage: sum valid pairs and ground truth across k, then divide
    coverage_df = aggregated_df.groupby(['model_name', 'method']).agg({
        'has_ground_truth': 'sum',
        'has_valid_pair': 'sum',
    }).reset_index()
    coverage_df['Cov'] = np.where(
        coverage_df['has_ground_truth'] > 0,
        coverage_df['has_valid_pair'] / coverage_df['has_ground_truth'],
        np.nan
    )
    coverage_df = coverage_df[['model_name', 'method', 'Cov']]
    
    # For metrics: take mean across k
    metrics_df = aggregated_df.groupby(['model_name', 'method']).agg({
        'grade_difference': 'mean',
        'grade_proximity': 'mean',
        'codebleu': 'mean',
    }).reset_index()
    
    # Merge coverage with metrics
    simple_table = coverage_df.merge(metrics_df, on=['model_name', 'method'])
    
    for col in ['Cov', 'grade_difference', 'grade_proximity', 'codebleu']:
        simple_table[col] = simple_table[col].round(3)
    
    simple_table = simple_table.rename(columns={
        'model_name': 'Model',
        'method': 'Method',
        'grade_proximity': 'GP',
        'codebleu': 'CB',
    })
    
    # GP (Grade Proximity) is the focus metric
    column_order = ['Model', 'Method', 'Cov', 'GP', 'CB']
    simple_table = simple_table[column_order]
    simple_table = simple_table.sort_values(['Model', 'Method'])
    
    return simple_table


def save_simple_metrics_table_to_latex(simple_table, filename='outputs/tables/rollout_results_simple.tex'):
    """Save simple metrics table to LaTeX with multirow format matching paper style."""
    method_order = ['BASE', 'PARA', 'SFT', 'DPO', 'DAPO', 'Baseline']
    model_order = ['Qwen3-4B', 'Qwen3-8B', 'GPT-5-mini']
    
    simple_table = simple_table.copy()
    
    # Map DPO_0.1 and DPO_0.5 to just DPO for cleaner table
    simple_table['Method'] = simple_table['Method'].replace({
        'DPO_0.1': 'DPO',
        'DPO_0.5': 'DPO',
        'CODE': 'SFT',  # Map CODE to SFT
    })
    
    # Sort methods
    existing_methods = simple_table['Method'].unique()
    ordered_methods = [m for m in method_order if m in existing_methods]
    for method in existing_methods:
        if method not in ordered_methods:
            ordered_methods.append(method)
    
    simple_table['Method'] = pd.Categorical(
        simple_table['Method'],
        categories=ordered_methods,
        ordered=True
    )
    
    # Sort models
    existing_models = simple_table['Model'].unique()
    ordered_models = [m for m in model_order if m in existing_models]
    for model in existing_models:
        if model not in ordered_models:
            ordered_models.append(model)
    
    simple_table['Model'] = pd.Categorical(
        simple_table['Model'],
        categories=ordered_models,
        ordered=True
    )
    
    simple_table = simple_table.sort_values(['Model', 'Method'])
    
    # Compute best values for each model
    best_values = {}
    for model in ordered_models:
        model_data = simple_table[simple_table['Model'] == model]
        best_values[model] = {
            'Cov': model_data['Cov'].max(),
            'GP': model_data['GP'].max(),
            'CB': model_data['CB'].max(),
        }
    
    # Build table rows manually
    rows = []
    current_model = None
    model_row_count = {}
    
    # Count rows per model
    for model in ordered_models:
        model_row_count[model] = len(simple_table[simple_table['Model'] == model])
    
    for _, row in simple_table.iterrows():
        model = row['Model']
        method = row['Method']
        cov = row['Cov']
        gp = row['GP']
        cb = row['CB']
        
        # Format values with bold for best
        def fmt(val, col, mdl):
            if pd.isna(val):
                return "--"
            is_best = abs(val - best_values[mdl][col]) < 1e-6
            if is_best:
                return f"\\textbf{{{val:.3f}}}"
            return f"{val:.3f}"
        
        cov_str = fmt(cov, 'Cov', model)
        gp_str = fmt(gp, 'GP', model)
        cb_str = fmt(cb, 'CB', model)
        
        if model != current_model:
            # New model group
            if current_model is not None:
                rows.append("\\cline{1-5}")
            
            n_rows = model_row_count.get(model, 1)
            if model == 'GPT-5-mini' or n_rows == 1:
                # Single row without multirow
                model_str = model
            else:
                model_str = f"\\multirow[t]{{{n_rows}}}{{*}}{{{model}}}"
            
            rows.append(f"{model_str} & {method} & {cov_str} & {gp_str} & {cb_str} \\\\")
            current_model = model
        else:
            # Continuation of model group
            rows.append(f" & {method} & {cov_str} & {gp_str} & {cb_str} \\\\")
    
    rows_str = "\n".join(rows)
    
    caption = (
        "\\textbf{Rollout performance metrics.} "
        "Performance metrics averaged across k=1 to k=5 steps ahead. "
        "Metrics are computed \\textit{only} on steps where model generated code, "
        "avoiding bias against models that correctly stop early. "
        "Arrows: $\\uparrow$ higher is better. "
        "\\textbf{Bold} indicates best performance within each model."
    )
    
    full_latex = f"""\\begin{{table}}[htbp]
\\centering
\\caption{{{caption}}}
\\label{{tab:rollout_simple}}
\\resizebox{{\\columnwidth}}{{!}}{{%
\\begin{{tabular}}{{ll|ccc}}
\\toprule
Model & Method & Coverage$\\uparrow$ & Grade-proximity$\\uparrow$ & CodeBLEU$\\uparrow$ \\\\
\\midrule
{rows_str}
\\bottomrule
\\end{{tabular}}
}}
\\end{{table}}
"""
    
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as f:
        f.write(full_latex)
    
    print(f"Simple metrics LaTeX table saved to: {filename}")


def save_averaged_degradation_table_to_latex(averaged_table, filename='outputs/tables/rollout_degradation.tex'):
    """Save averaged degradation table to LaTeX with multirow format matching paper style."""
    method_order = ['BASE', 'PARA', 'SFT', 'DPO', 'DAPO']
    model_order = ['Qwen3-4B', 'Qwen3-8B', 'GPT-5-mini']
    
    table = averaged_table.copy()
    
    # Map DPO_0.1 and DPO_0.5 to just DPO for cleaner table
    table['Method'] = table['Method'].replace({
        'DPO_0.1': 'DPO',
        'DPO_0.5': 'DPO',
        'CODE': 'SFT',  # Also map CODE to SFT
    })
    
    # Apply MODEL_MAPPING to ensure consistent model names
    table['Model'] = table['Model'].map(lambda x: MODEL_MAPPING.get(x, x))
    
    # Sort methods
    existing_methods = table['Method'].unique()
    ordered_methods = [m for m in method_order if m in existing_methods]
    for method in existing_methods:
        if method not in ordered_methods:
            ordered_methods.append(method)
    
    table['Method'] = pd.Categorical(
        table['Method'],
        categories=ordered_methods,
        ordered=True
    )
    
    # Sort models
    existing_models = table['Model'].unique()
    ordered_models = [m for m in model_order if m in existing_models]
    for model in existing_models:
        if model not in ordered_models:
            ordered_models.append(model)
    
    table['Model'] = pd.Categorical(
        table['Model'],
        categories=ordered_models,
        ordered=True
    )
    
    table = table.sort_values(['Model', 'Method'])
    
    # Compute best values per model per column
    # For average metrics: higher is better
    # For degradation: less negative (closer to 0) is better
    best_values = {}
    for model in ordered_models:
        model_data = table[table['Model'] == model]
        if len(model_data) == 0:
            continue
        best_values[model] = {
            'Cov': model_data['Cov'].max(),
            'GP': model_data['GP'].max(),
            'CB': model_data['CB'].max(),
            # For degradation, closer to 0 is better (max of negative values)
            'ΔCov': model_data['ΔCov'].max() if 'ΔCov' in model_data.columns else np.nan,
            'ΔGP': model_data['ΔGP'].max() if 'ΔGP' in model_data.columns else np.nan,
            'ΔCB': model_data['ΔCB'].max() if 'ΔCB' in model_data.columns else np.nan,
        }
    
    # Count rows per model for multirow
    model_row_count = {}
    for model in ordered_models:
        model_row_count[model] = len(table[table['Model'] == model])
    
    # Build table rows manually
    rows = []
    current_model = None
    
    for _, row in table.iterrows():
        model = row['Model']
        method = row['Method']
        
        # Format values with bold for best
        def fmt(val, col, mdl):
            if pd.isna(val):
                return "--"
            if mdl in best_values and col in best_values[mdl]:
                is_best = abs(val - best_values[mdl][col]) < 1e-6
                if is_best:
                    return f"\\textbf{{{val:.3f}}}"
            return f"{val:.3f}"
        
        cov_str = fmt(row.get('Cov', np.nan), 'Cov', model)
        gp_str = fmt(row.get('GP', np.nan), 'GP', model)
        cb_str = fmt(row.get('CB', np.nan), 'CB', model)
        dcov_str = fmt(row.get('ΔCov', np.nan), 'ΔCov', model)
        dgp_str = fmt(row.get('ΔGP', np.nan), 'ΔGP', model)
        dcb_str = fmt(row.get('ΔCB', np.nan), 'ΔCB', model)
        
        if model != current_model:
            # New model group
            if current_model is not None:
                rows.append("\\cline{1-8}")
            
            n_rows = model_row_count.get(model, 1)
            if model == 'GPT-5-mini' or n_rows == 1:
                # Single row without multirow
                model_str = str(model)
            else:
                model_str = f"\\multirow[t]{{{n_rows}}}{{*}}{{{model}}}"
            
            rows.append(f"{model_str} & {method} & {cov_str} & {gp_str} & {cb_str} & {dcov_str} & {dgp_str} & {dcb_str} \\\\")
            current_model = model
        else:
            # Continuation of model group
            rows.append(f" & {method} & {cov_str} & {gp_str} & {cb_str} & {dcov_str} & {dgp_str} & {dcb_str} \\\\")
    
    rows_str = "\n".join(rows)
    
    caption = (
        "\\textbf{Rollout performance metrics.} "
        "Legend: Cov (Coverage), GP (Grade Proximity), CB (CodeBLEU). "
        "Metrics averaged across k=1 to k=5 steps ahead, with degradation ($\\Delta$) showing "
        "the average drop from k=1 to k=2..5. "
        "Negative $\\Delta$ values indicate performance worsens over longer rollouts. "
        "\\textbf{Bold}: best performance within each model."
    )
    
    full_latex = f"""\\begin{{table}}[htbp]
\\centering
\\caption{{{caption}}}
\\label{{tab:rollout_degradation}}
\\resizebox{{\\columnwidth}}{{!}}{{%
\\begin{{tabular}}{{ll|ccc|ccc}}
\\toprule
 &  & \\multicolumn{{3}}{{c|}}{{Average}} & \\multicolumn{{3}}{{c}}{{Degradation}} \\\\
Model & Method & Cov & GP & CB & $\\Delta$Cov & $\\Delta$GP & $\\Delta$CB \\\\
\\midrule
{rows_str}
\\bottomrule
\\end{{tabular}}
}}
\\end{{table}}
"""
    
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as f:
        f.write(full_latex)
    
    print(f"Averaged degradation LaTeX table saved to: {filename}")


def create_averaged_results_table(aggregated_df):
    """
    Create a table showing metrics averaged across all k values with degradation.
    
    Metrics: Cov, GP, CB + their degradation (ΔCov, ΔGP, ΔCB)
    """
    # For coverage: sum valid pairs and ground truth across k, then divide
    coverage_df = aggregated_df.groupby(['model_name', 'method']).agg({
        'has_ground_truth': 'sum',
        'has_valid_pair': 'sum',
    }).reset_index()
    coverage_df['Cov'] = np.where(
        coverage_df['has_ground_truth'] > 0,
        coverage_df['has_valid_pair'] / coverage_df['has_ground_truth'],
        np.nan
    )
    coverage_df = coverage_df[['model_name', 'method', 'Cov']]
    
    # For metrics: take mean across k
    metrics_df = aggregated_df.groupby(['model_name', 'method']).agg({
        'grade_proximity': 'mean',
        'codebleu': 'mean',
    }).reset_index()
    
    # Merge coverage with metrics
    averaged = coverage_df.merge(metrics_df, on=['model_name', 'method'])
    
    # Add degradation metrics
    degradation = compute_average_degradation(aggregated_df)
    
    full_table = averaged.merge(
        degradation,
        on=['model_name', 'method'],
        how='left'
    )
    
    for col in ['Cov', 'grade_proximity', 'codebleu', 'delta_gp', 'delta_cb', 'delta_cov']:
        if col in full_table.columns:
            full_table[col] = full_table[col].round(3)
    
    full_table = full_table.rename(columns={
        'model_name': 'Model',
        'method': 'Method',
        'grade_proximity': 'GP',
        'codebleu': 'CB',
        'delta_gp': 'ΔGP',
        'delta_cb': 'ΔCB',
        'delta_cov': 'ΔCov',
    })
    
    column_order = ['Model', 'Method', 'Cov', 'GP', 'CB', 'ΔCov', 'ΔGP', 'ΔCB']
    column_order = [c for c in column_order if c in full_table.columns]
    full_table = full_table[column_order]
    full_table = full_table.sort_values(['Model', 'Method'])
    
    return full_table


def compute_average_degradation(aggregated_df, max_k=5):
    """
    Compute average degradation for each model-method combination.
    
    Degradation = mean(metric[k] - metric[1]) for k in 2..max_k
    
    For GP: negative degradation = performance worsens (GP decreases)
    For GD: positive degradation = performance worsens (GD increases)
    For CB: negative degradation = performance worsens (CB decreases)
    For Cov: negative degradation = coverage drops
    """
    results = []
    
    for (model, method), group_df in aggregated_df.groupby(['model_name', 'method']):
        group_df = group_df.sort_values('k')
        k1_data = group_df[group_df['k'] == 1]
        
        if len(k1_data) == 0:
            continue
        
        k1_row = k1_data.iloc[0]
        gd_1 = k1_row['grade_difference']
        gp_1 = k1_row['grade_proximity']
        cb_1 = k1_row['codebleu']
        cov_1 = k1_row['coverage'] if 'coverage' in k1_row else None
        
        gd_diffs, gp_diffs, cb_diffs, cov_diffs = [], [], [], []
        
        for k in range(2, max_k + 1):
            k_data = group_df[group_df['k'] == k]
            if len(k_data) > 0:
                k_row = k_data.iloc[0]
                if pd.notna(k_row['grade_difference']) and pd.notna(gd_1):
                    gd_diffs.append(k_row['grade_difference'] - gd_1)
                if pd.notna(k_row['grade_proximity']) and pd.notna(gp_1):
                    gp_diffs.append(k_row['grade_proximity'] - gp_1)
                if pd.notna(k_row['codebleu']) and pd.notna(cb_1):
                    cb_diffs.append(k_row['codebleu'] - cb_1)
                if cov_1 is not None and 'coverage' in k_row and pd.notna(k_row['coverage']) and pd.notna(cov_1):
                    cov_diffs.append(k_row['coverage'] - cov_1)
        
        avg_gd_deg = sum(gd_diffs) / len(gd_diffs) if gd_diffs else np.nan
        avg_gp_deg = sum(gp_diffs) / len(gp_diffs) if gp_diffs else np.nan
        avg_cb_deg = sum(cb_diffs) / len(cb_diffs) if cb_diffs else np.nan
        avg_cov_deg = sum(cov_diffs) / len(cov_diffs) if cov_diffs else np.nan
        
        results.append({
            'model_name': model,
            'method': method,
            'delta_gd': avg_gd_deg,
            'delta_gp': avg_gp_deg,
            'delta_cb': avg_cb_deg,
            'delta_cov': avg_cov_deg,
        })
    
    return pd.DataFrame(results)


def create_success_stratified_table(aggregated_df, detailed_df):
    """
    Create a table showing metrics stratified by trajectory success.
    
    Columns: Cov, GP, CB for overall, solved (✓), and unsolved (✗) trajectories.
    """
    def compute_stratum_metrics(df):
        """Compute metrics for a stratum of the data."""
        results = []
        for (model, method), group_df in df.groupby(['model_name', 'method']):
            n_gt = group_df['has_ground_truth'].sum()
            n_valid = group_df['has_valid_pair'].sum()
            coverage = n_valid / n_gt if n_gt > 0 else np.nan
            
            results.append({
                'model_name': model,
                'method': method,
                'coverage': coverage,
                'grade_difference': group_df['grade_difference'].mean(),
                'grade_proximity': group_df['grade_proximity'].mean(),
                'codebleu': group_df['codebleu'].mean(),
            })
        return pd.DataFrame(results)
    
    # Overall metrics
    overall = compute_stratum_metrics(detailed_df)
    
    if 'trajectory_solved' not in detailed_df.columns or detailed_df['trajectory_solved'].isna().all():
        overall = overall.rename(columns={
            'model_name': 'Model', 'method': 'Method', 'coverage': 'Cov',
            'grade_proximity': 'GP', 'codebleu': 'CB',
        })
        for col in ['Cov', 'GP', 'CB']:
            if col in overall.columns:
                overall[col] = overall[col].round(3)
        return overall
    
    # Solved trajectories
    solved_df = detailed_df[detailed_df['trajectory_solved'] == True]
    if len(solved_df) > 0:
        solved_metrics = compute_stratum_metrics(solved_df)
        solved_metrics = solved_metrics.rename(columns={
            'coverage': 'Cov_solved',
            'grade_proximity': 'GP_solved',
            'codebleu': 'CB_solved',
        })
    else:
        solved_metrics = pd.DataFrame(columns=['model_name', 'method'])
    
    # Unsolved trajectories
    unsolved_df = detailed_df[detailed_df['trajectory_solved'] == False]
    if len(unsolved_df) > 0:
        unsolved_metrics = compute_stratum_metrics(unsolved_df)
        unsolved_metrics = unsolved_metrics.rename(columns={
            'coverage': 'Cov_unsolved',
            'grade_proximity': 'GP_unsolved',
            'codebleu': 'CB_unsolved',
        })
    else:
        unsolved_metrics = pd.DataFrame(columns=['model_name', 'method'])
    
    # Merge all
    full_table = overall.merge(solved_metrics, on=['model_name', 'method'], how='left')
    full_table = full_table.merge(unsolved_metrics, on=['model_name', 'method'], how='left')
    
    numeric_cols = ['coverage', 'grade_proximity', 'codebleu',
                    'Cov_solved', 'GP_solved', 'CB_solved',
                    'Cov_unsolved', 'GP_unsolved', 'CB_unsolved']
    for col in numeric_cols:
        if col in full_table.columns:
            full_table[col] = full_table[col].round(3)
    
    full_table = full_table.rename(columns={
        'model_name': 'Model', 'method': 'Method', 'coverage': 'Cov',
        'grade_proximity': 'GP', 'codebleu': 'CB',
        'Cov_solved': 'Cov(✓)', 'GP_solved': 'GP(✓)', 'CB_solved': 'CB(✓)',
        'Cov_unsolved': 'Cov(✗)', 'GP_unsolved': 'GP(✗)', 'CB_unsolved': 'CB(✗)',
    })
    
    column_order = ['Model', 'Method', 'Cov', 'GP', 'CB',
                    'Cov(✓)', 'GP(✓)', 'CB(✓)',
                    'Cov(✗)', 'GP(✗)', 'CB(✗)']
    column_order = [c for c in column_order if c in full_table.columns]
    full_table = full_table[column_order]
    full_table = full_table.sort_values(['Model', 'Method'])
    
    return full_table


# =============================================================================
# SECTION: VISUALIZATION
# =============================================================================

def create_grade_progression_plot(detailed_df, output_path='outputs/figures/grade_progression.pdf'):
    """
    Create a figure showing grade progression across normalized trajectory position.
    
    Layout:
    - Columns: Model (Qwen3-4B, Qwen3-8B)
    - X-axis: Normalized trajectory position (0 = start, 1 = end)
    - Y-axis: Normalized grade (0-1)
    - Hue: Method
    - Also shows Student (actual) as reference line
    - GPT-5-mini shown as reference line (not separate column)
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Prepare data
    plot_data = detailed_df.copy()
    
    # Apply mappings
    plot_data['Method'] = plot_data['method'].map(lambda x: METHOD_DISPLAY_MAPPING.get(x, x))
    plot_data['Model'] = plot_data['model_name'].map(lambda x: MODEL_MAPPING.get(x, x))
    
    # Compute absolute position in trajectory (1-indexed)
    # starting_position is where rollout started, k is steps into rollout
    plot_data['absolute_position'] = plot_data['starting_position'] + plot_data['k'] - 1
    
    # Get trajectory length for each student/problem combo
    # Use the maximum absolute position observed across ALL experiments for this student/problem
    traj_length = plot_data.groupby(['student_id', 'problem_id'])['absolute_position'].max().reset_index()
    traj_length.columns = ['student_id', 'problem_id', 'trajectory_length']
    
    plot_data = plot_data.merge(traj_length, on=['student_id', 'problem_id'], how='left')
    
    # Normalized position = absolute_position / trajectory_length
    # This puts all trajectories on [0, 1] scale regardless of where rollout started
    plot_data['norm_position'] = plot_data['absolute_position'] / plot_data['trajectory_length']
    plot_data['norm_position'] = plot_data['norm_position'].clip(0, 1)
    
    # Bin positions for smoother visualization (10 bins)
    plot_data['position_bin'] = (plot_data['norm_position'] * 10).round() / 10
    
    # Normalize grades to 0-1
    plot_data['norm_true_grade'] = plot_data['true_grade'] / 100.0
    plot_data['norm_pred_grade'] = plot_data['pred_grade'] / 100.0
    
    # Filter to rows with valid predicted grades
    plot_data_filtered = plot_data.dropna(subset=['norm_pred_grade'])
    
    # Define method styling
    method_order = ['BASE', 'PARA', 'SFT', 'DPO', 'DAPO']
    method_palette = {
        'BASE': '#ff0000',      # Bright Red
        'PARA': '#1f77b4',      # Blue
        'SFT': '#2ca02c',       # Green
        'DPO': '#ff7f0e',       # Orange
        'DAPO': '#e377c2',      # Pink
    }
    method_markers = {
        'BASE': 'o', 'PARA': 's', 'SFT': '^', 'DPO': 'D', 'DAPO': 'X',
    }
    method_linestyles = {
        'BASE': '-', 'PARA': '-', 'SFT': '-', 'DPO': '--', 'DAPO': '-.',
    }
    
    existing_methods = [m for m in method_order if m in plot_data_filtered['Method'].unique()]
    all_models = sorted(plot_data_filtered['Model'].unique())
    
    # Separate GPT-5-mini to use as reference line
    gpt_reference_data = None
    if 'GPT-5-mini' in all_models:
        gpt_data = plot_data_filtered[plot_data_filtered['Model'] == 'GPT-5-mini']
        if len(gpt_data) > 0:
            gpt_reference_data = gpt_data.groupby('position_bin').agg({
                'norm_pred_grade': ['mean', 'std', 'count']
            }).reset_index()
            gpt_reference_data.columns = ['position_bin', 'mean', 'std', 'count']
            gpt_reference_data['se'] = gpt_reference_data['std'] / np.sqrt(gpt_reference_data['count'])
    
    # Exclude GPT-5-mini from main columns
    models = [m for m in all_models if m != 'GPT-5-mini']
    
    if len(models) == 0:
        print("Warning: No models to plot")
        return
    
    # Prepare student reference data
    student_ref_data = plot_data.dropna(subset=['norm_true_grade'])
    
    # Set up plot
    sns.set_style("whitegrid")
    n_models = len(models)
    fig, axes = plt.subplots(1, n_models, figsize=(7 * n_models, 5.5), sharey=True)
    
    if n_models == 1:
        axes = [axes]
    
    for idx, model in enumerate(models):
        ax = axes[idx]
        model_data = plot_data_filtered[plot_data_filtered['Model'] == model]
        
        # Plot student reference line
        student_model_data = student_ref_data[student_ref_data['Model'] == model]
        if len(student_model_data) > 0:
            student_agg = student_model_data.groupby('position_bin').agg({
                'norm_true_grade': ['mean', 'std', 'count']
            }).reset_index()
            student_agg.columns = ['position_bin', 'mean', 'std', 'count']
            student_agg['se'] = student_agg['std'] / np.sqrt(student_agg['count'])
            
            ax.plot(student_agg['position_bin'], student_agg['mean'], 
                   color='black', linewidth=2.5, linestyle='--', label='Student (actual)', zorder=10)
            ax.fill_between(student_agg['position_bin'], 
                           student_agg['mean'] - 1.96 * student_agg['se'],
                           student_agg['mean'] + 1.96 * student_agg['se'],
                           color='black', alpha=0.1, zorder=1)
        
        # Plot GPT-5-mini reference line
        if gpt_reference_data is not None and len(gpt_reference_data) > 0:
            ax.plot(gpt_reference_data['position_bin'], gpt_reference_data['mean'], 
                   color='#7f7f7f', linewidth=2.5, linestyle=':', marker='d', markersize=6,
                   label='GPT-5-mini (BASE)', zorder=9)
            ax.fill_between(gpt_reference_data['position_bin'], 
                           gpt_reference_data['mean'] - 1.96 * gpt_reference_data['se'],
                           gpt_reference_data['mean'] + 1.96 * gpt_reference_data['se'],
                           color='#7f7f7f', alpha=0.1, zorder=1)
        
        # Plot each method
        for method in existing_methods:
            method_data = model_data[model_data['Method'] == method]
            if len(method_data) == 0:
                continue
            
            agg_data = method_data.groupby('position_bin').agg({
                'norm_pred_grade': ['mean', 'std', 'count']
            }).reset_index()
            agg_data.columns = ['position_bin', 'mean', 'std', 'count']
            agg_data['se'] = agg_data['std'] / np.sqrt(agg_data['count'])
            
            color = method_palette.get(method, '#333333')
            marker = method_markers.get(method, 'o')
            linestyle = method_linestyles.get(method, '-')
            zorder = 5 if method == 'BASE' else 3
            
            ax.plot(agg_data['position_bin'], agg_data['mean'], 
                   color=color, linewidth=2.5, linestyle=linestyle, 
                   marker=marker, markersize=8, label=method, zorder=zorder)
            ax.fill_between(agg_data['position_bin'], 
                           agg_data['mean'] - 1.96 * agg_data['se'],
                           agg_data['mean'] + 1.96 * agg_data['se'],
                           color=color, alpha=0.15, zorder=1)
        
        ax.set_xlabel('Normalized Trajectory Position', fontsize=17, fontweight='bold')
        if idx == 0:
            ax.set_ylabel('Normalized Grade', fontsize=17, fontweight='bold')
        ax.set_title(model, fontsize=17, fontweight='bold')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.05)
        ax.tick_params(labelsize=15, width=2, length=6)
    
    # Create unified legend - horizontal on top
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, 
               loc='upper center',
               bbox_to_anchor=(0.5, 1.02),
               ncol=len(labels),
               fontsize=13,
               frameon=True,
               fancybox=True)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for legend on top
    
    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    print(f"Grade progression plot saved to: {output_path}")
    plt.close()


def create_grade_progression_with_coverage_plot(detailed_df, output_path='outputs/figures/grade_progression_with_coverage.pdf'):
    """
    Create a figure showing GP, CB, and Coverage across normalized trajectory position.
    
    Layout:
    - 3 rows: Grade Proximity (top), CodeBLEU (middle), Coverage (bottom)
    - Columns: Model (Qwen3-4B, Qwen3-8B)
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Prepare data
    plot_data = detailed_df.copy()
    
    # Apply mappings
    plot_data['Method'] = plot_data['method'].map(lambda x: METHOD_DISPLAY_MAPPING.get(x, x))
    plot_data['Model'] = plot_data['model_name'].map(lambda x: MODEL_MAPPING.get(x, x))
    
    # Compute absolute position in trajectory (1-indexed)
    plot_data['absolute_position'] = plot_data['starting_position'] + plot_data['k'] - 1
    
    # Get trajectory length for each student/problem combo
    traj_length = plot_data.groupby(['student_id', 'problem_id'])['absolute_position'].max().reset_index()
    traj_length.columns = ['student_id', 'problem_id', 'trajectory_length']
    plot_data = plot_data.merge(traj_length, on=['student_id', 'problem_id'], how='left')
    
    # Normalized position = absolute_position / trajectory_length
    plot_data['norm_position'] = plot_data['absolute_position'] / plot_data['trajectory_length']
    plot_data['norm_position'] = plot_data['norm_position'].clip(0, 1)
    plot_data['position_bin'] = (plot_data['norm_position'] * 10).round() / 10
    
    # Define method styling
    method_order = ['BASE', 'PARA', 'SFT', 'DPO', 'DAPO']
    method_palette = {
        'BASE': '#ff0000', 'PARA': '#1f77b4', 'SFT': '#2ca02c', 'DPO': '#ff7f0e',
        'DAPO': '#e377c2',
    }
    method_markers = {
        'BASE': 'o', 'PARA': 's', 'SFT': '^', 'DPO': 'D', 'DAPO': 'X',
    }
    method_linestyles = {
        'BASE': '-', 'PARA': '-', 'SFT': '-', 'DPO': '--', 'DAPO': '-.',
    }
    
    existing_methods = [m for m in method_order if m in plot_data['Method'].unique()]
    all_models = sorted(plot_data['Model'].unique())
    
    # Separate GPT-5-mini as reference
    gpt_ref_data = {}
    if 'GPT-5-mini' in all_models:
        gpt_data = plot_data[plot_data['Model'] == 'GPT-5-mini']
        if len(gpt_data) > 0:
            # GP reference
            gpt_gp = gpt_data.dropna(subset=['grade_proximity']).groupby('position_bin').agg({
                'grade_proximity': ['mean', 'std', 'count']
            }).reset_index()
            gpt_gp.columns = ['position_bin', 'mean', 'std', 'count']
            gpt_gp['se'] = gpt_gp['std'] / np.sqrt(gpt_gp['count'])
            gpt_ref_data['gp'] = gpt_gp
            
            # CB reference
            gpt_cb = gpt_data.dropna(subset=['codebleu']).groupby('position_bin').agg({
                'codebleu': ['mean', 'std', 'count']
            }).reset_index()
            gpt_cb.columns = ['position_bin', 'mean', 'std', 'count']
            gpt_cb['se'] = gpt_cb['std'] / np.sqrt(gpt_cb['count'])
            gpt_ref_data['cb'] = gpt_cb
            
            # Coverage reference
            gpt_cov = gpt_data.groupby('position_bin').agg({
                'has_ground_truth': 'sum',
                'has_model_code': 'sum'
            }).reset_index()
            gpt_cov['coverage'] = gpt_cov['has_model_code'] / gpt_cov['has_ground_truth'].clip(lower=1)
            gpt_ref_data['cov'] = gpt_cov
    
    models = [m for m in all_models if m != 'GPT-5-mini']
    
    if len(models) == 0:
        print("Warning: No models to plot")
        return
    
    # Set up plot: 3 rows x n_models columns
    sns.set_style("whitegrid")
    n_models = len(models)
    fig, axes = plt.subplots(3, n_models, figsize=(7 * n_models, 12), sharey='row')
    
    if n_models == 1:
        axes = axes.reshape(3, 1)
    
    for col_idx, model in enumerate(models):
        model_data = plot_data[plot_data['Model'] == model]
        
        # ===== ROW 0: GRADE PROXIMITY =====
        ax_gp = axes[0, col_idx]
        
        # GPT-5-mini reference
        if 'gp' in gpt_ref_data and len(gpt_ref_data['gp']) > 0:
            ax_gp.plot(gpt_ref_data['gp']['position_bin'], gpt_ref_data['gp']['mean'], 
                      color='#7f7f7f', linewidth=2.5, linestyle=':', marker='d', markersize=6,
                      label='GPT-5-mini', zorder=9)
            ax_gp.fill_between(gpt_ref_data['gp']['position_bin'], 
                              gpt_ref_data['gp']['mean'] - 1.96 * gpt_ref_data['gp']['se'],
                              gpt_ref_data['gp']['mean'] + 1.96 * gpt_ref_data['gp']['se'],
                              color='#7f7f7f', alpha=0.1, zorder=1)
        
        # Each method
        for method in existing_methods:
            method_data = model_data[model_data['Method'] == method].dropna(subset=['grade_proximity'])
            if len(method_data) == 0:
                continue
            
            agg_data = method_data.groupby('position_bin').agg({
                'grade_proximity': ['mean', 'std', 'count']
            }).reset_index()
            agg_data.columns = ['position_bin', 'mean', 'std', 'count']
            agg_data['se'] = agg_data['std'] / np.sqrt(agg_data['count'])
            
            color = method_palette.get(method, '#333333')
            marker = method_markers.get(method, 'o')
            linestyle = method_linestyles.get(method, '-')
            
            ax_gp.plot(agg_data['position_bin'], agg_data['mean'], 
                      color=color, linewidth=2.5, linestyle=linestyle, 
                      marker=marker, markersize=8, label=method)
            ax_gp.fill_between(agg_data['position_bin'], 
                              agg_data['mean'] - 1.96 * agg_data['se'],
                              agg_data['mean'] + 1.96 * agg_data['se'],
                              color=color, alpha=0.15)
        
        ax_gp.set_title(model, fontsize=17, fontweight='bold')
        if col_idx == 0:
            ax_gp.set_ylabel('Grade Proximity (GP)', fontsize=17, fontweight='bold')
        ax_gp.set_xlim(0, 1)
        ax_gp.set_ylim(0, 1.05)
        ax_gp.tick_params(labelsize=15, width=2, length=6)
        ax_gp.set_xlabel('')
        
        # ===== ROW 1: CODEBLEU =====
        ax_cb = axes[1, col_idx]
        
        # GPT-5-mini reference
        if 'cb' in gpt_ref_data and len(gpt_ref_data['cb']) > 0:
            ax_cb.plot(gpt_ref_data['cb']['position_bin'], gpt_ref_data['cb']['mean'], 
                      color='#7f7f7f', linewidth=2.5, linestyle=':', marker='d', markersize=6,
                      label='GPT-5-mini', zorder=9)
            ax_cb.fill_between(gpt_ref_data['cb']['position_bin'], 
                              gpt_ref_data['cb']['mean'] - 1.96 * gpt_ref_data['cb']['se'],
                              gpt_ref_data['cb']['mean'] + 1.96 * gpt_ref_data['cb']['se'],
                              color='#7f7f7f', alpha=0.1, zorder=1)
        
        # Each method
        for method in existing_methods:
            method_data = model_data[model_data['Method'] == method].dropna(subset=['codebleu'])
            if len(method_data) == 0:
                continue
            
            agg_data = method_data.groupby('position_bin').agg({
                'codebleu': ['mean', 'std', 'count']
            }).reset_index()
            agg_data.columns = ['position_bin', 'mean', 'std', 'count']
            agg_data['se'] = agg_data['std'] / np.sqrt(agg_data['count'])
            
            color = method_palette.get(method, '#333333')
            marker = method_markers.get(method, 'o')
            linestyle = method_linestyles.get(method, '-')
            
            ax_cb.plot(agg_data['position_bin'], agg_data['mean'], 
                      color=color, linewidth=2.5, linestyle=linestyle, 
                      marker=marker, markersize=8, label=method)
            ax_cb.fill_between(agg_data['position_bin'], 
                              agg_data['mean'] - 1.96 * agg_data['se'],
                              agg_data['mean'] + 1.96 * agg_data['se'],
                              color=color, alpha=0.15)
        
        if col_idx == 0:
            ax_cb.set_ylabel('CodeBLEU (CB)', fontsize=17, fontweight='bold')
        ax_cb.set_xlim(0, 1)
        ax_cb.set_ylim(0, 1.05)
        ax_cb.tick_params(labelsize=15, width=2, length=6)
        ax_cb.set_xlabel('')
        
        # ===== ROW 2: COVERAGE =====
        ax_cov = axes[2, col_idx]
        
        # GPT-5-mini coverage reference
        if 'cov' in gpt_ref_data and len(gpt_ref_data['cov']) > 0:
            ax_cov.plot(gpt_ref_data['cov']['position_bin'], gpt_ref_data['cov']['coverage'], 
                       color='#7f7f7f', linewidth=2.5, linestyle=':', marker='d', markersize=6,
                       label='GPT-5-mini')
        
        # Each method coverage
        for method in existing_methods:
            method_data_all = model_data[model_data['Method'] == method]
            if len(method_data_all) == 0:
                continue
            
            cov_agg = method_data_all.groupby('position_bin').agg({
                'has_ground_truth': 'sum',
                'has_model_code': 'sum'
            }).reset_index()
            cov_agg['coverage'] = cov_agg['has_model_code'] / cov_agg['has_ground_truth'].clip(lower=1)
            
            color = method_palette.get(method, '#333333')
            marker = method_markers.get(method, 'o')
            linestyle = method_linestyles.get(method, '-')
            
            ax_cov.plot(cov_agg['position_bin'], cov_agg['coverage'], 
                       color=color, linewidth=2.5, linestyle=linestyle, 
                       marker=marker, markersize=8, label=method)
        
        ax_cov.set_xlabel('Normalized Trajectory Position', fontsize=17, fontweight='bold')
        if col_idx == 0:
            ax_cov.set_ylabel('Coverage', fontsize=17, fontweight='bold')
        ax_cov.set_xlim(0, 1)
        ax_cov.set_ylim(0, 1.05)
        ax_cov.tick_params(labelsize=15, width=2, length=6)
    
    # Legend - horizontal on top
    handles, labels = axes[0, -1].get_legend_handles_labels()
    fig.legend(handles, labels, 
               loc='upper center',
               bbox_to_anchor=(0.5, 1.02),
               ncol=len(labels),
               fontsize=13,
               frameon=True,
               fancybox=True)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave space for legend on top
    
    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    print(f"Grade progression with coverage plot saved to: {output_path}")
    plt.close()


def create_k_metrics_plot(aggregated_df, output_path='outputs/figures/metrics_by_k.pdf'):
    """
    Create a figure showing how metrics change across k steps ahead.
    
    Layout:
    - Columns: Model (Qwen3-4B, Qwen3-8B)
    - Rows: Metric (GP, CB)
    - X-axis: k (1-5 steps ahead)
    - Hue: Method
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Prepare data
    plot_data = aggregated_df.copy()
    
    # Apply mappings
    plot_data['Method'] = plot_data['method'].map(lambda x: METHOD_DISPLAY_MAPPING.get(x, x))
    plot_data['Model'] = plot_data['model_name'].map(lambda x: MODEL_MAPPING.get(x, x))
    
    # Melt to long format
    plot_data_long = plot_data.melt(
        id_vars=['Model', 'Method', 'k'],
        value_vars=['grade_proximity', 'codebleu'],
        var_name='Metric',
        value_name='Value'
    )
    
    metric_names = {
        'grade_proximity': 'Grade Proximity (GP)',
        'codebleu': 'CodeBLEU (CB)'
    }
    plot_data_long['Metric'] = plot_data_long['Metric'].map(metric_names)
    
    # Method styling
    method_order = ['BASE', 'PARA', 'SFT', 'DPO', 'DAPO']
    method_palette = {
        'BASE': '#ff0000', 'PARA': '#1f77b4', 'SFT': '#2ca02c', 'DPO': '#ff7f0e',
        'DAPO': '#e377c2',
    }
    
    existing_methods = [m for m in method_order if m in plot_data_long['Method'].unique()]
    
    # Set up plot
    sns.set_style("whitegrid")
    
    # Separate GPT-5-mini as reference
    all_models = sorted(plot_data_long['Model'].unique())
    gpt_ref_data = None
    if 'GPT-5-mini' in all_models:
        gpt_ref_data = plot_data_long[plot_data_long['Model'] == 'GPT-5-mini']
    
    models = [m for m in all_models if m != 'GPT-5-mini']
    n_models = len(models)
    
    if n_models == 0:
        print("Warning: No models to plot in k_metrics")
        return
    
    fig, axes = plt.subplots(2, n_models, figsize=(6 * n_models, 7), sharey='row')
    
    if n_models == 1:
        axes = axes.reshape(2, 1)
    
    metrics = ['Grade Proximity (GP)', 'CodeBLEU (CB)']
    
    for col_idx, model in enumerate(models):
        for row_idx, metric in enumerate(metrics):
            ax = axes[row_idx, col_idx]
            
            mask = (plot_data_long['Model'] == model) & (plot_data_long['Metric'] == metric)
            subset = plot_data_long[mask]
            
            sns.lineplot(
                data=subset,
                x='k',
                y='Value',
                hue='Method',
                hue_order=[m for m in existing_methods if m in subset['Method'].unique()],
                palette=method_palette,
                marker='o',
                markersize=10,
                linewidth=2.5,
                ax=ax,
                legend=(row_idx == 0 and col_idx == n_models - 1),
            )
            
            # GPT-5-mini reference
            if gpt_ref_data is not None:
                gpt_metric_data = gpt_ref_data[gpt_ref_data['Metric'] == metric]
                if len(gpt_metric_data) > 0:
                    gpt_agg = gpt_metric_data.groupby('k')['Value'].mean().reset_index()
                    ax.plot(gpt_agg['k'], gpt_agg['Value'], 
                           color='#7f7f7f', linewidth=2.5, linestyle=':', marker='d', markersize=8,
                           label='GPT-5-mini' if (row_idx == 0 and col_idx == n_models - 1) else None)
            
            if row_idx == 0:
                ax.set_title(model, fontsize=17, fontweight='bold')
            if col_idx == 0:
                ax.set_ylabel(metric, fontsize=17, fontweight='bold')
            else:
                ax.set_ylabel('')
            
            if row_idx == 1:
                ax.set_xlabel('Steps Ahead (k)', fontsize=17, fontweight='bold')
            else:
                ax.set_xlabel('')
            
            ax.set_xticks([1, 2, 3, 4, 5])
            ax.tick_params(labelsize=15, width=2, length=6)
    
    # Legend - horizontal on top
    if n_models > 0:
        handles, labels = axes[0, -1].get_legend_handles_labels()
        # Remove individual legend from subplot if exists
        if axes[0, -1].get_legend():
            axes[0, -1].get_legend().remove()
        fig.legend(handles, labels,
            loc='upper center',
            bbox_to_anchor=(0.5, 1.02),
            ncol=len(labels),
            fontsize=13,
            frameon=True,
            fancybox=True,
        )
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for legend on top
    
    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    print(f"K-metrics plot saved to: {output_path}")
    plt.close()


def create_k_metrics_plot_with_coverage(aggregated_df, output_path='outputs/figures/metrics_by_k_with_cov.pdf'):
    """
    Create a figure showing how metrics change across k steps ahead, including Coverage.
    
    Layout:
    - Columns: Model (Qwen3-4B, Qwen3-8B)
    - Rows: Metric (Coverage, GP, CB)
    - X-axis: k (1-5 steps ahead)
    - Hue: Method
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Prepare data
    plot_data = aggregated_df.copy()
    
    # Apply mappings
    plot_data['Method'] = plot_data['method'].map(lambda x: METHOD_DISPLAY_MAPPING.get(x, x))
    plot_data['Model'] = plot_data['model_name'].map(lambda x: MODEL_MAPPING.get(x, x))
    
    # Melt to long format - now including coverage
    plot_data_long = plot_data.melt(
        id_vars=['Model', 'Method', 'k'],
        value_vars=['coverage', 'grade_proximity', 'codebleu'],
        var_name='Metric',
        value_name='Value'
    )
    
    metric_names = {
        'coverage': 'Coverage (Cov)',
        'grade_proximity': 'Grade Proximity (GP)',
        'codebleu': 'CodeBLEU (CB)'
    }
    plot_data_long['Metric'] = plot_data_long['Metric'].map(metric_names)
    
    # Method styling
    method_order = ['BASE', 'PARA', 'SFT', 'DPO', 'DAPO']
    method_palette = {
        'BASE': '#ff0000', 'PARA': '#1f77b4', 'SFT': '#2ca02c', 'DPO': '#ff7f0e',
        'DAPO': '#e377c2',
    }
    
    existing_methods = [m for m in method_order if m in plot_data_long['Method'].unique()]
    
    # Set up plot
    sns.set_style("whitegrid")
    
    # Separate GPT-5-mini as reference
    all_models = sorted(plot_data_long['Model'].unique())
    gpt_ref_data = None
    if 'GPT-5-mini' in all_models:
        gpt_ref_data = plot_data_long[plot_data_long['Model'] == 'GPT-5-mini']
    
    models = [m for m in all_models if m != 'GPT-5-mini']
    n_models = len(models)
    
    if n_models == 0:
        print("Warning: No models to plot in k_metrics_with_coverage")
        return
    
    # 3 rows now: Coverage, GP, CB
    fig, axes = plt.subplots(3, n_models, figsize=(6 * n_models, 10), sharey='row')
    
    if n_models == 1:
        axes = axes.reshape(3, 1)
    
    metrics = ['Coverage (Cov)', 'Grade Proximity (GP)', 'CodeBLEU (CB)']
    
    for col_idx, model in enumerate(models):
        for row_idx, metric in enumerate(metrics):
            ax = axes[row_idx, col_idx]
            
            mask = (plot_data_long['Model'] == model) & (plot_data_long['Metric'] == metric)
            subset = plot_data_long[mask]
            
            sns.lineplot(
                data=subset,
                x='k',
                y='Value',
                hue='Method',
                hue_order=[m for m in existing_methods if m in subset['Method'].unique()],
                palette=method_palette,
                marker='o',
                markersize=10,
                linewidth=2.5,
                ax=ax,
                legend=(row_idx == 0 and col_idx == n_models - 1),
            )
            
            # GPT-5-mini reference
            if gpt_ref_data is not None:
                gpt_metric_data = gpt_ref_data[gpt_ref_data['Metric'] == metric]
                if len(gpt_metric_data) > 0:
                    gpt_agg = gpt_metric_data.groupby('k')['Value'].mean().reset_index()
                    ax.plot(gpt_agg['k'], gpt_agg['Value'], 
                           color='#7f7f7f', linewidth=2.5, linestyle=':', marker='d', markersize=8,
                           label='GPT-5-mini' if (row_idx == 0 and col_idx == n_models - 1) else None)
            
            if row_idx == 0:
                ax.set_title(model, fontsize=17, fontweight='bold')
            if col_idx == 0:
                ax.set_ylabel(metric, fontsize=17, fontweight='bold')
            else:
                ax.set_ylabel('')
            
            if row_idx == 2:  # Bottom row
                ax.set_xlabel('Steps Ahead (k)', fontsize=17, fontweight='bold')
            else:
                ax.set_xlabel('')
            
            ax.set_xticks([1, 2, 3, 4, 5])
            ax.tick_params(labelsize=15, width=2, length=6)
    
    # Legend - horizontal on top
    if n_models > 0:
        handles, labels = axes[0, -1].get_legend_handles_labels()
        # Remove individual legend from subplot if exists
        if axes[0, -1].get_legend():
            axes[0, -1].get_legend().remove()
        fig.legend(handles, labels,
            loc='upper center',
            bbox_to_anchor=(0.5, 1.02),
            ncol=len(labels),
            fontsize=13,
            frameon=True,
            fancybox=True,
        )
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave space for legend on top
    
    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    print(f"K-metrics plot (with coverage) saved to: {output_path}")
    plt.close()


# =============================================================================
# SECTION: MAIN ENTRY POINT
# =============================================================================

def main():
    """Main entry point for rollout evaluation."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate rollout experiments and generate tables')
    parser.add_argument('base_path', type=str, help='Base path containing experiment directories')
    parser.add_argument('--output-dir', type=str, default='outputs/tables',
                        help='Directory to save output tables (default: outputs/tables)')
    parser.add_argument('--max-k', type=int, default=5,
                        help='Maximum rollout steps to evaluate (default: 5)')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Loading experiments from: {args.base_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Max k: {args.max_k}")
    print()
    
    # Load all experiments
    results, detailed_df, aggregated_df = load_all_rollout_data(args.base_path)
    
    print(f"\nLoaded {len(results)} experiments")
    print(f"Detailed metrics: {len(detailed_df)} rows")
    print(f"Aggregated metrics: {len(aggregated_df)} rows")
    
    # Generate simple metrics table
    print("\n" + "="*80)
    print("SIMPLE METRICS TABLE (Cov, GP, CB)")
    print("="*80)
    simple_table = create_simple_metrics_table(aggregated_df)
    
    print(simple_table.to_string(index=False))
    
    simple_csv_path = os.path.join(args.output_dir, 'rollout_simple.csv')
    simple_table.to_csv(simple_csv_path, index=False)
    print(f"\nSaved to: {simple_csv_path}")
    
    simple_latex_path = os.path.join(args.output_dir, 'rollout_simple.tex')
    save_simple_metrics_table_to_latex(simple_table, simple_latex_path)
    print(f"LaTeX saved to: {simple_latex_path}")
    
    # Generate figures
    figures_dir = os.path.join(args.output_dir, '..', 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    
    # Generate k-metrics plot showing performance across rollout steps
    print("\n" + "="*80)
    print("GENERATING FIGURES")
    print("="*80)
    
    k_metrics_path = os.path.join(figures_dir, 'metrics_by_k.pdf')
    create_k_metrics_plot(aggregated_df, k_metrics_path)
    
    # Generate k-metrics plot with coverage (3 rows)
    k_metrics_cov_path = os.path.join(figures_dir, 'metrics_by_k_with_cov.pdf')
    create_k_metrics_plot_with_coverage(aggregated_df, k_metrics_cov_path)
    
    # Generate grade progression plot
    grade_prog_path = os.path.join(figures_dir, 'grade_progression.pdf')
    create_grade_progression_plot(detailed_df, grade_prog_path)
    
    # Generate grade progression with coverage plot (3 rows: GP, CB, Coverage)
    grade_prog_cov_path = os.path.join(figures_dir, 'grade_progression_with_coverage.pdf')
    create_grade_progression_with_coverage_plot(detailed_df, grade_prog_cov_path)
    
    # Generate averaged results table with degradation
    print("\n" + "="*80)
    print("AVERAGED RESULTS WITH DEGRADATION")
    print("="*80)
    averaged_table = create_averaged_results_table(aggregated_df)
    print(averaged_table.to_string(index=False))
    
    averaged_csv_path = os.path.join(args.output_dir, 'rollout_averaged.csv')
    averaged_table.to_csv(averaged_csv_path, index=False)
    print(f"\nSaved to: {averaged_csv_path}")
    
    # Save averaged degradation to LaTeX
    averaged_latex_path = os.path.join(args.output_dir, 'rollout_degradation.tex')
    save_averaged_degradation_table_to_latex(averaged_table, averaged_latex_path)
    
    # Generate success-stratified table
    print("\n" + "="*80)
    print("SUCCESS-STRATIFIED TABLE")
    print("="*80)
    stratified_table = create_success_stratified_table(aggregated_df, detailed_df)
    print(stratified_table.to_string(index=False))
    
    stratified_csv_path = os.path.join(args.output_dir, 'rollout_stratified.csv')
    stratified_table.to_csv(stratified_csv_path, index=False)
    print(f"\nSaved to: {stratified_csv_path}")
    
    # Compute and display degradation metrics
    print("\n" + "="*80)
    print("DEGRADATION ANALYSIS")
    print("="*80)
    degradation_df = compute_average_degradation(aggregated_df, max_k=args.max_k)
    print(degradation_df.to_string(index=False))
    
    degradation_csv_path = os.path.join(args.output_dir, 'rollout_degradation.csv')
    degradation_df.to_csv(degradation_csv_path, index=False)
    print(f"\nSaved to: {degradation_csv_path}")
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)
    
    return results, detailed_df, aggregated_df


if __name__ == "__main__":
    main()