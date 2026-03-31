"""
Code similarity and distance metrics.

Provides multiple distance functions between two Python code strings:
  - **String edit distance** (``str_dist``): character-level SED.
  - **Sequence edit distance** (``seq_dist``): token-level SED.
  - **Tree edit distance** (``ted_dist``): AST-level TED.
  - Normalized versions (divided by max length) and relative patch size
    versions (divided by source length) are provided for each.
  - **BLEU** and **ROUGE** variants adapted for code tokens.

Note: ``codebleu_dist`` is commented out and requires additional setup.
``REPO_ABSOLUTE_PATH`` must be set if using ``codebleu_dist``.
"""

import os
import ast
from edist.ted import standard_ted
from edist.sed import sed_string, standard_sed
from tokenize_rt import src_to_tokens
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu
# from src.codebleu.my_codebleu import calc_codebleu

# Absolute path to repo root — only needed for codebleu_dist
REPO_ABSOLUTE_PATH = ""


# =============================================================================
# Classical distances
# =============================================================================

def str_dist(buggy, corrected):
    """Character-level string edit distance."""
    return sed_string(buggy, corrected)


def seq_dist(buggy, corrected):
    """Token-level sequence edit distance."""
    b_tokens = [t.src for t in src_to_tokens(buggy)]
    c_tokens = [t.src for t in src_to_tokens(corrected)]
    return standard_sed(b_tokens, c_tokens)


def ted_dist(buggy, corrected):
    """AST tree edit distance."""
    x_nodes, x_adj = ast_to_passen_repre(ast.parse(buggy))
    y_nodes, y_adj = ast_to_passen_repre(ast.parse(corrected))
    return standard_ted(x_nodes, x_adj, y_nodes, y_adj)


# =============================================================================
# Normalized distances (divided by max sequence length)
# =============================================================================

def str_norm_dist(buggy, corrected):
    """Character-level SED normalized by the length of the longer string."""
    return sed_string(buggy, corrected) / max(len(buggy), len(corrected))


def seq_norm_dist(buggy, corrected):
    """Token-level SED normalized by the length of the longer token sequence."""
    b_tokens = [t.src for t in src_to_tokens(buggy)]
    c_tokens = [t.src for t in src_to_tokens(corrected)]
    return standard_sed(b_tokens, c_tokens) / max(len(b_tokens), len(c_tokens))


def ted_norm_dist(buggy, corrected):
    """AST TED normalized by the size of the larger tree."""
    x_nodes, x_adj = ast_to_passen_repre(ast.parse(buggy))
    y_nodes, y_adj = ast_to_passen_repre(ast.parse(corrected))
    return standard_ted(x_nodes, x_adj, y_nodes, y_adj) / max(len(x_nodes), len(y_nodes))


# =============================================================================
# Relative patch size distances (divided by source length)
# =============================================================================

def str_rps_dist(buggy, corrected):
    """Character-level SED normalized by the length of the source string."""
    return sed_string(buggy, corrected) / len(buggy)


def seq_rps_dist(buggy, corrected):
    """Token-level SED normalized by the number of tokens in the source."""
    b_tokens = [t.src for t in src_to_tokens(buggy)]
    c_tokens = [t.src for t in src_to_tokens(corrected)]
    return standard_sed(b_tokens, c_tokens) / len(b_tokens)


def ted_rps_dist(buggy, corrected):
    """AST TED normalized by the size of the source tree."""
    x_nodes, x_adj = ast_to_passen_repre(ast.parse(buggy))
    y_nodes, y_adj = ast_to_passen_repre(ast.parse(corrected))
    return standard_ted(x_nodes, x_adj, y_nodes, y_adj) / len(x_nodes)


# =============================================================================
# NLP-adapted distances
# =============================================================================

def bleu_dist(buggy, corrected):
    """BLEU-based distance: ``1 - BLEU(buggy, corrected)``."""
    b_tokens = [t.src for t in src_to_tokens(buggy)]
    c_tokens = [t.src for t in src_to_tokens(corrected)]
    return 1 - sentence_bleu([b_tokens], c_tokens)

def codebleu_dist(buggy, corrected):
    return 1 - (calc_codebleu(lang="python", 
                         references = [[buggy]], predictions = [corrected], 
                         kw_dir=os.path.join(REPO_ABSOLUTE_PATH, "src/codebleu"),
                         langso_dir=os.path.join(REPO_ABSOLUTE_PATH, "src/codebleu/my-languages.so"))['CodeBLEU'])

scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL', 'rougeLsum'])
    
def rouge1_dist(buggy, correction):
    """ROUGE-1 F1 distance: ``1 - ROUGE-1 F1``."""
    return 1 - scorer.score(buggy, correction)['rouge1'][-1]


def rouge2_dist(buggy, correction):
    """ROUGE-2 F1 distance: ``1 - ROUGE-2 F1``."""
    return 1 - scorer.score(buggy, correction)['rouge2'][-1]


def rougel_dist(buggy, correction):
    """ROUGE-L F1 distance: ``1 - ROUGE-L F1``."""
    return 1 - scorer.score(buggy, correction)['rougeL'][-1]


def rougelcsum_dist(buggy, correction, get_score=False):
    """
    ROUGE-Lsum F1 distance.

    Args:
        get_score (bool): If True, return the raw F1 score instead of the distance.
    """
    score = scorer.score(buggy, correction)['rougeLsum'][-1]
    return score if get_score else 1 - score

def ast_to_passen_repre(sc_ast):
    """ Transforms a Python AST into the representation
    used for computing the tree edit distance used in 
    the python-edit-distance library 
    """
    adj_list = []
    n_list = []
    i = 0
    
    def dfs(node, i):
        node_name = str(node.__class__.__name__)
        adj_list.append([])
        n_list.append(node_name)
        node_adj_list = []
        for j, c in enumerate(ast.iter_child_nodes(node)):
            dfs(c, i + 1 + j)
            node_adj_list.append(i + 1 + j)
        adj_list[i] = node_adj_list
        
    dfs(sc_ast, i)
    
    return n_list, adj_list