"""
Code normalization utilities for robust code comparison.

Provides :func:`robust_normalize` which converts a Python program into a
canonical hash that is invariant to identifier renaming and minor formatting
differences.  Used throughout DPO and GRPO to detect duplicate submissions.

Normalization pipeline:
  1. Parse with ``ast`` and rename all identifiers to ``var_N`` / ``func_N``.
  2. Dump the normalized AST to a string and MD5-hash it.
  3. If parsing fails, fall back to ``libcst`` (tolerant parser) and hash the
     canonical source it produces.
  4. If both parsers fail, hash the raw source string.
"""

import ast
import hashlib
import libcst as cst


class NormalizeIdentifiers(ast.NodeTransformer):
    """
    AST node transformer that renames all variables and functions to canonical
    names (``var_0``, ``var_1``, …, ``func_0``, ``func_1``, …).

    This makes two programs that differ only in identifier names compare as
    equal after normalization.
    """
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

    def visit_ClassDef(self, node):
        # Optional: you could normalize class names as well
        return self.generic_visit(node)


def normalize_code_to_ast_string(source_code):
    """
    Parse ``source_code``, rename all identifiers, and return the AST dump.

    Returns:
        str or None: Normalized AST string, or ``None`` if parsing fails.
    """
    try:
        # Step 1: Parse code to AST
        tree = ast.parse(source_code)
    except SyntaxError:
        return None

    # Step 2: Normalize identifiers
    normalizer = NormalizeIdentifiers()
    normalized_tree = normalizer.visit(tree)
    ast.fix_missing_locations(normalized_tree)

    # Step 3: Dump normalized AST (without line numbers or column offsets)
    normalized_ast = ast.dump(normalized_tree, annotate_fields=True, include_attributes=False)
    return normalized_ast


def normalize_with_libcst(code):
    """
    Produce a canonical source string using the libcst tolerant parser.

    Returns:
        str or None: Canonical source, or ``None`` if libcst cannot parse the code.
    """
    try:
        module = cst.parse_module(code)
        # You can traverse the CST like an AST, but it never crashes
        return module.code  # Returns the canonicalized version
    except Exception:
        return None
    


def code_to_hash(normalized_ast: str) -> str:
    """Return the MD5 hex digest of the given string."""
    return hashlib.md5(normalized_ast.encode('utf-8')).hexdigest()


def robust_normalize(code):
    """
    Produce a stable hash for a Python program, invariant to identifier names.

    Tries the full AST-based normalization first, then libcst, then raw source.

    Args:
        code (str): Python source code.

    Returns:
        str: MD5 hex string that can be compared across programs.
    """
    normalized_ast = normalize_code_to_ast_string(code)
    if normalized_ast is not None:
        return code_to_hash(normalized_ast)
    
    # Try tolerant parser
    red = normalize_with_libcst(code)
    if red is not None:
        return code_to_hash(red)

    # Final fallback: hash raw source
    return code_to_hash(code)
