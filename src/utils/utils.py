"""
General-purpose string and collection utilities.
"""

from string import Formatter
import difflib


def format_string(info_source, string):
    """
    Format ``string`` by filling placeholders from ``info_source``.

    Works with both attribute-style objects and dict-like objects.

    Args:
        info_source: Object or dict providing placeholder values.
        string (str): Format string with ``{name}``-style placeholders.

    Returns:
        str: Formatted string.
    """
    if hasattr(info_source, string):
        getter = lambda n: getattr(info_source, n) 
    else:
        getter = lambda n: info_source[n]
        
    mapping = {name: getter(name)
               for name in get_format_names(string)}
    
    return  string.format(**mapping)


def get_format_names(string):
    """Return a list of all placeholder names in a format string."""
    return [fn for _, fn, _, _ in Formatter().parse(string) if fn is not None]


def duplicate(l, n):
    """
    Repeat each element of ``l`` exactly ``n`` times in place.

    Example: ``duplicate([1, 2], 3)`` → ``[1, 1, 1, 2, 2, 2]``.
    """
    return [val for val in l for _ in range(n)]


def find_closest_key(d, target_key):
    """
    Return the key in ``d`` whose string is most similar to ``target_key``.

    Uses ``difflib.SequenceMatcher`` ratio as the similarity measure.

    Args:
        d (dict): Dictionary to search.
        target_key (str): Key to match against.

    Returns:
        The key in ``d`` with the highest similarity ratio to ``target_key``.
    """
    # Helper function to compute the similarity ratio between two strings
    def string_overlap(key1, key2):
        return difflib.SequenceMatcher(None, key1, key2).ratio()
    
    # Find the key with the maximum string overlap
    closest_key = max(d.keys(), key=lambda k: string_overlap(k, target_key))
    
    # Return the value associated with the closest key
    return closest_key