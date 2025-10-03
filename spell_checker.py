"""
spell_checker.py

A simple spell-checker and auto-suggestion module with persistent storage.

Features:
- Download or load a word list.
- Build or load a serialized Trie for prefix-based suggestions.
- Compute enhanced Levenshtein distance using keyboard proximity for substitutions.
- Rank suggestions by edit distance and weighted word frequency.
- Persist the Trie to disk to avoid rebuilding on each run.
"""

import os
import pickle
import requests
import math

# Optional: use wordfreq for ranking by frequency
try:
    from wordfreq import zipf_frequency
except ImportError:
    def zipf_frequency(word, lang):
        return 0.0

# URL to the word list (DWYL words_alpha)
WORD_LIST_URL = "https://raw.githubusercontent.com/dwyl/english-words/master/words_alpha.txt"

# Cache filenames
TRIE_FILEPATH = "trie.pkl"
WORDS_FILEPATH = "words.pkl"

# Weights for composite scoring
DIST_WEIGHT = 1.0      # weight for edit distance
FREQ_WEIGHT = 17.0      # weight for frequency importance (higher => more emphasis)

# Keyboard layout positions for QWERTY
# Coordinates approximate: row, column
_KEYBOARD_POS = {}
_rows = [
    list("qwertyuiop"),
    list("asdfghjkl"),
    list("zxcvbnm")
]
for row_idx, row in enumerate(_rows):
    for col_idx, key in enumerate(row):
        _KEYBOARD_POS[key] = (row_idx, col_idx)


def keyboard_distance(k1, k2):
    """
    Euclidean distance between two keys on the keyboard; fallback cost=1 if unknown.
    """
    p1 = _KEYBOARD_POS.get(k1.lower())
    p2 = _KEYBOARD_POS.get(k2.lower())
    if p1 and p2:
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])
    return 1.0


def download_word_list(url=WORD_LIST_URL):
    """
    Download the word list from the given URL and return a list of words.
    """
    resp = requests.get(url)
    resp.raise_for_status()
    return resp.text.splitlines()

# --------------------
# Trie Implementation
# --------------------
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        node = self.root
        for char in word:
            node = node.children.setdefault(char, TrieNode())
        node.is_end_of_word = True

    def prefix_suggestions(self, prefix, limit=10):
        """
        Return up to `limit` words starting with `prefix`.
        """
        results = []
        node = self.root
        for char in prefix:
            if char not in node.children:
                return results
            node = node.children[char]
        self._dfs(node, prefix, results, limit)
        return results

    def _dfs(self, node, prefix, results, limit):
        if len(results) >= limit:
            return
        if node.is_end_of_word:
            results.append(prefix)
        for char, child in node.children.items():
            self._dfs(child, prefix + char, results, limit)

# --------------------
# Persistence Helpers
# --------------------

def save_to_disk(obj, filepath):
    """Serialize an object to disk using pickle."""
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)


def load_from_disk(filepath):
    """Load a pickled object from disk."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)

# --------------------
# Enhanced Levenshtein Distance
# --------------------

def levenshtein_distance(s1, s2):
    """
    Compute the Levenshtein distance between two strings, using keyboard distance for substitutions.
    """
    m, n = len(s1), len(s2)
    # initialize dp matrix with floats
    dp = [[0.0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        dp[i][0] = i
    for j in range(1, n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                cost = 0.0
            else:
                cost = keyboard_distance(s1[i - 1], s2[j - 1])
            dp[i][j] = min(
                dp[i - 1][j] + 1.0,        # deletion
                dp[i][j - 1] + 1.0,        # insertion
                dp[i - 1][j - 1] + cost    # substitution
            )
    return dp[m][n]

# ---------------------------------
# Approximate suggestions & ranking
# ---------------------------------
def get_approximate_suggestions(word, word_list, max_distance=2.0, limit=10):
    """
    Return up to `limit` words within `max_distance` edit distance from `word`,
    ranked by a composite score combining edit distance and frequency.
    """
    candidates = []
    for w in word_list:
        dist = levenshtein_distance(word, w)
        if dist <= max_distance:
            freq = zipf_frequency(w, 'en')
            score = DIST_WEIGHT * dist - FREQ_WEIGHT * freq
            candidates.append((w, score))
    candidates.sort(key=lambda x: (x[1], x[0]))
    return [w for w, _ in candidates[:limit]]

# --------------------
# Module Initialization
# --------------------
# Load or build word list
if os.path.exists(WORDS_FILEPATH):
    words = load_from_disk(WORDS_FILEPATH)
else:
    words = download_word_list()
    save_to_disk(words, WORDS_FILEPATH)

# Load or build Trie
if os.path.exists(TRIE_FILEPATH):
    trie = load_from_disk(TRIE_FILEPATH)
else:
    trie = Trie()
    for w in words:
        trie.insert(w)
    save_to_disk(trie, TRIE_FILEPATH)

# --------------------
# Example Usage
# --------------------
if __name__ == "__main__":
    prefix = "exam"
    print(f"Prefix suggestions for '{prefix}':", trie.prefix_suggestions(prefix, limit=10))

    test_word = "examl"
    print(f"Approximate suggestions for '{test_word}':", get_approximate_suggestions(test_word, words, max_distance=3.0, limit=10))
