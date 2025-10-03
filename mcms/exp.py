#!/usr/bin/env python3
"""
MCMC Substitution Cipher Solver with Simulated Annealing & Restarts

Usage:
  python mcmc_subcipher_solver.py ciphertext.txt \
      -q quadgrams.txt -i 300000 -r 5 -t 1.0 -c 0.999995 -p 50000 [--seed 123]

Notes:
- The quadgrams file must have lines like: "THHE 1234" (TOKEN COUNT).
- Scoring ignores non-letters and uses A–Z only, which is required for good results.
- Start with fewer iterations to sanity-check, then scale up.
"""

import argparse
import math
import random
import sys
from typing import Dict, Tuple

ALPHABET = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

# --------------------
# Quadgrams loader
# --------------------

def load_quadgrams(filename: str) -> Tuple[Dict[str, float], float]:
    """Load counts and convert to log-probabilities. Return (probs, floor)."""
    counts: Dict[str, int] = {}
    total = 0
    with open(filename, 'r', encoding='utf-8') as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            key, cnt = parts[0].upper(), parts[1]
            try:
                c = int(cnt)
            except ValueError:
                continue
            if len(key) != 4 or any(ch not in ALPHABET for ch in key):
                continue
            counts[key] = counts.get(key, 0) + c
            total += c
    if total == 0:
        raise ValueError("Quadgram file seems empty or invalid.")

    # Convert to natural log probabilities
    probs: Dict[str, float] = {}
    for k, c in counts.items():
        probs[k] = math.log(c / total)

    # Small floor probability for unseen n-grams
    floor = math.log(0.01 / total)
    return probs, floor

# --------------------
# Helpers
# --------------------

def normalize_letters(s: str) -> str:
    """Uppercase and strip to A–Z only for scoring."""
    return ''.join(ch for ch in s.upper() if 'A' <= ch <= 'Z')


def score_text(text: str, quadgrams: Dict[str, float], floor: float) -> float:
    """Compute log-score of text using quadgrams. Non-letters are ignored."""
    t = normalize_letters(text)
    if len(t) < 4:
        return float('-inf')
    score = 0.0
    for i in range(len(t) - 3):
        quad = t[i:i+4]
        score += quadgrams.get(quad, floor)
    return score


def random_key() -> str:
    letters = list(ALPHABET)
    random.shuffle(letters)
    return ''.join(letters)


def decrypt(text: str, key: str) -> str:
    """Map ciphertext letters (A..Z) -> plaintext letters given by key.
    key[i] is the plaintext for ciphertext chr(ord('A')+i).
    Case is preserved; non-letters pass through.
    """
    mapping = {ALPHABET[i]: key[i] for i in range(26)}
    out = []
    for c in text:
        u = c.upper()
        if u in mapping:
            dec = mapping[u]
            out.append(dec if c.isupper() else dec.lower())
        else:
            out.append(c)
    return ''.join(out)


def propose_key(key: str) -> str:
    k = list(key)
    i, j = random.sample(range(26), 2)
    k[i], k[j] = k[j], k[i]
    return ''.join(k)

# --------------------
# Core MCMC with annealing
# --------------------

def mcmc_chain(
    ciphertext: str,
    quadgrams: Dict[str, float],
    floor: float,
    iterations: int,
    temp: float,
    cooling_rate: float,
    print_every: int,
) -> Tuple[str, float]:
    current_key = random_key()
    current_plain = decrypt(ciphertext, current_key)
    current_score = score_text(current_plain, quadgrams, floor)

    best_key, best_score = current_key, current_score

    try:
        for it in range(1, iterations + 1):
            candidate_key = propose_key(current_key)
            cand_plain = decrypt(ciphertext, candidate_key)
            cand_score = score_text(cand_plain, quadgrams, floor)

            delta = cand_score - current_score
            T = temp * (cooling_rate ** it)  # exponential decay

            # Metropolis criterion with annealing
            if delta >= 0 or (T > 0 and random.random() < math.exp(delta / T)):
                current_key, current_score = candidate_key, cand_score

            if current_score > best_score:
                best_key, best_score = current_key, current_score

            if print_every and it % print_every == 0:
                print(f"Iteration {it}/{iterations}, Temp {T:.5e}, best score: {best_score:.4f}")
    except KeyboardInterrupt:
        print("\n[Interrupted] Returning best-so-far from this chain…")

    return best_key, best_score

# --------------------
# CLI
# --------------------

def main():
    parser = argparse.ArgumentParser(
        description='MCMC Substitution Cipher Solver with Annealing and Restarts'
    )
    parser.add_argument('ciphertext_file', help='Path to file containing ciphertext')
    parser.add_argument('-q', '--quadgrams', default='quadgrams.txt', help='Quadgrams file')
    parser.add_argument('-i', '--iterations', type=int, default=2_000_000, help='Iterations per chain')
    parser.add_argument('-r', '--restarts', type=int, default=1, help='Number of independent chains')
    parser.add_argument('-t', '--temp', type=float, default=1.0, help='Initial temperature for annealing')
    parser.add_argument('-c', '--cooling-rate', type=float, default=0.999995, help='Exponential cooling rate')
    parser.add_argument('-p', '--print-every', type=int, default=100_000, help='Progress report frequency')
    parser.add_argument('--seed', type=int, default=None, help='Random seed (for reproducibility)')

    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    try:
        quadgrams, floor = load_quadgrams(args.quadgrams)
    except FileNotFoundError:
        print(f"Error: Quadgrams file not found: {args.quadgrams}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error loading quadgrams: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        with open(args.ciphertext_file, 'r', encoding='utf-8') as f:
            ciphertext = f.read()
    except Exception as e:
        print(f"Error reading ciphertext file: {e}", file=sys.stderr)
        sys.exit(1)

    overall_best_key = None
    overall_best_score = float('-inf')

    for run in range(1, args.restarts + 1):
        print(f"\n=== Starting chain {run}/{args.restarts} ===")
        key, score = mcmc_chain(
            ciphertext,
            quadgrams,
            floor,
            iterations=args.iterations,
            temp=args.temp,
            cooling_rate=args.cooling_rate,
            print_every=args.print_every,
        )
        print(f"Chain {run} best score: {score:.4f}\n")

        if score > overall_best_score:
            overall_best_score = score
            overall_best_key = key

    plaintext = decrypt(ciphertext, overall_best_key)

    print('=== Overall Best Decryption ===')
    print(f'Score: {overall_best_score:.4f}')
    print(f'Key mapping (cipher A..Z -> plain): {overall_best_key}')
    print('Decrypted text:\n')
    print(plaintext)


if __name__ == '__main__':
    main()
