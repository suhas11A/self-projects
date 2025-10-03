import random
import math
import argparse

# --------------------
# Load English quadgrams for scoring (natural log probabilities)
# Quadgrams file format: each line "ABCD count"
# --------------------
def load_quadgrams(filename):
    quadgrams = {}
    total = 0
    with open(filename, 'r') as f:
        for line in f:
            key, count = line.split()
            count = int(count)
            quadgrams[key] = count
            total += count
    # convert counts to natural log probabilities
    for key in quadgrams:
        quadgrams[key] = math.log(quadgrams[key] / total)
    floor = math.log(0.01 / total)
    return quadgrams, floor

# Compute log-score of text using quadgrams
def score_text(text, quadgrams, floor):
    score = 0.0
    for i in range(len(text) - 3):
        quad = text[i:i+4]
        score += quadgrams.get(quad, floor)
    return score

# Generate a random substitution key (permutation of A–Z)
def random_key():
    letters = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
    random.shuffle(letters)
    return ''.join(letters)

# Decrypt ciphertext using key mapping ciphertext->plaintext
def decrypt(text, key):
    mapping = {chr(ord('A')+i): key[i] for i in range(26)}
    result = []
    for c in text:
        if c.upper() in mapping:
            dec = mapping[c.upper()]
            result.append(dec if c.isupper() else dec.lower())
        else:
            result.append(c)
    return ''.join(result)

# Propose new key by swapping two letters
def propose_key(key):
    k = list(key)
    i, j = random.sample(range(26), 2)
    k[i], k[j] = k[j], k[i]
    return ''.join(k)

# Single MCMC chain with simulated annealing
def mcmc_chain(ciphertext, quadgrams, floor, iterations, temp, cooling_rate, print_every):
    current_key = random_key()
    current_score = score_text(decrypt(ciphertext, current_key).upper(), quadgrams, floor)
    best_key, best_score = current_key, current_score

    for it in range(1, iterations+1):
        candidate_key = propose_key(current_key)
        cand_score = score_text(decrypt(ciphertext, candidate_key).upper(), quadgrams, floor)
        delta = cand_score - current_score
        # temperature decay (exponential)
        T = temp * (cooling_rate ** it)
        # Metropolis criterion with annealing
        if delta >= 0 or (T > 0 and random.random() < math.exp(delta / T)):
            current_key, current_score = candidate_key, cand_score
            if current_score > best_score:
                best_key, best_score = current_key, current_score
        if print_every and it % print_every == 0:
            print(f"Iteration {it}/{iterations}, Temp {T:.5e}, best score: {best_score:.4f}")

    return best_key, best_score

# Command-line interface
def main():
    parser = argparse.ArgumentParser(description='MCMC Substitution Cipher Solver with Annealing and Restarts')
    parser.add_argument('ciphertext_file', help='Path to file containing ciphertext')
    parser.add_argument('-q', '--quadgrams', default='quadgrams.txt', help='Quadgrams file')
    parser.add_argument('-i', '--iterations', type=int, default=2000000, help='Iterations per chain')
    parser.add_argument('-r', '--restarts', type=int, default=1, help='Number of independent chains')
    parser.add_argument('-t', '--temp', type=float, default=1.0, help='Initial temperature for annealing')
    parser.add_argument('-c', '--cooling-rate', type=float, default=0.999995, help='Exponential cooling rate')
    parser.add_argument('-p', '--print-every', type=int, default=100000, help='Progress report frequency')
    args = parser.parse_args()

    quadgrams, floor = load_quadgrams(args.quadgrams)
    with open(args.ciphertext_file, 'r') as f:
        ciphertext = f.read().strip()

    overall_best_key = None
    overall_best_score = float('-inf')

    for run in range(1, args.restarts + 1):
        print(f"\n=== Starting chain {run}/{args.restarts} ===")
        key, score = mcmc_chain(
            ciphertext, quadgrams, floor,
            iterations=args.iterations,
            temp=args.temp,
            cooling_rate=args.cooling_rate,
            print_every=args.print_every
        )
        print(f"Chain {run} best score: {score:.4f}\n")
        if score > overall_best_score:
            overall_best_score = score
            overall_best_key = key

    plaintext = decrypt(ciphertext, overall_best_key)
    print('=== Overall Best Decryption ===')
    print(f'Score: {overall_best_score:.4f}')
    print(f'Key mapping A->Z: {overall_best_key}')
    print('Decrypted text:\n', plaintext)

if __name__ == '__main__':
    main()
