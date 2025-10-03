# solver.py

from copy import deepcopy

# -- Cube representation --------------------------------------------------

class Cube:
    def __init__(self, state=None):
        # If you pass a 54-long list, use that; else start solved.
        self.state = state.copy() if state else [f for f in range(6) for _ in range(9)]

    def copy(self):
        return Cube(self.state)

    def is_solved(self):
        # each face has same color as its center
        return all(self.state[i] == self.state[9*(i//9)+4] for i in range(54))


# -- Moves definition ----------------------------------------------------

MOVE_CYCLES = {
    'U': [[0,2,8,6], [1,5,7,3],
          [36,18,9,45], [37,19,10,46], [38,20,11,47]],
    'R': [[9,11,17,15], [10,14,16,12],
          [2,20,29,47], [5,23,32,50], [8,26,35,53]],
    'F': [[18,20,26,24], [19,23,25,21],
          [6,36,27,11], [7,39,28,14], [8,42,29,17]],
    'D': [[27,33,35,29], [28,30,34,32],
          [24,42,51,15], [25,43,52,16], [26,44,53,17]],
    'L': [[36,38,44,42], [37,41,43,39],
          [0,45,33,18], [3,48,30,21], [6,51,27,24]],
    'B': [[45,47,53,51], [46,50,52,48],
          [2,13,31,44], [1,16,34,41], [0,19,37,28]],
}
MOVES = ['U','U\'','R','R\'','F','F\'','D','D\'','L','L\'','B','B\'']

def apply_move(cube, move):
    new = cube.state.copy()
    for cycle in MOVE_CYCLES[move]:
        a,b,c,d = cycle
        new[a],new[b],new[c],new[d] = cube.state[d],cube.state[a],cube.state[b],cube.state[c]
    cube.state = new

def do(cube, mv):
    if mv.endswith('\''):
        for _ in range(3):
            apply_move(cube, mv[0])
    else:
        apply_move(cube, mv)

# -- Heuristic -----------------------------------------------------------

def heuristic(cube):
    h = 0
    for face in range(6):
        center = cube.state[9*face + 4]
        for i in range(9):
            if cube.state[9*face + i] != center:
                h += 1
    return (h + 7)//8

# -- IDA* search ---------------------------------------------------------

def ida_star(root):
    bound = heuristic(root)
    path = []

    def search(node, g, bound, last_move, visited):
        f = g + heuristic(node)
        if f > bound:
            return f
        if node.is_solved():
            return True
        minimum = float('inf')
        for mv in MOVES:
            # prune immediate inverse
            if last_move and mv[0] == last_move[0] and len(mv) != len(last_move):
                continue
            child = node.copy()
            do(child, mv)
            key = tuple(child.state)
            if key in visited:
                continue
            visited.add(key)
            path.append(mv)
            t = search(child, g+1, bound, mv, visited)
            if t is True:
                return True
            if t < minimum:
                minimum = t
            path.pop()
            visited.remove(key)
        return minimum

    while True:
        t = search(root, 0, bound, None, {tuple(root.state)})
        if t is True:
            return path
        bound = t

# -- Wrapper to expose to Flask ------------------------------------------

def solve(state):
    """
    state: list of 54 ints (0–5)
    returns: list of moves, e.g. ['R','U','R\'',...]
    """
    cube = Cube(state)
    return ida_star(cube)
