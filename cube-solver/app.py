from flask import Flask, request, jsonify

# Tell Flask that static files live in the "static" folder,
# and that they should be served at the root URL.
app = Flask(
    __name__,
    static_folder='static',
    static_url_path=''     # <-- serve static/ at /
)

from solver import solve

@app.route('/solve', methods=['POST'])
def api_solve():
    data = request.get_json(force=True)
    state = data.get('state')
    if not isinstance(state, list) or len(state) != 54:
        return jsonify({'error': 'state must be a list of 54 ints'}), 400
    moves = solve(state)
    return jsonify({'moves': moves})

# When the user visits "/", serve static/index.html automatically:
@app.route('/')
def index():
    return app.send_static_file('index.html')

if __name__ == '__main__':
    app.run(debug=True)
