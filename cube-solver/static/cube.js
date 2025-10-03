document.getElementById('cubeForm').onsubmit = async e => {
    e.preventDefault();
    const stateStr = document.getElementById('stateInput').value;
    const state = stateStr.split('').map(n => parseInt(n, 10));
    const res = await fetch('/solve', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ state })
    });
    const json = await res.json();
    if (json.error) {
        document.getElementById('solution').textContent = 'Error: ' + json.error;
    } else {
        document.getElementById('solution').textContent = json.moves.join(' ');
    }
};
