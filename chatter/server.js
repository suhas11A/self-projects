// server.js
const express = require('express');
const fs = require('fs');
const path = require('path');
const app = express();
const PORT = process.env.PORT || 3000;

// ─── Middleware ────────────────────────────────────────────────────────────────
// Parse text/plain bodies so req.body is the raw string
app.use(express.text({ type: 'text/plain' }));
// (Optional) JSON parser if you ever need JSON endpoints
app.use(express.json());

// Serve all static files (index.html, your CSS, client script, .txt files if you want)
app.use(express.static(__dirname));

// ─── Helpers ──────────────────────────────────────────────────────────────────
function readFile(fname, res) {
    const p = path.join(__dirname, fname);
    fs.readFile(p, 'utf8', (err, data) => {
        if (err) {
            console.error(`Error reading ${fname}:`, err);
            return res.status(500).send(`Error reading ${fname}`);
        }
        res.type('text/plain').send(data);
    });
}

function writeFile(fname, body, res) {
    const p = path.join(__dirname, fname);
    fs.writeFile(p, body, 'utf8', err => {
        if (err) {
            console.error(`Error writing ${fname}:`, err);
            return res.status(500).send(`Error writing ${fname}`);
        }
        res.sendStatus(200);
    });
}

// ─── API Endpoints ────────────────────────────────────────────────────────────
// Read messages.txt
app.get('/api/messages', (req, res) => {
    readFile('messages.txt', res);
});
// Overwrite messages.txt with raw text body
app.post('/api/messages', (req, res) => {
    writeFile('messages.txt', req.body, res);
});

// Read status.txt
app.get('/api/status', (req, res) => {
    readFile('status.txt', res);
});
// Overwrite status.txt with raw text body
app.post('/api/status', (req, res) => {
    writeFile('status.txt', req.body, res);
});

// ─── Start Server ─────────────────────────────────────────────────────────────
app.listen(PORT, () => {
    console.log(`Server listening on http://localhost:${PORT}`);
});
