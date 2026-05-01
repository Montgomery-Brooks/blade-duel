/**
 * network.js — WebSocket client for Blade Duel online
 *
 * Auto-detects the server URL:
 *   - Local dev:  ws://localhost:8000/ws
 *   - Production: wss://your-app.railway.app/ws
 */

export class GameNetwork {
    constructor() {
        // Auto-detect URL based on where the page is hosted
        const protocol = location.protocol === "https:" ? "wss" : "ws";
        const host     = location.host;
        this.url       = `${protocol}://${host}/ws`;

        this.ws              = null;
        this.playerId        = null;
        this.matchId         = null;
        this._selectedCharacter = null;

        // Callbacks — override these in your game code
        this.onState    = (state) => {};
        this.onAssigned = (pid)   => {};
        this.onError    = (msg)   => {};
        this.onClose    = ()      => {};

        // Input state
        this._keys           = new Set();
        this._inputInterval  = null;
    }

    connect() {
        this.ws = new WebSocket(this.url);

        this.ws.onopen = () => {
            console.log(`[net] connected to ${this.url}`);
            this._startInputLoop();
        };

        this.ws.onmessage = (e) => {
            let msg;
            try { msg = JSON.parse(e.data); }
            catch { return; }

            if (msg.type === "assigned") {
                this.playerId = msg.player_id;
                this.matchId  = msg.match_id;
                this.onAssigned(msg.player_id);
            } else if (msg.type === "state") {
                this.onState(msg);
            } else if (msg.type === "error") {
                this.onError(msg.message);
            }
        };

        this.ws.onclose = () => {
            console.log("[net] disconnected");
            this._stopInputLoop();
            this.onClose();
        };

        this.ws.onerror = (e) => {
            console.error("[net] error", e);
        };

        this._bindKeys();
    }

    // ── Key bindings ──────────────────────────────────────────────────────────
    // Both players use their own device in online play
    // so all keys map to the local player's controls

    _bindKeys() {
        const keyMap = {
            // Movement
            KeyA: "left",    ArrowLeft:  "left",
            KeyD: "right",   ArrowRight: "right",
            // Attack
            KeyF: "attack",  KeyK: "attack",
            // Block
            KeyG: "block",   KeyL: "block",
            // Pause
            KeyP: "pause",
        };

        document.addEventListener("keydown", (e) => {
            const action = keyMap[e.code];
            if (!action) return;
            e.preventDefault();
            if (action === "pause") { this.pause(); return; }
            this._keys.add(action);
        });

        document.addEventListener("keyup", (e) => {
            const action = keyMap[e.code];
            if (action) this._keys.delete(action);
        });
    }

    _startInputLoop() {
        this._inputInterval = setInterval(() => {
            if (this._keys.size > 0) {
                this._send({ type: "input", keys: [...this._keys] });
            }
        }, 1000 / 60);
    }

    _stopInputLoop() {
        clearInterval(this._inputInterval);
    }

    // ── Public actions ────────────────────────────────────────────────────────

    ready(character) {
        this._send({ type: "ready", character });
    }

    pause() {
        this._send({ type: "pause" });
    }

    nextRound() {
        this._send({ type: "next_round" });
    }

    // ── Internal ──────────────────────────────────────────────────────────────

    _send(payload) {
        if (this.ws?.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify(payload));
        }
    }
}
