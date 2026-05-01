/**
 * network.js — WebSocket client for Blade Duel online
 * Supports pvp, pve, and ava modes via URL path: /ws/{mode}
 */

export class GameNetwork {
    constructor(mode = 'pvp') {
        const protocol = location.protocol === 'https:' ? 'wss' : 'ws';
        const host     = location.host;
        this.url       = `${protocol}://${host}/ws/${mode}`;
        this.mode      = mode;

        this.ws              = null;
        this.playerId        = null;
        this.matchId         = null;
        this._selectedCharacter = null;

        this.onState    = () => {};
        this.onAssigned = () => {};
        this.onError    = () => {};
        this.onClose    = () => {};

        this._keys          = new Set();
        this._inputInterval = null;
    }

    connect() {
        this.ws = new WebSocket(this.url);

        this.ws.onopen = () => {
            console.log(`[net] connected — mode: ${this.mode}`);
            this._startInputLoop();
        };

        this.ws.onmessage = (e) => {
            let msg;
            try { msg = JSON.parse(e.data); } catch { return; }
            if (msg.type === 'assigned') {
                this.playerId = msg.player_id;
                this.matchId  = msg.match_id;
                this.onAssigned(msg.player_id);
            } else if (msg.type === 'state') {
                this.onState(msg);
            } else if (msg.type === 'error') {
                this.onError(msg.message);
            }
        };

        this.ws.onclose  = () => { this._stopInputLoop(); this.onClose(); };
        this.ws.onerror  = (e) => console.error('[net] error', e);

        this._bindKeys();
    }

    _bindKeys() {
        const keyMap = {
            KeyA:'left', ArrowLeft:'left',
            KeyD:'right', ArrowRight:'right',
            KeyF:'attack', KeyK:'attack',
            KeyG:'block', KeyL:'block',
            KeyP:'pause',
        };
        document.addEventListener('keydown', (e) => {
            const a = keyMap[e.code];
            if (!a) return;
            e.preventDefault();
            if (a === 'pause') { this.pause(); return; }
            this._keys.add(a);
        });
        document.addEventListener('keyup', (e) => {
            const a = keyMap[e.code];
            if (a) this._keys.delete(a);
        });
    }

    _startInputLoop() {
        this._inputInterval = setInterval(() => {
            if (this._keys.size > 0)
                this._send({ type: 'input', keys: [...this._keys] });
        }, 1000 / 60);
    }

    _stopInputLoop() { clearInterval(this._inputInterval); }

    ready(character) { this._send({ type: 'ready', character }); }
    pause()          { this._send({ type: 'pause' }); }
    nextRound()      { this._send({ type: 'next_round' }); }

    _send(payload) {
        if (this.ws?.readyState === WebSocket.OPEN)
            this.ws.send(JSON.stringify(payload));
    }
}
