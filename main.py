"""
Blade Duel — Online Multiplayer Server
=======================================
Deployed on Railway. Serves both the WebSocket game server
and the HTML5 canvas client as static files.

Local dev:  uvicorn main:app --reload --port 8000
Production: Railway runs this automatically via Procfile
"""

import asyncio
import json
import uuid
import time
import os
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from game_state import GameState
import data_logger as db

# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(title="Blade Duel")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

db.init_db()

# ── Connection manager ────────────────────────────────────────────────────────

class ConnectionManager:

    def __init__(self):
        self.connections: dict[str, dict[int, WebSocket]] = {}

    def add(self, match_id: str, player_id: int, ws: WebSocket):
        self.connections.setdefault(match_id, {})[player_id] = ws

    def remove(self, match_id: str, player_id: int):
        if match_id in self.connections:
            self.connections[match_id].pop(player_id, None)

    async def send(self, ws: WebSocket, payload: dict):
        try:
            await ws.send_text(json.dumps(payload))
        except Exception:
            pass

    async def broadcast(self, match_id: str, payload: dict):
        text = json.dumps(payload)
        dead = []
        for pid, ws in list(self.connections.get(match_id, {}).items()):
            try:
                await ws.send_text(text)
            except Exception:
                dead.append(pid)
        for pid in dead:
            self.remove(match_id, pid)


manager = ConnectionManager()

# ── Matchmaking ───────────────────────────────────────────────────────────────

matches: dict[str, GameState] = {}
_lobby_match_id: Optional[str] = None


def get_or_create_lobby() -> GameState:
    global _lobby_match_id
    if _lobby_match_id and _lobby_match_id in matches:
        state = matches[_lobby_match_id]
        if len(state.players) < 2:
            return state
    mid = str(uuid.uuid4())[:8]
    matches[mid] = GameState(match_id=mid, phase="waiting")
    _lobby_match_id = mid
    return matches[mid]


def assign_player_id(state: GameState) -> Optional[int]:
    for pid in [1, 2]:
        if pid not in state.players:
            return pid
    return None

# ── Data logging ──────────────────────────────────────────────────────────────

def _log_round(state: GameState):
    p1 = state.players.get(1)
    p2 = state.players.get(2)
    db.log_round_result(
        match_id=state.match_id,
        round_number=state.round_number,
        winner_id=state.winner_id or 0,
        duration=99.0 - state.round_timer,
        p1_health=p1.health if p1 else 0,
        p2_health=p2.health if p2 else 0,
    )
    if state.phase == "game_over":
        db.log_match_end(state.match_id, state.match_winner_id or 0, state.round_number)

# ── Game loop ─────────────────────────────────────────────────────────────────

TICK_RATE = 1 / 60


async def game_loop(match_id: str):
    while match_id in matches:
        try:
            state = matches[match_id]
            if state.phase == "fighting":
                prev = state.phase
                state.tick_timer()
                if state.phase != prev and state.phase in ("round_end", "game_over"):
                    _log_round(state)
            await manager.broadcast(match_id, {"type": "state", **state.to_dict()})
        except Exception as e:
            print(f"[game_loop error] {e}")
        await asyncio.sleep(TICK_RATE)

# ── Input processing ──────────────────────────────────────────────────────────

ATTACK_COOLDOWN = 0.5
_last_attack: dict[str, float] = {}


def process_input(state: GameState, player_id: int, keys: list, match_id: str):
    player = state.players.get(player_id)
    if not player or state.phase != "fighting":
        return

    opponent_id = 2 if player_id == 1 else 1
    dx = -1 if "left" in keys else (1 if "right" in keys else 0)
    state.move_player(player_id, dx, 0)
    state.player_block(player_id, "block" in keys)

    if "attack" in keys:
        key = f"{match_id}:{player_id}"
        now = time.time()
        if now - _last_attack.get(key, 0) >= ATTACK_COOLDOWN:
            _last_attack[key] = now
            hit = state.player_attack(player_id, opponent_id)
            db.log_action(match_id, player_id, "attack")
            if hit:
                opp = state.players.get(opponent_id)
                db.log_hit(match_id, player_id, opponent_id,
                           state.attack_damage,
                           opp.action == "blocking" if opp else False,
                           opp.health if opp else 0)
                if state.phase in ("round_end", "game_over"):
                    _log_round(state)

# ── WebSocket ─────────────────────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    print("[ws] accepted")

    state     = get_or_create_lobby()
    match_id  = state.match_id
    player_id = assign_player_id(state)

    if player_id is None:
        await manager.send(ws, {"type": "error", "message": "Match is full."})
        await ws.close()
        return

    manager.add(match_id, player_id, ws)
    await manager.send(ws, {
        "type": "assigned", "player_id": player_id, "match_id": match_id
    })
    print(f"[ws] player {player_id} joined match {match_id}")

    if len(manager.connections.get(match_id, {})) == 1:
        asyncio.create_task(game_loop(match_id))

    try:
        async for raw in ws.iter_text():
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue

            msg_type = msg.get("type")

            if msg_type == "join":
                name = msg.get("name", f"Player {player_id}")
                state.add_player(player_id, name)
                db.log_match_start(match_id, {
                    pid: p.name for pid, p in state.players.items()
                })
                if state.all_players_connected():
                    state.phase = "character_select"
                await manager.broadcast(match_id, {"type": "state", **state.to_dict()})

            elif msg_type == "ready":
                character = msg.get("character", "warrior")
                state.set_player_ready(player_id, character)
                if state.phase == "character_select" and state.all_players_ready():
                    state.start_round()
                await manager.broadcast(match_id, {"type": "state", **state.to_dict()})

            elif msg_type == "input":
                process_input(state, player_id, msg.get("keys", []), match_id)

            elif msg_type == "pause":
                if state.phase in ("fighting", "paused"):
                    state.toggle_pause(player_id)
                    await manager.broadcast(match_id, {"type": "state", **state.to_dict()})

            elif msg_type == "next_round":
                if state.phase == "round_end":
                    state.next_round()
                    await manager.broadcast(match_id, {"type": "state", **state.to_dict()})

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[websocket error] player {player_id}: {e}")
    finally:
        manager.remove(match_id, player_id)
        player = state.players.get(player_id)
        if player:
            player.connected = False
            player.action = "idle"
        await manager.broadcast(match_id, {"type": "state", **state.to_dict()})
        print(f"[ws] player {player_id} disconnected from {match_id}")

# ── REST ──────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "active_matches": len(matches)}

@app.get("/matches/{match_id}/events")
def get_events(match_id: str):
    import sqlite3
    conn = sqlite3.connect(db.DB_PATH)
    rows = conn.execute(
        "SELECT timestamp, event_type, data FROM events WHERE match_id=? ORDER BY timestamp",
        (match_id,)
    ).fetchall()
    conn.close()
    return [{"ts": r[0], "type": r[1], "data": json.loads(r[2])} for r in rows]

# ── Serve client files ────────────────────────────────────────────────────────
# This serves index.html and network.js to browsers
# Must be mounted last so it doesn't shadow the API routes

client_dir = os.path.join(os.path.dirname(__file__), "client")
if os.path.exists(client_dir):
    app.mount("/", StaticFiles(directory=client_dir, html=True), name="client")

# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
