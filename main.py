"""
Blade Duel — Online Multiplayer Server with AI modes
=====================================================
Modes:
  pvp — Player vs Player (two humans)
  pve — Player vs AI     (one human, one AI)
  ava — AI vs AI         (spectate two AIs)

Local dev:  uvicorn main:app --reload --port 8000
Production: Railway runs this via Procfile
"""

import asyncio
import json
import uuid
import time
import os
import numpy as np
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from game_state import GameState
import data_logger as db

# ── AI setup ──────────────────────────────────────────────────────────────────

AI_AVAILABLE = False
ai_model     = None
ACTIONS      = ["idle", "move_left", "move_right", "jump", "attack", "block"]
SCREEN_W     = 800
GROUND_Y     = 400

def load_ai():
    global ai_model, AI_AVAILABLE
    try:
        from stable_baselines3 import PPO
        model_path = os.path.join(os.path.dirname(__file__), "sword_ai.zip")
        if os.path.exists(model_path):
            ai_model     = PPO.load(model_path)
            AI_AVAILABLE = True
            print("[ai] model loaded successfully")
        else:
            print("[ai] sword_ai.zip not found — AI modes disabled")
    except Exception as e:
        print(f"[ai] failed to load: {e}")

def build_obs(player, opponent) -> np.ndarray:
    """Build the observation vector the AI model expects."""
    return np.array([
        player.x / SCREEN_W,
        player.y / GROUND_Y,
        player.health / player.max_health,
        1.0 if player.action == "idle"      else 0.0,
        1.0 if player.action == "attacking" else 0.0,
        1.0 if player.action == "blocking"  else 0.0,
        opponent.x / SCREEN_W,
        opponent.y / GROUND_Y,
        opponent.health / opponent.max_health,
        1.0 if opponent.action == "attacking" else 0.0,
        1.0 if opponent.action == "blocking"  else 0.0,
        abs(player.x - opponent.x) / SCREEN_W,
    ], dtype=np.float32)

def get_ai_action(state: GameState, player_id: int) -> str:
    """
    Rule-based AI that always engages and attacks.
    Used as fallback when trained model is unavailable or misbehaving.
    """
    player   = state.players.get(player_id)
    opp_id   = 2 if player_id == 1 else 1
    opponent = state.players.get(opp_id)

    if not player or not opponent:
        return "idle"
    if player.is_ko() or opponent.is_ko():
        return "idle"

    dist = abs(player.x - opponent.x)

    # Always face the opponent first
    if opponent.x > player.x:
        player.facing = "right"
    else:
        player.facing = "left"

    # If far away — run toward opponent
    if dist > ATTACK_RANGE + 10:
        return "move_left" if opponent.x < player.x else "move_right"

    # If in attack range — attack most of the time
    import random
    r = random.random()
    if r < 0.70:
        return "attack"
    elif r < 0.80:
        return "block"
    elif r < 0.90:
        return "jump"
    else:
        return "move_left" if opponent.x < player.x else "move_right"


def apply_ai_action(state: GameState, player_id: int, action_name: str):
    """Apply an AI action to the game state."""
    opp_id   = 2 if player_id == 1 else 1
    player   = state.players.get(player_id)
    opponent = state.players.get(opp_id)

    if player and opponent:
        # Always face the opponent
        if opponent.x > player.x:
            player.facing = "right"
        else:
            player.facing = "left"

    if action_name == "move_left":
        state.move_player(player_id, -1, 0)
    elif action_name == "move_right":
        state.move_player(player_id,  1, 0)
    elif action_name == "jump":
        if player:
            player.action = "jumping"
    elif action_name == "attack":
        state.player_attack(player_id, opp_id)
    elif action_name == "block":
        state.player_block(player_id, True)
    else:
        state.player_block(player_id, False)
        state.move_player(player_id, 0, 0)

# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(title="Blade Duel")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

db.init_db()
load_ai()

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

# ── Match store ───────────────────────────────────────────────────────────────

matches:     dict[str, GameState] = {}
match_modes: dict[str, str]       = {}   # match_id → "pvp" | "pve" | "ava"
match_ai:    dict[str, list[int]] = {}   # match_id → [player_ids that are AI]
_lobby: dict[str, Optional[str]]  = {   # one lobby per mode
    "pvp": None, "pve": None, "ava": None
}


def get_or_create_lobby(mode: str) -> GameState:
    global _lobby
    mid = _lobby.get(mode)
    if mid and mid in matches:
        state = matches[mid]
        if len(state.players) < 2:
            return state
    mid = str(uuid.uuid4())[:8]
    matches[mid]      = GameState(match_id=mid, phase="waiting")
    match_modes[mid]  = mode
    match_ai[mid]     = []
    _lobby[mode]      = mid
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
        db.log_match_end(
            state.match_id,
            state.match_winner_id or 0,
            state.round_number
        )

# ── Game loop ─────────────────────────────────────────────────────────────────

TICK_RATE = 1 / 60
ATTACK_COOLDOWN = 0.5
_last_attack: dict[str, float] = {}


async def game_loop(match_id: str):
    while match_id in matches:
        try:
            state = matches[match_id]
            mode  = match_modes.get(match_id, "pvp")
            ai_players = match_ai.get(match_id, [])

            if state.phase == "fighting":
                prev = state.phase

                # Tick AI players
                for pid in ai_players:
                    action = get_ai_action(state, pid)
                    apply_ai_action(state, pid, action)
                    # AI attack logging
                    if action == "attack":
                        opp_id = 2 if pid == 1 else 1
                        opp    = state.players.get(opp_id)
                        if opp:
                            db.log_hit(match_id, pid, opp_id,
                                       state.attack_damage, False, opp.health)

                state.tick_timer()

                if state.phase != prev and state.phase in ("round_end", "game_over"):
                    _log_round(state)

            await manager.broadcast(match_id, {
                "type": "state",
                "ai_available": AI_AVAILABLE,
                "mode": mode,
                "ai_players": ai_players,
                **state.to_dict()
            })

        except Exception as e:
            print(f"[game_loop error] {match_id}: {e}")

        await asyncio.sleep(TICK_RATE)

# ── Input processing ──────────────────────────────────────────────────────────

def process_input(state: GameState, player_id: int, keys: list, match_id: str):
    player = state.players.get(player_id)
    if not player or state.phase != "fighting":
        return
    opp_id = 2 if player_id == 1 else 1
    dx = -1 if "left" in keys else (1 if "right" in keys else 0)
    state.move_player(player_id, dx, 0)
    state.player_block(player_id, "block" in keys)
    if "attack" in keys:
        key = f"{match_id}:{player_id}"
        now = time.time()
        if now - _last_attack.get(key, 0) >= ATTACK_COOLDOWN:
            _last_attack[key] = now
            hit = state.player_attack(player_id, opp_id)
            db.log_action(match_id, player_id, "attack")
            if hit:
                opp = state.players.get(opp_id)
                db.log_hit(match_id, player_id, opp_id,
                           state.attack_damage,
                           opp.action == "blocking" if opp else False,
                           opp.health if opp else 0)
                if state.phase in ("round_end", "game_over"):
                    _log_round(state)

# ── WebSocket ─────────────────────────────────────────────────────────────────

@app.websocket("/ws/{mode}")
async def websocket_endpoint(ws: WebSocket, mode: str = "pvp"):
    await ws.accept()

    # Validate mode
    if mode not in ("pvp", "pve", "ava"):
        mode = "pvp"

    # Block AI modes if model not loaded
    if mode in ("pve", "ava") and not AI_AVAILABLE:
        await manager.send(ws, {
            "type": "error",
            "message": "AI model not available on this server."
        })
        await ws.close()
        return

    state     = get_or_create_lobby(mode)
    match_id  = state.match_id
    player_id = assign_player_id(state)

    if player_id is None:
        await manager.send(ws, {"type": "error", "message": "Match is full."})
        await ws.close()
        return

    manager.add(match_id, player_id, ws)
    await manager.send(ws, {
        "type":      "assigned",
        "player_id": player_id,
        "match_id":  match_id,
        "mode":      mode,
        "ai_available": AI_AVAILABLE,
    })
    print(f"[ws] player {player_id} joined {mode} match {match_id}")

    # Start game loop on first connection
    if len(manager.connections.get(match_id, {})) == 1:
        asyncio.create_task(game_loop(match_id))

    # For PvE and AvA — add AI players automatically
    if mode == "pve" and player_id == 1:
        # AI is always player 2 in PvE
        if 2 not in match_ai.get(match_id, []):
            match_ai[match_id].append(2)
    elif mode == "ava":
        # Both players are AI in AvA — human just spectates
        for pid in [1, 2]:
            if pid not in match_ai.get(match_id, []):
                match_ai[match_id].append(pid)

    try:
        async for raw in ws.iter_text():
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue

            msg_type = msg.get("type")

            if msg_type == "join":
                name = msg.get("name", f"Player {player_id}")
                # In AvA the human is a spectator — add both AI players
                if mode == "ava":
                    state.add_player(1, "AI Alpha")
                    state.add_player(2, "AI Beta")
                    state.set_player_ready(1, "warrior")
                    state.set_player_ready(2, "warrior")
                    if state.all_players_ready():
                        state.start_round()
                elif mode == "pve":
                    state.add_player(player_id, name)
                    # Add AI player entry to state
                    opp_id = 2 if player_id == 1 else 1
                    if opp_id not in state.players:
                        state.add_player(opp_id, "AI Opponent")
                    db.log_match_start(match_id, {
                        pid: p.name for pid, p in state.players.items()
                    })
                    if state.all_players_connected():
                        state.phase = "character_select"
                else:
                    state.add_player(player_id, name)
                    db.log_match_start(match_id, {
                        pid: p.name for pid, p in state.players.items()
                    })
                    if state.all_players_connected():
                        state.phase = "character_select"
                await manager.broadcast(match_id, {
                    "type": "state", "mode": mode,
                    "ai_players": match_ai.get(match_id, []),
                    **state.to_dict()
                })

            elif msg_type == "ready":
                character = msg.get("character", "warrior")
                state.set_player_ready(player_id, character)
                # In PvE auto-ready the AI opponent
                if mode == "pve":
                    opp_id = 2 if player_id == 1 else 1
                    opp    = state.players.get(opp_id)
                    if opp and not opp.ready:
                        state.set_player_ready(opp_id, "warrior")
                if state.phase == "character_select" and state.all_players_ready():
                    state.start_round()
                await manager.broadcast(match_id, {
                    "type": "state", "mode": mode,
                    "ai_players": match_ai.get(match_id, []),
                    **state.to_dict()
                })

            elif msg_type == "input":
                # Only process input for human players
                if player_id not in match_ai.get(match_id, []):
                    process_input(state, player_id, msg.get("keys", []), match_id)

            elif msg_type == "pause":
                if state.phase in ("fighting", "paused"):
                    state.toggle_pause(player_id)
                    await manager.broadcast(match_id, {
                        "type": "state", "mode": mode,
                        "ai_players": match_ai.get(match_id, []),
                        **state.to_dict()
                    })

            elif msg_type == "next_round":
                if state.phase == "round_end":
                    state.next_round()
                    await manager.broadcast(match_id, {
                        "type": "state", "mode": mode,
                        "ai_players": match_ai.get(match_id, []),
                        **state.to_dict()
                    })

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[websocket error] player {player_id}: {e}")
    finally:
        manager.remove(match_id, player_id)
        player = state.players.get(player_id)
        if player:
            player.connected = False
            player.action    = "idle"
        await manager.broadcast(match_id, {
            "type": "state", "mode": mode,
            "ai_players": match_ai.get(match_id, []),
            **state.to_dict()
        })
        print(f"[ws] player {player_id} left {match_id}")

# ── REST ──────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status":       "ok",
        "active_matches": len(matches),
        "ai_available": AI_AVAILABLE,
    }

@app.get("/matches")
def get_matches():
    return db.get_all_matches()

@app.get("/matches/{match_id}/events")
def get_events(match_id: str):
    return db.get_match_events(match_id)

@app.get("/stats")
def stats_page():
    from fastapi.responses import FileResponse
    return FileResponse(
        os.path.join(os.path.dirname(__file__), "client", "stats.html")
    )

# ── Serve client ──────────────────────────────────────────────────────────────

client_dir = os.path.join(os.path.dirname(__file__), "client")
if os.path.exists(client_dir):
    app.mount("/", StaticFiles(directory=client_dir, html=True), name="client")

# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
