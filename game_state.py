from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, List
import time


# ── Player ────────────────────────────────────────────────────────────────────

@dataclass
class PlayerState:
    player_id: int
    name: str
    health: float = 100.0
    score: int = 0              # round wins
    x: float = 0.0             # canvas position
    y: float = 0.0
    action: str = "idle"       # idle | walking | attacking | blocking | stunned | ko
    facing: str = "right"      # right | left
    connected: bool = True
    character: str = "warrior" # warrior | knight | rogue — set during character select
    ready: bool = False        # confirmed ready on character select screen
    move_speed: float = 12.0   # pixels per tick — can differ between characters
    last_action_time: float = field(default_factory=time.time)
    on_ground: bool = True

    def take_damage(self, amount: float):
        self.health = max(0.0, self.health - amount)
        if self.health <= 0:
            self.action = "ko"

    def is_ko(self) -> bool:
        return self.health <= 0

    def is_blocking(self) -> bool:
        """Single source of truth for blocking state."""
        return self.action == "blocking"

    def reset_for_round(self, x: float):
        self.health = 100.0
        self.x = x
        self.y = 0.0
        self.action = "idle"
        self.ready = False
        self.last_action_time = time.time()
        # face inward at round start
        self.facing = "right" if self.player_id == 1 else "left"


# ── Match ─────────────────────────────────────────────────────────────────────

@dataclass
class GameState:
    match_id: str
    phase: str = "waiting"
    # phases: waiting | character_select | fighting | paused | round_end | game_over
    players: Dict[int, PlayerState] = field(default_factory=dict)
    round_number: int = 1
    max_rounds: int = 3
    round_timer: float = 99.0
    round_start_time: Optional[float] = None
    winner_id: Optional[int] = None
    match_winner_id: Optional[int] = None
    paused_by: Optional[int] = None
    round_history: List[dict] = field(default_factory=list)

    # arena dimensions — single source of truth used by both server and client
    arena_width: float = 800.0
    arena_height: float = 400.0

    # combat settings
    attack_damage: float = 10.0
    attack_range: float = 70.0

    # afk settings
    afk_timeout: float = 10.0          # seconds before afk damage kicks in
    afk_damage_per_tick: float = 0.25  # damage per tick while afk

    # internal pause tracking
    _paused_timer_remaining: float = 99.0

    # ── Player management ─────────────────────────────────────────────────────

    def add_player(self, player_id: int, name: str):
        spawn_x = 200.0 if player_id == 1 else 600.0
        self.players[player_id] = PlayerState(
            player_id=player_id,
            name=name,
            x=spawn_x
        )

    def all_players_connected(self) -> bool:
        """Both players have joined the lobby."""
        return len(self.players) >= 2

    def all_players_ready(self) -> bool:
        """Both players confirmed ready on character select screen."""
        return (
            self.all_players_connected() and
            all(p.ready for p in self.players.values())
        )

    def set_player_ready(self, player_id: int, character: str):
        """Called when a player confirms their character selection."""
        player = self.players.get(player_id)
        if player:
            player.character = character
            player.ready = True

    # ── Round management ──────────────────────────────────────────────────────

    def start_round(self):
        self.phase = "fighting"
        self.round_timer = 99.0
        self.round_start_time = time.time()
        self.winner_id = None
        spawn_positions = {1: 200.0, 2: 600.0}
        for pid, player in self.players.items():
            player.reset_for_round(spawn_positions.get(pid, 400.0))

    def next_round(self):
        if self.phase != "round_end":
            return
        self.round_number += 1
        self.start_round()

    # ── Tick ──────────────────────────────────────────────────────────────────

    def tick_timer(self):
        """Called every server tick while fighting."""
        if self.round_start_time is not None and self.phase == "fighting":
            elapsed = time.time() - self.round_start_time
            self.round_timer = max(0.0, 99.0 - elapsed)
            if self.round_timer == 0:
                self._resolve_timeout()

        if self.phase == "fighting":
            self._check_afk_damage()

    # ── Movement ──────────────────────────────────────────────────────────────

    def move_player(self, player_id: int, dx: float, dy: float):
        """
        Move a player by a direction vector.
        dx/dy should be -1, 0, or 1 — speed is applied inside.
        """
        if self.phase != "fighting":
            return

        player = self.players.get(player_id)
        if player is None or player.is_ko():
            return

        # standing still
        if dx == 0 and dy == 0:
            if player.action not in ("attacking", "blocking", "ko"):
                player.action = "idle"
            return

        # apply movement clamped to arena bounds
        player.x = max(0.0, min(self.arena_width,  player.x + dx * player.move_speed))
        player.y = max(0.0, min(self.arena_height, player.y + dy * player.move_speed))

        # update facing direction
        if dx > 0:
            player.facing = "right"
        elif dx < 0:
            player.facing = "left"

        # only switch to walking if not mid-attack or blocking
        if player.action not in ("attacking", "blocking"):
            player.action = "walking"

        player.last_action_time = time.time()

    # ── Combat ────────────────────────────────────────────────────────────────

    def player_attack(self, attacker_id: int, target_id: int) -> bool:
        """
        Attempt an attack. Returns True if hit landed.
        Uses action as single source of truth — no separate is_blocking flag.
        """
        if self.phase != "fighting":
            return False

        attacker = self.players.get(attacker_id)
        target = self.players.get(target_id)

        if attacker is None or target is None:
            return False
        if attacker.is_ko() or target.is_ko():
            return False

        attacker.action = "attacking"
        attacker.last_action_time = time.time()

        distance_x = abs(attacker.x - target.x)
        distance_y = abs(attacker.y - target.y)

        if distance_x <= self.attack_range and distance_y <= 50:
            return self.apply_hit(attacker_id, target_id, self.attack_damage)

        return False

    def apply_hit(self, attacker_id: int, target_id: int, damage: float) -> bool:
        """Apply damage to target. Returns True if KO occurred."""
        target = self.players.get(target_id)
        attacker = self.players.get(attacker_id)

        if target is None or attacker is None:
            return False
        if attacker.action != "attacking":
            return False

        # action is single source of truth for blocking
        if target.is_blocking():
            damage *= 0.1   # 90% damage reduction on block

        target.take_damage(damage)

        if target.is_ko():
            self._resolve_ko(winner_id=attacker_id)
            return True

        return False

    def player_block(self, player_id: int, blocking: bool):
        """
        Toggle blocking on or off.
        action field is the single source of truth — no separate flag.
        """
        if self.phase != "fighting":
            return

        player = self.players.get(player_id)
        if player is None or player.is_ko():
            return

        if blocking:
            player.action = "blocking"
            player.last_action_time = time.time()
        else:
            if player.action == "blocking":
                player.action = "idle"

    def stop_player_action(self, player_id: int):
        """Reset player back to idle after an attack animation finishes."""
        player = self.players.get(player_id)
        if player is None or player.is_ko():
            return
        if player.action in ("attacking", "walking"):
            player.action = "idle"

    # ── Resolution ────────────────────────────────────────────────────────────

    def _resolve_ko(self, winner_id: int):
        self.winner_id = winner_id
        winner = self.players.get(winner_id)

        if winner is not None:
            winner.score += 1

        # record this round in history for end screen / replays
        self.round_history.append({
            "round": self.round_number,
            "winner_id": winner_id,
            "duration": round(99.0 - self.round_timer, 2),
            "p1_health": self.players.get(1, PlayerState(0, "")).health,
            "p2_health": self.players.get(2, PlayerState(0, "")).health,
        })

        # check if match is over
        wins_needed = (self.max_rounds // 2) + 1
        if winner is not None and winner.score >= wins_needed:
            self.match_winner_id = winner_id
            self.phase = "game_over"
        else:
            self.phase = "round_end"

    def _resolve_timeout(self):
        """Player with highest HP wins when timer hits zero."""
        if not self.players:
            return
        winner = max(self.players.values(), key=lambda p: p.health)
        self._resolve_ko(winner_id=winner.player_id)

    # ── Pause ─────────────────────────────────────────────────────────────────

    def toggle_pause(self, player_id: int):
        if self.phase == "fighting":
            self.phase = "paused"
            self.paused_by = player_id
            if self.round_start_time is not None:
                elapsed = time.time() - self.round_start_time
                self._paused_timer_remaining = 99.0 - elapsed
        elif self.phase == "paused":
            self.phase = "fighting"
            self.round_start_time = time.time() - (99.0 - self._paused_timer_remaining)
            self.paused_by = None

    # ── AFK ───────────────────────────────────────────────────────────────────

    def _check_afk_damage(self):
        """Slowly drain health of any player doing nothing for too long."""
        now = time.time()
        for player in self.players.values():
            if player.is_ko():
                continue
            if player.action in ("attacking", "walking", "blocking"):
                continue
            if now - player.last_action_time >= self.afk_timeout:
                player.take_damage(self.afk_damage_per_tick)
                if player.is_ko():
                    for other in self.players.values():
                        if other.player_id != player.player_id:
                            self._resolve_ko(other.player_id)
                            return

    # ── Serialization ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "match_id": self.match_id,
            "phase": self.phase,
            "round_number": self.round_number,
            "max_rounds": self.max_rounds,
            "round_timer": round(self.round_timer, 1),
            "winner_id": self.winner_id,
            "match_winner_id": self.match_winner_id,
            "paused_by": self.paused_by,
            "arena_width": self.arena_width,
            "arena_height": self.arena_height,
            "round_history": self.round_history,
            "players": {
                str(pid): asdict(player)
                for pid, player in self.players.items()
            },
        }
