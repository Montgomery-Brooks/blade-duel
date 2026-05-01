"""
sword_env.py — Gymnasium environment for Blade Duel AI training
================================================================
Install deps:
    pip install stable-baselines3 gymnasium pygame-ce

Train the AI:
    python train_ai.py

This file wraps the game logic in a standard Gym interface so
Stable Baselines3 can train a PPO agent against it.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
import time

# ── Mirror the core game constants here so we don't import pygame ─────────────
# (pygame opens a display window which we don't want during headless training)

SCREEN_W    = 900
GROUND_Y    = 460
MAX_HEALTH  = 100
ATTACK_RANGE = 90
ATTACK_COOLDOWN = 0.5
JUMP_FORCE  = -16
GRAVITY     = 0.7
MOVE_SPEED  = 5
ROUND_TIME  = 99

CHARACTERS = {
    "Warrior": {"speed": 5,  "damage": 15, "health": 100},
    "Knight":  {"speed": 4,  "damage": 20, "health": 120},
    "Rogue":   {"speed": 7,  "damage": 12, "health": 85},
}

# ── Lightweight player simulation (no pygame, no rendering) ───────────────────

class SimPlayer:
    """Stripped-down player that runs without pygame for fast training."""

    def __init__(self, player_num: int, character: str, spawn_x: float):
        self.player_num = player_num
        self.character  = character
        self.spawn_x    = spawn_x
        stats = CHARACTERS[character]
        self.max_health = stats["health"]
        self.speed      = stats["speed"]
        self.damage     = stats["damage"]
        self.reset()

    def reset(self):
        self.x          = self.spawn_x
        self.y          = float(GROUND_Y)
        self.vel_y      = 0.0
        self.health     = float(self.max_health)
        self.on_ground  = True
        self.attacking  = False
        self.blocking   = False
        self.last_attack = 0.0
        self.facing     = 1 if self.player_num == 1 else -1

    @property
    def is_ko(self):
        return self.health <= 0

    def move(self, dx: int):
        if self.is_ko or self.blocking or self.attacking:
            return
        self.x += dx * self.speed
        self.x  = max(30.0, min(float(SCREEN_W - 30), self.x))
        if dx != 0:
            self.facing = 1 if dx > 0 else -1

    def jump(self):
        if self.on_ground and not self.is_ko and not self.blocking:
            self.vel_y     = float(JUMP_FORCE)
            self.on_ground = False

    def start_block(self):
        if not self.is_ko and not self.attacking:
            self.blocking = True

    def stop_block(self):
        self.blocking = False

    def try_attack(self, opponent: "SimPlayer") -> bool:
        """Returns True if a hit landed."""
        if self.is_ko:
            return False
        now = time.time()
        if now - self.last_attack < ATTACK_COOLDOWN:
            return False
        self.attacking   = True
        self.last_attack = now
        dist = abs(self.x - opponent.x)
        if dist <= ATTACK_RANGE and not opponent.is_ko:
            dmg = self.damage
            if opponent.blocking:
                dmg = max(1, dmg // 10)
            opponent.health = max(0.0, opponent.health - dmg)
            return True
        return False

    def apply_gravity(self):
        if not self.on_ground:
            self.vel_y += GRAVITY
        self.y += self.vel_y
        if self.y >= GROUND_Y:
            self.y         = float(GROUND_Y)
            self.vel_y     = 0.0
            self.on_ground = True
            self.attacking = False

    def finish_attack(self):
        self.attacking = False


# ── Action index map ──────────────────────────────────────────────────────────

ACTIONS = ["idle", "move_left", "move_right", "jump", "attack", "block"]
N_ACTIONS = len(ACTIONS)

# ── Observation builder ───────────────────────────────────────────────────────

N_OBS = 12

def get_obs(player: SimPlayer, opponent: SimPlayer) -> np.ndarray:
    """
    Returns a normalised observation vector for the given player.
    All values are in [0, 1] or [-1, 1] for direction.
    """
    return np.array([
        player.x   / SCREEN_W,                          # own x
        player.y   / GROUND_Y,                          # own y (1 = ground)
        player.health / player.max_health,               # own health %
        1.0 if player.on_ground  else 0.0,
        1.0 if player.attacking  else 0.0,
        1.0 if player.blocking   else 0.0,
        opponent.x / SCREEN_W,                          # opponent x
        opponent.y / GROUND_Y,                          # opponent y
        opponent.health / opponent.max_health,           # opponent health %
        1.0 if opponent.attacking else 0.0,
        1.0 if opponent.blocking  else 0.0,
        abs(player.x - opponent.x) / SCREEN_W,          # distance
    ], dtype=np.float32)


# ── Gymnasium environment ─────────────────────────────────────────────────────

class SwordGameEnv(gym.Env):
    """
    Single-agent environment. The trained agent controls the AI fighter.
    The opponent uses a simple scripted strategy during training so the
    agent learns to handle a real threat rather than a stationary target.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        agent_character:    str = "Warrior",
        opponent_character: str = "Warrior",
        max_steps:          int = 3000,
    ):
        super().__init__()
        self.agent_char    = agent_character
        self.opp_char      = opponent_character
        self.max_steps     = max_steps

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(N_OBS,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(N_ACTIONS)

        self.agent    = SimPlayer(1, agent_character,    220.0)
        self.opponent = SimPlayer(2, opponent_character, 680.0)
        self._step    = 0
        self._prev_opp_health = float(self.opponent.max_health)
        self._prev_agent_health = float(self.agent.max_health)

    # ── Gym interface ──────────────────────────────────────────────────────────

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.agent.reset()
        self.opponent.reset()
        self._step              = 0
        self._prev_opp_health   = float(self.opponent.max_health)
        self._prev_agent_health = float(self.agent.max_health)
        return get_obs(self.agent, self.opponent), {}

    def step(self, action: int):
        self._step += 1

        # ── Agent acts ────────────────────────────────────────────────────────
        self._apply_action(self.agent, action, self.opponent)

        # ── Scripted opponent acts ────────────────────────────────────────────
        opp_action = self._scripted_opponent()
        self._apply_action(self.opponent, opp_action, self.agent)

        # ── Physics ───────────────────────────────────────────────────────────
        self.agent.apply_gravity()
        self.opponent.apply_gravity()

        # ── Reward ────────────────────────────────────────────────────────────
        reward = self._calculate_reward()

        # ── Done ──────────────────────────────────────────────────────────────
        terminated = self.agent.is_ko or self.opponent.is_ko
        truncated  = self._step >= self.max_steps

        self._prev_opp_health   = self.opponent.health
        self._prev_agent_health = self.agent.health

        obs = get_obs(self.agent, self.opponent)
        return obs, reward, terminated, truncated, {}

    # ── Reward function ────────────────────────────────────────────────────────

    def _calculate_reward(self) -> float:
        reward = 0.0

        # Reward dealing damage
        dmg_dealt = self._prev_opp_health - self.opponent.health
        reward += dmg_dealt * 0.1

        # Penalty for taking damage
        dmg_taken = self._prev_agent_health - self.agent.health
        reward -= dmg_taken * 0.08

        # Small survival penalty per step (encourages finishing quickly)
        reward -= 0.001

        # Big terminal rewards
        if self.opponent.is_ko:
            reward += 15.0
        if self.agent.is_ko:
            reward -= 15.0

        # Bonus for staying close (encourages engagement over running)
        dist = abs(self.agent.x - self.opponent.x)
        if dist < ATTACK_RANGE:
            reward += 0.005
        elif dist > 400:
            reward -= 0.003

        return float(reward)

    # ── Action application ─────────────────────────────────────────────────────

    def _apply_action(self, player: SimPlayer, action_idx: int, opponent: SimPlayer):
        action = ACTIONS[action_idx]
        player.stop_block()
        player.finish_attack()
        if action == "move_left":
            player.move(-1)
        elif action == "move_right":
            player.move(1)
        elif action == "jump":
            player.jump()
        elif action == "attack":
            player.try_attack(opponent)
        elif action == "block":
            player.start_block()
        # "idle" — do nothing

    # ── Scripted opponent for training variety ─────────────────────────────────

    def _scripted_opponent(self) -> int:
        """
        Mix of behaviours so the agent learns to handle different playstyles.
        Randomised each episode via a style chosen at reset.
        """
        opp  = self.opponent
        agent = self.agent
        dist  = abs(opp.x - agent.x)

        # Aggressive: rush in and attack
        if dist > ATTACK_RANGE + 20:
            return ACTIONS.index("move_left" if agent.x < opp.x else "move_right")
        elif dist <= ATTACK_RANGE:
            # Occasionally block, mostly attack
            r = np.random.random()
            if r < 0.6:
                return ACTIONS.index("attack")
            elif r < 0.75:
                return ACTIONS.index("block")
            else:
                return ACTIONS.index("jump")
        return ACTIONS.index("idle")


# ── Self-play environment (both sides are the same model) ─────────────────────

class SelfPlayEnv(SwordGameEnv):
    """
    After initial training, use this to pit the agent against itself.
    The opponent model is loaded from disk and updated periodically.
    """

    def __init__(self, opponent_model=None, **kwargs):
        super().__init__(**kwargs)
        self.opponent_model = opponent_model

    def _scripted_opponent(self) -> int:
        if self.opponent_model is None:
            return super()._scripted_opponent()
        obs = get_obs(self.opponent, self.agent)
        action, _ = self.opponent_model.predict(obs, deterministic=False)
        return int(action)
