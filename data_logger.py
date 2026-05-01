import sqlite3
import json
import time
from pathlib import Path


DB_PATH = Path(__file__).parent / "db" / "matches.db"


def init_db():
    DB_PATH.parent.mkdir(exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS matches (
            match_id    TEXT PRIMARY KEY,
            started_at  REAL,
            ended_at    REAL,
            winner_id   INTEGER,
            total_rounds INTEGER
        );

        CREATE TABLE IF NOT EXISTS events (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            match_id    TEXT,
            timestamp   REAL,
            event_type  TEXT,
            data        TEXT
        );

        CREATE TABLE IF NOT EXISTS round_results (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            match_id    TEXT,
            round_number INTEGER,
            winner_id   INTEGER,
            duration    REAL,
            p1_health   REAL,
            p2_health   REAL
        );
    """)
    conn.commit()
    conn.close()


def _conn():
    return sqlite3.connect(DB_PATH)


def log_match_start(match_id: str, player_names: dict):
    with _conn() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO matches (match_id, started_at) VALUES (?, ?)",
            (match_id, time.time())
        )
        log_event(conn, match_id, "match_start", {"players": player_names})


def log_match_end(match_id: str, winner_id: int, total_rounds: int):
    with _conn() as conn:
        conn.execute(
            "UPDATE matches SET ended_at=?, winner_id=?, total_rounds=? WHERE match_id=?",
            (time.time(), winner_id, total_rounds, match_id)
        )
        log_event(conn, match_id, "match_end", {
            "winner_id": winner_id, "total_rounds": total_rounds
        })


def log_round_result(match_id: str, round_number: int, winner_id: int,
                     duration: float, p1_health: float, p2_health: float):
    with _conn() as conn:
        conn.execute(
            """INSERT INTO round_results
               (match_id, round_number, winner_id, duration, p1_health, p2_health)
               VALUES (?,?,?,?,?,?)""",
            (match_id, round_number, winner_id, duration, p1_health, p2_health)
        )
        log_event(conn, match_id, "round_end", {
            "round": round_number, "winner_id": winner_id,
            "duration": round(duration, 2),
            "p1_health": p1_health, "p2_health": p2_health,
        })


def log_hit(match_id: str, attacker_id: int, target_id: int,
            damage: float, blocked: bool, target_hp_after: float):
    with _conn() as conn:
        log_event(conn, match_id, "hit", {
            "attacker": attacker_id, "target": target_id,
            "damage": damage, "blocked": blocked,
            "target_hp_after": round(target_hp_after, 1),
        })


def log_action(match_id: str, player_id: int, action: str):
    with _conn() as conn:
        log_event(conn, match_id, "player_action", {
            "player_id": player_id, "action": action
        })


def log_event(conn, match_id: str, event_type: str, data: dict):
    conn.execute(
        "INSERT INTO events (match_id, timestamp, event_type, data) VALUES (?,?,?,?)",
        (match_id, time.time(), event_type, json.dumps(data))
    )
