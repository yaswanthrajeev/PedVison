# db.py
import sqlite3
from typing import List, Dict
import os

DB_NAME = "analytics.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            session_time TEXT DEFAULT CURRENT_TIMESTAMP,
            timestamps TEXT,
            people_counts TEXT,
            danger_counts TEXT,
            frames_saved INTEGER
        )
    """)
    conn.commit()
    conn.close()

def save_session(user_id: str, timestamps: List[str], people_counts: List[int], danger_counts: List[int], frames_saved: int):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO sessions (user_id, timestamps, people_counts, danger_counts, frames_saved)
        VALUES (?, ?, ?, ?, ?)
    """, (
        user_id,
        ",".join(timestamps),
        ",".join(map(str, people_counts)),
        ",".join(map(str, danger_counts)),
        frames_saved
    ))
    conn.commit()
    conn.close()

def get_sessions(user_id: str) -> List[Dict]:
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT session_time, timestamps, people_counts, danger_counts, frames_saved
        FROM sessions
        WHERE user_id = ?
        ORDER BY session_time DESC
    """, (user_id,))
    rows = cursor.fetchall()
    conn.close()

    sessions = []
    for row in rows:
        sessions.append({
            "session_time": row[0],
            "timestamps": row[1].split(","),
            "people_counts": list(map(int, row[2].split(","))),
            "danger_counts": list(map(int, row[3].split(","))),
            "frames_saved": row[4]
        })
    return sessions
