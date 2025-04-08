import sqlite3
import json
from datetime import datetime


def init_db():
    """Initialize the database with required tables"""
    conn = sqlite3.connect('pedestrian_detection.db')
    cursor = conn.cursor()

    # Create sessions table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS sessions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT NOT NULL,
        session_time TEXT NOT NULL,
        timestamps TEXT NOT NULL,
        people_counts TEXT NOT NULL,
        danger_counts TEXT NOT NULL,
        frames_saved INTEGER NOT NULL,
        collision_warnings TEXT
    )
    ''')

    conn.commit()
    conn.close()


def save_session(user_id, timestamps, people_counts, danger_counts, frames_saved, collision_warnings=None):
    """Save a detection session to the database"""
    conn = sqlite3.connect('pedestrian_detection.db')
    cursor = conn.cursor()

    session_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Convert lists to JSON strings for storage
    timestamps_json = json.dumps(timestamps)
    people_counts_json = json.dumps(people_counts)
    danger_counts_json = json.dumps(danger_counts)
    collision_warnings_json = json.dumps(collision_warnings if collision_warnings else [])

    cursor.execute('''
    INSERT INTO sessions 
    (user_id, session_time, timestamps, people_counts, danger_counts, frames_saved, collision_warnings)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (user_id, session_time, timestamps_json, people_counts_json, danger_counts_json, frames_saved,
          collision_warnings_json))

    conn.commit()
    conn.close()


def get_sessions(user_id):
    """Get all sessions for a user"""
    conn = sqlite3.connect('pedestrian_detection.db')
    cursor = conn.cursor()

    cursor.execute('''
    SELECT session_time, timestamps, people_counts, danger_counts, frames_saved, collision_warnings
    FROM sessions
    WHERE user_id = ?
    ORDER BY session_time DESC
    ''', (user_id,))

    sessions = []
    for row in cursor.fetchall():
        session_time, timestamps_json, people_counts_json, danger_counts_json, frames_saved, collision_warnings_json = row

        session = {
            'session_time': session_time,
            'timestamps': json.loads(timestamps_json),
            'people_counts': json.loads(people_counts_json),
            'danger_counts': json.loads(danger_counts_json),
            'frames_saved': frames_saved,
            'collision_warnings': json.loads(collision_warnings_json) if collision_warnings_json else []
        }

        sessions.append(session)

    conn.close()
    return sessions
