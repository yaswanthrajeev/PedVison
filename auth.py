import sqlite3
import hashlib


def create_users_table():
    """Create users table if it doesn't exist"""
    conn = sqlite3.connect('pedestrian_detection.db')
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL
    )
    ''')
    conn.commit()
    conn.close()


def add_user(username, password):
    """Add a new user to the database"""
    if not username or not password:
        return False

    # Hash the password
    hashed_password = hashlib.sha256(password.encode()).hexdigest()

    try:
        conn = sqlite3.connect('pedestrian_detection.db')
        cursor = conn.cursor()
        cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)",
                       (username, hashed_password))
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        # Username already exists
        return False


def login_user(username, password):
    """Verify user credentials"""
    if not username or not password:
        return False

    # Hash the password
    hashed_password = hashlib.sha256(password.encode()).hexdigest()

    conn = sqlite3.connect('pedestrian_detection.db')
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE username = ? AND password = ?",
                   (username, hashed_password))
    user = cursor.fetchone()
    conn.close()

    return user is not None
