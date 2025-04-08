import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime


def save_session_to_csv(timestamps, people_counts, danger_counts, collision_warnings):
    """Save session data to CSV file"""
    df = pd.DataFrame({
        "Timestamp": timestamps,
        "People Count": people_counts,
        "Danger Detected": danger_counts,
        "Collision Warnings": collision_warnings
    })
    df.to_csv("analytics_log.csv", index=False)


def create_people_count_chart(data):
    """Create people count chart"""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(data["Timestamp"], data["People Count"], marker='o', linestyle='-', color='#4361ee',
            label="People Count")
    ax.set_xlabel("Time")
    ax.set_ylabel("Number of People")
    ax.set_title("Pedestrian Count Over Time")
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()
    return fig


def create_alerts_chart(data):
    """Create alerts chart"""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(data["Timestamp"], data["Danger Detected"], marker='s', linestyle='-', color='#ef4444',
            label="Danger Alerts")
    ax.plot(data["Timestamp"], data["Collision Warnings"], marker='d', linestyle='-', color='#f97316',
            label="Collision Warnings")
    ax.set_xlabel("Time")
    ax.set_ylabel("Alerts")
    ax.set_title("Dangerous Situations Over Time")
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()
    return fig


def prepare_analytics_data(analytics_state):
    """Prepare analytics data for display"""
    if not analytics_state["timestamps"]:
        return None

    return pd.DataFrame({
        "Timestamp": analytics_state["timestamps"],
        "People Count": analytics_state["people_counts"],
        "Danger Detected": analytics_state["danger_counts"],
        "Collision Warnings": analytics_state.get("collision_warnings",
                                                  [0] * len(analytics_state["timestamps"]))
    })
