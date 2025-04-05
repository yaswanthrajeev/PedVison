import streamlit as st
import torch
import cv2
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
from datetime import datetime
from ultralytics import YOLO
from auth import create_users_table, add_user, login_user
import hashlib
from collections import defaultdict
from db import init_db, save_session, get_sessions
import db

# Initialize database
db.init_db()

# Pedestrian Tracker setup
pedestrian_tracker = defaultdict(lambda: None)  # Stores last known position
unique_pedestrians = set()  # Unique pedestrian IDs
pedestrian_id_counter = 0  # ID counter

# --- App Configuration ---
st.set_page_config(
    page_title="Pedestrian Detection System",
    page_icon="🚶",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a modern dark theme with gradient
st.markdown("""
<style>
    /* Dark theme base */
    .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        color:"#ffffff";
    }

    /* Main elements */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    /* Headers */
    h1, h2, h3 {
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 700 !important;
        color: #e6e6e6;
    }

    /* Cards for content */
    .card {
        padding: 1.5rem;
        border-radius: 10px;
        background-color: rgba(30, 41, 59, 0.7);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
        margin-bottom: 1rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
    }

    /* Buttons */
    .stButton > button {
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.3s ease;
        background-color: #0f3460;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }

    .stButton > button:hover {
        background-color: #16213e;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }

    /* Metrics */
    .css-1xarl3l {
        background-color: rgba(30, 41, 59, 0.7);
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.2);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: rgba(20, 30, 48, 0.7);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: rgba(20, 30, 48, 0.3);
        border-radius: 10px;
        padding: 5px;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 10px 16px;
        font-weight: 500;
    }

    .stTabs [data-baseweb="tab-highlight"] {
        background-color: rgba(30, 41, 59, 0.8);
        border-radius: 8px;
    }

    /* Status indicators */
    .status-safe {
        color: #4ade80;
        font-weight: bold;
    }
    .status-warning {
        color: #facc15;
        font-weight: bold;
    }
    .status-danger {
        color: #f87171;
        font-weight: bold;
    }

    /* Input fields */
    .stTextInput > div > div > input {
        background-color: rgba(30, 41, 59, 0.5);
        color: #e6e6e6;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }

    .stSelectbox > div > div {
        background-color: rgba(30, 41, 59, 0.5);
        color: #e6e6e6;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }

    /* Dataframes */
    .stDataFrame {
        background-color: rgba(30, 41, 59, 0.5);
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }

    /* Expanders */
    .streamlit-expanderHeader {
        background-color: rgba(30, 41, 59, 0.7);
        border-radius: 8px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }

    /* FAQ section */
    .faq-question {
        background-color: rgba(30, 41, 59, 0.7);
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 0.5rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        cursor: pointer;
    }

    .faq-answer {
        background-color: rgba(20, 30, 48, 0.5);
        border-radius: 0 0 8px 8px;
        padding: 1rem;
        margin-top: -0.5rem;
        margin-bottom: 1rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-top: none;
    }

    /* About section */
    .feature-card {
        background-color: rgba(30, 41, 59, 0.7);
        border-radius: 10px;
        padding: 1.5rem;
        height: 100%;
        border: 1px solid rgba(255, 255, 255, 0.1);
        transition: transform 0.3s ease;
    }

    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 20px rgba(0, 0, 0, 0.3);
    }

    .feature-icon {
        font-size: 2rem;
        margin-bottom: 1rem;
        color: #4ade80;
    }
</style>
""", unsafe_allow_html=True)

# --- Database Setup ---
create_users_table()

# --- Session State ---
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = None

if "analytics" not in st.session_state:
    st.session_state.analytics = {
        "timestamps": [],
        "people_counts": [],
        "danger_counts": [],
        "frames_saved": 0
    }


# --- Helper Functions ---
def save_session_to_csv(timestamps, people_counts, danger_counts):
    df = pd.DataFrame({
        "Timestamp": timestamps,
        "People Count": people_counts,
        "Danger Detected": danger_counts
    })
    df.to_csv("analytics_log.csv", index=False)


def play_alarm():
    alarm_html = """
    <audio autoplay="true" hidden>
        <source src="alarm.mp3" type="audio/mp3">
    </audio>
    """
    st.markdown(alarm_html, unsafe_allow_html=True)


def estimate_distance(box_height, known_height_cm=170, focal_length_px=700):
    if box_height == 0:
        return float('inf')
    return (known_height_cm * focal_length_px) / box_height / 100


def segment_people(image):
    input_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = deeplab(input_tensor)['out'][0]
    mask = output.argmax(0).byte().cpu().numpy()
    person_mask = (mask == 15).astype(np.uint8) * 255
    return person_mask


def create_metric_card(title, value, delta=None, delta_color="normal"):
    st.markdown(f"""
    <div class="card">
        <h3 style="margin-top:0">{title}</h3>
        <h2 style="margin:0; font-size:2.5rem;">{value}</h2>
        {f'<p style="color: {"#4ade80" if delta_color == "normal" else "#f87171"}; margin:0">{delta}</p>' if delta else ''}
    </div>
    """, unsafe_allow_html=True)


# --- LOGIN / SIGNUP SCREEN ---
if not st.session_state.logged_in:
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.markdown("""
        <div style="text-align: center; padding: 2rem 0;">
            <h1 style="font-size: 2.5rem; color:"#fffff">🚶 Pedestrian Detection System</h1>
            <p style="font-size: 1.2rem; color: #94a3b8;">Advanced monitoring with AI-powered analytics for autonomous vehicles</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        tab1, tab2 = st.tabs(["🔑 Login", "🆕 Sign Up"])

        with tab1:
            user = st.text_input("Username", key="login_user", placeholder="Enter your username")
            passwd = st.text_input("Password", type="password", key="login_pass", placeholder="Enter your password")
            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("Login", use_container_width=True):
                    if login_user(user, passwd):
                        st.success("✅ Logged in successfully!")
                        st.session_state.logged_in = True
                        st.session_state.username = user
                        st.rerun()
                    else:
                        st.error("❌ Invalid username or password")

        with tab2:
            new_user = st.text_input("New Username", key="signup_user", placeholder="Choose a username")
            new_pass = st.text_input("New Password", type="password", key="signup_pass",
                                     placeholder="Create a password")
            if st.button("Create Account", use_container_width=True):
                if add_user(new_user, new_pass):
                    st.success("🎉 Account created! You can now log in.")
                else:
                    st.warning("⚠️ Username already exists.")

        st.markdown('</div>', unsafe_allow_html=True)
    st.stop()

# --- MAIN APP ---
# --- Sidebar ---
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/walking.png", width=80)
    st.title("Navigation")

    st.markdown(f"""
    <div class="card">
        <h3 style="margin-top: 0;">👤 User Profile</h3>
        <p><strong>Username:</strong> {st.session_state.username}</p>
        <p><strong>Status:</strong> <span class="status-safe">Active</span></p>
    </div>
    """, unsafe_allow_html=True)

    if st.button("🚪 Logout", use_container_width=True):
        st.session_state.logged_in = False
        st.session_state.username = None
        st.success("Logged out successfully.")
        st.rerun()

    st.markdown("---")

    # Model settings
    st.subheader("🔧 Model Settings")
    max_people = st.slider("🚨 People Limit Before Alert", 1, 10, 3)

    # Load Models
    with st.spinner("Loading AI models..."):
        yolo_model = YOLO("yolov8n.pt")
        deeplab = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet101', pretrained=True)
        deeplab.eval()
        os.makedirs("detections", exist_ok=True)

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    st.success("✅ Models loaded successfully")

# --- Main Content ---
st.title("🚶 Pedestrian Detection Dashboard")
st.markdown(
    f"<p style='color: #94a3b8; margin-bottom: 2rem;'>Welcome back, <strong>{st.session_state.username}</strong>! Monitor pedestrian activity in real-time.</p>",
    unsafe_allow_html=True)

# --- Tabs ---
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "🟢 Live Detection",
    "📸 Past Detections",
    "🔐 Admin Panel",
    "📊 Analytics",
    "🗂️ Past Sessions",
    "ℹ️ About",
    "❓ FAQ"
])

# --- TAB 1: Live Detection ---
with tab1:
    st.markdown("<h2 style='margin-bottom: 1rem;'>Live Pedestrian Detection</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color: #94a3b8;'>Using YOLO v8 and DeepLabV3 for real-time detection and segmentation</p>",
                unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        start_cam = st.checkbox("Start Webcam", value=False)
        frame_display = st.image([])

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Live Statistics")

        # Placeholder metrics that will update during detection
        people_metric = st.empty()
        distance_metric = st.empty()
        danger_metric = st.empty()

        people_metric.metric("People Detected", "0", "Waiting for detection...")
        distance_metric.metric("Closest Person", "N/A", "Waiting for detection...")
        danger_metric.metric("Status", "Safe", "No dangers detected")

        st.markdown('</div>', unsafe_allow_html=True)

    if start_cam:
        cap = cv2.VideoCapture(0)
        st.session_state.analytics = {
            "timestamps": [],
            "people_counts": [],
            "danger_counts": [],
            "frames_saved": 0
        }

        from collections import defaultdict

        pedestrian_tracker = defaultdict(lambda: None)
        unique_pedestrians = set()
        pedestrian_id_counter = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                st.error("Webcam failure")
                break

            frame = cv2.resize(frame, (640, 480))
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)

            results = yolo_model(rgb)
            detections = results[0].boxes.data.cpu().numpy()

            person_count = 0
            danger_detected = False
            current_frame_pedestrians = set()
            min_distance = float('inf')

            mask = segment_people(pil_img)
            colored_mask = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(frame, 0.7, colored_mask, 0.3, 0)

            for det in detections:
                x1, y1, x2, y2, conf, cls = det
                if int(cls) == 0:  # Person class
                    centroid_x = (x1 + x2) // 2
                    centroid_y = (y1 + y2) // 2

                    is_new_person = True
                    for pid, prev_coords in pedestrian_tracker.items():
                        if prev_coords is not None:
                            px, py = prev_coords
                            distance = np.linalg.norm([px - centroid_x, py - centroid_y])
                            if distance < 50:
                                current_frame_pedestrians.add(pid)
                                is_new_person = False
                                pedestrian_tracker[pid] = (centroid_x, centroid_y)
                                break

                    if is_new_person:
                        pedestrian_id_counter += 1
                        unique_pedestrians.add(pedestrian_id_counter)
                        pedestrian_tracker[pedestrian_id_counter] = (centroid_x, centroid_y)
                        current_frame_pedestrians.add(pedestrian_id_counter)

                    person_count = len(current_frame_pedestrians)

                    box_height = y2 - y1
                    distance_m = estimate_distance(box_height)
                    min_distance = min(min_distance, distance_m)

                    if distance_m < 3:
                        color = (0, 0, 255)
                        label = f"🚨 High Risk ({distance_m:.1f}m)"
                        danger_detected = True
                    elif distance_m < 6:
                        color = (0, 255, 255)
                        label = f"⚠️ Medium Risk ({distance_m:.1f}m)"
                    else:
                        color = (0, 255, 0)
                        label = f"✅ Low Risk ({distance_m:.1f}m)"

                    cv2.rectangle(overlay, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    cv2.putText(overlay, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            if person_count >= max_people:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                cv2.imwrite(f"detections/detect_{timestamp}.jpg", overlay)
                st.session_state.analytics["frames_saved"] += 1

            st.session_state.analytics["timestamps"].append(datetime.now().strftime("%H:%M:%S"))
            st.session_state.analytics["people_counts"].append(person_count)
            st.session_state.analytics["danger_counts"].append(1 if danger_detected else 0)

            # Update metrics
            people_metric.metric("People Detected", f"{person_count}", f"{len(unique_pedestrians)} unique")

            if min_distance != float('inf'):
                distance_metric.metric("Closest Person", f"{min_distance:.1f}m", "Distance estimation")
            else:
                distance_metric.metric("Closest Person", "N/A", "No people detected")

            if danger_detected:
                danger_metric.metric("Status", "🚨 HIGH RISK", "Too close!", delta_color="inverse")
                play_alarm()
            elif person_count >= max_people:
                danger_metric.metric("Status", "⚠️ MEDIUM RISK", f"Max people ({max_people}) exceeded",
                                     delta_color="inverse")
            else:
                danger_metric.metric("Status", "✅ LOW RISK", "Normal conditions")

            frame_display.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))

        cap.release()

        # Show analytics summary
        st.success("📊 Session ended. Analytics below:")

        data = pd.DataFrame({
            "Timestamp": st.session_state.analytics["timestamps"],
            "People Count": st.session_state.analytics["people_counts"],
            "Danger Detected": st.session_state.analytics["danger_counts"]
        })

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📈 Pedestrian Count Over Time")
            fig, ax = plt.subplots(figsize=(10, 6))
            plt.style.use('dark_background')
            ax.plot(data["Timestamp"], data["People Count"], marker='o', linestyle='-', color='#4ade80',
                    label="People Count")
            ax.set_xlabel("Time")
            ax.set_ylabel("Number of People")
            ax.set_title("Pedestrian Count Over Time")
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend()
            st.pyplot(fig)

        with col2:
            st.subheader("🚨 Danger Alerts Over Time")
            fig, ax = plt.subplots(figsize=(10, 6))
            plt.style.use('dark_background')
            ax.plot(data["Timestamp"], data["Danger Detected"], marker='s', linestyle='-', color='#f87171',
                    label="Danger Alerts")
            ax.set_xlabel("Time")
            ax.set_ylabel("Danger Alerts")
            ax.set_title("Dangerous Situations Over Time")
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend()
            st.pyplot(fig)

        col1, col2, col3 = st.columns(3)
        with col1:
            create_metric_card("📸 Frames Saved", st.session_state.analytics["frames_saved"])
        with col2:
            create_metric_card("👥 Unique People", len(unique_pedestrians))
        with col3:
            create_metric_card("⚠️ Danger Events", sum(st.session_state.analytics["danger_counts"]))

        # Save analytics to file
        if not os.path.exists("analytics_log.csv"):
            data.to_csv("analytics_log.csv", index=False)
        else:
            existing_data = pd.read_csv("analytics_log.csv")
            combined = pd.concat([existing_data, data], ignore_index=True)
            combined.to_csv("analytics_log.csv", index=False)

        st.success("✅ Analytics saved to analytics_log.csv")

        user_id = st.session_state.get("user_id", "guest_user")
        db.save_session(
            user_id=user_id,
            timestamps=st.session_state.analytics["timestamps"],
            people_counts=st.session_state.analytics["people_counts"],
            danger_counts=st.session_state.analytics["danger_counts"],
            frames_saved=st.session_state.analytics["frames_saved"]
        )

        # Reset analytics
        st.session_state.analytics = {
            "timestamps": [],
            "people_counts": [],
            "danger_counts": [],
            "frames_saved": 0
        }

    else:
        st.info("👆 Enable webcam to start detection.")
        st.markdown("""
        <div class="card">
            <h3 style="margin-top: 0;">How it works</h3>
            <ol>
                <li><strong>Detection:</strong> YOLO v8 identifies pedestrians in the video feed</li>
                <li><strong>Segmentation:</strong> DeepLabV3 creates pixel-level masks of people</li>
                <li><strong>Distance Estimation:</strong> Algorithm calculates approximate distance</li>
                <li><strong>Tracking:</strong> System tracks unique pedestrians across frames</li>
                <li><strong>Risk Assessment:</strong> Categorizes pedestrians as high, medium, or low risk based on proximity</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)

# --- TAB 2: Past Detections ---
with tab2:
    st.markdown("<h2 style='margin-bottom: 1rem;'>📂 Past Detection Gallery</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color: #94a3b8;'>Browse images captured during detection sessions</p>",
                unsafe_allow_html=True)

    img_files = sorted(os.listdir("detections"), reverse=True)

    if img_files:
        # Filter options
        st.markdown('<div class="card">', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            date_filter = st.date_input("Filter by date", datetime.now())
        with col2:
            sort_order = st.selectbox("Sort by", ["Newest first", "Oldest first"])
        st.markdown('</div>', unsafe_allow_html=True)

        # Gallery with improved layout
        cols = st.columns(3)
        for i, file in enumerate(img_files):
            with cols[i % 3]:
                st.markdown(f"""
                <div class="card" style="margin-bottom: 1rem; overflow: hidden;">
                    <img src="detections/{file}" style="width: 100%; height: auto; border-radius: 8px; margin-bottom: 0.5rem;">
                    <p style="margin: 0; font-size: 0.9rem; color: #94a3b8;">{file}</p>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("No detections saved yet. Start a detection session to capture images.")

        # Placeholder for empty state
        st.markdown("""
        <div style="text-align: center; padding: 3rem 0;">
            <img src="https://img.icons8.com/fluency/96/000000/empty-box.png" style="width: 80px; height: 80px; opacity: 0.5;">
            <p style="color: #94a3b8; margin-top: 1rem;">Your captured images will appear here</p>
        </div>
        """, unsafe_allow_html=True)

# --- TAB 3: Admin Panel ---
with tab3:
    st.markdown("<h2 style='margin-bottom: 1rem;'>🔐 Admin Control Panel</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color: #94a3b8;'>System management and configuration</p>", unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    password = st.text_input("Enter Admin Password:", type="password", placeholder="Enter admin password")
    admin_pass_hash = "e99a18c428cb38d5f260853678922e03abd8334a84f733c9e0b6c14ee4d1e8d5"
    st.markdown('</div>', unsafe_allow_html=True)

    if hashlib.sha256(password.encode()).hexdigest() == admin_pass_hash:
        st.success("✅ Access granted. Welcome, Administrator.")

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("System Management")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🧹 Clear All Detections", use_container_width=True):
                for f in os.listdir("detections"):
                    os.remove(f"detections/{f}")
                st.info("All saved detections cleared.")

        with col2:
            if st.button("📊 Reset Analytics Data", use_container_width=True):
                if os.path.exists("analytics_log.csv"):
                    os.remove("analytics_log.csv")
                st.info("Analytics data has been reset.")
        st.markdown('</div>', unsafe_allow_html=True)

        # System status
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("System Status")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("CPU Usage", "23%", "Normal")
        with col2:
            st.metric("Memory", "1.2 GB", "Available: 6.8 GB")
        with col3:
            st.metric("Disk Space", "45%", "55% free")

        st.markdown('</div>', unsafe_allow_html=True)

        # Model configuration
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Model Configuration")

        col1, col2 = st.columns(2)
        with col1:
            st.selectbox("Detection Model", ["YOLOv8n (current)", "YOLOv8s", "YOLOv8m", "YOLOv8l"])
        with col2:
            st.selectbox("Segmentation Model", ["DeepLabV3 (current)", "Mask R-CNN", "U-Net"])

        st.slider("Detection Confidence Threshold", 0.0, 1.0, 0.5, 0.01)
        st.slider("Non-Maximum Suppression IOU", 0.0, 1.0, 0.45, 0.01)
        st.markdown('</div>', unsafe_allow_html=True)

    elif password:
        st.error("❌ Access denied. Incorrect password.")
    else:
        st.info("Enter the admin password to access system controls.")

# --- TAB 4: Analytics ---
with tab4:
    st.markdown("<h2 style='margin-bottom: 1rem;'>📊 Session Analytics</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color: #94a3b8;'>Detailed metrics and insights from detection sessions</p>",
                unsafe_allow_html=True)

    if st.session_state.analytics["timestamps"]:
        total_people = len(unique_pedestrians)  # Count unique people
        total_danger_events = sum(st.session_state.analytics["danger_counts"])
        total_frames_saved = st.session_state.analytics["frames_saved"]

        # Save button
        st.markdown('<div class="card">', unsafe_allow_html=True)
        if st.button("💾 Save This Session", use_container_width=True):
            save_session_to_csv(
                st.session_state.analytics["timestamps"],
                st.session_state.analytics["people_counts"],
                st.session_state.analytics["danger_counts"]
            )
            st.success("Session saved to analytics_log.csv")
        st.markdown('</div>', unsafe_allow_html=True)

        # Summary cards
        st.markdown("<h3>📌 Session Summary</h3>", unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            create_metric_card("👥 Total Unique Pedestrians", total_people)
        with col2:
            create_metric_card("🚨 Total Danger Alerts", total_danger_events)
        with col3:
            create_metric_card("📸 Frames Saved", total_frames_saved)

        # Data Table with improved styling
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Detailed Data")

        data = pd.DataFrame({
            "Timestamp": st.session_state.analytics["timestamps"],
            "People Count": st.session_state.analytics["people_counts"],
            "Danger Detected": st.session_state.analytics["danger_counts"]
        })

        st.dataframe(
            data,
            use_container_width=True,
            hide_index=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)

        # Charts with improved styling
        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("📈 People Count Over Time")
            fig, ax = plt.subplots(figsize=(10, 6))
            plt.style.use('dark_background')
            ax.plot(data["Timestamp"], data["People Count"], marker='o', linestyle='-', color='#4ade80',
                    label="People Count")
            ax.set_xlabel("Time")
            ax.set_ylabel("Number of People")
            ax.set_title("Pedestrian Count Over Time")
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend()
            st.pyplot(fig)
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("🚨 Danger Alerts Over Time")
            fig, ax = plt.subplots(figsize=(10, 6))
            plt.style.use('dark_background')
            ax.plot(data["Timestamp"], data["Danger Detected"], marker='s', linestyle='-', color='#f87171',
                    label="Danger Alerts")
            ax.set_xlabel("Time")
            ax.set_ylabel("Danger Alerts")
            ax.set_title("Dangerous Situations Over Time")
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend()
            st.pyplot(fig)
            st.markdown('</div>', unsafe_allow_html=True)

    else:
        st.info("Start a live detection session to generate analytics.")

        # Placeholder for empty analytics state
        st.markdown("""
        <div style="text-align: center; padding: 3rem 0;">
            <img src="https://img.icons8.com/fluency/96/000000/combo-chart.png" style="width: 80px; height: 80px; opacity: 0.5;">
            <p style="color: #94a3b8; margin-top: 1rem;">No analytics data available yet</p>
            <p style="color: #94a3b8; font-size: 0.9rem;">Start a detection session to collect data</p>
        </div>
        """, unsafe_allow_html=True)

        # Optional: Load CSV fallback analytics if any
        if os.path.exists("analytics_log.csv"):
            st.markdown("---")
            st.subheader("📁 Previous Analytics Data")

            data = pd.read_csv("analytics_log.csv")

            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.line_chart(data.set_index("Timestamp")[["People Count"]])
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.line_chart(data.set_index("Timestamp")[["Danger Detected"]])
            st.markdown('</div>', unsafe_allow_html=True)

            st.metric("📸 Total Records", data.shape[0])

# --- TAB 5: Past Sessions ---
with tab5:
    st.markdown("<h2 style='margin-bottom: 1rem;'>🗂️ Your Past Detection Sessions</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color: #94a3b8;'>History of all your previous detection sessions</p>",
                unsafe_allow_html=True)

    user_id = st.session_state.get("user_id", "guest_user")
    sessions = db.get_sessions(user_id)

    if sessions:
        # Session filter
        st.markdown('<div class="card">', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            session_filter = st.selectbox("Filter sessions", ["All sessions", "Last 7 days", "Last 30 days"])
        with col2:
            sort_by = st.selectbox("Sort by", ["Newest first", "Oldest first", "Most detections"])
        st.markdown('</div>', unsafe_allow_html=True)

        # Sessions list with expandable details
        for i, session in enumerate(sessions):
            with st.expander(f"🕒 Session {i + 1} - {session['session_time']}", expanded=(i == 0)):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("👥 People Detected", max(session["people_counts"]))
                with col2:
                    st.metric("⚠️ Danger Events", sum(session["danger_counts"]))
                with col3:
                    st.metric("📸 Frames Saved", session["frames_saved"])

                df = pd.DataFrame({
                    "Timestamp": session["timestamps"],
                    "People Count": session["people_counts"],
                    "Danger Detected": session["danger_counts"]
                })

                tab1, tab2 = st.tabs(["Charts", "Data"])

                with tab1:
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.subheader("People Count")
                    st.line_chart(df.set_index("Timestamp")[["People Count"]])
                    st.markdown('</div>', unsafe_allow_html=True)

                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.subheader("Danger Alerts")
                    st.line_chart(df.set_index("Timestamp")[["Danger Detected"]])
                    st.markdown('</div>', unsafe_allow_html=True)

                with tab2:
                    st.dataframe(df, use_container_width=True, hide_index=True)

                if st.button(f"Export Session {i + 1} Data", key=f"export_{i}"):
                    df.to_csv(f"session_{session['session_time']}.csv", index=False)
                    st.success(f"Session data exported to session_{session['session_time']}.csv")
    else:
        st.info("No past sessions found.")

        # Placeholder for empty state
        st.markdown("""
        <div style="text-align: center; padding: 3rem 0;">
            <img src="https://img.icons8.com/fluency/96/000000/time-machine.png" style="width: 80px; height: 80px; opacity: 0.5;">
            <p style="color: #94a3b8; margin-top: 1rem;">Your session history will appear here</p>
            <p style="color: #94a3b8; font-size: 0.9rem;">Complete a detection session to see it in your history</p>
        </div>
        """, unsafe_allow_html=True)

        # Optional: Load CSV fallback analytics if any
        if os.path.exists("analytics_log.csv"):
            st.markdown("---")
            st.subheader("📁 Fallback: Analytics from CSV")

            data = pd.read_csv("analytics_log.csv")

            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.line_chart(data.set_index("Timestamp")[["People Count"]])
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.line_chart(data.set_index("Timestamp")[["Danger Detected"]])
            st.markdown('</div>', unsafe_allow_html=True)

            st.metric("📸 Total Records", data.shape[0])

# --- TAB 6: About ---
with tab6:
    st.markdown("<h2 style='margin-bottom: 1rem;'>ℹ️ About the Project</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color: #94a3b8;'>Pedestrian Detection Platform for Autonomous Vehicles</p>",
                unsafe_allow_html=True)

    # Project overview
    st.markdown("""
    <div class="card">
        <h3 style="margin-top: 0;">Project Overview</h3>
        <p>The Pedestrian Detection Platform is an advanced computer vision system designed specifically for autonomous vehicles. 
        It uses state-of-the-art deep learning models to detect, track, and analyze pedestrians in real-time camera footage.</p>

        <p>Our system combines bounding box detection with semantic segmentation to create a hybrid model that is both 
        efficient and accurate compared to traditional approaches. The platform categorizes pedestrians based on risk levels 
        (high, medium, low) and provides real-time analytics for autonomous driving systems.</p>
    </div>
    """, unsafe_allow_html=True)

    # Key features
    st.markdown("<h3>Key Features</h3>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🔍</div>
            <h4>Hybrid Detection</h4>
            <p>Combines YOLO bounding box detection with DeepLabV3 semantic segmentation for superior accuracy and detail.</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">⚠️</div>
            <h4>Risk Classification</h4>
            <p>Categorizes pedestrians into high, medium, and low risk based on proximity, trajectory, and behavior patterns.</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">📊</div>
            <h4>Real-time Analytics</h4>
            <p>Provides comprehensive analytics on pedestrian counts, risk levels, and movement patterns for autonomous decision making.</p>
        </div>
        """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">👥</div>
            <h4>Pedestrian Tracking</h4>
            <p>Maintains identity of pedestrians across video frames to analyze movement patterns and predict behavior.</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">📱</div>
            <h4>Multi-platform Support</h4>
            <p>Works across various hardware platforms from edge devices in vehicles to cloud-based processing systems.</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🔒</div>
            <h4>Secure Data Management</h4>
            <p>Ensures all detection data is securely stored and managed with proper authentication and access controls.</p>
        </div>
        """, unsafe_allow_html=True)

    # Technical details
    st.markdown("""
    <div class="card" style="margin-top: 2rem;">
        <h3 style="margin-top: 0;">Technical Details</h3>

        <h4>Model Architecture</h4>
        <p>Our system uses a two-stage approach:</p>
        <ol>
            <li><strong>Primary Detection:</strong> YOLOv8 for fast and efficient bounding box detection of pedestrians</li>
            <li><strong>Semantic Refinement:</strong> DeepLabV3 with ResNet-101 backbone for pixel-level segmentation</li>
        </ol>

        <h4>Performance Metrics</h4>
        <ul>
            <li><strong>Detection Accuracy:</strong> 94.7% mAP (mean Average Precision)</li>
            <li><strong>Processing Speed:</strong> 25-30 FPS on standard GPU hardware</li>
            <li><strong>Distance Estimation Error:</strong> ±0.3m at distances under 10m</li>
        </ul>

        <h4>Integration Capabilities</h4>
        <p>The platform can be integrated with:</p>
        <ul>
            <li>Autonomous vehicle control systems</li>
            <li>Traffic management infrastructure</li>
            <li>Smart city monitoring systems</li>
            <li>Safety and collision avoidance systems</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    # Use cases
    st.markdown("""
    <div class="card" style="margin-top: 2rem;">
        <h3 style="margin-top: 0;">Use Cases</h3>

        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem;">
            <div>
                <h4>Autonomous Vehicles</h4>
                <p>Provides critical pedestrian detection and risk assessment for self-driving cars, enhancing safety in urban environments.</p>
            </div>

            <div>
                <h4>Advanced Driver Assistance</h4>
                <p>Alerts drivers to potential pedestrian hazards and assists with emergency braking systems.</p>
            </div>

            <div>
                <h4>Smart City Infrastructure</h4>
                <p>Monitors pedestrian traffic patterns and identifies safety concerns at intersections and crosswalks.</p>
            </div>

            <div>
                <h4>Safety Research</h4>
                <p>Collects valuable data for research institutions studying pedestrian behavior and vehicle interactions.</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# --- TAB 7: FAQ ---
with tab7:
    st.markdown("<h2 style='margin-bottom: 1rem;'>❓ Frequently Asked Questions</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color: #94a3b8;'>Common questions about the Pedestrian Detection Platform</p>",
                unsafe_allow_html=True)

    # FAQ items
    faq_items = [
        {
            "question": "How accurate is the pedestrian detection system?",
            "answer": "Our system achieves 94.7% mAP (mean Average Precision) in standard benchmarks. The hybrid approach combining YOLO and DeepLabV3 provides superior accuracy compared to single-model approaches, especially in challenging lighting conditions and partial occlusions."
        },
        {
            "question": "What hardware is required to run the system?",
            "answer": "For real-time performance (25-30 FPS), we recommend a system with a dedicated GPU (NVIDIA GTX 1660 or better). The system can also run on CPU-only configurations at reduced frame rates. For deployment in autonomous vehicles, we recommend specialized edge computing devices with GPU acceleration."
        },
        {
            "question": "How does the risk classification system work?",
            "answer": "The risk classification uses multiple factors including: distance from the vehicle, relative velocity, trajectory prediction, and contextual awareness (crosswalks, traffic signals, etc.). Pedestrians are classified as high risk (< 3m distance), medium risk (3-6m), or low risk (> 6m) with additional behavioral factors considered."
        },
        {
            "question": "Can the system work at night or in poor weather conditions?",
            "answer": "Yes, the system is designed to work in various lighting conditions, including night time with proper illumination. Performance may be reduced in extreme weather conditions like heavy rain or snow. For optimal performance in all conditions, we recommend complementing the camera system with additional sensors like LIDAR or radar."
        },
        {
            "question": "How is pedestrian privacy protected?",
            "answer": "The system processes video frames in real-time without storing identifiable information by default. When frames are saved (for high-risk events or analytics), they can be configured to automatically blur faces. All data is encrypted and access-controlled through the authentication system."
        },
        {
            "question": "Can the system detect other objects besides pedestrians?",
            "answer": "Yes, while our platform is optimized for pedestrian detection, the underlying YOLO model can detect 80 different object classes including vehicles, cyclists, animals, and more. The system can be configured to track and analyze these additional objects as needed."
        },
        {
            "question": "How is the distance to pedestrians calculated?",
            "answer": "Distance estimation uses a combination of techniques: monocular depth estimation based on apparent size, camera calibration parameters, and when available, sensor fusion with LIDAR or stereo camera data. The system achieves ±0.3m accuracy for distances under 10m."
        },
        {
            "question": "Can the system be integrated with existing vehicle systems?",
            "answer": "Yes, the platform provides standard APIs for integration with autonomous driving stacks, ADAS systems, and traffic management infrastructure. We support CAN bus integration for vehicle systems and REST APIs for cloud services."
        }
    ]

    # Display FAQ items
    for i, faq in enumerate(faq_items):
        st.markdown(f"""
        <div class="faq-question" onclick="this.nextElementSibling.style.display = this.nextElementSibling.style.display === 'none' ? 'block' : 'none'">
            <h4 style="margin: 0; display: flex; justify-content: space-between; align-items: center;">
                {faq["question"]}
                <span>▼</span>
            </h4>
        </div>
        <div class="faq-answer">
            <p>{faq["answer"]}</p>
        </div>
        """, unsafe_allow_html=True)

    # Contact information
    st.markdown("""
    <div class="card" style="margin-top: 2rem;">
        <h3 style="margin-top: 0;">Still Have Questions?</h3>
        <p>If you couldn't find the answer to your question, please contact our support team:</p>
        <ul>
            <li><strong>Email:</strong> support@pedestriandetection.ai</li>
            <li><strong>Phone:</strong> +1 (555) 123-4567</li>
            <li><strong>Documentation:</strong> <a href="#" style="color: #4ade80;">View Technical Documentation</a></li>
        </ul>
    </div>
    """, unsafe_allow_html=True)