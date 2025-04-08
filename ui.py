import streamlit as st


def setup_ui():
    """Configure the UI theme and styling for the Pedestrian Detection System."""
    st.set_page_config(
        page_title="Pedestrian Detection System",
        page_icon="🚶",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom CSS for a modern light theme with clean aesthetics
    st.markdown("""
    <style>
        /* Light theme base */
        .stApp {
            background: #f8f9fa;
            color: #212529;
        }

        /* Main elements */
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }

        /* Headers */
        h1, h2, h3 {
            font-family: 'Inter', 'Segoe UI', sans-serif;
            font-weight: 600 !important;
            color: #1a1a1a;
        }

        /* Cards for content */
        .card {
            padding: 1.5rem;
            border-radius: 12px;
            background-color: #ffffff;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
            margin-bottom: 1rem;
            border: 1px solid #f0f0f0;
        }

        /* Buttons */
        .stButton > button {
            border-radius: 8px;
            font-weight: 500;
            transition: all 0.3s ease;
            background-color: #4361ee;
            color: white;
            border: none;
        }

        .stButton > button:hover {
            background-color: #3a56d4;
            box-shadow: 0 4px 8px rgba(67, 97, 238, 0.15);
        }

        /* Metrics */
        .css-1xarl3l {
            background-color: #ffffff;
            border-radius: 10px;
            padding: 1rem;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05);
            border: 1px solid #f0f0f0;
        }

        /* Sidebar */
        [data-testid="stSidebar"] {
            background-color: #ffffff;
            border-right: 1px solid #f0f0f0;
        }

        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            background-color: #f8f9fa;
            border-radius: 10px;
            padding: 5px;
        }

        .stTabs [data-baseweb="tab"] {
            border-radius: 8px;
            padding: 10px 16px;
            font-weight: 500;
        }

        .stTabs [data-baseweb="tab-highlight"] {
            background-color: #ffffff;
            border-radius: 8px;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05);
        }

        /* Status indicators */
        .status-safe {
            color: #10b981;
            font-weight: bold;
        }
        .status-warning {
            color: #f59e0b;
            font-weight: bold;
        }
        .status-danger {
            color: #ef4444;
            font-weight: bold;
        }

        /* Input fields */
        .stTextInput > div > div > input {
            background-color: #ffffff;
            color: #212529;
            border: 1px solid #e5e7eb;
            border-radius: 8px;
        }

        .stSelectbox > div > div {
            background-color: #ffffff;
            color: #212529;
            border: 1px solid #e5e7eb;
            border-radius: 8px;
        }

        /* Dataframes */
        .stDataFrame {
            background-color: #ffffff;
            border-radius: 10px;
            border: 1px solid #f0f0f0;
        }

        /* Expanders */
        .streamlit-expanderHeader {
            background-color: #ffffff;
            border-radius: 8px;
            border: 1px solid #f0f0f0;
        }

        /* FAQ section */
        .faq-question {
            background-color: #ffffff;
            border-radius: 8px;
            padding: 1rem;
            margin-bottom: 0.5rem;
            border: 1px solid #f0f0f0;
            cursor: pointer;
            transition: all 0.2s ease;
        }

        .faq-question:hover {
            background-color: #f8f9fa;
        }

        .faq-answer {
            background-color: #f8f9fa;
            border-radius: 0 0 8px 8px;
            padding: 1rem;
            margin-top: -0.5rem;
            margin-bottom: 1rem;
            border: 1px solid #f0f0f0;
            border-top: none;
        }

        /* Feature cards */
        .feature-card {
            background-color: #ffffff;
            border-radius: 12px;
            padding: 1.5rem;
            height: 100%;
            border: 1px solid #f0f0f0;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }

        .feature-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 20px rgba(0, 0, 0, 0.08);
        }

        .feature-icon {
            font-size: 2rem;
            margin-bottom: 1rem;
            color: #4361ee;
        }

        /* Align elements better */
        .row-widget {
            display: flex;
            align-items: center;
            justify-content: space-between;
        }

        /* Better spacing */
        .stMarkdown {
            margin-bottom: 0.5rem;
        }

        /* Improved slider appearance */
        .stSlider > div > div {
            background-color: #f0f0f0;
        }

        .stSlider > div > div > div > div {
            background-color: #4361ee;
        }
    </style>
    """, unsafe_allow_html=True)


def create_metric_card(title, value, delta=None, delta_color="normal"):
    """Create a styled metric card with title and value."""
    delta_color_class = "text-green-500" if delta_color == "normal" else "text-red-500"
    delta_html = f'<p class="{delta_color_class}" style="margin:0">{delta}</p>' if delta else ''

    st.markdown(f"""
    <div class="card">
        <h3 style="margin-top:0">{title}</h3>
        <h2 style="margin:0; font-size:2.5rem;">{value}</h2>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)


def render_login_screen():
    """Render the login and signup screen."""
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.markdown("""
        <div style="text-align: center; padding: 2rem 0;">
            <h1 style="font-size: 2.5rem;">🚶 Pedestrian Detection System</h1>
            <p style="font-size: 1.2rem; color: #6b7280;">Advanced monitoring with AI-powered analytics for autonomous vehicles</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        tab1, tab2 = st.tabs(["🔑 Login", "🆕 Sign Up"])

        # Initialize variables
        login_button = None
        user = None
        passwd = None
        signup_button = None
        new_user = None
        new_pass = None

        with tab1:
            user = st.text_input("Username", key="login_user", placeholder="Enter your username")
            passwd = st.text_input("Password", type="password", key="login_pass", placeholder="Enter your password")
            col1, col2 = st.columns([1, 1])
            with col1:
                login_button = st.button("Login", use_container_width=True)

        with tab2:
            new_user = st.text_input("New Username", key="signup_user", placeholder="Choose a username")
            new_pass = st.text_input("New Password", type="password", key="signup_pass",
                                     placeholder="Create a password")
            signup_button = st.button("Create Account", use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)

        # Return all values at the end
        return login_button, user, passwd, signup_button, new_user, new_pass



def render_sidebar(username):
    """Render the sidebar with navigation and user info."""
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/000000/walking.png", width=80)
        st.title("Navigation")

        st.markdown(f"""
        <div class="card">
            <h3 style="margin-top: 0;">👤 User Profile</h3>
            <p><strong>Username:</strong> {username}</p>
            <p><strong>Status:</strong> <span class="status-safe">Active</span></p>
        </div>
        """, unsafe_allow_html=True)

        logout_button = st.button("🚪 Logout", use_container_width=True)

        st.markdown("---")

        # Model settings
        st.subheader("🔧 Model Settings")
        max_people = st.slider("🚨 People Limit Before Alert", 1, 10, 3)

        return logout_button, max_people


def render_live_detection_tab():
    """Render the live detection tab content."""
    st.markdown("<h2 style='margin-bottom: 1rem;'>Live Pedestrian Detection</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color: #6b7280;'>Using YOLO v8 and DeepLabV3 for real-time detection and segmentation</p>",
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

    return start_cam, frame_display, people_metric, distance_metric, danger_metric


def render_past_detections_tab(img_files):
    """Render the past detections tab content."""
    st.markdown("<h2 style='margin-bottom: 1rem;'>📂 Past Detection Gallery</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color: #6b7280;'>Browse images captured during detection sessions</p>",
                unsafe_allow_html=True)

    if img_files:
        # Filter options
        st.markdown('<div class="card">', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            date_filter = st.date_input("Filter by date")
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
                    <p style="margin: 0; font-size: 0.9rem; color: #6b7280;">{file}</p>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("No detections saved yet. Start a detection session to capture images.")

        # Placeholder for empty state
        st.markdown("""
        <div style="text-align: center; padding: 3rem 0;">
            <img src="https://img.icons8.com/fluency/96/000000/empty-box.png" style="width: 80px; height: 80px; opacity: 0.5;">
            <p style="color: #6b7280; margin-top: 1rem;">Your captured images will appear here</p>
        </div>
        """, unsafe_allow_html=True)


def render_admin_panel_tab():
    """Render the admin panel tab content."""
    st.markdown("<h2 style='margin-bottom: 1rem;'>🔐 Admin Control Panel</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color: #6b7280;'>System management and configuration</p>", unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    password = st.text_input("Enter Admin Password:", type="password", placeholder="Enter admin password")
    st.markdown('</div>', unsafe_allow_html=True)

    return password


def render_analytics_tab(analytics_data):
    """Render the analytics tab content."""
    st.markdown("<h2 style='margin-bottom: 1rem;'>📊 Session Analytics</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color: #6b7280;'>Detailed metrics and insights from detection sessions</p>",
                unsafe_allow_html=True)

    if analytics_data["timestamps"]:
        # Save button
        st.markdown('<div class="card">', unsafe_allow_html=True)
        save_button = st.button("💾 Save This Session", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Summary cards
        st.markdown("<h3>📌 Session Summary</h3>", unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            create_metric_card("👥 Total Unique Pedestrians", len(analytics_data.get("unique_pedestrians", set())))
        with col2:
            create_metric_card("🚨 Total Danger Alerts", sum(analytics_data["danger_counts"]))
        with col3:
            create_metric_card("📸 Frames Saved", analytics_data["frames_saved"])

        # Data Table with improved styling
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Detailed Data")

        data_placeholder = st.empty()
        st.markdown('</div>', unsafe_allow_html=True)

        # Charts placeholders
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("📈 People Count Over Time")
            chart1_placeholder = st.empty()
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("🚨 Danger Alerts Over Time")
            chart2_placeholder = st.empty()
            st.markdown('</div>', unsafe_allow_html=True)

        return save_button, data_placeholder, chart1_placeholder, chart2_placeholder
    else:
        st.info("Start a live detection session to generate analytics.")

        # Placeholder for empty analytics state
        st.markdown("""
        <div style="text-align: center; padding: 3rem 0;">
            <img src="https://img.icons8.com/fluency/96/000000/combo-chart.png" style="width: 80px; height: 80px; opacity: 0.5;">
            <p style="color: #6b7280; margin-top: 1rem;">No analytics data available yet</p>
            <p style="color: #6b7280; font-size: 0.9rem;">Start a detection session to collect data</p>
        </div>
        """, unsafe_allow_html=True)

        return None, None, None, None


def render_past_sessions_tab(sessions):
    """Render the past sessions tab content."""
    st.markdown("<h2 style='margin-bottom: 1rem;'>🗂️ Your Past Detection Sessions</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color: #6b7280;'>History of all your previous detection sessions</p>",
                unsafe_allow_html=True)

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
        session_expanders = []
        for i, session in enumerate(sessions):
            expander = st.expander(f"🕒 Session {i + 1} - {session['session_time']}", expanded=(i == 0))
            session_expanders.append((expander, session, i))

        return session_filter, sort_by, session_expanders
    else:
        st.info("No past sessions found.")

        # Placeholder for empty state
        st.markdown("""
        <div style="text-align: center; padding: 3rem 0;">
            <img src="https://img.icons8.com/fluency/96/000000/time-machine.png" style="width: 80px; height: 80px; opacity: 0.5;">
            <p style="color: #6b7280; margin-top: 1rem;">Your session history will appear here</p>
            <p style="color: #6b7280; font-size: 0.9rem;">Complete a detection session to see it in your history</p>
        </div>
        """, unsafe_allow_html=True)

        return None, None, None


def render_about_tab():
    """Render the about tab content."""
    st.markdown("<h2 style='margin-bottom: 1rem;'>ℹ️ About the Project</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color: #6b7280;'>Pedestrian Detection Platform for Autonomous Vehicles</p>",
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


def render_faq_tab():
    """Render the FAQ tab content."""
    st.markdown("<h2 style='margin-bottom: 1rem;'>❓ Frequently Asked Questions</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color: #6b7280;'>Common questions about the Pedestrian Detection Platform</p>",
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
            <li><strong>Documentation:</strong> <a href="#" style="color: #4361ee;">View Technical Documentation</a></li>
        </ul>
    </div>
    """, unsafe_allow_html=True)



def render_trajectory_explanation(generated_text, qa_pairs, predicted_question, ground_truth_question):
    st.header("Trajectory Explanation")

    st.subheader("Generated Text")
    st.write(generated_text)

    st.subheader("Question-Answer Pairs")
    for i, (q, a) in enumerate(qa_pairs, 1):
        st.markdown(f"**Q{i}:** {q}")
        st.markdown(f"**A{i}:** {a}")

    st.subheader("Predicted Question")
    st.markdown(f"**{predicted_question}**")

    st.subheader("Ground Truth Question")
    st.markdown(f"**{ground_truth_question}**")

def play_alarm():
    """Play an alarm sound."""
    alarm_html = """
    <audio autoplay="true" hidden>
        <source src="alarm.mp3" type="audio/mp3">
    </audio>
    """
    st.markdown(alarm_html, unsafe_allow_html=True)

