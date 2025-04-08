import streamlit as st
import os
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
from datetime import datetime
import hashlib
from PIL import Image
import ui
import db
import detection
import analytics
import config

def render_live_detection_tab(tab, max_people):
    """Render the live detection tab content"""
    start_cam, frame_display, people_metric, distance_metric, danger_metric = ui.render_live_detection_tab()

    # Add collision area toggle
    collision_enabled = st.checkbox("Enable Collision Area", value=st.session_state.collision_area["enabled"])
    st.session_state.collision_area["enabled"] = collision_enabled

    if collision_enabled:
        col1, col2 = st.columns(2)
        with col1:
            st.session_state.collision_area["x1"] = st.slider("Collision Area X1", 0, 640,
                                                              st.session_state.collision_area["x1"])
            st.session_state.collision_area["y1"] = st.slider("Collision Area Y1", 0, 480,
                                                              st.session_state.collision_area["y1"])
        with col2:
            st.session_state.collision_area["x2"] = st.slider("Collision Area X2", 0, 640,
                                                              st.session_state.collision_area["x2"])
            st.session_state.collision_area["y2"] = st.slider("Collision Area Y2", 0, 480,
                                                              st.session_state.collision_area["y2"])

    if start_cam:
        cap = cv2.VideoCapture(0)
        st.session_state.analytics = {
            "timestamps": [],
            "people_counts": [],
            "danger_counts": [],
            "collision_warnings": [],
            "frames_saved": 0,
            "unique_pedestrians": set()
        }

        pedestrian_tracker = {}
        unique_pedestrians = set()
        # Use the global variable from detection module
        detection.global_pedestrian_id_counter = 0  # Reset the counter at the start of detection

        # Load models
        with st.spinner("Loading AI models..."):
            yolo_model, deeplab, transform = detection.load_models()
            os.makedirs("detections", exist_ok=True)
            user_detection_dir = detection.get_user_detection_dir(st.session_state.username)

        st.success("✅ Models loaded successfully")

        while True:
            ret, frame = cap.read()
            if not ret:
                st.error("Webcam failure")
                break

            # Process frame - pass 0 as the pedestrian_id_counter parameter since we're using the global one
            overlay, person_count, danger_detected, collision_warning, min_distance, should_save, unique_pedestrians = detection.process_frame(
                frame, yolo_model, deeplab, transform, st.session_state.collision_area,
                pedestrian_tracker, unique_pedestrians, 0, max_people
            )

            # Save frame if needed
            if should_save:
                detection.save_detection_frame(overlay, st.session_state.username)
                st.session_state.analytics["frames_saved"] += 1

            # Update analytics
            st.session_state.analytics["timestamps"].append(datetime.now().strftime("%H:%M:%S"))
            st.session_state.analytics["people_counts"].append(person_count)
            st.session_state.analytics["danger_counts"].append(1 if danger_detected else 0)
            st.session_state.analytics["collision_warnings"].append(1 if collision_warning else 0)
            st.session_state.analytics["unique_pedestrians"].update(unique_pedestrians)

            # Update metrics
            people_metric.metric("People Detected", f"{person_count}", f"{len(unique_pedestrians)} unique")

            if min_distance != float('inf'):
                distance_metric.metric("Closest Person", f"{min_distance:.1f}m", "Distance estimation")
            else:
                distance_metric.metric("Closest Person", "N/A", "No people detected")

            if collision_warning:
                danger_metric.metric("Status", "🚨 COLLISION WARNING", "Trajectory intersects collision area!",
                                     delta_color="inverse")
                ui.play_alarm()
            elif danger_detected:
                danger_metric.metric("Status", "🚨 HIGH RISK", "Too close!", delta_color="inverse")
                ui.play_alarm()
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
            "Danger Detected": st.session_state.analytics["danger_counts"],
            "Collision Warnings": st.session_state.analytics["collision_warnings"]
        })

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📈 Pedestrian Count Over Time")
            fig = analytics.create_people_count_chart(data)
            st.pyplot(fig)

        with col2:
            st.subheader("🚨 Alerts Over Time")
            fig = analytics.create_alerts_chart(data)
            st.pyplot(fig)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            ui.create_metric_card("📸 Frames Saved", st.session_state.analytics["frames_saved"])
        with col2:
            ui.create_metric_card("👥 Unique People", len(st.session_state.analytics["unique_pedestrians"]))
        with col3:
            ui.create_metric_card("⚠️ Danger Events", sum(st.session_state.analytics["danger_counts"]))
        with col4:
            ui.create_metric_card("🚨 Collision Warnings", sum(st.session_state.analytics["collision_warnings"]))

        # Save analytics to file
        analytics.save_session_to_csv(
            st.session_state.analytics["timestamps"],
            st.session_state.analytics["people_counts"],
            st.session_state.analytics["danger_counts"],
            st.session_state.analytics["collision_warnings"]
        )
        st.success("✅ Analytics saved to analytics_log.csv")

        user_id = st.session_state.get("user_id", st.session_state.username)
        db.save_session(
            user_id=user_id,
            timestamps=st.session_state.analytics["timestamps"],
            people_counts=st.session_state.analytics["people_counts"],
            danger_counts=st.session_state.analytics["danger_counts"],
            frames_saved=st.session_state.analytics["frames_saved"],
            collision_warnings=st.session_state.analytics["collision_warnings"]
        )

        # Reset analytics
        st.session_state.analytics = config.default_analytics()

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
                <li><strong>Trajectory Analysis:</strong> Predicts future positions and checks for collision area intersection</li>
                <li><strong>Risk Assessment:</strong> Categorizes pedestrians as high, medium, or low risk based on proximity and trajectory</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)

def render_past_detections_tab(tab):
    """Render the past detections tab content"""
    # Get user-specific detection files
    user_detection_dir = detection.get_user_detection_dir(st.session_state.username)
    img_files = sorted(os.listdir(user_detection_dir), reverse=True) if os.path.exists(user_detection_dir) else []

    st.subheader("Your Saved Detections")

    if not img_files:
        st.info("No saved detections yet. Start the detection to capture frames.")
    else:
        st.success(f"Found {len(img_files)} saved detection frames.")

        # Display images in a grid without showing filenames
        cols = st.columns(3)
        for i, img_file in enumerate(img_files):
            # Get image timestamp from file metadata instead of filename
            img_path = os.path.join(user_detection_dir, img_file)
            timestamp = datetime.fromtimestamp(os.path.getmtime(img_path)).strftime("%Y-%m-%d %H:%M:%S")

            with cols[i % 3]:
                st.image(img_path, caption=f"Detection at {timestamp}", use_column_width=True)

                # Add download button with generic name
                with open(img_path, "rb") as file:
                    btn = st.download_button(
                        label="Download",
                        data=file,
                        file_name=f"detection_{timestamp.replace(':', '-')}.jpg",
                        mime="image/jpeg",
                        key=f"download_{i}"
                    )

def render_analytics_tab(tab):
    """Render the analytics tab content"""
    save_button, data_placeholder, chart1_placeholder, chart2_placeholder = ui.render_analytics_tab(
        st.session_state.analytics)

    if save_button:
        analytics.save_session_to_csv(
            st.session_state.analytics["timestamps"],
            st.session_state.analytics["people_counts"],
            st.session_state.analytics["danger_counts"],
            st.session_state.analytics.get("collision_warnings", [0] * len(st.session_state.analytics["timestamps"]))
        )
        st.success("Session saved to analytics_log.csv")

    if data_placeholder and st.session_state.analytics["timestamps"]:
        data = pd.DataFrame({
            "Timestamp": st.session_state.analytics["timestamps"],
            "People Count": st.session_state.analytics["people_counts"],
            "Danger Detected": st.session_state.analytics["danger_counts"],
            "Collision Warnings": st.session_state.analytics.get("collision_warnings",
                                                                 [0] * len(st.session_state.analytics["timestamps"]))
        })
        data_placeholder.dataframe(data, use_container_width=True, hide_index=True)

        # Charts
        if chart1_placeholder:
            fig1 = analytics.create_people_count_chart(data)
            chart1_placeholder.pyplot(fig1)

        if chart2_placeholder:
            fig2 = analytics.create_alerts_chart(data)
            chart2_placeholder.pyplot(fig2)

def render_past_sessions_tab(tab):
    """Render the past sessions tab content"""
    user_id = st.session_state.get("user_id", st.session_state.username)
    sessions = db.get_sessions(user_id)

    session_filter, sort_by, session_expanders = ui.render_past_sessions_tab(sessions)

    if session_expanders:
        for expander, session, i in session_expanders:
            with expander:
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("👥 People Detected", max(session["people_counts"]))
                with col2:
                    st.metric("⚠️ Danger Events", sum(session["danger_counts"]))
                with col3:
                    st.metric("🚨 Collision Warnings", sum(session.get("collision_warnings", [0])))
                with col4:
                    st.metric("📸 Frames Saved", session["frames_saved"])

                df = pd.DataFrame({
                    "Timestamp": session["timestamps"],
                    "People Count": session["people_counts"],
                    "Danger Detected": session["danger_counts"],
                    "Collision Warnings": session.get("collision_warnings", [0] * len(session["timestamps"]))
                })

                tab1, tab2 = st.tabs(["Charts", "Data"])

                with tab1:
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.subheader("People Count")
                    st.line_chart(df.set_index("Timestamp")[["People Count"]])
                    st.markdown('</div>', unsafe_allow_html=True)

                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.subheader("Alerts")
                    st.line_chart(df.set_index("Timestamp")[["Danger Detected", "Collision Warnings"]])
                    st.markdown('</div>', unsafe_allow_html=True)

                with tab2:
                    st.dataframe(df, use_container_width=True, hide_index=True)

                if st.button(f"Export Session {i + 1} Data", key=f"export_{i}"):
                    df.to_csv(f"session_{session['session_time']}.csv", index=False)
                    st.success(f"Session data exported to session_{session['session_time']}.csv")

