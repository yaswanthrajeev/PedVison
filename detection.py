import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO
import os
import hashlib
from datetime import datetime
import time


# Global variables for tracking
pedestrian_tracker = {}  # Stores position and velocity data for each pedestrian
unique_pedestrians = set()  # Unique pedestrian IDs
global_pedestrian_id_counter = 0  # ID counter - renamed to avoid parameter conflict

# Performance tracking
frame_times = []
detection_times = []
segmentation_times = []


# Load models
def load_models(use_tiny=True, skip_segmentation=False):
    """Load AI models for detection and segmentation with performance options"""
    # Use a smaller YOLO model for faster inference
    yolo_model = YOLO("yolov8n.pt" if use_tiny else "yolov8n.pt")

    # Configure YOLO for faster inference
    yolo_model.conf = 0.25  # Lower confidence threshold
    yolo_model.iou = 0.45  # IOU threshold for NMS

    deeplab = None
    transform = None

    if not skip_segmentation:
        # Use a smaller segmentation model (mobilenet instead of resnet101)
        deeplab = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_mobilenet_v3_large', pretrained=True)
        deeplab.eval()

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    return yolo_model, deeplab, transform


def get_user_detection_dir(username):
    """Get user-specific detection directory"""
    user_dir = f"detections/{username}"
    os.makedirs(user_dir, exist_ok=True)
    return user_dir


def estimate_distance(box_height, known_height_cm=170, focal_length_px=700):
    """Estimate distance based on bounding box height"""
    if box_height == 0:
        return float('inf')
    return (known_height_cm * focal_length_px) / box_height / 100


def segment_people(image, deeplab, transform):
    """Segment people in the image using DeepLabV3"""
    if deeplab is None or transform is None:
        # Return an empty mask if segmentation is disabled
        return np.zeros((image.height, image.width), dtype=np.uint8)

    start_time = time.time()
    input_tensor = transform(image).unsqueeze(0)

    # Use CUDA if available
    if torch.cuda.is_available():
        input_tensor = input_tensor.cuda()
        deeplab = deeplab.cuda()

    with torch.no_grad():
        output = deeplab(input_tensor)['out'][0]

    mask = output.argmax(0).byte().cpu().numpy()
    person_mask = (mask == 15).astype(np.uint8) * 255

    segmentation_times.append(time.time() - start_time)
    return person_mask


def is_trajectory_collision(position, velocity, collision_area, time_steps=5):  # Reduced from 10 to 5
    """Check if trajectory will intersect with collision area in the next few time steps"""
    x, y = position
    vx, vy = velocity

    # Collision area coordinates
    cx1, cy1 = collision_area["x1"], collision_area["y1"]
    cx2, cy2 = collision_area["x2"], collision_area["y2"]

    # Check current and future positions
    for i in range(time_steps):
        future_x = x + vx * i
        future_y = y + vy * i

        # Check if future position is in collision area
        if (cx1 <= future_x <= cx2) and (cy1 <= future_y <= cy2):
            return True, i  # Return True and time steps to collision

    return False, None


def process_frame(frame, yolo_model, deeplab, transform, collision_area, pedestrian_tracker,
                  unique_pedestrians, pedestrian_id_counter, max_people,
                  frame_count=0, skip_frames=2, skip_segmentation=False):
    """Process a single frame for pedestrian detection and tracking with performance optimizations"""
    global global_pedestrian_id_counter  # Use the renamed global variable

    start_time = time.time()

    # Skip frames to improve performance
    if frame_count % skip_frames != 0 and frame_count > 0 and pedestrian_tracker:
        # On skipped frames, just update positions based on velocity
        overlay = frame.copy()
        person_count = len(pedestrian_tracker)
        danger_detected = False
        collision_warning = False
        min_distance = float('inf')

        # Draw existing trackers with predicted positions
        for pid, data in pedestrian_tracker.items():
            if len(data) >= 2:
                pos, velocity = data[0], data[1]
                x, y = pos
                vx, vy = velocity

                # Update position based on velocity
                new_x = int(x + vx)
                new_y = int(y + vy)

                # Update tracker with new position
                pedestrian_tracker[pid] = ((new_x, new_y), velocity)

                # Draw a simple marker for the person
                cv2.circle(overlay, (new_x, new_y), 10, (0, 255, 0), -1)

        frame_times.append(time.time() - start_time)
        return overlay, person_count, danger_detected, collision_warning, min_distance, False, unique_pedestrians

    # Process frame at reduced resolution
    frame_height, frame_width = frame.shape[:2]
    scale_factor = 0.5  # Reduce resolution by half

    # Only resize if the frame is large
    if frame_width > 320:
        width = int(frame_width * scale_factor)
        height = int(frame_height * scale_factor)
        small_frame = cv2.resize(frame, (width, height))
    else:
        small_frame = frame
        width, height = frame_width, frame_height

    # Convert color only once
    rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Run YOLO detection
    detection_start = time.time()
    results = yolo_model(rgb, verbose=False)  # Turn off verbose output
    detections = results[0].boxes.data.cpu().numpy()
    detection_times.append(time.time() - detection_start)

    # Create output frame
    overlay = frame.copy()

    person_count = 0
    danger_detected = False
    collision_warning = False
    current_frame_pedestrians = set()
    min_distance = float('inf')

    # Get segmentation mask only if needed and not skipped
    if not skip_segmentation:
        pil_img = Image.fromarray(rgb)
        mask = segment_people(pil_img, deeplab, transform)

        # Scale mask back to original size if needed
        if frame_width > 320:
            mask = cv2.resize(mask, (frame_width, frame_height))

        colored_mask = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(overlay, 0.7, colored_mask, 0.3, 0, dtype=cv2.CV_8U)

    # Draw collision area if enabled
    if collision_area["enabled"]:
        cv2.rectangle(overlay,
                      (collision_area["x1"], collision_area["y1"]),
                      (collision_area["x2"], collision_area["y2"]),
                      (255, 165, 0), 2)  # Orange color for collision area
        cv2.putText(overlay, "Collision Zone",
                    (collision_area["x1"], collision_area["y1"] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 2)

    # Scale factor to map from resized frame to original frame
    x_scale = frame_width / width if frame_width > 320 else 1
    y_scale = frame_height / height if frame_width > 320 else 1

    # Process each detection
    for det in detections:
        x1, y1, x2, y2, conf, cls = det

        # Skip low confidence detections
        if conf < 0.3:
            continue

        if int(cls) == 0:  # Person class
            # Scale coordinates back to original frame size
            x1, y1, x2, y2 = int(x1 * x_scale), int(y1 * y_scale), int(x2 * x_scale), int(y2 * y_scale)

            centroid_x = int((x1 + x2) / 2)
            centroid_y = int((y1 + y2) / 2)

            # Simplified tracking with trajectory prediction
            is_new_person = True
            matched_pid = None

            for pid, prev_data in pedestrian_tracker.items():
                if prev_data is not None and len(prev_data) >= 2:
                    prev_pos, prev_velocity = prev_data[0], prev_data[1]
                    px, py = prev_pos

                    # Predict position based on previous velocity
                    predicted_x = px + prev_velocity[0]
                    predicted_y = py + prev_velocity[1]

                    # Calculate distance to predicted position
                    distance = np.sqrt((predicted_x - centroid_x) ** 2 + (predicted_y - centroid_y) ** 2)

                    if distance < 50:  # Threshold for same person
                        current_frame_pedestrians.add(pid)
                        is_new_person = False
                        matched_pid = pid

                        # Update velocity (simple moving average)
                        vx = 0.7 * prev_velocity[0] + 0.3 * (centroid_x - px)
                        vy = 0.7 * prev_velocity[1] + 0.3 * (centroid_y - py)

                        # Update tracker with new position and velocity
                        pedestrian_tracker[pid] = ((centroid_x, centroid_y), (vx, vy))
                        break

            if is_new_person:
                global_pedestrian_id_counter += 1  # Use the global counter
                unique_pedestrians.add(global_pedestrian_id_counter)
                # Initialize with zero velocity
                pedestrian_tracker[global_pedestrian_id_counter] = ((centroid_x, centroid_y), (0, 0))
                current_frame_pedestrians.add(global_pedestrian_id_counter)
                matched_pid = global_pedestrian_id_counter

            person_count = len(current_frame_pedestrians)

            # Distance estimation
            box_height = y2 - y1
            distance_m = estimate_distance(box_height)
            min_distance = min(min_distance, distance_m)

            # Risk classification
            if distance_m < 3:
                color = (0, 0, 255)  # Red
                label = f"🚨 High Risk ({distance_m:.1f}m)"
                danger_detected = True
            elif distance_m < 6:
                color = (0, 255, 255)  # Yellow
                label = f"⚠️ Medium Risk ({distance_m:.1f}m)"
            else:
                color = (0, 255, 0)  # Green
                label = f"✅ Low Risk ({distance_m:.1f}m)"

            # Check for collision trajectory only for high and medium risk pedestrians
            if matched_pid is not None and collision_area["enabled"] and distance_m < 6:
                pos, velocity = pedestrian_tracker[matched_pid][0], pedestrian_tracker[matched_pid][1]
                will_collide, time_to_collision = is_trajectory_collision(pos, velocity, collision_area)

                if will_collide:
                    collision_warning = True
                    color = (0, 0, 255)  # Red for collision warning
                    label = f"🚨 COLLISION WARNING! ({time_to_collision} frames)"

            # Draw bounding box
            cv2.rectangle(overlay, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(overlay, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Draw trajectory prediction only for high risk pedestrians to save computation
            if matched_pid is not None and distance_m < 3:
                pos, velocity = pedestrian_tracker[matched_pid][0], pedestrian_tracker[matched_pid][1]
                # Predict future positions (3 steps instead of 5)
                for i in range(1, 4):
                    future_x = int(pos[0] + velocity[0] * i)
                    future_y = int(pos[1] + velocity[1] * i)
                    # Draw prediction points with decreasing size
                    radius = 4 - i + 1
                    cv2.circle(overlay, (future_x, future_y), radius, color, -1)

                # Draw velocity vector
                end_x = int(pos[0] + velocity[0] * 5)  # Reduced from 10
                end_y = int(pos[1] + velocity[1] * 5)
                cv2.arrowedLine(overlay, pos, (end_x, end_y), color, 2)

                # Add trajectory info to label
                speed = np.sqrt(velocity[0] ** 2 + velocity[1] ** 2)
                direction = "→" if velocity[0] > 0 else "←" if velocity[0] < 0 else "•"
                cv2.putText(overlay, f"ID:{matched_pid} {direction} {speed:.1f}px/f",
                            (int(x1), int(y2) + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    # Clean up trackers for pedestrians no longer in frame
    current_pids = list(pedestrian_tracker.keys())
    for pid in current_pids:
        if pid not in current_frame_pedestrians:
            # Keep for a few frames before removing (helps with occlusions)
            if len(pedestrian_tracker[pid]) < 3:
                pedestrian_tracker[pid] = (*pedestrian_tracker[pid], 1)  # Add missing frame counter
            else:
                pos, vel, missing = pedestrian_tracker[pid]
                if missing > 3:  # Reduced from 5 to 3
                    del pedestrian_tracker[pid]
                else:
                    pedestrian_tracker[pid] = (pos, vel, missing + 1)

    # Determine if frame should be saved
    should_save = collision_warning or person_count >= max_people or danger_detected

    # Add performance metrics to the frame
    if len(frame_times) > 0:
        avg_frame_time = sum(frame_times[-10:]) / min(len(frame_times), 10)
        fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
        cv2.putText(overlay, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    frame_times.append(time.time() - start_time)
    return overlay, person_count, danger_detected, collision_warning, min_distance, should_save, unique_pedestrians


def save_detection_frame(frame, username):
    """Save detection frame to user's directory"""
    user_detection_dir = get_user_detection_dir(username)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Save to user-specific directory with encrypted filename
    filename = hashlib.md5(f"{timestamp}_{username}".encode()).hexdigest() + ".jpg"
    filepath = os.path.join(user_detection_dir, filename)
    cv2.imwrite(filepath, frame)
    return filepath


def get_performance_stats():
    """Return performance statistics"""
    stats = {
        "frame_times": frame_times,
        "detection_times": detection_times,
        "segmentation_times": segmentation_times,
        "avg_fps": 1.0 / (sum(frame_times[-30:]) / min(len(frame_times), 30)) if frame_times else 0,
        "avg_detection_time": sum(detection_times[-30:]) / min(len(detection_times), 30) if detection_times else 0,
        "avg_segmentation_time": sum(segmentation_times[-30:]) / min(len(segmentation_times),
                                                                     30) if segmentation_times else 0
    }
    return stats