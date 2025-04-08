# Default configuration values

def default_collision_area():
    """Default collision area configuration"""
    return {
        "x1": 200,  # Default values
        "y1": 300,
        "x2": 440,
        "y2": 450,
        "enabled": True
    }

def default_analytics():
    """Default analytics state"""
    return {
        "timestamps": [],
        "people_counts": [],
        "danger_counts": [],
        "collision_warnings": [],
        "frames_saved": 0,
        "unique_pedestrians": set()
    }

# Model configuration
DETECTION_MODEL = "yolov8n.pt"
SEGMENTATION_MODEL = "deeplabv3_resnet101"
CONFIDENCE_THRESHOLD = 0.5
NMS_IOU = 0.45

# Trajectory prediction settings
PREDICTION_TIME_STEPS = 10
VELOCITY_SMOOTHING = 0.7

# Admin password hash
ADMIN_PASS_HASH = "e99a18c428cb38d5f260853678922e03abd8334a84f733c9e0b6c14ee4d1e8d5"
