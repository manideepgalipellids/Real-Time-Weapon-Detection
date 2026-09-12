import streamlit as st
import av

from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Real-Time Weapon Detection",
    page_icon="🛡️",
    layout="wide"
)


# =========================================================
# MODEL
# =========================================================

MODEL_PATH = r"C:\weapon_clean\runs\weapon_clean\weights\best.pt"


@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)


model = load_model()


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("Detection Settings")

confidence = st.sidebar.slider(
    "Detection Confidence",
    min_value=0.10,
    max_value=0.90,
    value=0.25,
    step=0.05
)

st.sidebar.divider()

st.sidebar.subheader("Detection Classes")

st.sidebar.write("🔪 Knife")
st.sidebar.write("🔫 Gun")
st.sidebar.write("🎯 Rifle")

st.sidebar.divider()

st.sidebar.subheader("Model Information")

st.sidebar.write("**Model:**")
st.sidebar.write("YOLO Custom Trained")

st.sidebar.write("**Device:**")
st.sidebar.write("GPU")

st.sidebar.divider()

st.sidebar.subheader("System Status")

st.sidebar.success("Model Loaded")


# =========================================================
# MAIN TITLE
# =========================================================

st.title("🛡️ Real-Time Weapon Detection")

st.write(
    "AI-powered surveillance and threat detection "
    "using YOLO and computer vision."
)

st.info("● AI VISION SYSTEM ONLINE")


# =========================================================
# IMAGE DETECTION
# =========================================================

st.header("📷 Image Detection")

st.write(
    "Upload an image to detect knives, guns and rifles."
)

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    # Read uploaded image
    image_bytes = uploaded_file.read()

    # Run YOLO
    results = model.predict(
        source=image_bytes,
        imgsz=512,
        conf=confidence,
        device=0,
        verbose=False
    )

    result = results[0]

    # Display detection result
    annotated_image = result.plot()

    st.image(
        annotated_image,
        channels="BGR",
        use_container_width=True
    )

    # Count detections
    detections = []

    for box in result.boxes:

        cls_id = int(box.cls[0])
        conf = float(box.conf[0])

        class_name = model.names[cls_id]

        if class_name in ["knife", "gun", "rifle"]:
            detections.append(
                (class_name, conf)
            )

    # Detection summary
    if len(detections) > 0:

        st.error(
            f"⚠️ WEAPON DETECTED: {len(detections)}"
        )

        for name, conf in detections:

            st.write(
                f"**{name.upper()}** — "
                f"{conf * 100:.1f}% confidence"
            )

    else:

        st.success("✅ No weapon detected")


# =========================================================
# LIVE CAMERA DETECTION
# =========================================================

st.divider()

st.header("📹 Live Camera Detection")

st.write(
    "Start the camera to detect weapons in real time."
)

st.write(
    "Real-time YOLO detection • Knife • Gun • Rifle"
)


# =========================================================
# WEBRTC VIDEO PROCESSOR
# =========================================================

class VideoProcessor(VideoProcessorBase):

    def __init__(self):

        self.model = load_model()

    def recv(self, frame):

        img = frame.to_ndarray(format="bgr24")

        results = self.model.predict(
            source=img,
            imgsz=512,
            conf=confidence,
            device=0,
            verbose=False
        )

        result = results[0]

        # Draw YOLO boxes
        annotated = result.plot()

        # Count weapons
        weapon_count = 0

        for box in result.boxes:

            cls_id = int(box.cls[0])

            class_name = self.model.names[cls_id]

            if class_name in ["knife", "gun", "rifle"]:
                weapon_count += 1

        # Add detection message
        if weapon_count > 0:

            cv2 = __import__("cv2")

            cv2.rectangle(
                annotated,
                (20, 20),
                (600, 90),
                (0, 0, 180),
                -1
            )

            cv2.putText(
                annotated,
                f"WEAPON DETECTED: {weapon_count}",
                (40, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (255, 255, 255),
                3
            )

        return av.VideoFrame.from_ndarray(
            annotated,
            format="bgr24"
        )


# =========================================================
# CAMERA CONFIGURATION
# =========================================================

RTC_CONFIGURATION = RTCConfiguration(
    {
        "iceServers": [
            {
                "urls": [
                    "stun:stun.l.google.com:19302"
                ]
            }
        ]
    }
)


# =========================================================
# START CAMERA
# =========================================================

webrtc_streamer(
    key="weapon-detection-camera",
    video_processor_factory=VideoProcessor,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={
        "video": True,
        "audio": False
    },
    async_processing=True
)


# =========================================================
# SYSTEM INFORMATION
# =========================================================

st.divider()

st.header("📊 System Information")

col1, col2, col3 = st.columns(3)


with col1:

    st.subheader("⚙️ YOLO Model")

    st.write("Custom Trained Model")


with col2:

    st.subheader("🎯 Supported Classes")

    st.write("Knife • Gun • Rifle")


with col3:

    st.subheader("🟢 Application Status")

    st.write("System Online • Real-Time Detection")


# =========================================================
# PROJECT NAME
# =========================================================

st.divider()

st.caption("@manideep_projects")

st.caption(
    "Real-Time Weapon Detection • AI Vision"
)