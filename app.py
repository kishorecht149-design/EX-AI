"""
Streamlit web app for live and upload-based exercise analysis.
"""

from __future__ import annotations

import atexit

import av
import cv2
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer

from src.runtime_assets import ensure_runtime_assets
from src.web_pipeline import ExerciseAnalyzer, LiveExerciseMonitor


WORKOUT_OPTIONS = ["Auto detect", "Squat", "Pushup", "Lunge", "Plank"]


st.set_page_config(
    page_title="AI Exercise Coach",
    page_icon="AI",
    layout="wide",
)

bootstrap_error = None
try:
    ensure_runtime_assets()
except Exception as exc:
    bootstrap_error = str(exc)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

    /* Global overrides */
    .stApp {
        background:
            radial-gradient(circle at 10% 20%, rgba(139, 92, 246, 0.15) 0%, transparent 40%),
            radial-gradient(circle at 90% 10%, rgba(236, 72, 153, 0.12) 0%, transparent 35%),
            radial-gradient(circle at 50% 80%, rgba(59, 130, 246, 0.12) 0%, transparent 40%),
            #090514;
        color: #f1f0f7;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
    }

    /* Ensure all streamlit text elements use the font */
    .stApp h1, .stApp h2, .stApp h3, .stApp p, .stApp span, .stApp label, .stApp button, .stApp select {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
    }

    /* Custom Hero Section */
    .hero {
        position: relative;
        overflow: hidden;
        padding: 2.2rem 2rem;
        border-radius: 24px;
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.05), rgba(255, 255, 255, 0.01));
        border: 1px solid rgba(255, 255, 255, 0.08);
        box-shadow: 0 24px 70px rgba(0, 0, 0, 0.4);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        margin-bottom: 2rem;
    }
    .kicker {
        letter-spacing: 0.16em;
        text-transform: uppercase;
        font-size: 0.8rem;
        color: #a78bfa;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .hero h1 {
        font-size: 3.5rem !important;
        font-weight: 800 !important;
        margin: 0 0 0.8rem 0;
        background: linear-gradient(135deg, #ffffff 30%, #c084fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero p {
        font-size: 1.1rem;
        max-width: 52rem;
        color: #cbd5e1;
        line-height: 1.6;
    }

    /* Glass Card */
    .glass-card {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.04), rgba(255, 255, 255, 0.01));
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 20px;
        padding: 1.5rem;
        box-shadow: 0 16px 40px rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        transition: transform 0.3s ease, border-color 0.3s ease;
    }
    .glass-card:hover {
        transform: translateY(-2px);
        border-color: rgba(167, 139, 250, 0.25);
    }

    /* Dumbbell Animation Styling */
    .dumbbell-shell {
        position: absolute;
        top: 24px;
        right: 32px;
        width: 130px;
        height: 130px;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(167, 139, 250, 0.15), transparent 70%);
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .dumbbell {
        position: relative;
        width: 90px;
        height: 16px;
        background: linear-gradient(90deg, #a78bfa, #f472b6, #a78bfa);
        border-radius: 999px;
        animation: dumbbell-bob 3s ease-in-out infinite;
        box-shadow: 0 8px 24px rgba(167, 139, 250, 0.3);
    }
    .dumbbell::before,
    .dumbbell::after {
        content: "";
        position: absolute;
        top: -12px;
        width: 16px;
        height: 40px;
        border-radius: 6px;
        background: linear-gradient(180deg, #8b5cf6, #ec4899);
        box-shadow:
            8px 0 0 0 rgba(139, 92, 246, 0.8),
            16px 0 0 0 rgba(236, 72, 153, 0.8);
    }
    .dumbbell::before { left: -24px; }
    .dumbbell::after {
        right: -24px;
        box-shadow:
            -8px 0 0 0 rgba(139, 92, 246, 0.8),
            -16px 0 0 0 rgba(236, 72, 153, 0.8);
    }

    @keyframes dumbbell-bob {
        0% { transform: translateY(0px) rotate(-8deg); }
        50% { transform: translateY(-10px) rotate(8deg); }
        100% { transform: translateY(0px) rotate(-8deg); }
    }

    /* Status Chip */
    .status-chip {
        display: inline-block;
        padding: 0.45rem 1rem;
        border-radius: 999px;
        background: rgba(167, 139, 250, 0.1);
        color: #c084fc;
        font-weight: 700;
        border: 1px solid rgba(167, 139, 250, 0.2);
        font-size: 0.85rem;
    }

    /* Streamlit Button Overrides */
    div.stButton > button {
        background: linear-gradient(135deg, #7c3aed 0%, #4f46e5 100%) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.6rem 1.5rem !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        box-shadow: 0 4px 15px rgba(124, 58, 237, 0.3) !important;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
        width: 100% !important;
        height: auto !important;
    }
    div.stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 20px rgba(124, 58, 237, 0.45) !important;
        background: linear-gradient(135deg, #8b5cf6 0%, #6366f1 100%) !important;
    }
    div.stButton > button:active {
        transform: translateY(1px) !important;
        box-shadow: 0 2px 8px rgba(124, 58, 237, 0.2) !important;
    }
    /* Specifically target the Reset button to have a secondary style */
    div.stButton > button[key*="reset"] {
        background: rgba(255, 255, 255, 0.08) !important;
        color: #f1f0f7 !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        box-shadow: none !important;
    }
    div.stButton > button[key*="reset"]:hover {
        background: rgba(255, 255, 255, 0.12) !important;
        border-color: rgba(255, 255, 255, 0.25) !important;
    }

    /* Streamlit Native Selectbox & Form Inputs */
    div[data-baseweb="select"] {
        background-color: rgba(255, 255, 255, 0.04) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 12px !important;
        color: #ffffff !important;
    }
    div[data-baseweb="select"] > div {
        background-color: transparent !important;
        border: none !important;
        color: #ffffff !important;
    }
    div[data-baseweb="select"] span {
        color: #ffffff !important;
    }
    ul[role="listbox"] {
        background-color: #110c22 !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        border-radius: 12px !important;
    }
    ul[role="listbox"] li {
        color: #e2e8f0 !important;
        transition: background-color 0.2s;
    }
    ul[role="listbox"] li:hover {
        background-color: rgba(167, 139, 250, 0.15) !important;
    }

    /* Streamlit Tabs Overrides */
    button[data-baseweb="tab"] {
        font-weight: 600 !important;
        color: #94a3b8 !important;
        border-bottom: 2px solid transparent !important;
        transition: all 0.2s !important;
        background-color: transparent !important;
    }
    button[data-baseweb="tab"]:hover {
        color: #ffffff !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #c084fc !important;
        border-bottom: 2px solid #a78bfa !important;
    }

    /* File Uploader styling */
    section[data-testid="stFileUploader"] {
        background: rgba(255, 255, 255, 0.02) !important;
        border: 1px dashed rgba(255, 255, 255, 0.15) !important;
        border-radius: 16px !important;
        padding: 1.5rem !important;
    }
    section[data-testid="stFileUploader"] [data-testid="stFileUploadDropzone"] {
        background: transparent !important;
        border: none !important;
    }

    /* Metric styling */
    div[data-testid="stMetricValue"] {
        font-size: 2.2rem !important;
        font-weight: 800 !important;
        color: #ffffff !important;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 0.85rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.08em !important;
        color: #94a3b8 !important;
    }

    /* Video/Image captions and titles */
    .stApp h3 {
        font-weight: 700 !important;
        color: #ffffff !important;
        margin-top: 1.5rem !important;
    }
    .stApp label {
        font-weight: 600 !important;
        color: #e2e8f0 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_upload_analyzer() -> ExerciseAnalyzer:
    if bootstrap_error is not None:
        raise RuntimeError(bootstrap_error)
    analyzer = ExerciseAnalyzer()
    atexit.register(analyzer.close)
    return analyzer


def format_feedback(messages: list[str]) -> str:
    if not messages:
        return "No major form issues detected."
    return "\n".join(f"- {msg}" for msg in messages)


def render_live_metric_cards(metrics, selected_workout: str) -> None:
    chosen_label = selected_workout if selected_workout != "Auto detect" else metrics.exercise.title()
    cards = st.columns(5)
    items = [
        ("Workout", chosen_label),
        ("Status", metrics.status_badge),
        ("Reps / Hold", metrics.rep_status),
        ("Good Reps", str(metrics.proper_reps)),
        ("Needs Work", str(metrics.needs_work_reps)),
    ]
    for col, (label, value) in zip(cards, items):
        with col:
            st.markdown(
                f"""
                <div class="glass-card">
                    <div style="font-size:0.78rem; letter-spacing:0.08em; text-transform:uppercase; color:#a78bfa; font-weight:600;">{label}</div>
                    <div style="font-size:1.4rem; font-weight:800; color:#ffffff; margin-top:0.2rem;">{value}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def show_frame_result(result) -> None:
    col1, col2 = st.columns([1.35, 1], gap="large")
    with col1:
        rgb = cv2.cvtColor(result.annotated_frame, cv2.COLOR_BGR2RGB)
        st.image(rgb, caption="Pose overlay", use_container_width=True)
    with col2:
        st.metric("Exercise", result.exercise.title())
        st.metric("Form", result.form.title())
        st.metric("Confidence", f"{result.confidence:.0%}")
        st.markdown("**Feedback**")
        st.markdown(format_feedback(result.feedback))


def show_video_result(result) -> None:
    col1, col2 = st.columns([1.35, 1], gap="large")
    with col1:
        if result.preview_frame is not None:
            rgb = cv2.cvtColor(result.preview_frame, cv2.COLOR_BGR2RGB)
            st.image(rgb, caption="Highest-confidence sampled frame", use_container_width=True)
        else:
            st.info("No person was detected in the sampled frames.")
    with col2:
        st.metric("Exercise", result.exercise.title())
        st.metric("Form", result.form.title())
        st.metric("Confidence", f"{result.confidence:.0%}")
        st.metric("Detected Frames", f"{result.detected_frames}/{result.sampled_frames}")
        st.markdown("**Most common feedback**")
        st.markdown(format_feedback(result.common_feedback))


class LiveCameraProcessor:
    """WebRTC processor that runs the live exercise monitor."""

    def __init__(self) -> None:
        if bootstrap_error is not None:
            raise RuntimeError(bootstrap_error)
        self.monitor = LiveExerciseMonitor()
        self._applied_counter_version = -1

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        image = frame.to_ndarray(format="bgr24")
        annotated = self.monitor.process_frame(image)
        return av.VideoFrame.from_ndarray(annotated, format="bgr24")

    def get_metrics(self):
        return self.monitor.get_metrics()

    def configure(self, selected_exercise: str, session_active: bool) -> None:
        self.monitor.configure(selected_exercise, session_active)

    def end_session(self) -> None:
        self.monitor.end_session()

    def reset_session(self) -> None:
        self.monitor.reset_session()

    def reset_session_if_needed(self, counter_version: int) -> None:
        if counter_version != self._applied_counter_version:
            self.monitor.reset_session()
            self._applied_counter_version = counter_version

    def has_detector(self) -> bool:
        return self.monitor.detector is not None

    def get_detector_error(self) -> str | None:
        return self.monitor.detector_error


if "selected_workout" not in st.session_state:
    st.session_state.selected_workout = "Squat"
if "workout_picker" not in st.session_state:
    st.session_state.workout_picker = st.session_state.selected_workout
if "workout_session_active" not in st.session_state:
    st.session_state.workout_session_active = False
if "session_status" not in st.session_state:
    st.session_state.session_status = "Ready"
if "counter_version" not in st.session_state:
    st.session_state.counter_version = 0


def start_workout_session() -> None:
    """Start a fresh live workout session."""
    st.session_state.selected_workout = st.session_state.workout_picker
    st.session_state.workout_session_active = True
    st.session_state.session_status = "Active"
    st.session_state.counter_version += 1


def end_workout_session() -> None:
    """Stop the current workout session without losing the final count."""
    st.session_state.workout_session_active = False
    st.session_state.session_status = "Ended"


def request_counter_reset() -> None:
    """Reset the rep counter while keeping the chosen workout."""
    st.session_state.counter_version += 1
    if st.session_state.workout_session_active:
        st.session_state.session_status = "Active"
    else:
        st.session_state.session_status = "Ready"


st.markdown(
    """
    <section class="hero">
        <div class="kicker">Live Web Coach</div>
        <h1>AI Exercise Coach</h1>
        <div class="dumbbell-shell"><div class="dumbbell"></div></div>
        <p>
            Monitor exercises live from the browser camera, count reps, and
            flag whether posture looks right or wrong with tighter session-based
            logic, steadier rep counting, and a more polished coaching surface.
        </p>
    </section>
    """,
    unsafe_allow_html=True,
)

st.caption(
    "For live monitoring, the browser needs camera permission. In production, "
    "deploy over HTTPS so camera access works consistently."
)

if bootstrap_error is not None:
    st.error(
        "Startup bootstrap failed while preparing model assets for deployment:\n\n"
        f"{bootstrap_error}"
    )

live_tab, image_tab, video_tab = st.tabs(
    ["Live Monitor", "Image Review", "Video Review"]
)

with live_tab:
    st.subheader("Live camera analysis")
    st.markdown(
        """
        <div class="glass-card" style="margin-bottom:1rem;">
            Choose a workout, start a session, and keep your full body visible.
            The counter now uses tighter transition logic, confidence smoothing,
            and session-based rep tracking so it does not bounce back to zero
            when the classifier jitters.
        </div>
        """,
        unsafe_allow_html=True,
    )

    control_col1, control_col2, control_col3 = st.columns([1.2, 1, 1])
    with control_col1:
        st.selectbox(
            "Workout",
            WORKOUT_OPTIONS,
            index=WORKOUT_OPTIONS.index(st.session_state.workout_picker),
            key="workout_picker",
            disabled=st.session_state.workout_session_active,
        )
    with control_col2:
        st.button(
            "Start Workout Session",
            use_container_width=True,
            on_click=start_workout_session,
            disabled=st.session_state.workout_session_active,
        )
    with control_col3:
        st.button(
            "End Workout Session",
            use_container_width=True,
            on_click=end_workout_session,
            disabled=not st.session_state.workout_session_active,
        )

    st.markdown(
        f"""
        <div class="glass-card" style="margin-bottom:1rem;">
            <div style="display:flex; justify-content:space-between; gap:1rem; flex-wrap:wrap;">
                <div>
                    <div style="font-size:0.78rem; letter-spacing:0.08em; text-transform:uppercase; color:#a78bfa; font-weight:600;">Selected workout</div>
                    <div style="font-size:1.25rem; font-weight:800; color:#ffffff;">{st.session_state.selected_workout}</div>
                </div>
                <div>
                    <div style="font-size:0.78rem; letter-spacing:0.08em; text-transform:uppercase; color:#a78bfa; font-weight:600;">Session state</div>
                    <div style="font-size:1.25rem; font-weight:800; color:#ffffff;">
                        {st.session_state.session_status}
                    </div>
                </div>
                <div>
                    <div style="font-size:0.78rem; letter-spacing:0.08em; text-transform:uppercase; color:#a78bfa; font-weight:600;">How it works</div>
                    <div style="font-size:1rem; color:#cbd5e1;">Choose workout, start once, end to freeze the final count.</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    ctx = webrtc_streamer(
        key="exercise-live-monitor",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration={
            "iceServers": [
                {
                    "urls": ["stun:stun.l.google.com:19302"]
                },
                {
                    "urls": [
                        "stun:openrelay.metered.ca:80",
                        "turn:openrelay.metered.ca:80",
                        "turn:openrelay.metered.ca:443"
                    ],
                    "username": "openrelayproject",
                    "credential": "openrelayproject"
                }
            ]
        },
        media_stream_constraints={
            "video": {
                "width": {"ideal": 960},
                "height": {"ideal": 540},
                "facingMode": "user",
            },
            "audio": False,
        },
        async_processing=True,
        video_processor_factory=LiveCameraProcessor,
    )

    metrics_placeholder = st.empty()
    live_summary_placeholder = st.empty()

    if ctx.video_processor:
        selected_for_monitor = (
            "auto" if st.session_state.selected_workout == "Auto detect"
            else st.session_state.selected_workout
        )
        ctx.video_processor.reset_session_if_needed(st.session_state.counter_version)
        ctx.video_processor.configure(
            selected_for_monitor,
            st.session_state.workout_session_active,
        )
        if not st.session_state.workout_session_active:
            ctx.video_processor.end_session()
        metrics = ctx.video_processor.get_metrics()
        render_live_metric_cards(metrics, st.session_state.selected_workout)
        live_summary_placeholder.markdown(
            f"""
            <div class="glass-card" style="margin-top:1rem;">
                <div style="display:flex; gap:0.8rem; align-items:center; flex-wrap:wrap;">
                    <span class="status-chip">{metrics.status_badge}</span>
                    <span style="font-weight:700; color:#cbd5e1;">Form: {metrics.form.title()}</span>
                    <span style="font-weight:700; color:#cbd5e1;">Confidence: {metrics.confidence:.0%}</span>
                    <span style="font-weight:700; color:#cbd5e1;">Form score: {metrics.form_score:.0f}/100</span>
                    <span style="font-weight:700; color:#cbd5e1;">Session: {metrics.session_seconds:.1f}s</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        metrics_placeholder.markdown(
            "**Live feedback**\n\n" + format_feedback(metrics.feedback)
        )
        if not ctx.video_processor.has_detector():
            st.error(
                "Live pose detection could not start in this runtime. "
                f"Underlying error: {ctx.video_processor.get_detector_error()}"
            )
    else:
        class Waiting:
            exercise = "waiting"
            rep_status = "Reps: 0"
            proper_reps = 0
            needs_work_reps = 0
            status_badge = "Waiting"
        render_live_metric_cards(Waiting(), st.session_state.selected_workout)
        metrics_placeholder.info(
            "Allow camera access and press Start to begin live monitoring."
        )

    st.button(
        "Reset Counter",
        use_container_width=True,
        on_click=request_counter_reset,
        disabled=ctx.video_processor is None,
    )

with image_tab:
    st.subheader("Single image review")
    upload = st.file_uploader(
        "Upload a JPG or PNG image", type=["jpg", "jpeg", "png"], key="image-upload"
    )
    if upload is not None:
        upload_analyzer = get_upload_analyzer()
        if upload_analyzer.detector is None:
            st.error(
                "Image review is unavailable because the pose detector could not start in "
                f"this runtime: {upload_analyzer.detector_error}"
            )
        else:
            with st.spinner("Analyzing image..."):
                result = upload_analyzer.analyze_image_bytes(upload.getvalue())
            if result is None:
                st.error("No pose was detected in that image. Try a clearer full-body photo.")
            else:
                show_frame_result(result)

with video_tab:
    st.subheader("Short video review")
    upload = st.file_uploader(
        "Upload a short MP4, MOV, AVI, MKV, or WEBM clip",
        type=["mp4", "mov", "avi", "mkv", "webm"],
        key="video-upload",
    )
    sample_every = st.slider(
        "Frame sampling interval",
        min_value=5,
        max_value=30,
        value=10,
        help="Higher values analyze fewer frames and run faster.",
    )
    if upload is not None:
        upload_analyzer = get_upload_analyzer()
        if upload_analyzer.detector is None:
            st.error(
                "Video review is unavailable because the pose detector could not start in "
                f"this runtime: {upload_analyzer.detector_error}"
            )
        else:
            with st.spinner("Analyzing video..."):
                result = upload_analyzer.analyze_video_file(
                    upload.getvalue(), sample_every=sample_every
                )
            show_video_result(result)
