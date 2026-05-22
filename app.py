"""
AI Exercise Coach - Flask Web Application Backend
=================================================
Exposes REST APIs to process real-time landmarks sent from the browser camera,
and handles file uploads for static image & video postural analysis.
"""

from __future__ import annotations

import os
import sys
import base64
import numpy as np
import cv2
from flask import Flask, render_template, request, jsonify

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.runtime_assets import ensure_runtime_assets
from src.web_pipeline import ExerciseAnalyzer, LiveExerciseMonitor

app = Flask(__name__, template_folder='templates')

# Prepare model assets during bootstrap
bootstrap_error = None
try:
    ensure_runtime_assets()
except Exception as exc:
    bootstrap_error = str(exc)

# Lazy-loaded Analyzers to save memory during startup
_global_monitor = None
_global_analyzer = None

def get_monitor() -> LiveExerciseMonitor:
    global _global_monitor
    if bootstrap_error is not None:
        raise RuntimeError(f"Model assets failed to load: {bootstrap_error}")
    if _global_monitor is None:
        _global_monitor = LiveExerciseMonitor()
    return _global_monitor

def get_analyzer() -> ExerciseAnalyzer:
    global _global_analyzer
    if bootstrap_error is not None:
        raise RuntimeError(f"Model assets failed to load: {bootstrap_error}")
    if _global_analyzer is None:
        _global_analyzer = ExerciseAnalyzer()
    return _global_analyzer


# ======================================================================
# HTML Frontend Route
# ======================================================================

@app.route('/')
def index():
    """Serve the single-page glassmorphic coaching console."""
    return render_template('index.html')


# ======================================================================
# Real-time Browser API Endpoints
# ======================================================================

@app.route('/api/analyze_landmarks', methods=['POST'])
def analyze_landmarks():
    """
    Process coordinate landmarks computed in the browser locally.
    
    Accepts JSON containing:
      - landmarks: List of 33 landmarks [x, y, z, visibility]
      - selected_exercise: e.g. "Squat", "Pushup", "Auto detect"
      - session_active: boolean
    """
    data = request.get_json() or {}
    landmarks_list = data.get('landmarks')
    selected_exercise = data.get('selected_exercise', 'auto')
    session_active = data.get('session_active', False)

    if not landmarks_list or len(landmarks_list) != 33:
        return jsonify({'error': 'Invalid landmarks array. Expecting 33 keypoints.'}), 400

    try:
        monitor = get_monitor()
        metrics = monitor.process_landmarks(
            landmarks_list=landmarks_list,
            selected_exercise=selected_exercise,
            session_active=session_active
        )
        
        return jsonify({
            'exercise': metrics.exercise,
            'form': metrics.form,
            'confidence': metrics.confidence,
            'feedback': metrics.feedback,
            'rep_status': metrics.rep_status,
            'pose_detected': metrics.pose_detected,
            'session_active': metrics.session_active,
            'selected_exercise': metrics.selected_exercise,
            'form_score': metrics.form_score,
            'proper_reps': metrics.proper_reps,
            'needs_work_reps': metrics.needs_work_reps,
            'session_seconds': metrics.session_seconds,
            'status_badge': metrics.status_badge
        })
    except Exception as exc:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(exc)}), 500


@app.route('/api/reset_session', methods=['POST'])
def reset_session():
    """Reset the rep counting counts and active monitor timers."""
    data = request.get_json() or {}
    selected_exercise = data.get('selected_exercise', 'auto')
    session_active = data.get('session_active', False)
    freeze = request.args.get('freeze', 'false').lower() == 'true'

    try:
        monitor = get_monitor()
        if freeze:
            monitor.end_session()
        else:
            monitor.reset_session()
            monitor.configure(selected_exercise, session_active)
            
        metrics = monitor.get_metrics()
        return jsonify({
            'exercise': metrics.exercise,
            'form': metrics.form,
            'confidence': metrics.confidence,
            'feedback': metrics.feedback,
            'rep_status': metrics.rep_status,
            'pose_detected': metrics.pose_detected,
            'session_active': metrics.session_active,
            'selected_exercise': metrics.selected_exercise,
            'form_score': metrics.form_score,
            'proper_reps': metrics.proper_reps,
            'needs_work_reps': metrics.needs_work_reps,
            'session_seconds': metrics.session_seconds,
            'status_badge': metrics.status_badge
        })
    except Exception as exc:
        return jsonify({'error': str(exc)}), 500


# ======================================================================
# File Upload Posture Analysis Endpoints
# ======================================================================

@app.route('/api/analyze_image', methods=['POST'])
def analyze_image():
    """Run full posture skeleton extraction and form analysis on an uploaded image."""
    if 'image' not in request.files:
        return jsonify({'error': 'No image file uploaded'}), 400
    
    file = request.files['image']
    img_bytes = file.read()
    
    try:
        analyzer = get_analyzer()
        result = analyzer.analyze_image_bytes(img_bytes)
        if result is None:
            return jsonify({'error': 'No human pose was detected in this photo. Please stand further back.'}), 422
        
        # Calculate standard 100-scale form score
        buf = np.frombuffer(img_bytes, dtype=np.uint8)
        frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        landmarks = analyzer.detector.get_landmarks(frame)
        form_score = 95.0
        if landmarks is not None:
            form_score = analyzer.form_analyzer.get_form_score(result.exercise, landmarks)

        # Base64 encode the BGR annotated result
        _, buffer = cv2.imencode('.jpg', result.annotated_frame)
        base64_str = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            'exercise': result.exercise,
            'form': result.form,
            'confidence': result.confidence,
            'feedback': result.feedback,
            'annotated_image': base64_str,
            'form_score': form_score
        })
    except Exception as exc:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(exc)}), 500


@app.route('/api/analyze_video', methods=['POST'])
def analyze_video():
    """Sample video frames, calculate keypoint features, and return form statistics."""
    if 'video' not in request.files:
        return jsonify({'error': 'No video file uploaded'}), 400
    
    file = request.files['video']
    video_bytes = file.read()
    sample_every = int(request.form.get('sample_every', 10))
    
    try:
        analyzer = get_analyzer()
        result = analyzer.analyze_video_file(video_bytes, sample_every=sample_every)
        
        preview_base64 = None
        if result.preview_frame is not None:
            _, buffer = cv2.imencode('.jpg', result.preview_frame)
            preview_base64 = base64.b64encode(buffer).decode('utf-8')
            
        return jsonify({
            'detected_frames': result.detected_frames,
            'sampled_frames': result.sampled_frames,
            'exercise': result.exercise,
            'form': result.form,
            'confidence': result.confidence,
            'common_feedback': result.common_feedback,
            'preview_frame': preview_base64
        })
    except Exception as exc:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(exc)}), 500


# ======================================================================
# Server Boot
# ======================================================================

if __name__ == '__main__':
    # Serve on Render designated PORT or default local development port 8501
    port = int(os.environ.get('PORT', 8501))
    print(f"Starting AI Exercise Coach Flask App on port {port}...")
    app.run(host='0.0.0.0', port=port, debug=False)
