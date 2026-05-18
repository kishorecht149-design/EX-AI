# 🏋️ AI Exercise Detection & Form Correction System

An end-to-end AI-powered system that detects exercises, analyzes form correctness, counts repetitions, and provides real-time feedback using **MediaPipe** pose estimation and **Scikit-learn** classification.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Pose-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## ✨ Features

- **Pose Detection** — 33 body landmarks via MediaPipe Pose
- **Exercise Classification** — Squats, Push-ups, Lunges, Plank
- **Form Analysis** — Rule-based posture checks with corrective feedback
- **Rep Counting** — State-machine–based repetition tracking
- **Real-Time Feedback** — Skeleton overlay, exercise name, rep count, form tips
- **Colab Ready** — Runs entirely on Google Colab (CPU only)
- **Synthetic Dataset** — Generate training data without real videos

---

## 📁 Project Structure

```
AI-Exercise-Detection/
├── README.md
├── requirements.txt
├── setup_colab.ipynb
│
├── dataset/
│   ├── raw_videos/            # Place exercise videos here
│   ├── processed_keypoints/   # Extracted keypoint CSVs
│   ├── labels.csv
│   ├── prepare_dataset.py     # Video → keypoints pipeline
│   └── generate_synthetic.py  # Synthetic data generator
│
├── src/
│   ├── pose_detector.py
│   ├── angle_calculator.py
│   ├── feature_extractor.py
│   ├── exercise_classifier.py
│   ├── form_analyzer.py
│   ├── rep_counter.py
│   └── realtime_feedback.py
│
├── models/
│   └── exercise_model.pkl     # Trained classifier
│
├── notebooks/
│   └── model_training.ipynb
│
└── demo/
    └── run_webcam_demo.py
```

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
python3.11 --version  # recommended
git clone https://github.com/kuro-cybet/AI-EX.git
cd AI-EX
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Generate Synthetic Dataset (no videos needed)

```bash
python dataset/generate_synthetic.py
```

This creates `dataset/processed_keypoints/keypoints.csv` with labelled pose data for all four exercises.

### 3. Train the Model

```bash
# Open the notebook
jupyter notebook notebooks/model_training.ipynb
```

Or run from the command line:

```bash
python train_model.py
```

### 4. Run the Live Web App

```bash
streamlit run app.py
```

This opens a browser app with:

- **Live Monitor** — browser camera + live pose, reps, and form status
- **Image Review** — upload a single frame for analysis
- **Video Review** — upload a short clip for sampled analysis

For live camera access in production, serve the app over **HTTPS**.

### 5. Optional: Run the Desktop Demo

```bash
python demo/run_webcam_demo.py
```

Press **q** to quit.

---

## 📊 Dataset Preparation (from real videos)

Organise your videos like this:

```
dataset/raw_videos/
├── squat_correct/
├── squat_incorrect/
├── pushup_correct/
├── pushup_incorrect/
├── lunge_correct/
├── lunge_incorrect/
├── plank_correct/
└── plank_incorrect/
```

Then run:

```bash
python dataset/prepare_dataset.py --frame-skip 5
python train_model.py
```

This will:

- extract 33 MediaPipe keypoints per sampled frame
- compute the feature vector used by the classifier
- save metadata such as `video_id`, `source_video`, and `frame_idx`
- train a model with a **grouped-by-video** split when real workout footage is available

Training outputs are saved to:

- `models/exercise_model.pkl`
- `models/training_report.json`
- `models/confusion_matrix.csv`

Using `video_id` in the split is important because it avoids leaking frames
from the same source video into both train and test sets, which would make
the reported accuracy look better than real-world performance.

## 🏋️ Real-Data Workflow

For better workout detection than the synthetic baseline, use real clips for
each class:

1. Record or collect multiple short videos per class with different people,
   camera heights, clothes, lighting conditions, and room setups.
2. Place them in:

```text
dataset/raw_videos/
  squat_correct/
  squat_incorrect/
  pushup_correct/
  pushup_incorrect/
  lunge_correct/
  lunge_incorrect/
  plank_correct/
  plank_incorrect/
```

3. Extract features:

```bash
python dataset/prepare_dataset.py --frame-skip 5
```

4. Train and evaluate:

```bash
python train_model.py --n-estimators 300
```

5. Review `models/training_report.json` and `models/confusion_matrix.csv`
   before replacing the live app model.

This real-data path is the highest-leverage way to improve workout recognition.
Inference smoothing can help, but model quality comes primarily from better
labeled footage.

---

## 🧪 Evaluation Metrics

The training notebook reports:

| Metric    | Description                        |
|-----------|------------------------------------|
| Accuracy  | Overall classification accuracy    |
| Precision | Positive predictive value per class|
| Recall    | Sensitivity per class              |
| F1-Score  | Harmonic mean of precision & recall|

---

## 🔮 Future Improvements

- Add more exercises (deadlifts, bicep curls, shoulder press)
- LSTM/Transformer model for temporal sequence classification
- 3D pose estimation with depth cameras
- Mobile deployment via TensorFlow Lite
- Gamification & workout tracking dashboard
- Multi-person pose detection

---

## 📄 License

This project is released under the **MIT License**.

---

## 🤝 Contributing

Contributions are welcome! Please open an issue or submit a pull request.

---

*Built for AI research, computer vision coursework, rehabilitation monitoring, and fitness prototyping.*
