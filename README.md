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
git clone https://github.com/your-username/AI-Exercise-Detection.git
cd AI-Exercise-Detection
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
python -c "
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib, os

df = pd.read_csv('dataset/processed_keypoints/keypoints.csv')
X = df.drop(columns=['label'])
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
print(classification_report(y_test, y_train[:len(y_test)] if len(y_test) > len(y_train) else clf.predict(X_test)))
os.makedirs('models', exist_ok=True)
joblib.dump(clf, 'models/exercise_model.pkl')
print('Model saved to models/exercise_model.pkl')
"
```

### 4. Run the Real-Time Demo

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
python dataset/prepare_dataset.py
```

This will extract 33 MediaPipe keypoints per frame, compute joint angles, and save everything to `dataset/processed_keypoints/keypoints.csv`.

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
