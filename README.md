```python
# Write and render README.md for the Face Skin Disease Detection project (no images)
from IPython.display import Markdown, display

readme = r"""
# 👩‍⚕️ Face Skin Disease Detection with Attention Fusion (TensorFlow/Keras)

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.9%2B-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-2.9%2B-D00000?logo=keras&logoColor=white)](https://keras.io/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?logo=opencv&logoColor=white)](https://opencv.org/)
[![Albumentations](https://img.shields.io/badge/Albumentations-1.x-00A896)](https://albumentations.ai/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

End‑to‑end pipeline to detect common face skin conditions from selfie video using dual backbones (ResNet101 + DenseNet121) with attention fusion, skin segmentation, and majority‑vote inference.

</div>

---

## 📌 Overview

This project provides:
- A training notebook to build an attention‑fused model on curated face‑skin datasets.
- An inference notebook that records a 5‑second selfie video, extracts frames, performs skin detection, predicts per‑frame, and uses majority voting for a final label.

> ⚠️ Medical disclaimer: This project is for research/education only and is not a medical device. Do not use it for diagnosis or treatment. Consult qualified healthcare professionals for medical advice.

---

## ✨ Key Features

| Feature | Description |
| :--- | :--- |
| 🧠 Attention Fusion | Dual ImageNet backbones (ResNet101 + DenseNet121) fused via learnable soft attention (2‑way weights). |
| 🎥 Video → Frames | Record 5s selfie (OpenCV), extract frames for robust voting. |
| 🎯 Skin Segmentation | HSV‑based skin mask to focus the model on skin regions. |
| 🧪 Strong Augmentations | Albumentations pipeline; classes balanced to a target count (e.g., 500/class). |
| 🧰 Transfer Learning | Pretrained weights; GAP + Dense layers with BN/Dropout. |
| 🗳️ Majority Voting | Stable final prediction across frames. |

---

## 📂 Project Structure

```plaintext
face-skin-disease/
├── Face disease detection final.ipynb   # Inference pipeline (video → frames → skin → predict → vote)
├── skin disease attention.ipynb         # Training pipeline (data prep, augment, train, eval)
├── skin_disease.h5                      # Trained model weights (not in repo)
├── frames/                              # Extracted frames (generated)
├── processed_skin_frames/               # Skin-masked frames (generated)
├── balanced_train_df.csv                # Saved balanced metadata (generated during training)
├── README.md
└── LICENSE
```

---

## 📦 Datasets

- Face Skin Disease dataset (Kaggle) paths used in training:
  - Train: `/kaggle/input/face-skin-disease/DATA/train`
  - Test: `/kaggle/input/face-skin-disease/DATA/testing`
- Additional “normal” class images sourced from:
  - `/kaggle/input/selfies-id-images-dataset/Selfies ID Images dataset`
  - 236 normal images sampled and added.
- Label harmonization:
  - Ensure consistent class names. If any label appears as `Eczemaa`, map it to `Eczema`.
- Final class set (6):
  - `['normal', 'Eczema', 'Acne', 'Rosacea', 'Actinic Keratosis', 'Basal Cell Carcinoma']`

Note: Datasets are large and not included in the repo. Update paths as needed.

---

## 🧠 Model Architecture (Training)

- Backbones: ResNet101 + DenseNet121 (include_top=False, ImageNet weights)
- Heads: GAP → Dense(512) → BN → Dense(256) → BN → Dropout(0.3) → Dense(128)
- Attention fusion:
  - Concatenate(128+128) → Dense(2, softmax) → weights [α_resnet, α_densenet]
  - Multiply each branch by its weight, then Add
- Classifier:
  - Dense(256) → BN → Dropout(0.3) → Dense(128) → BN → Dropout(0.3) → Dense(6, softmax)
- Training:
  - Optimizer: Adam(lr=1e-4)
  - Loss: sparse_categorical_crossentropy
  - Input size: 224×224, batch size: 16
  - Augmentation: Albumentations; classes balanced to target_count=500

Results (your run):
- Validation Accuracy: ~0.8500
- Test Accuracy: ~0.8475

---

## 🎛️ Inference Pipeline (Notebook: Face disease detection final.ipynb)

1) Record selfie video (5 seconds, 20 FPS) with OpenCV → `selfie_video.avi`
2) Extract frames → `frames/`
3) Skin detection (HSV range: lower=(0,48,80), upper=(20,255,255)) → `processed_skin_frames/`
4) Load trained model weights `skin_disease.h5`
5) Predict per frame (224×224, scaled 0–1), map to class names
6) Majority vote across frames → Final predicted condition

Tip: Ensure `class_names` matches the order the model was trained on.

---

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- GPU recommended for training

### Installation
```bash
pip install tensorflow keras opencv-python albumentations pandas numpy scikit-learn pillow tqdm
```

### Train (optional)
- Open `skin disease attention.ipynb`
- Set dataset paths
- Run cells to:
  - Build `train_df`, add 236 “normal” images, balance to ~500/class with Albumentations
  - Split into train/val
  - Train attention‑fusion model
  - Save weights to `skin_disease.h5`

### Inference
- Place `skin_disease.h5` alongside `Face disease detection final.ipynb`
- Run the notebook cells in order
- Final console output shows majority‑vote prediction

---

## ⚖️ Limitations

- Not a diagnostic tool; class labels may be visually similar.
- Dataset domain shift (lighting, camera, ethnicity, makeup, artifacts) can degrade accuracy.
- HSV skin masking is heuristic; may exclude/inflate regions in varied lighting.
- Majority voting helps stability but cannot correct systematic bias.

---

## 🔒 Privacy & Safety

- Obtain explicit consent for recording.
- Prefer local execution; avoid uploading selfies to external servers.
- Anonymize and securely delete data after use.

---

## 🧪 Reproducibility

- Fix seeds where possible; document data versions.
- Save label encoder and class index mapping used at train time.
- Keep the exact preprocessing consistent between train and inference.

---

## 📄 License

Released under the MIT License. See `LICENSE`.

---

## 👨‍💻 Authors

- Your Name — add GitHub/LinkedIn links
- Contributors welcome! Open an issue or PR.

---

⭐️ If this repo helps your work, consider giving it a star!
"""

with open("README.md", "w", encoding="utf-8") as f:
    f.write(readme.strip() + "\n")

display(Markdown(readme))
print("README.md written.")
```
