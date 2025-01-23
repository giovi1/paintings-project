> Check the *pdf* for more info.

# Painting Recognition and People Localization in Art Galleries

<img src="/img/example.png" alt="image" style="width:300px;height:auto;">

This repository contains the implementation of a pipeline for **painting detection, rectification, recognition**, and **people localization** in art galleries. The solution combines deep learning, image processing, and geometry techniques to achieve robust results in detecting and identifying paintings, as well as localizing individuals within a gallery.

---

## Features

- **Painting Detection**: Identifies paintings in video frames using semantic segmentation.
- **Painting Rectification**: Corrects perspective distortions in detected paintings.
- **Painting Recognition**: Matches detected paintings to a pre-existing database.
- **People Detection**: Detects and tracks people in gallery footage.
- **Room Localization**: Assigns detected people to specific rooms based on recognized paintings.

---

## Methodology

The pipeline is divided into two primary phases:

1. **Detection and Segmentation**:
   - Utilizes the ADE20K dataset and pretrained models (e.g., HRNet) for semantic segmentation.
   - Extracts paintings and people from video frames.

2. **Post-Processing and Matching**:
   - Rectifies detected paintings using convex hull and homography-based methods.
   - Matches rectified paintings to a database using the ORB feature-matching algorithm.
   - Localizes detected people to specific rooms by linking them to recognized paintings.

### Key Challenges Addressed

- Handling varying lighting conditions and perspective distortions.
- Managing false positives caused by sculptures, shadows, or similar objects.
- Achieving high precision in painting recognition while maintaining a reasonable recall.

---

## Results

- **Painting Detection and Recognition**:
  - Precision: 97%
  - Recall: 68%
- **People Detection**:
  - Precision: 33%
  - Recall: 80%

---

## Requirements

- Python 3.7+
- OpenCV
- PyTorch
- ADE20K Dataset pretrained models
- Additional dependencies listed in `requirements.txt`

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/painting-people-localization.git
   cd painting-people-localization
