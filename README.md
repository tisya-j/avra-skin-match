---
title: Avra Shade Matcher
emoji: 🧪
colorFrom: purple
colorTo: indigo
sdk: gradio
sdk_version: 5.7.1
python_version: '3.10'
app_file: app.py
pinned: false
license: mit
---

# Avra

**AI-powered foundation shade matcher for Indian skin tones**

`Python` · `OpenCV` · `MediaPipe` · `scikit-image` · `Gradio`

---
<img width="1462" height="905" alt="image" src="https://github.com/user-attachments/assets/3daf7151-9e35-49d6-931f-627f7849e048" />

## Overview

Avra predicts foundation shade matches from facial photos using computer vision, LAB color space analysis, and perceptual distance ranking.

Instead of using static shade charts or classification-based tone buckets, Avra models foundation matching as a **continuous similarity problem in perceptual space**, similar to nearest-neighbor retrieval systems in machine learning.

Given 1–4 selfies, the system:
- extracts skin regions using facial landmarks  
- computes a stable LAB representation of skin tone  
- ranks foundation shades by perceptual distance  
- returns top-k matches with confidence scores  

---

## Key Idea

Most foundation matchers:
> skin tone classification → fixed shade bucket

Avra:
> skin pixels → LAB space → similarity search → ranked shades

This avoids early discretization and preserves fine-grained variation in undertone and depth.

---

## Pipeline

### 1. Face and skin sampling
- Detect facial landmarks using MediaPipe  
- Sample skin pixels from forehead and cheek regions  
- Filter shadows, highlights, and noise  
- Aggregate pixels into a stable skin representation  

---

### 2. Color analysis
- Convert BGR → LAB using OpenCV  
- Compute median LAB per image  
- Aggregate across 2–4 selfies to reduce lighting noise  

L — lightness (0–100)  
A — green–red axis  
B — blue–yellow axis  

---

### 3. Depth and undertone classification

Depth is used only for visualization and does not affect matching.

| Depth | L range |
|------|--------|
| Fair | 75+ |
| Fair-Light | 71–74 |
| Light | 67–70 |
| Light-Medium | 63–66 |
| Medium | 59–62 |
| Medium-Tan | 50–58 |
| Tan | 35–49 |
| Deep | < 35 |

Undertones:
Cool Pink · Cool Neutral · Neutral · Neutral Warm · Warm Golden · Olive · Warm Olive  

---

### 4. Foundation database

Includes:
Fenty Beauty Pro Filt'r · Maybelline Fit Me · Lakmé Powerplay · Kay Beauty  

Each entry:
shade_id · brand · shade_code · shade_name · depth · undertone · L · A · B · oxidation · notes  

---

### 5. Matching engine

Shades are ranked by Euclidean distance in LAB space:

d = √((L₁−L₂)² + (A₁−A₂)² + (B₁−B₂)²)

Top 3 matches returned per brand.

---

### 6. Confidence scoring

c = 1 / (1 + d)

Lower distance → higher confidence. Scores indicate perceptual closeness and dataset coverage density.

---

### 7. Undertone bias correction

- Penalizes overly warm shades for olive undertones  
- Reduces orange bias in warm categories  
- Boosts neutral / muted matches where appropriate  

---

## Results

Tested across Fair → Deep skin depth range:

| Subject | L | A | B | Depth | Fenty match | Confidence |
|--------|----|----|----|--------|--------------|------------|
| Aishwarya Rai | 63.8 | 14.0 | 14.0 | Light | 185 | 0.053 |
| Deepika Padukone | 62.3 | 23.5 | 29.5 | Medium | 260 | 0.028 |
| Simone Ashley | 42.1 | 22.0 | 26.0 | Tan | 370 | 0.039 |

A/B channel differences drive separation more strongly than L values alone. Lower confidence reflects boundary regions in shade coverage.

---

## Interface

Built with Gradio.

Outputs:
- Skin swatch (HEX)
- Depth classification
- Undertone classification
- Top 3 matches per brand
- Confidence scores
- Debug skin sampling visualization

---

## Why LAB over RGB

RGB encodes display color space. LAB encodes perceptual differences.

- Euclidean distance ≈ human perception  
- More robust to lighting variation  
- Better separation of skin tone variation  

---

## Design Decisions

Multi-image aggregation:
- reduces lighting variance  
- stabilizes skin estimation  

Distance-based retrieval:
- avoids rigid classification  
- preserves continuous tone representation  

---

## Limitations

- Sensitive to extreme lighting conditions  
- Rule-based undertone classification  
- Limited medium-deep shade density in some brands  
- No personalization or feedback loop  

---

## Roadmap

- Illumination normalization before LAB conversion  
- Learning-to-rank model from user feedback  
- Skin embedding space via metric learning  
- Brand-specific oxidation calibration  
- Expanded shade database (MAC, TIRTIR, etc.)  

---



## Project Structure

```text
avra/
│
├── .gitattributes                 # Git attributes and file handling rules
├── .gitignore                     # Ignored files and directories
├── app.py                         # Gradio web application
├── README.md                      # Project documentation
├── requirements.txt               # Python dependencies
├── test_env.py                    # Environment validation script
│
├── data/
│   ├── Foundation_shades.csv
│   ├── Foundation_shades_with_LAB.csv
│   └── Recommendation_lookup.csv
│
└── utils/
    ├── cv_pipeline.py             # MediaPipe face detection and skin sampling
    ├── colour_analyzer.py         # LAB extraction, depth and undertone analysis
    └── recommender.py             # Foundation retrieval and ranking engine
```

---

## Scope

Designed for:
- Indian and South Asian skin tones  
- Continuous perceptual matching instead of classification  

Evaluated across:
Fair → Deep range  

---

## Built with

Python · OpenCV · MediaPipe · scikit-image · Gradio
