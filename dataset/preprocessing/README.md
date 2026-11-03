# Dataset Preprocessing

This document summarizes the image and facial landmark preprocessing
procedure described in the MVGP-FER manuscript for RAF-DB and FER2013.

## 1. Image Preprocessing

Each facial image undergoes the following steps:

1. Convert the input image to grayscale.
2. Replicate the grayscale image across three channels.
3. Resize the image to 224 x 224 pixels using bilinear interpolation.
4. Scale pixel intensities from [0, 255] to [0, 1].
5. Apply channel-wise ImageNet normalization.

Normalization parameters:

- Mean: [0.485, 0.456, 0.406]
- Standard deviation: [0.229, 0.224, 0.225]

Final image dimensions: 3 x 224 x 224.

## 2. Facial Landmark Extraction

Facial landmarks are extracted using MediaPipe Face Mesh.

The proposed method selects 44 expression-relevant landmarks
from the facial contour, eyes, eyebrows, nose, mouth, and cheeks.

These landmarks provide 88 coordinate values (44 x 2).

## 3. Landmark Coordinate Normalization

The normalization center is calculated using the outer-eye landmarks:

    center = (r33 + r263) / 2

The normalization scale is:

    scale = max(
        distance(r33, r263),
        0.35 * distance(r10, r152),
        1e-6
    )

Each landmark is normalized as:

    p_j = (r_j - center) / scale

The resulting coordinates form an 88-dimensional vector.

## 4. Geometric Expression Features

Sixteen additional geometric cues are constructed:

- Mouth opening
- Mouth width
- Smile curvature
- Mouth downturn
- Left-eye openness
- Right-eye openness
- Mean eye openness
- Left brow-eye distance
- Right brow-eye distance
- Mean brow-eye distance
- Inner-brow gap
- Upper-lip raise
- Jaw drop
- Mouth asymmetry
- Cheek-mouth ratio
- Face-tension proxy

The final geometric representation contains:

    88 normalized coordinates + 16 geometric cues = 104 features

## 5. Feature Standardization

Each of the 104 dimensions is standardized using the mean
and standard deviation estimated from valid training samples.

The same training-set statistics are applied during evaluation.

## 6. Missing Landmark Handling

If valid facial landmarks cannot be extracted, geometric
features are treated as unavailable.

The manuscript specifies a visual-only prediction fallback
rather than inserting fabricated geometric features.

## 7. Expression Labels

The seven expression categories are:

1. Angry
2. Disgust
3. Fear
4. Happy
5. Neutral
6. Sad
7. Surprise

## Reproducibility Note

This document describes the preprocessing methodology from
the manuscript. It is not an executable preprocessing pipeline.

The complete landmark index selection, individual geometric
cue formulas, and training-set normalization statistics are
not included in this public repository.

Datasets must be obtained from their original providers.
