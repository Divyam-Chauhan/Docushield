# Signature Verification — Research Methodology

## Phase 1: Initial Approach Using Scaled Normalization

### The Problem
Pixel-comparison algorithms such as SSIM (Structural Similarity Index Measure) and edge differencing are too rigid for human handwriting. A person never signs their name identically twice — scale, offset, pen pressure, and stroke width all vary naturally between attempts.

In preliminary tests, these methods frequently scored a traced forgery higher than a genuine second attempt. A tracing tries to overlap the original strokes exactly, which inflates pixel-based similarity. A genuine re-signing flows naturally and misses exact pixel overlap, but preserves the overall shape and rhythm of the signature.

### Interim Solution
A scaled normalization method was applied to make the algorithm more tolerant:
- The raw SSIM range for signatures (typically `0.05` to `0.13`) was mapped to a `0%`–`100%` scale.
- Scoring weights were set to 85% normalized SSIM, 15% edge matching.
- **Result:** Genuine signatures scored >75%, forgeries scored <50%.

---

## Phase 2: Cross-Testing and Breakdown

When signatures from multiple people were introduced (e.g., "Divya" vs. "Ayush"), the normalization approach failed — two entirely different signatures scored ~90% similarity.

### Why It Failed
The algorithm resized bounding boxes and applied Gaussian blurring to handle natural variation. But blurring two different signatures compresses them into similar horizontal shapes. The overly forgiving SSIM scoring then treated these blurred shapes as matching, producing false positives.

### Other Approaches Tested
Five additional computer vision methods were evaluated to find a solution that did not require neural networks:

1. **SIFT / ORB Keypoint Matching + RANSAC Homography**
   - *Idea:* If two signatures are from the same person, their keypoints should form a consistent geometric transformation.
   - *Result:* Failed. Signatures are thin lines without distinct corners or texture. SIFT typically found fewer than 5 keypoints, making matching unreliable.

2. **Histogram of Oriented Gradients (HOG) + Cosine Similarity**
   - *Idea:* HOG captures gradient distribution and should tolerate small local changes.
   - *Result:* Failed. HOG depends on a fixed spatial grid. Since aspect ratios vary between signing attempts (e.g., 1.52 vs. 1.89), the grid cells misaligned. Genuine pairs scored lower than tracings.

3. **Dynamic Time Warping (DTW) on Projection Profiles**
   - *Idea:* Sum pixels across rows and columns to create projection profiles, then use DTW to align and compare them.
   - *Result:* Failed. Variations in pen pressure and stroke thickness distorted the profiles completely, producing near-zero correlation between genuine pairs.

4. **Grid-Based Density Extraction**
   - *Idea:* Divide the signature into an 8×8 grid, extract ink density and center of mass per cell.
   - *Result:* Failed. The same aspect ratio problem occurred. Different signatures matched at 80%+ because the overall ink density in an 8×8 grid is similar across most handwriting.

5. **Center-of-Mass Padding + Aspect Ratio Penalties**
   - *Idea:* Pad signatures to a standard canvas, align by center of mass, and penalize aspect ratio differences instead of resizing.
   - *Result:* Failed. The penalties were not strong enough to separate same-author variation from different-author differences.

---

## Conclusion After Traditional Methods
These tests showed that pixel-level comparison methods cannot reliably verify offline signatures. The core problem is that intra-class variation (how much one person's signature changes) overlaps with inter-class variation (how different two people's signatures are). No amount of normalization, blurring, or grid subdivision resolves this overlap.

**Decision:** The system was redesigned to use deep learning. Instead of comparing pixels, signatures are passed through a pre-trained convolutional neural network (MobileNetV2) to produce feature embeddings — numerical vectors that represent the high-level structure and style of the signature. These embeddings are then compared using cosine similarity.

---

## Phase 3: Deep Learning Implementation

### Model Setup
The verification engine uses **MobileNetV2**, pre-trained on ImageNet. The classification layer was removed and replaced with an identity function, so the network outputs its internal feature representation (a 1280-dimensional vector) instead of a class label. The model runs in evaluation mode with no gradient computation.

### Preprocessing (Initial Version)
1. **Thresholding:** The image is converted to grayscale and binarized using Otsu's method to separate ink from background.
2. **Cropping:** The bounding box of all non-zero pixels (the ink) is found, and the image is cropped to remove whitespace.
3. **Resizing and normalization:** The cropped image is resized to 224×224 pixels, converted to a tensor, and normalized using ImageNet statistics (mean and standard deviation per color channel).

### Similarity Metric
The system computes **cosine similarity** between the two embedding vectors. Cosine similarity measures the angle between vectors rather than their magnitude, which makes it more stable when image brightness varies. The output ranges from 0.0 (completely different) to 1.0 (identical).

### Interface Changes
Since the system no longer produces per-pixel comparison maps (like SSIM heatmaps or edge overlays), the UI was simplified to show three values: cosine similarity, match percentage, and the verdict (genuine / suspicious / forged).

---

## Phase 4: Empirical Testing and Preprocessing Refinement

### Test Setup
All 8 signatures from 3 authors were tested in every pairwise combination (28 pairs total):
- **Divya (DPSChauhan):** `Original.jpeg`, `original 2.jpeg`, `duplicate.jpeg`
- **Ayush:** `Ayush Original 1.jpeg`, `Ayush Original 2.jpeg`, `Ayush Duplicate.jpeg`
- **Third author (A):** `A signature Original.jpeg`, `A signature original 2.jpeg`

This produced 7 genuine pairs (same author) and 21 impostor pairs (different authors).

### Problem Found: Otsu Thresholding Fails on Shadows
`Original.jpeg` has a dark shadow across the right half of the image from the camera angle. Otsu thresholding computes a single threshold for the entire image, so it classified the shadow as ink. This corrupted the embedding and inflated similarity scores with unrelated authors.

The same problem affected `Ayush Original 1.jpeg`, which has faint bleed-through from the other side of the paper.

With Otsu preprocessing:
```
Original.jpeg  vs  original 2.jpeg:     0.7956  (should be HIGH — same author)
original 2.jpeg  vs  Ayush Duplicate:   0.8458  (should be LOW — different author)
```
A different-author pair scored higher than a same-author pair. The system was producing inverted results.

### Fix: Blue Channel Isolation + Adaptive Thresholding
Ballpoint pen ink absorbs blue light. Shadows and paper do not — they appear as uniform grey across all color channels. This physical difference allows the blue channel to isolate ink from shadows.

The revised preprocessing pipeline:
1. **Extract the blue channel** from the RGB image and invert it (ink becomes bright, background becomes dark).
2. **Apply adaptive Gaussian thresholding** with a 51×51 pixel block size. Each region is thresholded independently, so a shadow on one side of the image does not affect the threshold on the other side.
3. **Morphological opening** with a 2×2 kernel removes single-pixel noise from paper texture without erasing ink strokes.
4. **Square padding** preserves the aspect ratio by padding the cropped signature to a square canvas before resizing to 224×224.

### Results After Fix

| Metric | Value |
|---|---|
| Genuine pairs — average | **0.892** |
| Genuine pairs — minimum | 0.824 |
| Genuine pairs — maximum | 0.954 |
| Impostor pairs — average | **0.702** |
| Impostor pairs — minimum | 0.557 |
| Impostor pairs — maximum | 0.857 |
| Separation gap (genuine avg − impostor avg) | **+0.190** |

The 0.19 gap provides enough separation to set a threshold. The highest-scoring impostor pairs (`original 2 vs Ayush Duplicate: 0.857`, `Original vs Ayush Duplicate: 0.852`) are genuinely ambiguous — both authors use a similar horizontal cursive style, which pushes their embeddings closer together.

### Threshold Settings
Based on these distributions:
- **>= 0.88 → Genuine:** Above the genuine average. High confidence of same author.
- **0.76 – 0.87 → Suspicious:** The overlap zone between genuine and impostor distributions. Manual review recommended.
- **< 0.76 → Likely Forged:** Well below the impostor average. Different author.

### Preprocessing Comparison

| Approach | Genuine Avg | Impostor Avg | Gap |
|---|---|---|---|
| Otsu (grayscale, global) | 0.87 | 0.79 | +0.08 |
| Otsu + square padding | 0.89 | 0.79 | +0.10 |
| Spatial features (7×7 grid) | 0.76 | 0.67 | +0.09 |
| Blue channel + adaptive threshold | 0.892 | 0.702 | +0.190 |
| **Black-on-White + Closing** | **0.887** | **0.664** | **+0.223** |

The blue channel approach was further refined by inverting the signature to black-on-white and padding with white. This nearly doubled the separation gap compared to earlier methods, as the pre-trained CNN features are more effective at identifying patterns against a white background than a black one. This approach was adopted as the production pipeline.

---

## Phase 5: Pipeline Finalization & Threshold Recalibration

Following the move to Black-on-White preprocessing, the gap between genuine and impostor distributions widened to a robust **0.223**. This allowed for safer, more discriminative threshold boundaries.

### Final Results
- **Genuine Avg:** 0.887
- **Impostor Avg:** 0.664
- **Separation Gap:** +0.223

### Revised Verdict Thresholds
- **>= 0.82 → Genuine:** Significant stylistic alignment.
- **0.74 – 0.81 → Suspicious:** Boundary region requiring manual forensic review.
- **< 0.74 → Likely Forged:** Structural and stylistic divergence confirmed.
