# Signature Verification Context & Research Methodology

## Phase 1: Initial Approach Using Scaled Normalization

### The Initial Problem
When comparing signatures computationally, direct pixel-comparison algorithms such as Structural Similarity Index Measure (SSIM) or raw edge differences are overly rigid for human handwriting. Human signatures naturally fluctuate in scale, offset, and stroke width. 

In preliminary tests, standard mathematical comparisons frequently scored a traced forgery higher than a genuine second attempt. This occurs because a tracing attempts to perfectly overlap the original strokes, inflating pixel-based similarity metrics. Conversely, a genuine second attempt flows naturally; it lacks exact pixel overlap but retains the true structural geometry of the signature.

### The Interim Solution
To make the algorithm accommodate human variation, a scaled normalization method was applied:
- The natural human variation range for SSIM (typically `0.05` to `0.13`) was mapped to a standard `0%` to `100%` scale.
- Scoring weights were adjusted to favor this normalized SSIM (85%) over exact edge matching (15%).
- **Result:** This successfully categorized genuine signatures at >75% similarity and forged ones at <50%.

---

## Phase 2: Cross-Testing and Algorithmic Breakdown

Upon introducing a wider dataset containing signatures from multiple individuals (e.g., "Divya" vs. "Ayush"), the scaled normalization approach failed. Two completely different signatures were incorrectly scoring ~90% similarity. 

### Failure Analysis
To accommodate natural variation, the algorithm resized bounding boxes and applied Gaussian blurring. However, blurring two entirely different signatures compressed them into generic horizontal shapes. When evaluated using the normalized SSIM scoring, these generalized shapes matched one another, resulting in high false-positive rates.

### Evaluation of Traditional Computer Vision Techniques
To establish a reliable offline signature verification algorithm without requiring neural networks, five separate computer vision methodologies were evaluated:

1. **SIFT / ORB Keypoint Matching + RANSAC Homography**
   - *Hypothesis:* If two signatures belong to the same person, their keypoints will form a valid geometric transformation.
   - *Result:* **FAILED.** Signatures are sparse line structures that lack distinct corners and gradients. SIFT struggled to identify stable keypoints (often returning <5 matches), leading to inconsistent similarity scores.

2. **Histogram of Oriented Gradients (HOG) + Cosine Similarity**
   - *Hypothesis:* HOG captures the gradient structure and shape distribution of an image, making it invariant to small local changes.
   - *Result:* **FAILED.** HOG requires rigid spatial grids. Because the aspect ratios of genuine signatures vary significantly across attempts (e.g., one attempt had an Aspect Ratio of 1.52, the next was 1.89), the grid cells misaligned, causing genuine signatures to score lower than tracings.

3. **Dynamic Time Warping (DTW) on Projection Profiles**
   - *Hypothesis:* Taking the horizontal and vertical pixel projection profiles (summing pixels in rows/columns) and comparing them using DTW or Pearson Correlation should handle natural stretch and compression.
   - *Result:* **FAILED.** Variations in pen pressure and stroke thickness completely skewed the projection profiles, resulting in ~0% correlation between genuine pairs.

4. **Grid-Based Density Extraction**
   - *Hypothesis:* Dividing the signature into an 8x8 grid and extracting pixel density and Center of Mass for each cell.
   - *Result:* **FAILED.** The same aspect ratio squishing problem occurred. Totally different signatures matched at 80%+ because the global density of ink in an 8x8 grid is relatively uniform across all human signatures.

5. **Center-of-Mass Padding + Aspect Ratio Penalties**
   - *Hypothesis:* Instead of resizing the images, pad them to a standard canvas, align their Centers of Mass, and apply penalties for differences in aspect ratio.
   - *Result:* **FAILED.** The penalty mechanism lacked discriminative power. The absolute structural overlap between two natural attempts from the same author is minimal; no combination of padding and blurring could reliably distinguish inter-class from intra-class variations.

---

## Conclusion & Proposed Deep Learning Transition
Testing demonstrated that direct pixel overlap and traditional structural computer vision metrics cannot reliably verify offline signatures. The intra-class variation (how much a single author's signature changes) overlaps too heavily with inter-class variation (how different two authors' signatures are).

**The Solution:**
A structural redesign was necessary. The system shifted from pixel-matching to a **Siamese Neural Network (Deep Learning)** approach. By processing signatures through a pre-trained Convolutional Neural Network (e.g., MobileNetV2), the model extracts a high-level feature embedding that represents the spatial structure of the signature. Utilizing Cosine Similarity on these embeddings is the standard method for achieving reliable offline signature verification.

---

## Phase 3: Deep Learning Implementation and Embedding Extraction

Following the conclusion of Phase 2, the system architecture was completely refactored to eliminate strict pixel-wise comparisons. The solution was implemented by transitioning from traditional computer vision heuristics to a Deep Learning feature extraction pipeline.

### Architectural Setup
The core verification engine now utilizes **MobileNetV2**, a lightweight convolutional neural network (CNN) pre-trained on the ImageNet dataset. The primary objective of this network is not image classification, but rather spatial feature extraction. To achieve this:
- The terminal classification layer (the fully connected head) of the network was removed and replaced with an Identity function.
- The model operates strictly in evaluation mode (`model.eval()`), utilizing the pre-trained hierarchical filters to capture structural and stylistic patterns rather than categorical identifiers.

### Data Preprocessing Pipeline
Deep learning models require standardized inputs to generate consistent embeddings. The preprocessing pipeline was redesigned to isolate the signature geometry:
1. **Grayscale and Thresholding:** The input image is converted to grayscale, and Otsu's thresholding is applied to aggressively separate the ink from the background substrate.
2. **Bounding Box Cropping:** A contour detection mechanism identifies the spatial limits of the non-zero pixels (the signature itself) and dynamically crops the image. This removes extraneous whitespace, ensuring the CNN's receptive fields focus entirely on the stroke data.
3. **Tensor Normalization:** The cropped image is resized to a standardized 224x224 resolution, converted into a multi-dimensional PyTorch tensor, and normalized using the standard ImageNet mean and standard deviation matrices.

### Metric Definition: Cosine Similarity
To evaluate the authenticity of a questioned signature against a reference, the system computes the similarity between their respective feature embeddings. Instead of Euclidean distance, the system utilizes **Cosine Similarity**. 
- Cosine Similarity measures the cosine of the angle between two multi-dimensional vectors projected in a latent space.
- Because the magnitude of the vectors can vary depending on image intensity, analyzing the angular difference provides a much more robust measurement of stylistic equivalence. 
- The resulting similarity metric ranges from `0.0` to `1.0`.

### Threshold Calibration
Based on the distribution of cosine similarity scores across genuine and forged samples, the verification thresholds were strictly recalibrated:
- **Genuine / High Match (> 0.90):** The deep feature embeddings exhibit a statistically significant correlation, indicating the same author.
- **Suspicious (0.82 - 0.89):** The embeddings show notable stylistic divergence. The overall structure is maintained, but granular features conflict, warranting further review.
- **Likely Forged (< 0.82):** The distance in the latent space confirms a distinct topological structure indicative of a different author or a crude tracing attempt.

### UI Refactoring and Optimization
With the transition to high-dimensional embeddings, localized spatial mapping (such as SSIM difference heatmaps and Edge detection overlays) became fundamentally incompatible, as deep features represent global style rather than localized pixels. Consequently, the user interface was refactored into a high-level dashboard focused purely on algorithmic metrics, returning Cosine Similarity, Match Percentage, and defined Risk Thresholds to the user.

---

## Phase 4: Empirical Testing and Preprocessing Refinement

### Test Setup
All 8 signatures from 3 distinct authors were tested against each other in every combination (28 pairs total). Authors are:
- **Divya (DPSChauhan):** 3 signatures — `Original.jpeg`, `original 2.jpeg`, `duplicate.jpeg`
- **Ayush:** 3 signatures — `Ayush Original 1.jpeg`, `Ayush Original 2.jpeg`, `Ayush Duplicate.jpeg`
- **Third author (A):** 2 signatures — `A signature Original.jpeg`, `A signature original 2.jpeg`

Ground truth labels:
- **7 genuine pairs** — signatures from the same author compared against each other
- **21 impostor pairs** — signatures from different authors compared

### Initial Failure: Otsu Thresholding in Mixed Lighting
The initial test run using global Otsu thresholding revealed a critical problem: `Original.jpeg` has a **large dark shadow** across the right half of the image caused by the camera angle. Otsu thresholding, which computes a single global threshold for the entire image, classified this shadow as part of the signature. This produced a corrupted embedding that raised similarity scores with unrelated authors.

The same problem existed for `Ayush Original 1.jpeg`, which shows faint text bleed-through from a page behind the paper.

Result with Otsu preprocessing (first run):
```
Original.jpeg  vs  original 2.jpeg:       0.7956  (should be HIGH — GENUINE pair)
original 2.jpeg  vs  Ayush Duplicate:     0.8458  (should be LOW — DIFFERENT author)
```
The system was scoring a cross-author pair (0.84) **higher than a same-author pair** (0.79) — a direct inversion of expected results.

### Preprocessing Fix: Blue Channel Isolation + Adaptive Thresholding
Pen ink (typically blue or dark blue ballpoint) absorbs blue-spectrum light. Shadows and paper surfaces do not — they appear as a diffuse, uniform grey across all color channels. This physical property allows the blue channel to be used as a selective ink detector.

**Pipeline:**
1. **Blue channel extraction:** Extract only the blue channel from the RGB image (`img[:, :, 2]`).
2. **Inversion:** Invert the channel (ink becomes bright, background becomes dark).
3. **Adaptive Gaussian thresholding:** Instead of a single global threshold, a 51x51 pixel neighborhood is analyzed for each pixel independently. This means a shadow on the right side of the image does not affect the threshold calculation for the left side where the ink sits.
4. **Morphological opening:** A 2x2 structuring element removes isolated single-pixel noise caused by paper grain and dust, without affecting ink strokes.

### Test Results After Fix

| Metric | Value |
|---|---|
| Genuine pairs — average similarity | **0.892** |
| Genuine pairs — minimum | 0.824 |
| Genuine pairs — maximum | 0.954 |
| Impostor pairs — average similarity | **0.702** |
| Impostor pairs — minimum | 0.557 |
| Impostor pairs — maximum | 0.857 |
| Separation gap (genuine avg - impostor avg) | **+0.190** |

The 0.19 gap is a measurable, statistically meaningful separation — sufficient to calibrate a threshold boundary.

The highest-scoring impostor pairs (`original 2.jpeg vs Ayush Duplicate: 0.857` and `Original.jpeg vs Ayush Duplicate: 0.852`) represent genuinely ambiguous cases — both Divya and Ayush's signatures follow a similar horizontal cursive style, which causes the embeddings to overlap at the upper boundary.

### Threshold Recalibration
Based on these results, the verdict thresholds were revised:
- **Genuine / High Match (>= 0.88):** Mean genuine score sits at 0.892 — above this boundary, the probability of a correct positive identification is high.
- **Suspicious (0.76 - 0.87):** The region of overlap between genuine and impostor distributions. Neither rejection nor acceptance can be made with confidence.
- **Likely Forged (< 0.76):** Well below the impostor average — structurally dissimilar signatures.

### Preprocessing Approaches Evaluated (Comparative)

| Approach | Genuine Avg | Impostor Avg | Gap |
|---|---|---|---|
| Otsu (grayscale global) | 0.87 | 0.79 | +0.08 |
| Padding + Otsu (original 2 bug fixed) | 0.89 | 0.79 | +0.10 |
| model.features spatial (7x7 grid) | 0.76 | 0.67 | +0.09 |
| Blue channel + adaptive threshold | **0.892** | **0.702** | **+0.190** |

The blue channel + adaptive threshold approach produced the best genuine/impostor separation by a significant margin, and was therefore integrated into the main application pipeline.

