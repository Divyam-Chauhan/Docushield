# Signature Verification Context & Research Methodology

## Phase 1: The Forgiving Normalization Attempt

### The Initial Problem
When comparing signatures computationally, pixel-perfect algorithms like Structural Similarity Index Measure (SSIM) or raw Edge Differences are inherently **too strict** for human behavior. A human will never sign their name exactly the same way twice—the scale, offset, and stroke width naturally fluctuate. 

In our initial tests using sample signatures, standard mathematical comparisons often scored a **forgery higher than the genuine signature**. This occurs because a forged "tracing" attempts to perfectly overlap the original strokes, artificially boosting pixel-based metrics. Meanwhile, a genuine second attempt flows naturally, missing exact pixel overlaps but retaining the true structural essence of the signature. 

### The Interim Solution
To make the algorithm "forgiving," we applied a **Forgiving Normalization scale**:
- We mapped the natural human variation range (SSIM `0.05` to `0.13`) to a human-readable `0%` to `100%` scale.
- We recalibrated the weights to heavily favor this normalized SSIM (85%) over Edge exactness (15%).
- **Result:** It successfully categorized the genuine signature as >75% and the forged as <50%.

---

## Phase 2: Rigorous Cross-Testing and Algorithmic Breakdown

Upon introducing a wider dataset containing signatures from multiple individuals (e.g., "Divya" vs. "Ayush"), the Forgiving Normalization approach immediately broke down. Two completely different signatures were scoring ~90% similarity. 

### Why did it fail?
To accommodate natural variation, the algorithm resized bounding boxes and applied Gaussian Blurs. However, blurring two totally different signatures squishes them into generic "horizontal blobs." When passed through the highly-normalized SSIM scoring, these blobs matched each other, resulting in catastrophic false positives.

### Trial & Error of Traditional Computer Vision Techniques
To find a robust offline signature verification algorithm without resorting to Deep Learning, we exhaustively tested five separate Computer Vision methodologies:

1. **SIFT / ORB Keypoint Matching + RANSAC Homography**
   - *Hypothesis:* If two signatures are from the same person, their keypoints will form a valid geometric transformation.
   - *Result:* **FAILED.** Signatures are fundamentally thin line-drawings that lack distinct corners and textures. SIFT struggled to find stable keypoints (often returning <5 matches), leading to completely random and inconsistent matching scores.

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
   - *Hypothesis:* Instead of resizing the images, pad them to a standard canvas, align their Centers of Mass, and penalize differences in Aspect Ratio.
   - *Result:* **FAILED.** The penalty wasn't discriminative enough. The absolute structural overlap between two natural attempts from the same human is so small that no combination of padding and blurring could reliably distinguish "different human" from "same human."

---

## Conclusion & Proposed Deep Learning Pivot
The exhaustive testing definitively proves that **pixel-level overlap and traditional structural computer vision metrics cannot reliably verify offline signatures.** The intra-class variation (how much a single person's signature changes) overlaps too heavily with inter-class variation (how different two people's signatures are).

**The Solution:**
A massive architectural pivot is required. We must abandon pixel-matching and adopt a **Siamese Neural Network (Deep Learning)** approach. By passing signatures through a pre-trained Convolutional Neural Network (e.g., MobileNetV2), we can extract a high-level "Feature Embedding" that represents the stylistic flow and spatial structure of the signature. Using Cosine Similarity on these deep embeddings is the industry-standard method for achieving robust offline signature verification.
