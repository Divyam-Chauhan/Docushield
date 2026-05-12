# DocuShield

DocuShield is an offline signature and document verification system. It provides a computational method for determining whether a questioned signature is authentic or a potential forgery, and includes Error Level Analysis (ELA) for detecting digital manipulation in scanned documents.

## Methodology

### Signature Verification (Deep Learning)
Initial tests demonstrated that direct pixel-comparison algorithms (such as Structural Similarity Index Measure or edge difference mapping) are overly rigid for human handwriting, as natural signatures fluctuate in scale, offset, and stroke width. 

To resolve this, DocuShield uses a deep learning feature extraction pipeline:
- **Model:** MobileNetV2 (pre-trained on ImageNet), with the classification layer removed so it outputs a 1280-dimensional feature vector.
- **Preprocessing:** The blue channel of the image is inverted and adaptive Gaussian thresholding is applied to isolate ink from shadows and paper texture. The signature is then cropped to its bounding box, padded to a square, and resized to 224×224.
- **Evaluation:** Cosine similarity is computed between the feature vectors of the reference and questioned signatures.
- **Thresholds:** ≥88% = genuine, 76–87% = suspicious (manual review), <76% = likely forged. These were calibrated from exhaustive testing across 28 signature pairs from 3 authors.

### Document Tampering Detection (Error Level Analysis)
For full-page document scans, DocuShield applies Error Level Analysis (ELA) to detect digital alterations (such as spliced text or pasted signatures). ELA re-saves the image at a lower JPEG quality and highlights regions that recompress differently, indicating manipulation.

## Installation & Usage

1. **Clone the repository and set up a virtual environment:**
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

2. **Install dependencies:**
   *(Ensure PyTorch, Torchvision, OpenCV, Streamlit, and SciPy are installed)*
   ```bash
   pip install torch torchvision opencv-python streamlit scipy pillow matplotlib numpy
   ```

3. **Run the Application:**
   ```bash
   streamlit run docushield_app.py
   ```

## Repository Structure
- `docushield_app.py`: The main Streamlit application — preprocessing, embedding extraction, verification logic, and ELA.
- `Research/Research.md`: Full documentation of every approach tested, why each failed or succeeded, and the empirical data behind the current thresholds.
- `example signatures/`: Sample signatures from 3 authors used for testing.
