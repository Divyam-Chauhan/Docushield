# DocuShield

DocuShield is an offline signature and document verification system. It provides a computational method for determining whether a questioned signature is authentic or a potential forgery, and includes Error Level Analysis (ELA) for detecting digital manipulation in scanned documents.

## Methodology

### Signature Verification (Deep Learning)
Initial tests demonstrated that direct pixel-comparison algorithms (such as Structural Similarity Index Measure or edge difference mapping) are overly rigid for human handwriting, as natural signatures fluctuate in scale, offset, and stroke width. 

To resolve this, DocuShield utilizes a Deep Learning feature extraction pipeline:
- **Model:** MobileNetV2 (pre-trained on ImageNet), adapted for spatial feature extraction rather than classification.
- **Preprocessing:** Signatures are converted to grayscale, thresholded using Otsu's method, dynamically cropped to their bounding boxes, and normalized as tensors.
- **Evaluation:** The system computes the **Cosine Similarity** between the feature embeddings of a reference signature and a questioned signature.
- **Thresholds:** A high match (>90%) indicates an authentic signature, while lower scores flag the signature as suspicious or likely forged based on stylistic and topological divergence.

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
   pip install torch torchvision opencv-python streamlit scipy pillow matplotlib scikit-image numpy
   ```

3. **Run the Application:**
   ```bash
   streamlit run docushield_app.py
   ```

## Repository Structure
- `docushield_app.py`: The main Streamlit dashboard and inference logic.
- `Research/Research.md`: Detailed documentation of the architectural testing, failures of traditional computer vision, and the implementation of the deep learning pipeline.
- `example signatures/`: Sample data for testing verification thresholds.
