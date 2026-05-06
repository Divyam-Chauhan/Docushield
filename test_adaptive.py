import os
import itertools
import numpy as np
import cv2
import torch
from PIL import Image, ImageOps
import torchvision.models as models
import torchvision.transforms as transforms
from scipy.spatial.distance import cosine

import warnings
warnings.filterwarnings("ignore")

# ─── Ground truth for our 8 signatures ───────────────────────────────────────
# Group 1 (Divya): Original, original 2, duplicate
# Group 2 (A sig): A signature Original, A signature original 2
# Group 3 (Ayush): Ayush Original 1, Ayush Original 2, Ayush Duplicate
# Expected: intra-group >> inter-group

GROUND_TRUTH = {
    ("Original.jpeg",                "original 2.jpeg"):              "GENUINE_PAIR",
    ("Original.jpeg",                "duplicate.jpeg"):               "GENUINE_PAIR",
    ("original 2.jpeg",              "duplicate.jpeg"):               "GENUINE_PAIR",
    ("A signature Original.jpeg",    "A signature original 2.jpeg"):  "GENUINE_PAIR",
    ("Ayush Original 1.jpeg",        "Ayush Original 2 .jpeg"):       "GENUINE_PAIR",
    ("Ayush Original 1.jpeg",        "Ayush Duplicate.jpeg"):         "GENUINE_PAIR",
    ("Ayush Original 2 .jpeg",       "Ayush Duplicate.jpeg"):         "GENUINE_PAIR",
}


def load_feature_extractor():
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    model.classifier = torch.nn.Identity()
    model.eval()
    return model


def preprocess_ink_only(img_rgb: np.ndarray) -> np.ndarray:
    """
    Isolate ink from background using blue-channel inversion + 
    adaptive thresholding to handle shadows and uneven lighting.
    Returns a clean binary image (white ink on black).
    """
    # Step 1: Use blue channel — pen ink absorbs blue, shadows don't
    b_channel = img_rgb[:, :, 2]
    
    # Step 2: Invert (so ink = bright)
    b_inv = 255 - b_channel
    
    # Step 3: Adaptive threshold on a wider block size to handle shadows/gradients
    binary = cv2.adaptiveThreshold(
        b_inv, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=51,
        C=-10
    )
    
    # Step 4: Morphological cleanup — remove isolated noise pixels
    kernel = np.ones((2, 2), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    
    return cleaned


def get_embedding(img_rgb: np.ndarray, model) -> np.ndarray:
    binary = preprocess_ink_only(img_rgb)
    coords = cv2.findNonZero(binary)
    
    if coords is not None and len(coords) > 100:
        x, y, w, h = cv2.boundingRect(coords)
        # Small margin
        margin = 10
        y1 = max(0, y - margin)
        x1 = max(0, x - margin)
        y2 = min(img_rgb.shape[0], y + h + margin)
        x2 = min(img_rgb.shape[1], x + w + margin)
        cropped_bin = binary[y1:y2, x1:x2]
    else:
        cropped_bin = binary
    
    # Convert binary to 3-channel PIL
    img_pil = Image.fromarray(cropped_bin).convert("RGB")
    
    # Pad to square (preserve aspect ratio)
    w, h = img_pil.size
    max_dim = max(w, h)
    pl = (max_dim - w) // 2
    pt = (max_dim - h) // 2
    pr = (max_dim - w + 1) // 2
    pb = (max_dim - h + 1) // 2
    img_padded = ImageOps.expand(img_pil, (pl, pt, pr, pb), fill=(0, 0, 0))
    
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    tensor = preprocess(img_padded).unsqueeze(0)
    with torch.no_grad():
        feat = model(tensor)
    return feat.numpy().flatten()


def compute_similarity(img1, img2, model):
    e1 = get_embedding(img1, model)
    e2 = get_embedding(img2, model)
    return float(1.0 - cosine(e1, e2))


# ─── Run tests ────────────────────────────────────────────────────────────────
sig_dir = "C:\\Users\\divya\\Desktop\\Docushield\\example signatures"
files = [
    "Original.jpeg",
    "original 2.jpeg",
    "duplicate.jpeg",
    "A signature Original.jpeg",
    "A signature original 2.jpeg",
    "Ayush Original 1.jpeg",
    "Ayush Original 2 .jpeg",
    "Ayush Duplicate.jpeg"
]

model = load_feature_extractor()

results = []
for f1, f2 in itertools.combinations(files, 2):
    img1 = np.array(Image.open(os.path.join(sig_dir, f1)).convert("RGB"))
    img2 = np.array(Image.open(os.path.join(sig_dir, f2)).convert("RGB"))
    sim = compute_similarity(img1, img2, model)
    label = GROUND_TRUTH.get((f1, f2), GROUND_TRUTH.get((f2, f1), "DIFFERENT_PERSON"))
    results.append((sim, f1, f2, label))

results.sort(reverse=True, key=lambda x: x[0])

print(f"{'Similarity':<12} | {'Label':<15} | {'Pair'}")
print("-" * 90)
for sim, f1, f2, label in results:
    tag = "[OK]" if label == "GENUINE_PAIR" else "[--]"
    print(f"{sim:.4f}       | {tag} {label:<13} | {f1}  vs  {f2}")

# Diagnostics
genuine_scores  = [r[0] for r in results if r[3] == "GENUINE_PAIR"]
impostor_scores = [r[0] for r in results if r[3] == "DIFFERENT_PERSON"]

print(f"\n--- Diagnostics ---")
print(f"Genuine pairs  — min: {min(genuine_scores):.4f}  max: {max(genuine_scores):.4f}  avg: {np.mean(genuine_scores):.4f}")
print(f"Impostor pairs — min: {min(impostor_scores):.4f}  max: {max(impostor_scores):.4f}  avg: {np.mean(impostor_scores):.4f}")
print(f"Gap (avg genuine - avg impostor): {np.mean(genuine_scores) - np.mean(impostor_scores):.4f}")
