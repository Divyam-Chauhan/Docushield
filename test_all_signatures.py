import os
import itertools
from docushield_app import load_gray, load_feature_extractor, compute_dl_similarity

import warnings
warnings.filterwarnings("ignore")

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

# Test all combinations
for f1, f2 in itertools.combinations(files, 2):
    p1 = os.path.join(sig_dir, f1)
    p2 = os.path.join(sig_dir, f2)
    
    img1 = load_gray(p1)
    img2 = load_gray(p2)
    
    sim = compute_dl_similarity(img1, img2, model)
    results.append((sim, f1, f2))

results.sort(reverse=True, key=lambda x: x[0])

print(f"{'Similarity':<12} | {'Signature 1':<30} | {'Signature 2':<30}")
print("-" * 75)
for sim, f1, f2 in results:
    print(f"{sim:.4f}       | {f1:<30} | {f2:<30}")
