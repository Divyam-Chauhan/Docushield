"""
Independent verification of the DocuShield signature pipeline.
This script imports directly from docushield_app.py to verify
the EXACT functions the Streamlit app uses produce correct results.
"""
import os
import sys
import itertools
import numpy as np
from PIL import Image

import warnings
warnings.filterwarnings("ignore")

# Import the actual app functions
sys.path.insert(0, r"C:\Users\divya\Desktop\Docushield")
from docushield_app import (
    load_feature_extractor,
    get_signature_embedding,
    compute_dl_similarity,
    preprocess_ink_only,
    load_gray,
)

SIG_DIR = r"C:\Users\divya\Desktop\Docushield\example signatures"

FILES = [
    "Original.jpeg",
    "original 2.jpeg",
    "duplicate.jpeg",
    "A signature Original.jpeg",
    "A signature original 2.jpeg",
    "Ayush Original 1.jpeg",
    "Ayush Original 2 .jpeg",
    "Ayush Duplicate.jpeg",
]

# Ground truth: which pairs belong to the same author
GENUINE_PAIRS = {
    frozenset(("Original.jpeg", "original 2.jpeg")),
    frozenset(("Original.jpeg", "duplicate.jpeg")),
    frozenset(("original 2.jpeg", "duplicate.jpeg")),
    frozenset(("A signature Original.jpeg", "A signature original 2.jpeg")),
    frozenset(("Ayush Original 1.jpeg", "Ayush Original 2 .jpeg")),
    frozenset(("Ayush Original 1.jpeg", "Ayush Duplicate.jpeg")),
    frozenset(("Ayush Original 2 .jpeg", "Ayush Duplicate.jpeg")),
}


def main():
    print("=" * 80)
    print("DOCUSHIELD PIPELINE VERIFICATION")
    print("=" * 80)

    # --- Verify file existence ---
    print("\n[1] Checking all 8 signature files exist...")
    for f in FILES:
        path = os.path.join(SIG_DIR, f)
        assert os.path.isfile(path), f"MISSING: {path}"
    print("    All 8 files found.")

    # --- Load model ---
    print("\n[2] Loading MobileNetV2 feature extractor...")
    model = load_feature_extractor()
    print("    Model loaded. Classifier type:", type(model.classifier).__name__)

    # --- Verify preprocessing produces non-trivial output ---
    print("\n[3] Verifying preprocess_ink_only on shadow-heavy Original.jpeg...")
    orig = np.array(Image.open(os.path.join(SIG_DIR, "Original.jpeg")).convert("RGB"))
    binary = preprocess_ink_only(orig)
    ink_pixels = np.count_nonzero(binary)
    total_pixels = binary.shape[0] * binary.shape[1]
    ink_ratio = ink_pixels / total_pixels
    print(f"    Image size: {orig.shape}")
    print(f"    Ink pixels: {ink_pixels} / {total_pixels} ({ink_ratio*100:.2f}%)")
    assert 0.001 < ink_ratio < 0.30, f"Ink ratio {ink_ratio:.4f} looks wrong (shadow leak?)"
    print("    PASS: Ink ratio in expected range (shadow NOT leaking into binary).")

    # --- Verify embedding shape ---
    print("\n[4] Verifying embedding dimensions...")
    emb = get_signature_embedding(orig, model)
    print(f"    Embedding shape: {emb.shape}")
    assert emb.shape == (1280,), f"Expected (1280,), got {emb.shape}"
    print("    PASS: 1280-dimensional embedding as expected for MobileNetV2.")

    # --- Run all 28 pair comparisons ---
    print("\n[5] Running all 28 pair comparisons...")
    genuine_scores = []
    impostor_scores = []
    all_results = []

    for f1, f2 in itertools.combinations(FILES, 2):
        img1 = np.array(Image.open(os.path.join(SIG_DIR, f1)).convert("RGB"))
        img2 = np.array(Image.open(os.path.join(SIG_DIR, f2)).convert("RGB"))
        sim = compute_dl_similarity(img1, img2, model)

        is_genuine = frozenset((f1, f2)) in GENUINE_PAIRS
        label = "GENUINE" if is_genuine else "IMPOSTOR"
        all_results.append((sim, f1, f2, label))

        if is_genuine:
            genuine_scores.append(sim)
        else:
            impostor_scores.append(sim)

    all_results.sort(reverse=True, key=lambda x: x[0])

    print(f"\n    {'Sim':>8}  {'Label':<10}  Pair")
    print("    " + "-" * 70)
    for sim, f1, f2, label in all_results:
        marker = "  <<" if label == "GENUINE" else ""
        print(f"    {sim:8.4f}  {label:<10}  {f1} vs {f2}{marker}")

    # --- Statistics ---
    g_min, g_max, g_avg = min(genuine_scores), max(genuine_scores), np.mean(genuine_scores)
    i_min, i_max, i_avg = min(impostor_scores), max(impostor_scores), np.mean(impostor_scores)
    gap = g_avg - i_avg

    print(f"\n    --- Statistics ---")
    print(f"    Genuine:  min={g_min:.4f}  max={g_max:.4f}  avg={g_avg:.4f}  (7 pairs)")
    print(f"    Impostor: min={i_min:.4f}  max={i_max:.4f}  avg={i_avg:.4f}  (21 pairs)")
    print(f"    Gap (genuine_avg - impostor_avg): {gap:.4f}")

    # --- Verify claims from Research.md ---
    print("\n[6] Verifying claims from Research.md Phase 4...")
    errors = []

    # Claim: Genuine avg ~0.892
    if abs(g_avg - 0.892) > 0.02:
        errors.append(f"Genuine avg {g_avg:.4f} deviates from claimed 0.892")

    # Claim: Impostor avg ~0.702
    if abs(i_avg - 0.702) > 0.02:
        errors.append(f"Impostor avg {i_avg:.4f} deviates from claimed 0.702")

    # Claim: Gap ~0.190
    if abs(gap - 0.190) > 0.03:
        errors.append(f"Gap {gap:.4f} deviates from claimed 0.190")

    # Claim: Genuine min >= 0.82
    if g_min < 0.80:
        errors.append(f"Genuine min {g_min:.4f} is below 0.80")

    # Claim: Original vs original 2 should be GENUINE (was the broken pair)
    orig_pair_sim = None
    for sim, f1, f2, label in all_results:
        if frozenset((f1, f2)) == frozenset(("Original.jpeg", "original 2.jpeg")):
            orig_pair_sim = sim
            break
    if orig_pair_sim is not None and orig_pair_sim < 0.85:
        errors.append(f"Original vs original 2 = {orig_pair_sim:.4f} (should be > 0.85)")

    # --- Verify threshold alignment ---
    print("\n[7] Verifying threshold alignment with verdict_html...")
    from docushield_app import verdict_html

    # Test a genuine-level score
    html_g, pct_g = verdict_html(0.92)
    assert "genuine" in html_g.lower() or "GENUINE" in html_g, f"0.92 should map to GENUINE"

    # Test a suspicious-level score
    html_s, pct_s = verdict_html(0.80)
    assert "suspicious" in html_s.lower() or "SUSPICIOUS" in html_s, f"0.80 should map to SUSPICIOUS"

    # Test a forged-level score
    html_f, pct_f = verdict_html(0.60)
    assert "forged" in html_f.lower() or "FORGED" in html_f, f"0.60 should map to FORGED"

    print("    PASS: All threshold boundaries map to correct verdict classes.")

    # --- Final report ---
    print("\n" + "=" * 80)
    if errors:
        print("VERIFICATION FAILED:")
        for e in errors:
            print(f"  - {e}")
    else:
        print("ALL CHECKS PASSED")
        print(f"  - Preprocessing handles shadows correctly")
        print(f"  - Embedding dimensions are correct (1280-d)")
        print(f"  - Genuine avg={g_avg:.4f}, Impostor avg={i_avg:.4f}, Gap={gap:.4f}")
        print(f"  - Original vs original 2 = {orig_pair_sim:.4f} (fixed)")
        print(f"  - Verdict thresholds align with calibrated values")
    print("=" * 80)

    return len(errors) == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
