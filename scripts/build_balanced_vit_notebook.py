"""Build the 7,685-image ViT experiment from the clean full-data notebook."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "notebooks/vision-transformer-improved.source.ipynb"
DEST = ROOT / "notebooks/vision-transformer-balanced-7685.ipynb"


def replace_once(source: str, old: str, new: str) -> str:
    if source.count(old) != 1:
        raise ValueError(f"Expected one match for {old[:80]!r}, got {source.count(old)}")
    return source.replace(old, new)


book = json.loads(SOURCE.read_text())
for cell in book["cells"]:
    src = "".join(cell["source"])
    if src.startswith("# ChestNet: improved Vision Transformer"):
        src = """# ChestNet: pretrained ViT on the original 7,685-image subset

This experiment recreates the original rarest-first, seeded subset and its patient-level 70/15/15 split. It trains the improved ViT-B/16 on the original eight labels and evaluates on the original test patients. The original validation patients are divided into model-selection and threshold-tuning groups; this difference and the changed training procedure mean the comparison is informative but not a single-variable architecture ablation.

Run on Kaggle with the NIH Chest X-rays dataset, Internet enabled, and GPU T4 x2. All code and outputs are saved in this notebook version; the inference weights and metric files are exported separately.
"""
    elif src.startswith("## 2. Audit the full attached data"):
        src = """## 2. Recreate and audit the original balanced subset

The original script samples up to 1,500 images per class, processing rarer classes first and allowing one image to count for multiple labels. It uses seed 42 and preserves the CSV row order during sampling. We verify all selected images and record the exact subset and split identities. The test patient split is checked against the original notebook's saved counts.
"""
    elif src.startswith("## 3. Fixed patient splits"):
        src = """## 3. Recreate the original patient split

The original code tries 200 patient-group split candidates and selects the one with the closest class distribution. We reproduce that rule and require its train, validation, and test image counts to match the original run (5,400 / 1,109 / 1,176). We then split the original validation patients into model-selection and threshold-tuning groups. The original test patients remain untouched.
"""
    elif "'output_dir': '/kaggle/working/chestnet_vit_improved'" in src:
        src = replace_once(src, "'output_dir': '/kaggle/working/chestnet_vit_improved'", "'output_dir': '/kaggle/working/chestnet_vit_balanced_7685'")
        src = replace_once(src, "'cache_dir': '/kaggle/temp/chestnet_vit_cache'", "'cache_dir': '/kaggle/temp/chestnet_vit_balanced_7685_cache'")
        src = replace_once(src, "'cache_images': True,", "'cache_images': True,\n    'max_images_per_class': 1500,\n    'split_candidates': 200,\n    'min_positives_per_split': 30,")
    elif src.startswith("def load_metadata(root_override=None):"):
        src = replace_once(src, "df = df.sort_values('image').reset_index(drop=True)\n    df['cache_index'] = np.arange(len(df))", "df = df.reset_index(drop=True)\n    df['cache_index'] = np.arange(len(df))")
        src = replace_once(src, "'image_downsampling': False,", "'image_downsampling': True,")
        src = replace_once(src, "df, audit = load_metadata(CFG['data_root'])\nDATA_HASH", """df, audit = load_metadata(CFG['data_root'])
eligible = df[df[CLASSES].sum(axis=1) > 0].reset_index(drop=True)
# Match src/preprocessing/splits.py: metadata order, rarest class first, RNG seed 42.
rng = np.random.default_rng(SEED)
chosen = np.zeros(len(eligible), dtype=bool)
for c in sorted(CLASSES, key=lambda c: eligible[c].sum()):
    already = int(eligible.loc[chosen, c].sum())
    pool = np.flatnonzero((eligible[c].to_numpy() == 1) & ~chosen)
    take = max(0, min(CFG['max_images_per_class'] - already, len(pool)))
    if take:
        chosen[rng.choice(pool, size=take, replace=False)] = True
df = eligible.loc[chosen].reset_index(drop=True)
df['patient_id'] = pd.to_numeric(df.patient_id)
df['cache_index'] = np.arange(len(df))
assert len(df) == 7685, f'Original subset should have 7,685 images; got {len(df)}.'
audit.update({'subset_images': len(df), 'subset_patients': int(df.patient_id.nunique()),
              'eligible_selected_label_images': len(eligible),
              'max_images_per_class': CFG['max_images_per_class']})
df[['image','patient_id',*CLASSES]].to_csv(OUT / 'balanced_subset.csv', index=False)
DATA_HASH""")
        src = replace_once(src, "audit['verified_images'] = len(df)", "audit['verified_subset_images'] = len(df)")
    elif src.startswith("def group_holdout(frame, fraction, seed):"):
        start = src.index("development, test_df =")
        end = src.index("PARTS =", start)
        src = src[:start] + """def original_group_split(frame, val_size, test_size, seed):
    trval, test = group_holdout(frame, test_size, seed)
    train, validation = group_holdout(trval, val_size / (1.0 - test_size), seed)
    return train, validation, test

def split_score(parts, frame, sizes, min_pos):
    overall = frame[CLASSES].mean().to_numpy()
    score = 0.0
    for part, target in zip(parts, sizes):
        score += np.abs(part[CLASSES].mean().to_numpy() - overall).sum()
        score += abs(len(part) / len(frame) - target)
        score += 10.0 * int((part[CLASSES].sum().to_numpy() < min_pos).any())
    return score

rng = np.random.default_rng(SEED)
best, best_score, best_seed = None, np.inf, None
for candidate in rng.integers(0, 2**31 - 1, size=CFG['split_candidates']):
    parts = original_group_split(df, 0.15, 0.15, int(candidate))
    score = split_score(parts, df, (0.70, 0.15, 0.15), CFG['min_positives_per_split'])
    if score < best_score:
        best, best_score, best_seed = parts, score, int(candidate)
train_df, original_val_df, test_df = best
assert (len(train_df), len(original_val_df), len(test_df)) == (5400, 1109, 1176), (
    'Original split counts differ; compare subset and metadata before training.')
select_df, tune_df = group_holdout(original_val_df, 1 / 3, SEED + 2)
""" + src[end:]
        src = replace_once(src, "'overlap':overlap}", "'overlap':overlap, 'original_split_seed':best_seed,\n    'original_split_score':best_score, 'original_validation_images':len(original_val_df)}")
    elif "'Original notebook test AUC 0.7630" in src:
        src = replace_once(src, "'Original notebook test AUC 0.7630 and F1 0.4691 came from a balanced 7,685-image subset; this run uses different patients and prevalence. Those numbers cannot establish improvement.',", "'Original notebook test AUC 0.7630 and F1 0.4691 used this balanced subset and original test split. The new model-selection/threshold-tuning separation changes the validation procedure; model training also changed.',")
        src = replace_once(src, "archive=OUT.parent/'chestnet_vit_improved_bundle.zip'", "archive=OUT.parent/'chestnet_vit_balanced_7685_bundle.zip'")
        src = replace_once(src, "f'Images audited: {len(df):,}; patients: {df.patient_id.nunique():,}; PNG storage: {audit[\"image_gib\"]:.2f} GiB.'", "f'Original subset images: {len(df):,}; patients: {df.patient_id.nunique():,}; full PNG storage: {audit[\"image_gib\"]:.2f} GiB.'")
    cell["source"] = src.splitlines(keepends=True)

DEST.write_text(json.dumps(book, indent=1, ensure_ascii=False) + "\n")
print(DEST)
