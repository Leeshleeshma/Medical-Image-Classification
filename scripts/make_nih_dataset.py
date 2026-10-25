"""Build the NIH ChestX-ray14 external test set for Group 5 (ICSI 535), in the SAME format as
Ariana's data/processed_224_pad_gray (notebook 04 / xray_data.py).

Run from the project root (the folder with data/ and notebooks/):
    python scripts/make_nih_dataset.py --nih-dir /path/to/NIH

Output: data/processed_224_pad_gray_nih/test.npz and test.csv (+ reports/nih_summary.csv)
Label rule: PNEUMONIA = "Pneumonia" is among the findings.
NORMAL = the only finding is "No Finding", and the patient has NO pneumonia image anywhere.
Default: at most 1 image per patient, and as many NORMAL as PNEUMONIA images.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nih-dir", required=True)
    ap.add_argument("--project-dir", default=".")
    ap.add_argument("--neg-per-pos", type=float, default=1.0, help="NORMAL images per PNEUMONIA image")
    ap.add_argument("--max-per-patient", type=int, default=1, help="0 = no limit")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    project = Path(args.project_dir).resolve()
    nih = Path(args.nih_dir).resolve()
    for p in [project / "notebooks", project, Path.cwd()]:
        sys.path.insert(0, str(p))
    from xray_data import CLASS_NAMES, IMAGE_SIZE, load_image, load_npz  # shared code, unchanged

    out_dir = project / "data" / "processed_224_pad_gray_nih"
    (out_dir / "reports").mkdir(parents=True, exist_ok=True)

    # 1. Labels
    csvs = sorted(nih.rglob("Data_Entry_2017*.csv")) + sorted(nih.rglob("sample_labels.csv"))
    if not csvs:
        sys.exit(f"No Data_Entry_2017*.csv or sample_labels.csv found under {nih}")
    meta = pd.read_csv(csvs[0])
    print("labels file:", csvs[0], "|", len(meta), "rows,", meta["Patient ID"].nunique(), "patients")

    # 2. Which images do we actually have?
    on_disk = {p.name: p for p in nih.rglob("*.png")}
    meta["has_image"] = meta["Image Index"].isin(on_disk)
    print("images found on disk:", int(meta["has_image"].sum()))

    findings = meta["Finding Labels"].str.split("|")
    meta["is_pneu"] = findings.map(lambda f: "Pneumonia" in f)
    meta["is_normal"] = findings.map(lambda f: f == ["No Finding"])
    pneu_patients = set(meta.loc[meta["is_pneu"], "Patient ID"])
    meta["age"] = pd.to_numeric(meta["Patient Age"].astype(str).str.extract(r"(\d+)")[0], errors="coerce")

    def one_per_patient(df, shuffle_seed=None):
        if shuffle_seed is not None:
            df = df.sample(frac=1, random_state=shuffle_seed)
        else:
            df = df.sort_values(["Patient ID", "Follow-up #"])
        if args.max_per_patient > 0:
            df = df.groupby("Patient ID", sort=False).head(args.max_per_patient)
        return df

    pos = one_per_patient(meta[meta.has_image & meta.is_pneu])
    neg_pool = one_per_patient(
        meta[meta.has_image & meta.is_normal & ~meta["Patient ID"].isin(pneu_patients)], args.seed)
    n_neg = min(len(neg_pool), int(round(len(pos) * args.neg_per_pos)))
    if n_neg < len(pos) * args.neg_per_pos:
        print(f"WARNING: only {len(neg_pool)} NORMAL images available for {len(pos)} PNEUMONIA")
    neg = neg_pool.sample(n=n_neg, random_state=args.seed)
    if len(pos) == 0:
        sys.exit("No pneumonia images found in the downloaded folder (the sample set has very few).")

    # 3. Build the CSV
    rows = []
    for label, d in [(0, neg), (1, pos)]:
        for r in d.itertuples(index=False):
            path = on_disk[r[meta.columns.get_loc("Image Index")]]
            with Image.open(path) as im:
                w, h = im.size
            rows.append({
                "path": path.relative_to(nih).as_posix(), "label": label,
                "class_name": CLASS_NAMES[label], "patient_id": f"NIH-{r[meta.columns.get_loc('Patient ID')]}",
                "split": "test", "original_split": "nih", "width": w, "height": h,
                "finding_labels": r[meta.columns.get_loc("Finding Labels")],
                "patient_age": r[meta.columns.get_loc("age")],
                "patient_gender": r[meta.columns.get_loc("Patient Gender")],
                "view_position": r[meta.columns.get_loc("View Position")],
            })
    df = pd.DataFrame(rows).sort_values("path").reset_index(drop=True)
    df["npz_index"] = np.arange(len(df))
    cols = ["path", "label", "class_name", "patient_id", "split", "original_split", "npz_index",
            "width", "height", "finding_labels", "patient_age", "patient_gender", "view_position"]
    df = df[cols]
    df.to_csv(out_dir / "test.csv", index=False)

    # 4. Images: the SAME pad_and_resize as the Kaggle data
    images = np.empty((len(df), IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    for i, rel in enumerate(df["path"]):
        images[i] = load_image(nih / rel)
        if i % 500 == 0:
            print(f"  {i}/{len(df)}")
    np.savez_compressed(out_dir / "test.npz", images=images, labels=df["label"].to_numpy(np.int64))

    # 5. Checks and summary
    assert df["patient_id"].is_unique or args.max_per_patient != 1, "patient repeated"
    raw_images, raw_labels = load_npz("test", out_dir)
    assert raw_images.shape == (len(df), IMAGE_SIZE, IMAGE_SIZE) and raw_images.dtype == np.uint8
    assert np.array_equal(raw_labels, df["label"].to_numpy()) and raw_labels.dtype == np.int64
    assert not set(df.loc[df.label == 0, "patient_id"]) & set(df.loc[df.label == 1, "patient_id"])
    summary = pd.DataFrame([{
        "split": "nih_test", "images": len(df), "patients": df["patient_id"].nunique(),
        "NORMAL": int((df.label == 0).sum()), "PNEUMONIA": int((df.label == 1).sum()),
        "% NORMAL": round(100 * (df.label == 0).mean(), 1),
        "median_age": df["patient_age"].median(), "pct_PA_view": round(100 * (df.view_position == "PA").mean(), 1),
    }])
    summary.to_csv(out_dir / "reports" / "nih_summary.csv", index=False)
    print("\n", summary.to_string(index=False))
    print("\nSaved to", out_dir, "| images", images.shape, images.dtype)
    print("SLIDE: NIH ChestX-ray14 external test set:",
          f"{len(df)} adult X-rays (median age {df['patient_age'].median():.0f}), "
          f"{int((df.label == 1).sum())} pneumonia / {int((df.label == 0).sum())} normal, 1 image per patient, "
          "same padding + 224x224 preprocessing as the Kaggle data.")


if __name__ == "__main__":
    main()
