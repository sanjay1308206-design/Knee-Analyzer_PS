"""
D:\knee-ai-analyzer\backend\generate_dataset.py
================================================
High-Precision 30,000-sample Clinical Morphometry Dataset Generator.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import truncnorm
from sklearn.model_selection import StratifiedShuffleSplit

OUT_DIR = Path("D:/knee-ai-analyzer/data")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = OUT_DIR / "synthetic_knee_dataset_30k.csv"


def generate_synthetic_knee_dataset(n_samples: int = 30000, seed: int = 42) -> pd.DataFrame:
    print(f"Generating {n_samples} high-precision clinical knee morphometry samples...")
    np.random.seed(seed)

    patient_ids = [f"P{i:05d}" for i in range(1, n_samples + 1)]
    gender_probs = [0.51, 0.47, 0.02]
    gender = np.random.choice([0, 1, 2], size=n_samples, p=gender_probs)

    age_low, age_high, age_mean, age_sd = 35.0, 88.0, 61.2, 10.5
    a_clip = (age_low - age_mean) / age_sd
    b_clip = (age_high - age_mean) / age_sd
    age_vals = np.round(truncnorm.rvs(a_clip, b_clip, loc=age_mean, scale=age_sd, size=n_samples), 1)

    is_age_missing = np.random.binomial(n=1, p=0.04, size=n_samples)
    age_series = np.where(is_age_missing == 1, np.nan, age_vals)

    z_global = np.random.normal(0.0, 1.0, size=n_samples)
    gender_offset = np.where(gender == 0, -0.85, np.where(gender == 1, 0.85, 0.0))
    latent_size = 0.75 * gender_offset + 0.65 * z_global

    def make_correlated_feature(mean_f, sd_f, mean_m, sd_m, min_f, max_f, min_m, max_m, weight_latent=0.88):
        noise = np.random.normal(0.0, 1.0, size=n_samples)
        z = weight_latent * latent_size + np.sqrt(1.0 - weight_latent**2) * noise
        val_f = np.clip(mean_f + z * sd_f, min_f, max_f)
        val_m = np.clip(mean_m + z * sd_m, min_m, max_m)
        val_u = np.clip((val_f + val_m) / 2.0, min_f, max_m)
        res = np.where(gender == 0, val_f, np.where(gender == 1, val_m, val_u))
        return np.round(res, 2)

    p1 = make_correlated_feature(3.70, 0.22, 4.35, 0.24, 3.20, 4.20, 3.80, 4.90, weight_latent=0.85)
    p2 = make_correlated_feature(4.10, 0.22, 4.80, 0.26, 3.60, 4.60, 4.20, 5.40, weight_latent=0.90)
    p3 = make_correlated_feature(3.55, 0.20, 4.10, 0.22, 3.10, 4.00, 3.60, 4.60, weight_latent=0.85)

    femur_ap = make_correlated_feature(51.0, 1.30, 58.5, 1.50, 48.0, 54.0, 55.0, 62.0, weight_latent=0.95)
    femur_ml = make_correlated_feature(65.5, 1.50, 76.0, 1.70, 62.0, 69.0, 72.0, 80.0, weight_latent=0.95)

    tibia_ap = make_correlated_feature(43.0, 1.30, 50.0, 1.30, 40.0, 46.0, 47.0, 53.0, weight_latent=0.94)
    tibia_ml = make_correlated_feature(67.5, 1.50, 77.0, 1.70, 64.0, 71.0, 73.0, 81.0, weight_latent=0.94)

    # Deterministic surgical sizing metric (without artificial boundary noise)
    size_metric = (
        0.40 * ((femur_ap - 54.75) / 4.0) +
        0.35 * ((femur_ml - 70.75) / 5.5) +
        0.15 * ((tibia_ap - 46.50) / 3.8) +
        0.10 * ((tibia_ml - 72.25) / 5.0)
    )

    thresholds = [-1.50, -1.00, -0.50, 0.00, 0.50, 1.00, 1.50]
    implant_size = np.digitize(size_metric, bins=thresholds) + 1

    df = pd.DataFrame({
        "patient_id": patient_ids,
        "age": age_series,
        "is_age_missing": is_age_missing,
        "gender": gender,
        "outer_meniscus_p1_mm": p1,
        "middle_meniscus_p2_mm": p2,
        "inner_meniscus_p3_mm": p3,
        "femur_ap_mm": femur_ap,
        "femur_ml_mm": femur_ml,
        "tibia_ap_mm": tibia_ap,
        "tibia_ml_mm": tibia_ml,
        "implant_size": implant_size,
    })

    df["strata"] = df["gender"].astype(str) + "_" + df["implant_size"].astype(str)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.10, random_state=seed)
    train_idx, test_idx = next(sss.split(df, df["strata"]))
    df["dataset_split"] = "train"
    df.loc[test_idx, "dataset_split"] = "test"
    df = df.drop(columns=["strata"])

    expected_cols = [
        "patient_id", "age", "is_age_missing", "gender",
        "outer_meniscus_p1_mm", "middle_meniscus_p2_mm", "inner_meniscus_p3_mm",
        "femur_ap_mm", "femur_ml_mm", "tibia_ap_mm", "tibia_ml_mm",
        "implant_size", "dataset_split",
    ]
    return df[expected_cols]


if __name__ == "__main__":
    df = generate_synthetic_knee_dataset(n_samples=30000, seed=42)
    df.to_csv(OUT_FILE, index=False)
    print(f"\n[SUCCESS] Updated dataset saved at:\n -> {OUT_FILE.resolve()}")