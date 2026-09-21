"""
Synthetic Malware Dataset Generator
====================================
Generates a realistic 50,000-sample malware detection dataset with 30+ features
inspired by the EMBER (Endgame Malware BEnchmark for Research) dataset.

Feature Groups:
    1. PE Header Features
    2. Section Features
    3. Import Features
    4. String Features
    5. Behavioral Features
    6. Network Features

Target: label (0=benign, 1=malware), ~20% malware (class imbalance)
"""

import numpy as np
import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ─── Constants ────────────────────────────────────────────────────────────────
N_SAMPLES = 50_000
MALWARE_RATIO = 0.20
RANDOM_SEED = 42

FEATURE_NAMES = [
    # PE Header Features
    "file_size", "num_sections", "entry_point_offset", "image_base",
    "has_debug", "has_signature", "dll_characteristics",
    # Section Features
    "avg_section_entropy", "max_section_entropy", "section_size_variance",
    "num_executable_sections", "num_writable_sections",
    # Import Features
    "num_imports", "num_suspicious_imports", "num_unique_dlls",
    "uses_crypto_api", "uses_network_api", "uses_process_api", "uses_registry_api",
    # String Features
    "num_urls", "num_ips", "num_registry_keys", "num_file_paths",
    "avg_string_length", "num_printable_strings",
    # Behavioral Features
    "polymorphic_score", "obfuscation_level", "code_mutation_rate",
    "api_call_frequency", "memory_allocation_pattern",
    # Network Features
    "packet_entropy", "connection_attempts", "c2_similarity_score",
]


def generate_dataset(
    n_samples: int = N_SAMPLES,
    malware_ratio: float = MALWARE_RATIO,
    random_seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Generate a synthetic malware detection dataset with realistic feature correlations.

    Args:
        n_samples: Total number of samples to generate.
        malware_ratio: Fraction of samples that are malware.
        random_seed: RNG seed for reproducibility.

    Returns:
        pd.DataFrame with features and 'label' column.
    """
    rng = np.random.default_rng(random_seed)
    n_malware = int(n_samples * malware_ratio)
    n_benign = n_samples - n_malware

    logger.info(f"Generating {n_samples} samples ({n_benign} benign, {n_malware} malware)...")

    # ── Benign samples ──────────────────────────────────────────────────────
    benign = _generate_benign(n_benign, rng)
    benign["label"] = 0

    # ── Malware samples ─────────────────────────────────────────────────────
    malware = _generate_malware(n_malware, rng)
    malware["label"] = 1

    df = pd.concat([benign, malware], ignore_index=True)
    df = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)

    # Add realistic noise
    _add_noise(df, rng)

    # Clip to physical ranges
    df = _clip_features(df)

    logger.info(f"Dataset generated: {df.shape}, malware={df['label'].mean():.2%}")
    return df


# ─── Benign sample generator ──────────────────────────────────────────────────

def _generate_benign(n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Generate benign (clean) executable features."""
    data = {}

    # PE Header
    data["file_size"] = rng.lognormal(mean=13.0, sigma=1.8, size=n).astype(int)  # bytes
    data["num_sections"] = rng.integers(3, 8, size=n)
    data["entry_point_offset"] = rng.integers(0x1000, 0x10000, size=n)
    data["image_base"] = rng.choice([0x400000, 0x10000000, 0x140000000], size=n)
    data["has_debug"] = rng.binomial(1, 0.6, size=n)        # benign often have debug info
    data["has_signature"] = rng.binomial(1, 0.75, size=n)   # benign often signed
    data["dll_characteristics"] = rng.integers(0, 256, size=n)

    # Section features (benign: low entropy)
    data["avg_section_entropy"] = rng.beta(2, 5, size=n) * 8          # 0–8, skewed low
    data["max_section_entropy"] = data["avg_section_entropy"] + rng.beta(1, 4, size=n) * 1.5
    data["section_size_variance"] = rng.lognormal(3.0, 1.5, size=n)
    data["num_executable_sections"] = rng.integers(1, 3, size=n)
    data["num_writable_sections"] = rng.integers(0, 3, size=n)

    # Import features (benign: fewer suspicious imports)
    data["num_imports"] = rng.integers(20, 300, size=n)
    data["num_suspicious_imports"] = rng.integers(0, 5, size=n)
    data["num_unique_dlls"] = rng.integers(2, 20, size=n)
    data["uses_crypto_api"] = rng.binomial(1, 0.15, size=n)
    data["uses_network_api"] = rng.binomial(1, 0.40, size=n)
    data["uses_process_api"] = rng.binomial(1, 0.30, size=n)
    data["uses_registry_api"] = rng.binomial(1, 0.35, size=n)

    # String features
    data["num_urls"] = rng.integers(0, 10, size=n)
    data["num_ips"] = rng.integers(0, 5, size=n)
    data["num_registry_keys"] = rng.integers(0, 30, size=n)
    data["num_file_paths"] = rng.integers(1, 50, size=n)
    data["avg_string_length"] = rng.normal(15, 8, size=n).clip(3, 100)
    data["num_printable_strings"] = rng.integers(100, 5000, size=n)

    # Behavioral features (benign: low polymorphic scores)
    data["polymorphic_score"] = rng.beta(1, 8, size=n)
    data["obfuscation_level"] = rng.beta(1, 6, size=n)
    data["code_mutation_rate"] = rng.beta(1, 10, size=n)
    data["api_call_frequency"] = rng.normal(50, 30, size=n).clip(1, 500)
    data["memory_allocation_pattern"] = rng.beta(2, 5, size=n)

    # Network features (benign: low activity)
    data["packet_entropy"] = rng.normal(4.5, 1.2, size=n).clip(0, 8)
    data["connection_attempts"] = rng.integers(0, 20, size=n)
    data["c2_similarity_score"] = rng.beta(1, 10, size=n)

    return pd.DataFrame(data)


# ─── Malware sample generator ─────────────────────────────────────────────────

def _generate_malware(n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Generate malware feature distributions with polymorphic variants."""
    data = {}

    # ── Sub-type ratios: 40% polymorphic, 30% obfuscated, 30% classic ──────
    is_polymorphic = rng.binomial(1, 0.40, size=n).astype(bool)
    is_obfuscated = rng.binomial(1, 0.50, size=n).astype(bool)  # can overlap

    # PE Header (malware often unusual sizes, fewer or more sections)
    data["file_size"] = np.where(
        is_polymorphic,
        rng.lognormal(12.5, 2.2, size=n),   # polymorphic: highly variable
        rng.lognormal(13.5, 1.5, size=n),
    ).astype(int)
    data["num_sections"] = np.where(
        is_polymorphic,
        rng.integers(1, 12, size=n),         # unusual section counts
        rng.integers(2, 6, size=n),
    )
    data["entry_point_offset"] = rng.integers(0x0100, 0x200000, size=n)  # wider range
    data["image_base"] = rng.choice(
        [0x400000, 0x10000000, 0x1000000, 0x3000000], size=n
    )
    data["has_debug"] = rng.binomial(1, 0.05, size=n)        # rarely have debug info
    data["has_signature"] = rng.binomial(1, 0.08, size=n)    # rarely signed
    data["dll_characteristics"] = rng.integers(128, 512, size=n)

    # Section features (malware: HIGH entropy due to packing/encryption)
    base_entropy = np.where(
        is_obfuscated,
        rng.beta(8, 2, size=n) * 8,          # very high entropy when obfuscated
        rng.beta(5, 3, size=n) * 8,
    )
    data["avg_section_entropy"] = base_entropy.clip(0, 8)
    data["max_section_entropy"] = (base_entropy + rng.beta(3, 2, size=n) * 1.0).clip(0, 8)
    data["section_size_variance"] = rng.lognormal(5.0, 2.0, size=n)
    data["num_executable_sections"] = np.where(
        is_polymorphic,
        rng.integers(2, 8, size=n),          # multiple executable sections
        rng.integers(1, 4, size=n),
    )
    data["num_writable_sections"] = rng.integers(1, 6, size=n)

    # Import features (malware: more suspicious, crypto, network, process APIs)
    data["num_imports"] = rng.integers(5, 200, size=n)       # sometimes very few (packed)
    data["num_suspicious_imports"] = rng.integers(3, 30, size=n)
    data["num_unique_dlls"] = rng.integers(1, 25, size=n)
    data["uses_crypto_api"] = rng.binomial(1, 0.70, size=n)
    data["uses_network_api"] = rng.binomial(1, 0.80, size=n)
    data["uses_process_api"] = rng.binomial(1, 0.85, size=n)
    data["uses_registry_api"] = rng.binomial(1, 0.75, size=n)

    # String features (malware: embedded IPs, URLs, encrypted strings)
    data["num_urls"] = rng.integers(0, 50, size=n)
    data["num_ips"] = rng.integers(0, 20, size=n)
    data["num_registry_keys"] = rng.integers(5, 100, size=n)
    data["num_file_paths"] = rng.integers(0, 30, size=n)
    data["avg_string_length"] = np.where(
        is_obfuscated,
        rng.normal(5, 3, size=n).clip(1, 30),   # very short encrypted strings
        rng.normal(20, 12, size=n).clip(3, 200),
    )
    data["num_printable_strings"] = np.where(
        is_obfuscated,
        rng.integers(5, 200, size=n),            # fewer printable strings when obfuscated
        rng.integers(50, 3000, size=n),
    )

    # Behavioral features (malware: HIGH polymorphic/obfuscation scores)
    data["polymorphic_score"] = np.where(
        is_polymorphic,
        rng.beta(8, 2, size=n),                  # very high for polymorphic
        rng.beta(4, 3, size=n),
    )
    data["obfuscation_level"] = np.where(
        is_obfuscated,
        rng.beta(7, 2, size=n),
        rng.beta(3, 4, size=n),
    )
    data["code_mutation_rate"] = np.where(
        is_polymorphic,
        rng.beta(6, 2, size=n),                  # high mutation for polymorphic
        rng.beta(2, 5, size=n),
    )
    data["api_call_frequency"] = rng.normal(180, 80, size=n).clip(1, 1000)
    data["memory_allocation_pattern"] = rng.beta(6, 3, size=n)

    # Network features (malware: suspicious beaconing, high entropy traffic)
    data["packet_entropy"] = rng.normal(7.0, 0.8, size=n).clip(0, 8)   # near-max entropy
    data["connection_attempts"] = rng.integers(5, 500, size=n)
    data["c2_similarity_score"] = rng.beta(6, 3, size=n)                # high C2 similarity

    return pd.DataFrame(data)


# ─── Noise & clipping ─────────────────────────────────────────────────────────

def _add_noise(df: pd.DataFrame, rng: np.random.Generator, noise_fraction: float = 0.02) -> None:
    """Add realistic measurement noise and label flipping (sensor error simulation)."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.drop("label")
    for col in numeric_cols:
        noise = rng.normal(0, df[col].std() * 0.01, size=len(df))
        df[col] = df[col] + noise

    # Flip ~1% labels to simulate mis-labeled samples
    n_flip = int(len(df) * 0.01)
    flip_idx = rng.choice(len(df), size=n_flip, replace=False)
    df.loc[flip_idx, "label"] = 1 - df.loc[flip_idx, "label"]


def _clip_features(df: pd.DataFrame) -> pd.DataFrame:
    """Clip features to physically valid ranges."""
    clips = {
        "avg_section_entropy": (0, 8),
        "max_section_entropy": (0, 8),
        "packet_entropy": (0, 8),
        "polymorphic_score": (0, 1),
        "obfuscation_level": (0, 1),
        "code_mutation_rate": (0, 1),
        "memory_allocation_pattern": (0, 1),
        "c2_similarity_score": (0, 1),
        "api_call_frequency": (1, 2000),
        "avg_string_length": (1, 500),
        "file_size": (1_024, 500_000_000),
        "num_sections": (1, 20),
        "num_imports": (0, 1000),
        "num_suspicious_imports": (0, 100),
        "connection_attempts": (0, 10_000),
    }
    for col, (lo, hi) in clips.items():
        if col in df.columns:
            df[col] = df[col].clip(lo, hi)

    # Ensure integer columns stay integer
    int_cols = [
        "file_size", "num_sections", "entry_point_offset", "image_base",
        "has_debug", "has_signature", "dll_characteristics", "section_size_variance",
        "num_executable_sections", "num_writable_sections", "num_imports",
        "num_suspicious_imports", "num_unique_dlls", "uses_crypto_api",
        "uses_network_api", "uses_process_api", "uses_registry_api",
        "num_urls", "num_ips", "num_registry_keys", "num_file_paths",
        "num_printable_strings", "connection_attempts", "label",
    ]
    for col in int_cols:
        if col in df.columns:
            df[col] = df[col].round().astype(int)

    return df


# ─── Sample preset generators ─────────────────────────────────────────────────

def generate_benign_sample() -> dict:
    """Return a single benign sample dict."""
    rng = np.random.default_rng(100)
    df = _generate_benign(1, rng)
    return df.iloc[0].to_dict()


def generate_malware_sample() -> dict:
    """Return a single classic malware sample dict."""
    rng = np.random.default_rng(200)
    df = _generate_malware(1, rng)
    return df.iloc[0].to_dict()


def generate_polymorphic_sample() -> dict:
    """Return a single polymorphic malware sample dict — hardest to detect."""
    rng = np.random.default_rng(300)
    n = 1
    # Polymorphic characteristics
    sample = {
        "file_size": int(rng.lognormal(12.0, 2.5)),
        "num_sections": int(rng.integers(6, 12)),
        "entry_point_offset": int(rng.integers(0x50000, 0x200000)),
        "image_base": 0x1000000,
        "has_debug": 0,
        "has_signature": 0,
        "dll_characteristics": int(rng.integers(300, 512)),
        "avg_section_entropy": float(rng.beta(9, 1.5) * 8),
        "max_section_entropy": 7.95,
        "section_size_variance": float(rng.lognormal(6.0, 1.5)),
        "num_executable_sections": int(rng.integers(4, 8)),
        "num_writable_sections": int(rng.integers(3, 6)),
        "num_imports": int(rng.integers(5, 30)),       # packed → few imports visible
        "num_suspicious_imports": int(rng.integers(15, 30)),
        "num_unique_dlls": int(rng.integers(2, 8)),
        "uses_crypto_api": 1,
        "uses_network_api": 1,
        "uses_process_api": 1,
        "uses_registry_api": 1,
        "num_urls": int(rng.integers(10, 50)),
        "num_ips": int(rng.integers(5, 20)),
        "num_registry_keys": int(rng.integers(30, 100)),
        "num_file_paths": int(rng.integers(0, 5)),
        "avg_string_length": float(np.clip(rng.normal(4, 2), 1, 10)),
        "num_printable_strings": int(rng.integers(5, 50)),
        "polymorphic_score": float(rng.beta(9, 1.2)),
        "obfuscation_level": float(rng.beta(8, 1.5)),
        "code_mutation_rate": float(rng.beta(8, 1.5)),
        "api_call_frequency": float(np.clip(rng.normal(350, 50), 200, 600)),
        "memory_allocation_pattern": float(rng.beta(8, 2)),
        "packet_entropy": float(np.clip(rng.normal(7.8, 0.15), 7, 8)),
        "connection_attempts": int(rng.integers(100, 500)),
        "c2_similarity_score": float(rng.beta(8, 1.5)),
    }
    return sample


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    output_dir = Path(__file__).parent
    sample_dir = output_dir / "sample_inputs"
    sample_dir.mkdir(exist_ok=True)

    # Generate main dataset
    df = generate_dataset()
    out_path = output_dir / "synthetic_malware_data.csv"
    df.to_csv(out_path, index=False)
    logger.info(f"Saved dataset to {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")

    # Generate sample presets
    import csv

    benign = generate_benign_sample()
    malware = generate_malware_sample()
    polymorphic = generate_polymorphic_sample()

    fieldnames = list(FEATURE_NAMES)

    for name, sample in [("benign_sample", benign), ("malware_sample", malware),
                          ("polymorphic_sample", polymorphic)]:
        path = sample_dir / f"{name}.csv"
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow({k: sample.get(k, 0) for k in fieldnames})
        logger.info(f"Saved {path}")

    # Print summary statistics
    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)
    print(f"Total samples   : {len(df):,}")
    print(f"Features        : {len(FEATURE_NAMES)}")
    print(f"Benign          : {(df['label']==0).sum():,} ({(df['label']==0).mean():.1%})")
    print(f"Malware         : {(df['label']==1).sum():,} ({(df['label']==1).mean():.1%})")
    print(f"\nTop correlations with label:")
    corr = df.corr()["label"].drop("label").abs().sort_values(ascending=False)
    print(corr.head(10).to_string())


if __name__ == "__main__":
    main()
