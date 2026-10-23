"""Patient-level data partitioning across simulated hospital silos.

Supports both IID (balanced) and non-IID (unbalanced) distributions across
hospitals while preserving strict patient-level isolation (no patient scans
appear in multiple silos) and holding out a dedicated global validation set
for server-side evaluation.
"""

from __future__ import annotations

import json
from pathlib import Path
import random
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from monai.data import DataLoader

from fedmed.data.datasets import discover_cases
from fedmed.data.loaders import build_loader
from fedmed.utils.config import load_config
from fedmed.utils.logging import get_logger

logger = get_logger(__name__)


class HospitalLoaders(tuple):
    """Named tuple-like container for hospital train and validation DataLoaders.

    Supports tuple unpacking:
        train_loader, val_loader = get_hospital_loaders(...)
    As well as attribute and dictionary-style access:
        loaders.train, loaders["val"]
    """

    def __new__(cls, train_loader: DataLoader, val_loader: DataLoader) -> HospitalLoaders:
        return super().__new__(cls, (train_loader, val_loader))

    @property
    def train(self) -> DataLoader:
        """Local training DataLoader for the hospital silo."""
        return self[0]

    @property
    def val(self) -> DataLoader:
        """Local validation DataLoader for the hospital silo."""
        return self[1]

    def __getitem__(self, item: Union[int, str]) -> DataLoader:
        if item in ("train", "train_loader"):
            return self[0]
        if item in ("val", "val_loader", "local_val"):
            return self[1]
        return super().__getitem__(item)

    def get(self, key: str, default: Optional[Any] = None) -> Any:
        if key in ("train", "train_loader"):
            return self[0]
        if key in ("val", "val_loader", "local_val"):
            return self[1]
        return default


def partition_patients(
    patient_ids: Sequence[str],
    num_hospitals: int = 3,
    mode: str = "iid",
    proportions: Optional[Sequence[float]] = None,
    seed: int = 42,
    global_val_ratio: float = 0.15,
    global_test_ids: Optional[Sequence[str]] = None,
    local_val_ratio: float = 0.20,
    hospital_names: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Deterministically partition patient IDs across hospital silos and global sets.

    Strict patient-level separation:
      1. If ``global_test_ids`` are provided, they are excluded from the partitioning pool.
      2. A global validation set (``global_val_ratio``) is carved out for server evaluation.
      3. The remaining patients are partitioned across hospitals in either:
         - "iid" mode: equal/balanced distribution across silos.
         - "non_iid" mode: unequal distribution according to proportions (e.g. 50% / 30% / 20%).
      4. Within each hospital silo, patients are partitioned into local train and local val sets.

    Args:
        patient_ids: Full collection of candidate patient IDs.
        num_hospitals: Number of simulated hospital nodes (default: 3).
        mode: Partitioning strategy ("iid" or "non_iid").
        proportions: Relative weights for "non_iid" (e.g. [0.5, 0.3, 0.2]).
        seed: Random seed for deterministic reproducibility.
        global_val_ratio: Proportion of patients reserved for server global validation.
        global_test_ids: Optional pre-existing test patient IDs from baseline.
        local_val_ratio: Fraction of each hospital's patients reserved for local validation.
        hospital_names: Optional custom names/keys for the hospitals (defaults to hospital_1, ...).

    Returns:
        Structured partition manifest mapping containing patient IDs only.
    """
    if num_hospitals <= 0:
        raise ValueError(f"num_hospitals must be positive, got {num_hospitals}")

    mode_normalized = mode.strip().lower()
    if mode_normalized not in ("iid", "non_iid"):
        raise ValueError(f"Invalid partitioning mode '{mode}'. Expected 'iid' or 'non_iid'.")

    # Establish names for hospital keys
    if hospital_names is not None:
        if len(hospital_names) != num_hospitals:
            raise ValueError(
                f"hospital_names length ({len(hospital_names)}) must match num_hospitals ({num_hospitals})"
            )
        hosp_keys = list(hospital_names)
    else:
        hosp_keys = [f"hospital_{i + 1}" for i in range(num_hospitals)]

    # Deterministic shuffling with fixed seed
    rng = random.Random(seed)
    unique_ids = sorted(set(patient_ids))

    # Exclude global test cases if provided
    test_set = set(global_test_ids) if global_test_ids else set()
    pool = [pid for pid in unique_ids if pid not in test_set]
    rng.shuffle(pool)

    total_pool_size = len(pool)
    if total_pool_size == 0:
        raise ValueError("Patient pool is empty after filtering out test set.")

    # 1. Hold out Global Validation Set (belongs to NO hospital)
    n_global_val = int(round(total_pool_size * global_val_ratio))
    # If pool is large enough, guarantee at least 1 global val case if ratio > 0
    if global_val_ratio > 0 and n_global_val == 0 and total_pool_size > num_hospitals:
        n_global_val = 1

    global_val_ids = sorted(pool[:n_global_val])
    train_pool = pool[n_global_val:]

    n_train_pool = len(train_pool)
    if n_train_pool < num_hospitals:
        raise ValueError(
            f"Not enough patients ({n_train_pool}) to assign at least 1 patient to each of {num_hospitals} hospitals."
        )

    # 2. Partition training pool across hospitals
    hosp_patient_buckets: List[List[str]] = [[] for _ in range(num_hospitals)]

    if mode_normalized == "iid":
        # Balanced assignment
        base_count = n_train_pool // num_hospitals
        remainder = n_train_pool % num_hospitals
        cursor = 0
        for i in range(num_hospitals):
            count = base_count + (1 if i < remainder else 0)
            hosp_patient_buckets[i] = train_pool[cursor : cursor + count]
            cursor += count
    else:
        # Non-IID assignment with unequal sizes (e.g. 50% / 30% / 20%)
        if proportions is None:
            # Default non-IID proportions for 3 hospitals: 50%, 30%, 20%
            if num_hospitals == 3:
                proportions = [0.50, 0.30, 0.20]
            else:
                # Generate decreasing geometric proportions if unspecified
                weights = [1.0 / (2**i) for i in range(num_hospitals)]
                total_w = sum(weights)
                proportions = [w / total_w for w in weights]

        if len(proportions) != num_hospitals:
            raise ValueError(
                f"Proportions length ({len(proportions)}) must match num_hospitals ({num_hospitals})"
            )

        norm_props = [float(p) / sum(proportions) for p in proportions]
        counts = [max(1, int(round(n_train_pool * p))) for p in norm_props]

        # Adjust total sum of counts to match n_train_pool exactly
        diff = n_train_pool - sum(counts)
        # Distribute remaining difference starting with largest hospital
        order = sorted(range(num_hospitals), key=lambda i: norm_props[i], reverse=True)
        idx_step = 0
        while diff != 0:
            target_idx = order[idx_step % num_hospitals]
            if diff > 0:
                counts[target_idx] += 1
                diff -= 1
            elif diff < 0 and counts[target_idx] > 1:
                counts[target_idx] -= 1
                diff += 1
            idx_step += 1

        cursor = 0
        for i, count in enumerate(counts):
            hosp_patient_buckets[i] = train_pool[cursor : cursor + count]
            cursor += count

    # 3. Create hospital structures and local train/val splits
    hospitals_dict: Dict[str, Dict[str, Any]] = {}
    manifest: Dict[str, Any] = {
        "seed": seed,
        "mode": mode_normalized,
        "num_hospitals": num_hospitals,
        "global_val": global_val_ids,
        "global_test": sorted(list(test_set)),
        "hospitals": hospitals_dict,
    }

    for i, hosp_key in enumerate(hosp_keys):
        patients = sorted(hosp_patient_buckets[i])
        n_p = len(patients)

        # Local train vs validation split within this silo
        if n_p >= 2:
            n_local_val = max(1, int(round(n_p * local_val_ratio)))
            # Ensure at least 1 local training case
            if n_local_val >= n_p:
                n_local_val = n_p - 1
        else:
            # Single patient silo: use for training, fallback to same for local val
            n_local_val = 0

        train_ids = patients[: n_p - n_local_val]
        val_ids = patients[n_p - n_local_val :] if n_local_val > 0 else list(patients)

        hosp_data = {
            "patient_ids": patients,
            "train_ids": train_ids,
            "val_ids": val_ids,
            # Aliases for compatibility
            "patients": patients,
            "train": train_ids,
            "val": val_ids,
        }
        hospitals_dict[hosp_key] = hosp_data
        # Top-level alias for convenience: manifest["hospital_1"] -> patient IDs list
        manifest[hosp_key] = patients

    return manifest


def save_partition_manifest(
    manifest: Dict[str, Any],
    manifest_path: Union[str, Path] = "partitions/partition_manifest.json",
) -> Path:
    """Save partition manifest containing patient IDs only to disk.

    HIPAA / GDPR compliance:
        Stores purely string patient identifiers, never imaging file paths or labels.

    Args:
        manifest: Partition manifest dictionary.
        manifest_path: Destination JSON filepath.

    Returns:
        Path to written file.
    """
    path = Path(manifest_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Clean serializable payload
    serializable: Dict[str, Any] = {
        "seed": manifest.get("seed", 42),
        "mode": manifest.get("mode", "iid"),
        "num_hospitals": manifest.get("num_hospitals", len(manifest.get("hospitals", {}))),
        "global_val": manifest.get("global_val", []),
        "global_test": manifest.get("global_test", []),
        "hospitals": {},
    }

    for hosp_id, data in manifest.get("hospitals", {}).items():
        if isinstance(data, dict):
            serializable["hospitals"][hosp_id] = {
                "patient_ids": list(data.get("patient_ids", data.get("patients", []))),
                "train_ids": list(data.get("train_ids", data.get("train", []))),
                "val_ids": list(data.get("val_ids", data.get("val", []))),
            }
        elif isinstance(data, (list, tuple)):
            serializable["hospitals"][hosp_id] = {
                "patient_ids": list(data),
                "train_ids": list(data),
                "val_ids": list(data),
            }

    with path.open("w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)

    logger.info("Saved partition manifest to %s", path)
    return path


def load_partition_manifest(
    manifest_path: Union[str, Path] = "partitions/partition_manifest.json",
) -> Dict[str, Any]:
    """Load partition manifest from JSON file and rebuild convenience aliases."""
    path = Path(manifest_path)
    if not path.is_file():
        raise FileNotFoundError(f"Partition manifest not found at: {path}")

    with path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    # Inject convenience aliases if missing
    for hosp_id, data in manifest.get("hospitals", {}).items():
        if isinstance(data, dict):
            data.setdefault("patients", data.get("patient_ids", []))
            data.setdefault("train", data.get("train_ids", []))
            data.setdefault("val", data.get("val_ids", []))
            manifest.setdefault(hosp_id, data.get("patient_ids", []))

    return manifest


def print_partition_table(manifest: Dict[str, Any]) -> str:
    """Format and print an informative ASCII table of the data partitions.

    Displays: Hospital ID, Total Patients, Number of Train, and Number of Val.

    Args:
        manifest: Partition manifest dictionary.

    Returns:
        Formatted table as a string.
    """
    col_hosp = "Hospital / Set"
    col_total = "Total Patients"
    col_train = "Train Scans"
    col_val = "Val Scans"

    w_hosp = 24
    w_total = 16
    w_train = 14
    w_val = 14

    border = f"+{'-' * (w_hosp + 2)}+{'-' * (w_total + 2)}+{'-' * (w_train + 2)}+{'-' * (w_val + 2)}+"
    header = f"| {col_hosp:<{w_hosp}} | {col_total:<{w_total}} | {col_train:<{w_train}} | {col_val:<{w_val}} |"

    lines = [
        border,
        header,
        border,
    ]

    hospitals = manifest.get("hospitals", {})
    total_train_scans = 0
    total_hosp_val_scans = 0
    total_hosp_patients = 0

    for hosp_id, data in hospitals.items():
        p_count = len(data.get("patient_ids", data.get("patients", [])))
        t_count = len(data.get("train_ids", data.get("train", [])))
        v_count = len(data.get("val_ids", data.get("val", [])))

        total_hosp_patients += p_count
        total_train_scans += t_count
        total_hosp_val_scans += v_count

        lines.append(
            f"| {hosp_id:<{w_hosp}} | {p_count:^{w_total}} | {t_count:^{w_train}} | {v_count:^{w_val}} |"
        )

    lines.append(border)

    # Server global hold-out sets
    g_val_count = len(manifest.get("global_val", []))
    lines.append(
        f"| {'Global Validation (Server)':<{w_hosp}} | {g_val_count:^{w_total}} | {'-':^{w_train}} | {g_val_count:^{w_val}} |"
    )

    g_test_count = len(manifest.get("global_test", []))
    if g_test_count > 0:
        lines.append(
            f"| {'Global Test (Baseline)':<{w_hosp}} | {g_test_count:^{w_total}} | {'-':^{w_train}} | {g_test_count:^{w_val}} |"
        )

    lines.append(border)
    table_str = "\n".join(lines)
    logger.info("\n%s\n", table_str)
    print(table_str)
    return table_str


def create_or_load_partitions(
    config: Union[Mapping[str, Any], str, Path],
    patient_ids: Optional[Sequence[str]] = None,
    manifest_path: Optional[Union[str, Path]] = None,
    print_table: bool = True,
) -> Dict[str, Any]:
    """Retrieve existing partition manifest or generate a fresh one matching configuration.

    Args:
        config: Loaded config dictionary or path to configuration YAML.
        patient_ids: Optional explicit patient IDs. If None, discovered via ``discover_cases(cfg)``.
        manifest_path: Destination path (defaults to config partition path or ``partitions/partition_manifest.json``).
        print_table: Whether to print the summary table to stdout.

    Returns:
        Partition manifest mapping.
    """
    cfg = load_config(config) if isinstance(config, (str, Path)) else dict(config)

    # Determine destination manifest path
    if manifest_path is None:
        manifest_path = (
            cfg.get("federated", {}).get("manifest_path")
            or cfg.get("partition", {}).get("manifest_path")
            or "partitions/partition_manifest.json"
        )
    dest_path = Path(manifest_path)

    # Re-use existing manifest if present and valid
    if dest_path.is_file():
        try:
            manifest = load_partition_manifest(dest_path)
            logger.info("Loaded existing partition manifest from %s", dest_path)
            if print_table:
                print_partition_table(manifest)
            return manifest
        except Exception as exc:
            logger.warning("Could not read existing manifest %s (%s); regenerating", dest_path, exc)

    # Discover patients if not provided
    all_discovered = None
    if patient_ids is None:
        all_discovered = discover_cases(cfg)
        patient_ids = [c["patient_id"] for c in all_discovered]

    # Check if Week 1 baseline splits already exist to reuse baseline test split
    baseline_splits_path = cfg.get("data", {}).get("splits_path")
    global_test_ids: List[str] = []
    if baseline_splits_path and Path(baseline_splits_path).is_file():
        try:
            with Path(baseline_splits_path).open("r", encoding="utf-8") as f:
                base_splits = json.load(f)
            global_test_ids = base_splits.get("test", [])
            logger.info("Reusing %s baseline test cases from %s", len(global_test_ids), baseline_splits_path)
        except Exception as exc:
            logger.warning("Failed to load baseline splits for test set reuse: %s", exc)

    # Extract configuration parameters
    mode = (
        cfg.get("federated", {}).get("mode")
        or cfg.get("partition", {}).get("mode")
        or "iid"
    )
    seed = int(cfg.get("seed", 42))

    hospitals_cfg = cfg.get("hospitals", [])
    num_hospitals = len(hospitals_cfg) if hospitals_cfg else 3
    hospital_names = [h["id"] for h in hospitals_cfg] if hospitals_cfg else None

    proportions = (
        cfg.get("federated", {}).get("proportions")
        or cfg.get("partition", {}).get("proportions")
        or ([0.50, 0.30, 0.20] if mode == "non_iid" and num_hospitals == 3 else None)
    )

    global_val_ratio = float(
        cfg.get("federated", {}).get("global_val_ratio")
        or cfg.get("partition", {}).get("global_val_ratio")
        or cfg.get("data", {}).get("val_ratio", 0.15)
    )

    local_val_ratio = float(
        cfg.get("federated", {}).get("local_val_ratio")
        or cfg.get("partition", {}).get("local_val_ratio", 0.20)
    )

    manifest = partition_patients(
        patient_ids=patient_ids,
        num_hospitals=num_hospitals,
        mode=mode,
        proportions=proportions,
        seed=seed,
        global_val_ratio=global_val_ratio,
        global_test_ids=global_test_ids,
        local_val_ratio=local_val_ratio,
        hospital_names=hospital_names,
    )

    save_partition_manifest(manifest, dest_path)
    if print_table:
        print_partition_table(manifest)

    return manifest


def get_hospital_loaders(
    hospital_id: str,
    config: Union[Mapping[str, Any], str, Path],
    manifest: Optional[Dict[str, Any]] = None,
) -> HospitalLoaders:
    """Build train and local-val DataLoaders for a specific hospital node.

    Reuses existing MONAI transforms, batch sizing, and caching mechanisms from Week 1.

    Args:
        hospital_id: Identifier of the hospital node (e.g. 'hospital_1').
        config: Validated configuration mapping or YAML path.
        manifest: Optional pre-loaded partition manifest. If None, loaded/created automatically.

    Returns:
        HospitalLoaders: Container yielding (train_loader, val_loader).
    """
    cfg = load_config(config) if isinstance(config, (str, Path)) else dict(config)

    if manifest is None:
        manifest = create_or_load_partitions(cfg, print_table=False)

    hospitals = manifest.get("hospitals", {})
    if hospital_id not in hospitals:
        raise KeyError(
            f"Hospital '{hospital_id}' not found in partition manifest. Available: {list(hospitals.keys())}"
        )

    hosp_info = hospitals[hospital_id]
    train_ids = hosp_info.get("train_ids", hosp_info.get("train", []))
    val_ids = hosp_info.get("val_ids", hosp_info.get("val", []))

    # Pre-discover cases once to avoid multiple disk scans
    cases = discover_cases(cfg)

    train_loader = build_loader(cfg, train_ids, mode="train", cases=cases)
    val_loader = build_loader(cfg, val_ids, mode="val", cases=cases)

    logger.info(
        "Built HospitalLoaders for [%s]: %s train patients, %s val patients",
        hospital_id,
        len(train_ids),
        len(val_ids),
    )
    return HospitalLoaders(train_loader, val_loader)


def get_global_eval_loaders(
    config: Union[Mapping[str, Any], str, Path],
    manifest: Optional[Dict[str, Any]] = None,
) -> Dict[str, DataLoader]:
    """Build global validation (and test) DataLoaders for server-side evaluation.

    Args:
        config: Configuration mapping or YAML path.
        manifest: Optional partition manifest.

    Returns:
        Dict mapping 'val' (and optionally 'test') to MONAI DataLoaders.
    """
    cfg = load_config(config) if isinstance(config, (str, Path)) else dict(config)
    if manifest is None:
        manifest = create_or_load_partitions(cfg, print_table=False)

    cases = discover_cases(cfg)
    loaders: Dict[str, DataLoader] = {}

    global_val_ids = manifest.get("global_val", [])
    if global_val_ids:
        loaders["val"] = build_loader(cfg, global_val_ids, mode="val", cases=cases)

    global_test_ids = manifest.get("global_test", [])
    if global_test_ids:
        loaders["test"] = build_loader(cfg, global_test_ids, mode="test", cases=cases)

    return loaders
