"""Dataset acquisition and loading.

Primary path : download the real NSL-KDD files (KDDTrain+ / KDDTest+) from a
               public mirror, verify their SHA-256, cache them in data/raw/.
Fallback path: if no mirror is reachable (offline machine, blocked network) or
               the user passes --synthetic, generate a synthetic dataset with
               the same schema (see synthetic.py).

Whichever path is taken is recorded in results/run_info.json so that every
reported number can be traced back to its data source.
"""
from __future__ import annotations

import hashlib
import logging
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from . import config

log = logging.getLogger(__name__)


@dataclass
class Dataset:
    """Container for the train/test split plus provenance metadata."""

    train: pd.DataFrame
    test: pd.DataFrame
    source: str  # "nsl-kdd" or "synthetic"


# --------------------------------------------------------------------------- #
# Download helpers
# --------------------------------------------------------------------------- #
def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download_one(name: str, expected_sha: str, dest_dir: Path, timeout: int = 60) -> Path:
    """Fetch one file from the first mirror whose content matches the hash."""
    dest = dest_dir / name
    if dest.exists() and _sha256(dest) == expected_sha:
        log.info("Using cached %s", dest)
        return dest

    dest_dir.mkdir(parents=True, exist_ok=True)
    last_err: Exception | None = None
    for template in config.NSL_KDD_MIRRORS:
        url = template.format(name=name)
        tmp = dest.with_suffix(dest.suffix + ".part")
        try:
            log.info("Downloading %s", url)
            with urllib.request.urlopen(url, timeout=timeout) as resp, open(tmp, "wb") as out:
                out.write(resp.read())
            got = _sha256(tmp)
            if got != expected_sha:
                raise ValueError(f"checksum mismatch for {url}: {got}")
            tmp.replace(dest)
            return dest
        except Exception as err:  # network error, HTTP error, bad hash ...
            last_err = err
            log.warning("Mirror failed (%s): %s", url, err)
            tmp.unlink(missing_ok=True)
    raise RuntimeError(f"Could not obtain {name} from any mirror") from last_err


def download_nsl_kdd(dest_dir: Path = config.RAW_DIR) -> dict[str, Path]:
    """Download (or reuse cached) NSL-KDD train and test files."""
    return {name: _download_one(name, sha, dest_dir) for name, sha in config.NSL_KDD_FILES.items()}


# --------------------------------------------------------------------------- #
# Loading / labelling
# --------------------------------------------------------------------------- #
def read_kdd_file(path: Path) -> pd.DataFrame:
    """Read a raw NSL-KDD text file (CSV without header)."""
    df = pd.read_csv(path, header=None, names=config.ALL_COLUMNS)
    return add_labels(df)


def add_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Derive the two prediction targets from the raw attack name.

    * ``category`` - 5-way label: normal / DoS / Probe / R2L / U2R
    * ``is_attack`` - binary label: 0 = normal, 1 = any attack
    """
    df = df.copy()
    df["attack"] = df["attack"].str.strip().str.rstrip(".")
    unknown = set(df["attack"]) - set(config.ATTACK_CATEGORY)
    if unknown:
        raise ValueError(f"Unmapped attack labels: {sorted(unknown)}")
    df["category"] = df["attack"].map(config.ATTACK_CATEGORY)
    df["is_attack"] = (df["category"] != "normal").astype(int)
    return df


def load_dataset(force_synthetic: bool = False) -> Dataset:
    """Return the experiment dataset, falling back to synthetic data if needed."""
    if not force_synthetic:
        try:
            paths = download_nsl_kdd()
            train = read_kdd_file(paths["KDDTrain+.txt"])
            test = read_kdd_file(paths["KDDTest+.txt"])
            log.info("Loaded NSL-KDD: train=%d test=%d", len(train), len(test))
            return Dataset(train=train, test=test, source="nsl-kdd")
        except Exception as err:
            log.warning("NSL-KDD unavailable (%s); falling back to synthetic data.", err)

    from .synthetic import generate_synthetic_dataset

    train, test = generate_synthetic_dataset(random_state=config.RANDOM_STATE)
    return Dataset(train=add_labels(train), test=add_labels(test), source="synthetic")
