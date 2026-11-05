"""Central configuration: paths, random seed, dataset schema and label taxonomy.

Everything that another module might need to agree on (column names, the
attack -> category mapping, where files live) is defined once here so the rest
of the code never hard-codes it.
"""
from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
RESULTS_DIR = PROJECT_ROOT / "results"

# Single global seed used by every stochastic component (splits, models,
# permutation importance, synthetic data) so a re-run is bit-for-bit stable.
RANDOM_STATE = 42

# --------------------------------------------------------------------------- #
# NSL-KDD download sources
# --------------------------------------------------------------------------- #
# The canonical host (University of New Brunswick, CIC) sits behind a form, so
# we use well-known public GitHub mirrors of the unmodified files and verify the
# SHA-256 of what we receive. If the hash does not match, the file is rejected.
NSL_KDD_FILES = {
    "KDDTrain+.txt": "1b86d2f957b33082081bba410fe129b475efebcc13c9014c3f447c8271aadf95",
    "KDDTest+.txt": "fa46b0935342616aa83b7c2578db355b6a7aaabbc492248172c7a1e8b7ab8f84",
}
NSL_KDD_MIRRORS = [
    "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/{name}",
    "https://raw.githubusercontent.com/jmnwong/NSL-KDD-Dataset/master/{name}",
]

# --------------------------------------------------------------------------- #
# Schema (41 connection features + label + difficulty level)
# --------------------------------------------------------------------------- #
FEATURE_COLUMNS = [
    # --- basic features of individual TCP connections ---
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent",
    # --- content features (domain knowledge, payload-derived) ---
    "hot", "num_failed_logins", "logged_in", "num_compromised", "root_shell",
    "su_attempted", "num_root", "num_file_creations", "num_shells",
    "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login",
    # --- time-based traffic features (2-second window) ---
    "count", "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate",
    "srv_rerror_rate", "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
    # --- host-based traffic features (last 100 connections to same host) ---
    "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
]
ALL_COLUMNS = FEATURE_COLUMNS + ["attack", "difficulty"]

CATEGORICAL_FEATURES = ["protocol_type", "service", "flag"]

# Heavy-tailed count/byte features. They span many orders of magnitude
# (src_bytes reaches ~1.4e9), so they are log1p-compressed before scaling;
# otherwise a handful of huge values dominate a linear model.
LOG_FEATURES = [
    "duration", "src_bytes", "dst_bytes", "wrong_fragment", "urgent", "hot",
    "num_failed_logins", "num_compromised", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "count", "srv_count",
    "dst_host_count", "dst_host_srv_count",
]

# Constant (always 0) in NSL-KDD -> carries no information; dropped.
DROP_FEATURES = ["num_outbound_cmds"]

# --------------------------------------------------------------------------- #
# Label taxonomy
# --------------------------------------------------------------------------- #
# The 39 concrete attack names in NSL-KDD are grouped into the four standard
# DARPA-1998 categories. Attacks marked (*) appear ONLY in KDDTest+ and are
# therefore "novel" to any model trained on KDDTrain+.
ATTACK_CATEGORY = {
    "normal": "normal",
    # Denial of Service: exhaust a resource so legitimate users are refused.
    "back": "DoS", "land": "DoS", "neptune": "DoS", "pod": "DoS",
    "smurf": "DoS", "teardrop": "DoS",
    "apache2": "DoS", "mailbomb": "DoS", "processtable": "DoS",  # *
    "udpstorm": "DoS", "worm": "DoS",  # *
    # Probe: surveillance / scanning to map hosts, ports and services.
    "ipsweep": "Probe", "nmap": "Probe", "portsweep": "Probe", "satan": "Probe",
    "mscan": "Probe", "saint": "Probe",  # *
    # Remote-to-Local: an outsider gains local (user) access over the network.
    "ftp_write": "R2L", "guess_passwd": "R2L", "imap": "R2L", "multihop": "R2L",
    "phf": "R2L", "spy": "R2L", "warezclient": "R2L", "warezmaster": "R2L",
    "httptunnel": "R2L", "named": "R2L", "sendmail": "R2L",  # *
    "snmpgetattack": "R2L", "snmpguess": "R2L", "xlock": "R2L",  # *
    "xsnoop": "R2L",  # *
    # User-to-Root: a local user escalates to super-user privileges.
    "buffer_overflow": "U2R", "loadmodule": "U2R", "perl": "U2R",
    "rootkit": "U2R",
    "ps": "U2R", "sqlattack": "U2R", "xterm": "U2R",  # *
}

# Fixed class order used everywhere (tables, confusion matrices, colours).
CATEGORY_ORDER = ["normal", "DoS", "Probe", "R2L", "U2R"]
BINARY_ORDER = ["normal", "attack"]
