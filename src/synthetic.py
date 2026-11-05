"""Synthetic NSL-KDD-like data generator (fallback only).

Used when the real NSL-KDD files cannot be downloaded (or when the pipeline is
run with ``--synthetic``, e.g. for the offline smoke test). The generator
produces the *same 41-column schema* as NSL-KDD so the rest of the pipeline is
unchanged.

Each traffic type is described by a small "profile" that encodes the
qualitative signature documented for that attack in the DARPA-98 / KDD-99
literature, e.g.

* neptune (SYN flood)  : many half-open TCP connections, flag S0, serror_rate~1
* smurf (ICMP flood)   : icmp ecr_i, ~1 kB echo replies, count saturates at 511
* portsweep / satan    : many REJ/RSTO connections across many services
* guess_passwd         : telnet/ftp logins with num_failed_logins > 0
* buffer_overflow      : long interactive session, root_shell = 1

Per-feature noise and deliberately overlapping profiles keep the task from
being trivially separable, and the *test* split contains attack types that
never appear in training (mimicking the novel attacks in KDDTest+). Numbers
obtained on synthetic data are NOT comparable with published NSL-KDD results;
this path exists only so the project runs end-to-end without network access.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

# Services used by benign traffic and their relative frequencies.
_NORMAL_SERVICES = ["http", "smtp", "ftp_data", "domain_u", "private", "ftp",
                    "telnet", "ecr_i", "urp_i", "other", "pop_3", "finger"]
_NORMAL_SERVICE_P = [0.45, 0.12, 0.10, 0.09, 0.05, 0.04, 0.02, 0.03, 0.03, 0.04, 0.02, 0.01]
_SCAN_SERVICES = ["private", "other", "http", "telnet", "ftp", "smtp", "finger",
                  "domain", "imap4", "sunrpc", "eco_i", "ecr_i", "auth", "link"]


def _lognormal(rng, mean, sigma, n):
    return np.round(rng.lognormal(mean, sigma, n)).astype(np.int64)


def _rate(rng, center, spread, n):
    """A rate feature in [0, 1] concentrated around ``center``."""
    return np.clip(rng.normal(center, spread, n), 0.0, 1.0).round(2)


def _base(n: int, rng) -> dict:
    """Default feature values shared by all profiles (benign-looking)."""
    return {
        "duration": np.zeros(n, dtype=np.int64),
        "protocol_type": np.full(n, "tcp", dtype=object),
        "service": rng.choice(_NORMAL_SERVICES, n, p=_NORMAL_SERVICE_P),
        "flag": np.full(n, "SF", dtype=object),
        "src_bytes": _lognormal(rng, 5.5, 1.2, n),
        "dst_bytes": _lognormal(rng, 7.0, 1.8, n),
        "land": np.zeros(n, dtype=np.int64),
        "wrong_fragment": np.zeros(n, dtype=np.int64),
        "urgent": np.zeros(n, dtype=np.int64),
        "hot": rng.binomial(1, 0.05, n),
        "num_failed_logins": np.zeros(n, dtype=np.int64),
        "logged_in": rng.binomial(1, 0.7, n),
        "num_compromised": np.zeros(n, dtype=np.int64),
        "root_shell": np.zeros(n, dtype=np.int64),
        "su_attempted": np.zeros(n, dtype=np.int64),
        "num_root": np.zeros(n, dtype=np.int64),
        "num_file_creations": rng.binomial(1, 0.02, n),
        "num_shells": np.zeros(n, dtype=np.int64),
        "num_access_files": rng.binomial(1, 0.01, n),
        "num_outbound_cmds": np.zeros(n, dtype=np.int64),
        "is_host_login": np.zeros(n, dtype=np.int64),
        "is_guest_login": np.zeros(n, dtype=np.int64),
        "count": rng.integers(1, 20, n),
        "srv_count": rng.integers(1, 30, n),
        "serror_rate": _rate(rng, 0.0, 0.05, n),
        "srv_serror_rate": _rate(rng, 0.0, 0.05, n),
        "rerror_rate": _rate(rng, 0.0, 0.05, n),
        "srv_rerror_rate": _rate(rng, 0.0, 0.05, n),
        "same_srv_rate": _rate(rng, 0.95, 0.1, n),
        "diff_srv_rate": _rate(rng, 0.03, 0.05, n),
        "srv_diff_host_rate": _rate(rng, 0.1, 0.15, n),
        "dst_host_count": rng.integers(1, 256, n),
        "dst_host_srv_count": rng.integers(1, 256, n),
        "dst_host_same_srv_rate": _rate(rng, 0.8, 0.25, n),
        "dst_host_diff_srv_rate": _rate(rng, 0.03, 0.05, n),
        "dst_host_same_src_port_rate": _rate(rng, 0.1, 0.2, n),
        "dst_host_srv_diff_host_rate": _rate(rng, 0.03, 0.05, n),
        "dst_host_serror_rate": _rate(rng, 0.01, 0.05, n),
        "dst_host_srv_serror_rate": _rate(rng, 0.01, 0.05, n),
        "dst_host_rerror_rate": _rate(rng, 0.02, 0.08, n),
        "dst_host_srv_rerror_rate": _rate(rng, 0.02, 0.08, n),
    }


def _profile(name: str, n: int, rng) -> pd.DataFrame:
    """Generate ``n`` records of traffic type ``name``."""
    f = _base(n, rng)
    if name == "normal":
        udp = rng.random(n) < 0.12
        icmp = rng.random(n) < 0.05
        f["protocol_type"] = np.where(udp, "udp", np.where(icmp, "icmp", "tcp"))
        f["duration"] = np.where(rng.random(n) < 0.1, _lognormal(rng, 4, 2, n), 0)
    elif name in ("neptune", "processtable"):  # SYN flood / process exhaustion
        f["service"] = rng.choice(_SCAN_SERVICES, n)
        f["flag"] = rng.choice(["S0", "S0", "S0", "REJ"], n)
        f["src_bytes"] = np.zeros(n, dtype=np.int64)
        f["dst_bytes"] = np.zeros(n, dtype=np.int64)
        f["logged_in"] = np.zeros(n, dtype=np.int64)
        f["count"] = rng.integers(80, 512, n)
        f["srv_count"] = rng.integers(1, 30, n)
        for c in ("serror_rate", "srv_serror_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate"):
            f[c] = _rate(rng, 0.95, 0.08, n)
        f["same_srv_rate"] = _rate(rng, 0.08, 0.06, n)
        f["diff_srv_rate"] = _rate(rng, 0.07, 0.04, n)
        f["dst_host_count"] = np.full(n, 255)
        f["dst_host_srv_count"] = rng.integers(1, 30, n)
        f["dst_host_same_srv_rate"] = _rate(rng, 0.05, 0.05, n)
        if name == "processtable":  # long-lived connections that hold resources
            f["flag"] = np.full(n, "SF", dtype=object)
            f["duration"] = rng.integers(500, 3000, n)
            f["count"] = rng.integers(1, 10, n)
            for c in ("serror_rate", "srv_serror_rate", "dst_host_serror_rate"):
                f[c] = _rate(rng, 0.1, 0.1, n)
    elif name == "smurf":  # ICMP echo-reply flood
        f["protocol_type"] = np.full(n, "icmp", dtype=object)
        f["service"] = np.full(n, "ecr_i", dtype=object)
        f["src_bytes"] = rng.choice([520, 1032], n)
        f["dst_bytes"] = np.zeros(n, dtype=np.int64)
        f["logged_in"] = np.zeros(n, dtype=np.int64)
        f["count"] = rng.integers(300, 512, n)
        f["srv_count"] = f["count"]
        f["dst_host_count"] = np.full(n, 255)
        f["dst_host_srv_count"] = np.full(n, 255)
        f["dst_host_same_srv_rate"] = np.ones(n)
        f["dst_host_same_src_port_rate"] = _rate(rng, 0.9, 0.1, n)
    elif name in ("back", "apache2"):  # malformed / oversized HTTP requests
        f["service"] = np.full(n, "http", dtype=object)
        f["src_bytes"] = _lognormal(rng, 10.9, 0.15, n) if name == "back" else _lognormal(rng, 6.5, 0.6, n)
        f["dst_bytes"] = _lognormal(rng, 9.0, 0.4, n)
        f["hot"] = rng.integers(1, 3, n)
        f["logged_in"] = np.ones(n, dtype=np.int64)
        if name == "apache2":
            f["flag"] = rng.choice(["SF", "RSTR", "S3"], n)
            f["count"] = rng.integers(10, 80, n)
    elif name in ("teardrop", "pod"):  # fragmentation attacks
        f["protocol_type"] = np.full(n, "udp" if name == "teardrop" else "icmp", dtype=object)
        f["service"] = np.full(n, "private" if name == "teardrop" else "ecr_i", dtype=object)
        f["wrong_fragment"] = np.full(n, 3 if name == "teardrop" else 1)
        f["src_bytes"] = np.full(n, 28 if name == "teardrop" else 1480)
        f["dst_bytes"] = np.zeros(n, dtype=np.int64)
        f["logged_in"] = np.zeros(n, dtype=np.int64)
    elif name == "mailbomb":  # many SMTP messages to one host
        f["service"] = np.full(n, "smtp", dtype=object)
        f["src_bytes"] = _lognormal(rng, 7.6, 0.2, n)
        f["dst_bytes"] = _lognormal(rng, 5.8, 0.3, n)
        f["count"] = rng.integers(20, 120, n)
        f["srv_count"] = f["count"]
        f["dst_host_srv_count"] = np.full(n, 255)
        f["dst_host_same_srv_rate"] = _rate(rng, 0.98, 0.02, n)
    elif name in ("portsweep", "satan", "ipsweep", "nmap", "mscan", "saint"):  # probes
        f["service"] = rng.choice(_SCAN_SERVICES, n)
        f["flag"] = rng.choice(["REJ", "RSTO", "SH", "S0", "RSTOS0", "SF"], n,
                               p=[0.35, 0.2, 0.1, 0.1, 0.1, 0.15])
        f["src_bytes"] = np.where(rng.random(n) < 0.7, 0, _lognormal(rng, 3, 1, n))
        f["dst_bytes"] = np.where(rng.random(n) < 0.8, 0, _lognormal(rng, 4, 1, n))
        f["logged_in"] = np.zeros(n, dtype=np.int64)
        f["rerror_rate"] = _rate(rng, 0.6, 0.35, n)
        f["srv_rerror_rate"] = _rate(rng, 0.6, 0.35, n)
        f["same_srv_rate"] = _rate(rng, 0.25, 0.25, n)
        f["diff_srv_rate"] = _rate(rng, 0.5, 0.3, n)
        f["dst_host_diff_srv_rate"] = _rate(rng, 0.5, 0.3, n)
        f["dst_host_rerror_rate"] = _rate(rng, 0.5, 0.35, n)
        f["dst_host_same_src_port_rate"] = _rate(rng, 0.6, 0.35, n)
        if name in ("ipsweep", "mscan"):  # sweep across hosts (ICMP)
            f["protocol_type"] = rng.choice(["icmp", "tcp"], n, p=[0.7, 0.3])
            f["service"] = np.where(f["protocol_type"] == "icmp", "eco_i", f["service"])
            f["srv_diff_host_rate"] = _rate(rng, 0.8, 0.2, n)
            f["dst_host_srv_diff_host_rate"] = _rate(rng, 0.5, 0.3, n)
        if name in ("mscan", "saint"):  # novel scanners: slower, fewer errors
            f["rerror_rate"] = _rate(rng, 0.3, 0.3, n)
            f["dst_host_count"] = rng.integers(100, 256, n)
    elif name in ("guess_passwd", "snmpguess"):  # credential guessing
        if name == "guess_passwd":
            f["service"] = rng.choice(["telnet", "ftp", "pop_3", "imap4"], n)
            f["flag"] = rng.choice(["RSTO", "SF"], n, p=[0.6, 0.4])
            f["num_failed_logins"] = np.ones(n, dtype=np.int64)
            f["src_bytes"] = _lognormal(rng, 4.8, 0.3, n)
            f["dst_bytes"] = _lognormal(rng, 5.0, 0.4, n)
            f["logged_in"] = np.zeros(n, dtype=np.int64)
        else:  # SNMP community-string guessing: tiny UDP packets
            f["protocol_type"] = np.full(n, "udp", dtype=object)
            f["service"] = np.full(n, "snmp", dtype=object)
            f["src_bytes"] = rng.integers(30, 60, n)
            f["dst_bytes"] = np.zeros(n, dtype=np.int64)
            f["logged_in"] = np.zeros(n, dtype=np.int64)
            f["count"] = rng.integers(1, 5, n)
    elif name == "snmpgetattack":  # indistinguishable from benign SNMP polling
        f["protocol_type"] = np.full(n, "udp", dtype=object)
        f["service"] = np.full(n, "snmp", dtype=object)
        f["src_bytes"] = rng.integers(100, 110, n)
        f["dst_bytes"] = rng.integers(100, 120, n)
        f["logged_in"] = np.zeros(n, dtype=np.int64)
    elif name in ("warezclient", "warezmaster", "ftp_write", "httptunnel", "xlock"):
        f["service"] = rng.choice(["ftp_data", "ftp"], n) if name != "httptunnel" else np.full(n, "http", dtype=object)
        f["duration"] = _lognormal(rng, 5, 1.5, n)
        f["src_bytes"] = _lognormal(rng, 9.5 if name == "warezmaster" else 7.5, 1.0, n)
        f["hot"] = rng.integers(0, 28, n)
        f["is_guest_login"] = rng.binomial(1, 0.8 if name.startswith("warez") else 0.1, n)
        f["logged_in"] = np.ones(n, dtype=np.int64)
        f["num_access_files"] = rng.binomial(1, 0.2, n)
    elif name in ("buffer_overflow", "rootkit", "perl", "loadmodule", "xterm", "ps", "sqlattack"):
        f["service"] = rng.choice(["telnet", "ftp", "ftp_data"], n, p=[0.7, 0.15, 0.15])
        f["duration"] = _lognormal(rng, 5.5, 1.2, n)
        f["src_bytes"] = _lognormal(rng, 7.0, 1.0, n)
        f["dst_bytes"] = _lognormal(rng, 8.0, 1.0, n)
        f["hot"] = rng.integers(0, 6, n)
        f["logged_in"] = np.ones(n, dtype=np.int64)
        f["root_shell"] = rng.binomial(1, 0.6, n)
        f["num_file_creations"] = rng.integers(0, 4, n)
        f["num_shells"] = rng.binomial(1, 0.3, n)
        f["num_root"] = rng.integers(0, 5, n)
        f["count"] = rng.integers(1, 3, n)
    else:
        raise KeyError(name)

    df = pd.DataFrame(f)[config.FEATURE_COLUMNS]
    df["attack"] = name
    df["difficulty"] = rng.integers(10, 22, n)
    return df


# (attack, share) pairs. Shares roughly follow the NSL-KDD class balance; the
# test split adds attacks absent from training and has more R2L/U2R mass.
_TRAIN_MIX = [("normal", 0.534), ("neptune", 0.30), ("smurf", 0.02), ("back", 0.008),
              ("teardrop", 0.007), ("pod", 0.002), ("satan", 0.029), ("ipsweep", 0.029),
              ("portsweep", 0.023), ("nmap", 0.012), ("warezclient", 0.007),
              ("guess_passwd", 0.0015), ("warezmaster", 0.0008), ("ftp_write", 0.0004),
              ("buffer_overflow", 0.0004), ("rootkit", 0.0002), ("perl", 0.0001),
              ("loadmodule", 0.0001)]
_TEST_MIX = [("normal", 0.43), ("neptune", 0.21), ("smurf", 0.03), ("back", 0.016),
             ("teardrop", 0.001), ("pod", 0.002), ("apache2", 0.033), ("processtable", 0.03),
             ("mailbomb", 0.013), ("satan", 0.033), ("ipsweep", 0.006), ("portsweep", 0.007),
             ("nmap", 0.003), ("mscan", 0.044), ("saint", 0.013), ("guess_passwd", 0.054),
             ("warezmaster", 0.042), ("snmpgetattack", 0.008), ("snmpguess", 0.015),
             ("httptunnel", 0.006), ("xlock", 0.001), ("buffer_overflow", 0.001),
             ("xterm", 0.001), ("ps", 0.001), ("rootkit", 0.0005), ("sqlattack", 0.0005)]


def _draw(mix, n_total, rng) -> pd.DataFrame:
    parts = [_profile(name, max(2, int(round(share * n_total))), rng) for name, share in mix]
    return pd.concat(parts, ignore_index=True).sample(frac=1.0, random_state=int(rng.integers(1 << 31)))


def generate_synthetic_dataset(n_train: int = 25_000, n_test: int = 5_000,
                               random_state: int = config.RANDOM_STATE):
    """Return (train_df, test_df) with NSL-KDD columns incl. ``attack``/``difficulty``."""
    rng = np.random.default_rng(random_state)
    train = _draw(_TRAIN_MIX, n_train, rng).reset_index(drop=True)
    test = _draw(_TEST_MIX, n_test, rng).reset_index(drop=True)
    return train, test
