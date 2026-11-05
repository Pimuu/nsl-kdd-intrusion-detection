#!/usr/bin/env python3
"""Reproduce every result in results/ and the paper with one command:

    python run_pipeline.py              # real NSL-KDD (downloaded + checksum-verified)
    python run_pipeline.py --synthetic  # offline fallback with synthetic data
"""
from src.pipeline import main

if __name__ == "__main__":
    main()
