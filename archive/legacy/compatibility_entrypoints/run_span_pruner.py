#!/usr/bin/env python3
"""Compatibility entrypoint. New code lives under experiments/ner."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.ner.run_span_pruner import main


if __name__ == "__main__":
    main()
