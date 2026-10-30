"""Sliding-window reconstruction of a clip from simulated or measured bucket values.

    python scripts/reconstruct.py --config configs/reconstruct_m1000_k5.yaml
"""

import argparse
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True)
    config = yaml.safe_load(parser.parse_args().config.read_text())
    print(f"K={config['sampling']['steps']}, M={config['measurements']['count']}")
    raise SystemExit("Pretrained MGV-Flow weights are released upon publication.")


if __name__ == "__main__":
    main()
