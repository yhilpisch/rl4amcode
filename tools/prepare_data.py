"""Cache the exact market snapshot used by the CQF case study."""

import argparse
import hashlib
from pathlib import Path
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parents[1]
SOURCE = "https://hilpisch.com/eod_data.csv"
SHA256 = "949904965a389d65dd26848d6f5ef32b90ba16b3fd2454ba87080a9acbfdecf0"


def prepare(source: str, destination: Path) -> Path:
    """Validate bytes before creating a local snapshot; never replace data."""
    if destination.exists():
        data = destination.read_bytes()
    elif source.startswith("https://"):
        with urlopen(source, timeout=30) as response:
            data = response.read()
    else:
        data = Path(source).read_bytes()
    if hashlib.sha256(data).hexdigest() != SHA256:
        raise ValueError(
            "Snapshot checksum differs; provide the frozen lecture CSV "
            "with --source. Existing files have not been changed."
        )
    if not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=SOURCE)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "data/eod_data.csv",
    )
    args = parser.parse_args()
    print(prepare(args.source, args.output))
