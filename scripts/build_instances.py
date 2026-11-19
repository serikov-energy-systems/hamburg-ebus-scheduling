"""Create the instance files in data/instances from the HVV GTFS feed (data/raw/gtfs).

Usage: python scripts/build_instances.py [gtfs_dir]
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ebus.gtfs import build_trips  # noqa: E402
from ebus.model import make_instance, min_fleet_unlimited  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
GTFS = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "data/raw/gtfs")
DATE = "2024-12-10"   # Tuesday in the feed of 5-14 December 2024

# name: (lines, window start, window end in minutes after midnight)
SPEC = {
    "S": (["20"], 0, 30 * 60),
    "T": (["7"], 0, 30 * 60),
    "M": (["20", "25"], 0, 30 * 60),
    "L": (["3", "4", "5", "20", "25", "19", "6", "7"], 0, 30 * 60),
}


def main():
    out = ROOT / "data/instances"
    out.mkdir(parents=True, exist_ok=True)
    for name, (lines, a, b) in SPEC.items():
        df = build_trips(GTFS, DATE, lines)
        df = df[(df.start_min >= a) & (df.start_min < b)].reset_index(drop=True)
        inst = make_instance(name, df)
        (out / f"{name}.json").write_text(json.dumps(inst.to_json()))
        print(name, "lines", lines, "trips", inst.n, "terminals", inst.T, "depot", inst.term_name[inst.depot],
              "service km", round(float(inst.km.sum())), "min fleet without range limit", min_fleet_unlimited(inst))


if __name__ == "__main__":
    main()
