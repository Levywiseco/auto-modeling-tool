#!/usr/bin/env python3
"""Validate a saved scoring artifact before release."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from auto_modeling_tool.evaluation.quality_gate import validate_release


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, help="Output directory or scoring artifact")
    parser.add_argument("--report", help="Model_Report_N.xlsx path")
    parser.add_argument("--min-auc", type=float)
    parser.add_argument("--max-psi", type=float)
    parser.add_argument("--json", dest="json_path")
    args = parser.parse_args()

    result = validate_release(
        args.model_dir,
        report_path=args.report,
        min_auc=args.min_auc,
        max_psi=args.max_psi,
    )
    payload = result.as_dict()
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if args.json_path:
        output = Path(args.json_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
