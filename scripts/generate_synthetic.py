"""Write the synthetic survey and its human-coded sample to data/.

    python scripts/generate_synthetic.py            # ~3,000 respondents, 9 countries, 4 languages
    python scripts/generate_synthetic.py --scale 0.2  # smaller file for quick tests
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pulse.synthetic import generate, human_coded_sample  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--out", default="data")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    survey, gold = generate(seed=args.seed, scale=args.scale)
    sample = human_coded_sample(gold)

    survey.to_csv(out / "synthetic_survey.csv", index=False, encoding="utf-8")
    sample.to_csv(out / "human_coded_sample.csv", index=False, encoding="utf-8")
    gold.to_csv(out / "synthetic_labels_full.csv", index=False, encoding="utf-8")

    commented = survey[survey["comment"] != ""]
    print(f"Respondents: {len(survey):,} in {survey['country'].nunique()} countries, {survey['team'].nunique()} teams")
    print(f"Comments:    {len(commented):,}  ({gold['language'].value_counts().to_dict()})")
    print(f"Sensitive:   {(gold['sensitive_category'] != 'none').sum()}  {gold.loc[gold['sensitive_category'] != 'none', 'sensitive_category'].value_counts().to_dict()}")
    print(f"Human-coded sample: {len(sample)} comments ({sample['sample_stratum'].value_counts().to_dict()})")


if __name__ == "__main__":
    main()
