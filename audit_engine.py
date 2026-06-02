import json
from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).parent

def clamp(x, lo=0, hi=100):
    return max(lo, min(hi, x))

def visibility_score(row):
    size_component = min(row["bbox_area"] / 0.18, 1.0)
    duration_component = min(row["visible_seconds"] / 4.0, 1.0)
    occlusion_component = 1.0 - row["occlusion"]
    return clamp(100 * (0.35 * size_component + 0.35 * duration_component + 0.30 * occlusion_component))

def readability_score(row):
    return clamp(100 * (0.65 * row["ocr_confidence"] + 0.35 * row["contrast"]))

def sign_quality(row):
    return clamp(0.55 * visibility_score(row) + 0.45 * readability_score(row))

def main():
    building = json.loads((DATA_DIR / "building.json").read_text())
    scenarios = json.loads((DATA_DIR / "scenarios.json").read_text())

    for scenario, signs in scenarios.items():
        df = pd.DataFrame(signs)
        df["visibility_score"] = df.apply(visibility_score, axis=1).round(1)
        df["readability_score"] = df.apply(readability_score, axis=1).round(1)
        df["quality_score"] = df.apply(sign_quality, axis=1).round(1)
        print("\\nScenario:", scenario)
        print(df[["sign_id", "node", "text", "visibility_score", "readability_score", "quality_score"]])

if __name__ == "__main__":
    main()