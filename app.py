import json
from pathlib import Path
import pandas as pd
import streamlit as st

DATA_DIR = Path(__file__).parent

def load_data():
    with open(DATA_DIR / "building.json", "r", encoding="utf-8") as f:
        building = json.load(f)
    with open(DATA_DIR / "scenarios.json", "r", encoding="utf-8") as f:
        scenarios = json.load(f)
    return building, scenarios

def clamp(x, lo=0, hi=100):
    return max(lo, min(hi, x))

def visibility_score(row):
    # bbox_area is normalized from 0 to 1; visible_seconds is capped at 5 sec.
    size_component = min(row["bbox_area"] / 0.18, 1.0)
    duration_component = min(row["visible_seconds"] / 4.0, 1.0)
    occlusion_component = 1.0 - row["occlusion"]
    return clamp(100 * (0.35 * size_component + 0.35 * duration_component + 0.30 * occlusion_component))

def readability_score(row):
    return clamp(100 * (0.65 * row["ocr_confidence"] + 0.35 * row["contrast"]))

def sign_quality(row):
    return clamp(0.55 * visibility_score(row) + 0.45 * readability_score(row))

def status(score):
    if score >= 80:
        return "Good"
    if score >= 60:
        return "Acceptable"
    return "Problem"

def audit_scenario(building, signs, selected_route_id):
    route = next(r for r in building["routes"] if r["id"] == selected_route_id)
    route_nodes = set(route["path"])
    expected_terms = [d.lower() for d in route["expected_destinations"]]

    df = pd.DataFrame(signs)
    if df.empty:
        return df, {}

    df["visibility_score"] = df.apply(visibility_score, axis=1).round(1)
    df["readability_score"] = df.apply(readability_score, axis=1).round(1)
    df["quality_score"] = df.apply(sign_quality, axis=1).round(1)
    df["status"] = df["quality_score"].apply(status)
    df["on_route"] = df["node"].isin(route_nodes)

    route_df = df[df["on_route"]].copy()

    decision_points = [n["id"] for n in building["nodes"] if n["is_decision_point"] and n["id"] in route_nodes]
    decision_point_count = len(decision_points)
    covered_decision_points = route_df[route_df["directional"]]["node"].nunique()

    route_text = " ".join(route_df["text"].str.lower().tolist())
    destination_covered = any(term.split()[0] in route_text for term in expected_terms)

    distance = 0
    for e in building["edges"]:
        for a, b in zip(route["path"], route["path"][1:]):
            if {e["from"], e["to"]} == {a, b}:
                distance += e["distance_m"]

    metrics = {
        "Route": route["name"],
        "Route distance (m)": distance,
        "Signs visible on route": int(len(route_df)),
        "Average sign quality": round(route_df["quality_score"].mean(), 1) if len(route_df) else 0,
        "Decision-point coverage (%)": round(100 * covered_decision_points / decision_point_count, 1) if decision_point_count else 0,
        "Destination guidance found": "Yes" if destination_covered else "No",
        "Sign density (signs / 10m)": round(len(route_df) / distance * 10, 2) if distance else 0,
        "Problem signs": int((route_df["status"] == "Problem").sum())
    }
    return df, metrics

st.set_page_config(page_title="Smart Glasses Signage Audit POC", layout="wide")

building, scenarios = load_data()

st.title("Smart Glasses Signage Audit POC")
st.caption("Minimum proof of concept: synthetic building + simulated smart-glasses observations + signage audit metrics.")

left, right = st.columns([1, 2])

with left:
    scenario_name = st.selectbox("Select signage scenario", list(scenarios.keys()))
    route_name_to_id = {r["name"]: r["id"] for r in building["routes"]}
    route_name = st.selectbox("Select route", list(route_name_to_id.keys()))
    selected_route_id = route_name_to_id[route_name]

df, metrics = audit_scenario(building, scenarios[scenario_name], selected_route_id)

with right:
    st.subheader("Audit Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Avg Quality", metrics["Average sign quality"])
    c2.metric("Decision Coverage", f'{metrics["Decision-point coverage (%)"]}%')
    c3.metric("Problem Signs", metrics["Problem signs"])
    c4.metric("Destination Found", metrics["Destination guidance found"])

st.subheader("Building Graph")
edges_text = "\n".join([f'{e["from"]} -- {e["to"]} ({e["distance_m"]}m)' for e in building["edges"]])
st.code(edges_text)

st.subheader("Detected / Simulated Signs")
display_cols = ["sign_id", "node", "text", "bbox_area", "visible_seconds", "ocr_confidence", "contrast", "occlusion", "visibility_score", "readability_score", "quality_score", "status", "on_route"]
st.dataframe(df[display_cols], use_container_width=True)

st.subheader("Interpretation")
if metrics["Destination guidance found"] == "No":
    st.warning("Destination-specific guidance is missing or not detected on this route.")
elif metrics["Problem signs"] > 0:
    st.warning("Some signs are detected but have poor visibility/readability scores.")
else:
    st.success("The route has adequate signage coverage in this simulated scenario.")

st.subheader("POC logic")
st.markdown("""
This prototype simulates what smart glasses would provide after video processing:

- detected sign text
- approximate sign size in view
- visibility duration
- OCR confidence
- occlusion estimate
- contrast estimate

The audit engine converts these observations into signage quality metrics.
""")