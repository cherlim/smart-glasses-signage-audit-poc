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

def risk_label(score):
    if score >= 75:
        return "Low"
    if score >= 55:
        return "Medium"
    return "High"

def risk_icon(risk):
    return {"Low": "🟢", "Medium": "🟡", "High": "🔴"}.get(risk, "⚪")

def quality_colour(score):
    if score >= 80:
        return "palegreen"
    if score >= 60:
        return "khaki"
    return "lightcoral"

def prepare_sign_dataframe(signs):
    df = pd.DataFrame(signs)
    if df.empty:
        return df
    df["visibility_score"] = df.apply(visibility_score, axis=1).round(1)
    df["readability_score"] = df.apply(readability_score, axis=1).round(1)
    df["quality_score"] = df.apply(sign_quality, axis=1).round(1)
    df["status"] = df["quality_score"].apply(status)
    return df

def route_distance(building, route_path):
    distance = 0
    for a, b in zip(route_path, route_path[1:]):
        for e in building["edges"]:
            if {e["from"], e["to"]} == {a, b}:
                distance += e["distance_m"]
    return distance

def get_route(building, route_id):
    return next(r for r in building["routes"] if r["id"] == route_id)

def audit_route(building, df, selected_route_id):
    route = get_route(building, selected_route_id)
    route_nodes = set(route["path"])
    expected_terms = [d.lower() for d in route["expected_destinations"]]

    if df.empty:
        return df, {}

    df = df.copy()
    df["on_route"] = df["node"].isin(route_nodes)
    route_df = df[df["on_route"]].copy()

    decision_points = [
        n["id"] for n in building["nodes"]
        if n["is_decision_point"] and n["id"] in route_nodes
    ]

    decision_point_count = len(decision_points)
    covered_decision_points = route_df[route_df["directional"]]["node"].nunique()

    route_text = " ".join(route_df["text"].str.lower().tolist())
    destination_covered = any(term.split()[0] in route_text for term in expected_terms)

    dist = route_distance(building, route["path"])
    avg_quality = round(route_df["quality_score"].mean(), 1) if len(route_df) else 0
    decision_coverage = round(100 * covered_decision_points / decision_point_count, 1) if decision_point_count else 0
    effective_coverage = round(decision_coverage * avg_quality / 100, 1)

    confusion_risk_score = clamp(
        100
        - (0.45 * avg_quality)
        - (0.35 * effective_coverage)
        + (8 * int((route_df["status"] == "Problem").sum()))
        + (5 * max(decision_point_count - 2, 0))
    )

    # Convert risk score into label. High numeric confusion risk means worse condition.
    if confusion_risk_score >= 65:
        confusion_risk = "High"
    elif confusion_risk_score >= 40:
        confusion_risk = "Medium"
    else:
        confusion_risk = "Low"

    metrics = {
        "Route": route["name"],
        "Route distance (m)": dist,
        "Signs visible on route": int(len(route_df)),
        "Average sign quality": avg_quality,
        "Decision-point coverage (%)": decision_coverage,
        "Effective decision coverage (%)": effective_coverage,
        "Destination guidance found": "Yes" if destination_covered else "No",
        "Sign density (signs / 10m)": round(len(route_df) / dist * 10, 2) if dist else 0,
        "Problem signs": int((route_df["status"] == "Problem").sum()),
        "Confusion risk score": round(confusion_risk_score, 1),
        "Confusion risk": confusion_risk,
    }

    return df, metrics

def compute_node_metrics(building, df, selected_route_id):
    route = get_route(building, selected_route_id)
    route_nodes = set(route["path"])

    rows = []
    for node in building["nodes"]:
        node_id = node["id"]
        node_signs = df[df["node"] == node_id] if not df.empty else pd.DataFrame()

        if len(node_signs):
            sq = round(node_signs["quality_score"].mean(), 1)
            problem_count = int((node_signs["status"] == "Problem").sum())
            critical = node_signs.sort_values("quality_score").iloc[0]["sign_id"]
        else:
            sq = 0 if node["is_decision_point"] else 100
            problem_count = 1 if node["is_decision_point"] else 0
            critical = "No sign"

        if node["is_decision_point"]:
            edc = round(sq, 1) if len(node_signs) else 0
        else:
            edc = None

        if node["is_decision_point"]:
            risk_base = sq
        else:
            risk_base = sq if len(node_signs) else 85

        risk = risk_label(risk_base)

        rows.append({
            "node": node_id,
            "name": node["name"],
            "type": node["type"],
            "on_route": node_id in route_nodes,
            "is_decision_point": node["is_decision_point"],
            "sign_quality": sq,
            "effective_decision_coverage": edc,
            "problem_signs": problem_count,
            "critical_sign": critical,
            "risk": risk
        })

    return pd.DataFrame(rows)

def recommendation_for_sign(row):
    recommendations = []
    issues = []

    if row["visibility_score"] < 60:
        issues.append("low visibility")
        recommendations.append("Increase sign size or place the sign earlier along the route.")
    if row["readability_score"] < 60:
        issues.append("low readability")
        recommendations.append("Improve font clarity and increase text/background contrast.")
    if row["occlusion"] >= 0.25:
        issues.append("high occlusion")
        recommendations.append("Remove obstruction or reposition the sign to a clearer sight line.")
    if row["visible_seconds"] < 2:
        issues.append("short viewing time")
        recommendations.append("Add advance signage before the decision point.")
    if row["ocr_confidence"] < 0.65:
        issues.append("low OCR confidence")
        recommendations.append("Simplify wording and use larger, clearer lettering.")

    if not recommendations:
        issues.append("no major issue detected")
        recommendations.append("Maintain current signage condition.")

    return issues, recommendations

def build_graphviz(building, node_metrics, selected_route_id):
    route = get_route(building, selected_route_id)
    route_edges = set()
    for a, b in zip(route["path"], route["path"][1:]):
        route_edges.add(tuple(sorted([a, b])))

    metric_lookup = node_metrics.set_index("node").to_dict(orient="index")

    lines = []
    lines.append("graph G {")
    lines.append('  graph [rankdir=TB, bgcolor="transparent", splines=true, nodesep=0.65, ranksep=0.8];')
    lines.append('  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=12, margin=0.12];')
    lines.append('  edge [fontname="Helvetica", fontsize=10, color="gray55"];')

    for node in building["nodes"]:
        node_id = node["id"]
        m = metric_lookup[node_id]
        sq = m["sign_quality"]
        risk = m["risk"]
        colour = quality_colour(sq)
        penwidth = "3" if m["on_route"] else "1"
        border = "black" if m["on_route"] else "gray70"
        edc_text = "EDC: n/a" if m["effective_decision_coverage"] is None else f'EDC: {m["effective_decision_coverage"]}%'
        label = f'{node["name"]}\\nSQ: {sq}\\n{edc_text}\\nRisk: {risk}'
        lines.append(f'  {node_id} [label="{label}", fillcolor="{colour}", color="{border}", penwidth={penwidth}];')

    for e in building["edges"]:
        a, b = e["from"], e["to"]
        key = tuple(sorted([a, b]))
        if key in route_edges:
            color = "black"
            penwidth = "3"
        else:
            color = "gray70"
            penwidth = "1"
        lines.append(f'  {a} -- {b} [label="{e["distance_m"]}m", color="{color}", penwidth={penwidth}];')

    lines.append("}")
    return "\n".join(lines)

st.set_page_config(page_title="Smart Glasses Signage Audit POC", layout="wide")

building, scenarios = load_data()

st.title("Smart Glasses Signage Audit POC")
st.caption("Synthetic building + simulated smart-glasses observations + graph-based signage audit metrics.")

left, right = st.columns([1, 2])

with left:
    scenario_name = st.selectbox("Select signage scenario", list(scenarios.keys()))
    route_name_to_id = {r["name"]: r["id"] for r in building["routes"]}
    route_name = st.selectbox("Select route", list(route_name_to_id.keys()))
    selected_route_id = route_name_to_id[route_name]

df = prepare_sign_dataframe(scenarios[scenario_name])
df, metrics = audit_route(building, df, selected_route_id)
node_metrics = compute_node_metrics(building, df, selected_route_id)

with right:
    st.subheader("Route Audit Summary")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Avg Quality", metrics["Average sign quality"])
    c2.metric("Decision Coverage", f'{metrics["Decision-point coverage (%)"]}%')
    c3.metric("Effective Coverage", f'{metrics["Effective decision coverage (%)"]}%')
    c4.metric("Problem Signs", metrics["Problem signs"])
    c5.metric("Confusion Risk", f'{risk_icon(metrics["Confusion risk"])} {metrics["Confusion risk"]}')

st.divider()

st.subheader("Visual Building Graph with Signage Metrics")
st.caption("SQ = Sign Quality. EDC = Effective Decision Coverage. Thick black edges show the selected route.")
dot = build_graphviz(building, node_metrics, selected_route_id)
st.graphviz_chart(dot, use_container_width=True)

route_nodes = set(get_route(building, selected_route_id)["path"])
route_node_metrics = node_metrics[node_metrics["on_route"]].copy()
critical_node = route_node_metrics.sort_values("sign_quality").iloc[0]

route_signs = df[df["on_route"]].copy()
if len(route_signs):
    critical_sign = route_signs.sort_values("quality_score").iloc[0]
    issues, recommendations = recommendation_for_sign(critical_sign)
else:
    critical_sign = None
    issues, recommendations = [], ["Add wayfinding signage along this route."]

st.subheader("Key Findings")
k1, k2, k3 = st.columns(3)

with k1:
    st.markdown("### Critical Node")
    st.write(f"**{critical_node['name']}**")
    st.write(f"Sign quality: **{critical_node['sign_quality']}**")
    st.write(f"Risk: **{risk_icon(critical_node['risk'])} {critical_node['risk']}**")

with k2:
    st.markdown("### Critical Sign")
    if critical_sign is not None:
        st.write(f"**{critical_sign['sign_id']} — {critical_sign['text']}**")
        st.write(f"Quality score: **{critical_sign['quality_score']}**")
        st.write(f"Status: **{critical_sign['status']}**")
    else:
        st.write("No sign detected on route.")

with k3:
    st.markdown("### Route Interpretation")
    if metrics["Destination guidance found"] == "No":
        st.error("Destination-specific guidance is missing or not detected.")
    elif metrics["Confusion risk"] == "High":
        st.error("This route has high potential confusion risk.")
    elif metrics["Confusion risk"] == "Medium":
        st.warning("This route may require signage improvement.")
    else:
        st.success("This route appears relatively well supported.")

st.subheader("Recommendations")
for rec in recommendations:
    st.write(f"- {rec}")
if issues:
    st.caption("Detected issue basis: " + ", ".join(issues))

st.divider()

st.subheader("Node-Level Audit Metrics")
st.dataframe(
    node_metrics[[
        "node", "name", "type", "on_route", "is_decision_point",
        "sign_quality", "effective_decision_coverage", "problem_signs",
        "critical_sign", "risk"
    ]],
    use_container_width=True
)

st.subheader("Detected / Simulated Signs")
display_cols = [
    "sign_id", "node", "text", "bbox_area", "visible_seconds",
    "ocr_confidence", "contrast", "occlusion",
    "visibility_score", "readability_score", "quality_score",
    "status", "on_route"
]
st.dataframe(df[display_cols], use_container_width=True)

st.subheader("POC Logic")
st.markdown("""
This dashboard still uses simulated smart-glasses observations. The graph now acts as a wayfinding audit map:

- each node represents a location in the building
- node colour shows signage quality
- the selected route is highlighted
- each node displays sign quality, effective decision coverage, and confusion risk
- the dashboard identifies the weakest node and weakest sign, then generates recommendations

This is a mockup of the audit framework before adding real video and OCR.
""")
