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

def prepare_sign_dataframe(signs):
    df = pd.DataFrame(signs)
    if df.empty:
        return df
    df["visibility_score"] = df.apply(visibility_score, axis=1).round(1)
    df["readability_score"] = df.apply(readability_score, axis=1).round(1)
    df["quality_score"] = df.apply(sign_quality, axis=1).round(1)
    df["status"] = df["quality_score"].apply(status)
    return df

def get_route(building, route_id):
    return next(r for r in building["routes"] if r["id"] == route_id)

def route_distance(building, route_path):
    distance = 0
    for a, b in zip(route_path, route_path[1:]):
        for e in building["edges"]:
            if {e["from"], e["to"]} == {a, b}:
                distance += e["distance_m"]
    return distance

def audit_route(building, df, selected_route_id):
    route = get_route(building, selected_route_id)
    route_nodes = set(route["path"])
    expected_terms = [d.lower() for d in route["expected_destinations"]]

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

        risk_base = sq if node["is_decision_point"] or len(node_signs) else 85
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

def get_node_row(node_metrics, node_id):
    return node_metrics[node_metrics["node"] == node_id].iloc[0]

def node_button_label(node_metrics, node_id):
    r = get_node_row(node_metrics, node_id)
    icon = risk_icon(r["risk"])
    return f'{icon} {r["name"]}'

def click_node(node_id):
    st.session_state["selected_node"] = node_id

def render_clickable_graph(node_metrics, selected_route_nodes):
    st.markdown("#### Clickable Wayfinding Heat Map")
    st.caption("Click a location to inspect node-level metrics. Nodes on the selected route are marked with ⭐.")

    row1 = st.columns([1.1, 0.25, 1.1, 0.25, 1.1])
    with row1[0]:
        label = node_button_label(node_metrics, "E")
        if "E" in selected_route_nodes:
            label = "⭐ " + label
        st.button(label, key="node_E", on_click=click_node, args=("E",), use_container_width=True)
    with row1[1]:
        st.markdown("<h2 style='text-align:center;'>→</h2>", unsafe_allow_html=True)
    with row1[2]:
        label = node_button_label(node_metrics, "C1")
        if "C1" in selected_route_nodes:
            label = "⭐ " + label
        st.button(label, key="node_C1", on_click=click_node, args=("C1",), use_container_width=True)
    with row1[3]:
        st.markdown("<h2 style='text-align:center;'>→</h2>", unsafe_allow_html=True)
    with row1[4]:
        label = node_button_label(node_metrics, "J1")
        if "J1" in selected_route_nodes:
            label = "⭐ " + label
        st.button(label, key="node_J1", on_click=click_node, args=("J1",), use_container_width=True)

    st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)

    row2 = st.columns([1.1, 1.1, 1.1])
    for col, node_id in zip(row2, ["L", "CL", "T"]):
        with col:
            label = node_button_label(node_metrics, node_id)
            if node_id in selected_route_nodes:
                label = "⭐ " + label
            st.button(label, key=f"node_{node_id}", on_click=click_node, args=(node_id,), use_container_width=True)

    st.caption("Legend: 🟢 Low risk / good signage · 🟡 medium risk · 🔴 high risk or poor signage · ⭐ selected route")

def selected_node_detail_panel(building, df, node_metrics, node_id):
    node_row = get_node_row(node_metrics, node_id)
    node_signs = df[df["node"] == node_id].copy()

    st.markdown(f"### Selected Node: {node_row['name']}")
    st.write(f"Type: **{node_row['type']}**")
    st.write(f"On selected route: **{'Yes' if node_row['on_route'] else 'No'}**")
    st.write(f"Decision point: **{'Yes' if node_row['is_decision_point'] else 'No'}**")

    c1, c2, c3 = st.columns(3)
    c1.metric("Sign Quality", node_row["sign_quality"])
    if pd.isna(node_row["effective_decision_coverage"]):
        c2.metric("Effective Coverage", "n/a")
    else:
        c2.metric("Effective Coverage", f'{node_row["effective_decision_coverage"]}%')
    c3.metric("Risk", f'{risk_icon(node_row["risk"])} {node_row["risk"]}')

    if len(node_signs):
        critical_sign = node_signs.sort_values("quality_score").iloc[0]
        issues, recommendations = recommendation_for_sign(critical_sign)

        st.markdown("#### Critical Sign at This Node")
        st.write(f"**{critical_sign['sign_id']} — {critical_sign['text']}**")
        st.write(f"Quality score: **{critical_sign['quality_score']}** | Status: **{critical_sign['status']}**")

        st.markdown("#### Recommendations")
        for rec in recommendations:
            st.write(f"- {rec}")

        st.caption("Detected issue basis: " + ", ".join(issues))

        with st.expander("Show signs at this node"):
            st.dataframe(
                node_signs[[
                    "sign_id", "text", "visibility_score", "readability_score",
                    "quality_score", "status", "visible_seconds", "ocr_confidence",
                    "contrast", "occlusion"
                ]],
                use_container_width=True
            )
    else:
        if node_row["is_decision_point"]:
            st.error("No sign is detected at this decision point.")
            st.write("- Add clear directional signage at this location.")
            st.write("- Ensure the sign is visible before the user reaches the decision point.")
        else:
            st.info("No signage issue detected for this non-decision point.")

# ---------- Streamlit App ----------

st.set_page_config(page_title="Smart Glasses Signage Audit POC", layout="wide")

building, scenarios = load_data()

st.title("Smart Glasses Signage Audit POC")
st.caption("Synthetic building + simulated smart-glasses observations + clickable wayfinding heat map.")

left, right = st.columns([1, 2])

with left:
    scenario_name = st.selectbox("Select signage scenario", list(scenarios.keys()))
    route_name_to_id = {r["name"]: r["id"] for r in building["routes"]}
    route_name = st.selectbox("Select route", list(route_name_to_id.keys()))
    selected_route_id = route_name_to_id[route_name]

df = prepare_sign_dataframe(scenarios[scenario_name])
df, metrics = audit_route(building, df, selected_route_id)
node_metrics = compute_node_metrics(building, df, selected_route_id)

selected_route_nodes = set(get_route(building, selected_route_id)["path"])

if "selected_node" not in st.session_state:
    st.session_state["selected_node"] = "J1"

with right:
    st.subheader("Route Audit Summary")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Avg Quality", metrics["Average sign quality"])
    c2.metric("Decision Coverage", f'{metrics["Decision-point coverage (%)"]}%')
    c3.metric("Effective Coverage", f'{metrics["Effective decision coverage (%)"]}%')
    c4.metric("Problem Signs", metrics["Problem signs"])
    c5.metric("Confusion Risk", f'{risk_icon(metrics["Confusion risk"])} {metrics["Confusion risk"]}')

st.divider()

graph_col, detail_col = st.columns([1.25, 1])

with graph_col:
    render_clickable_graph(node_metrics, selected_route_nodes)

with detail_col:
    selected_node_detail_panel(building, df, node_metrics, st.session_state["selected_node"])

st.divider()

st.subheader("Route-Level Interpretation")
if metrics["Destination guidance found"] == "No":
    st.error("Destination-specific guidance is missing or not detected on this route.")
elif metrics["Confusion risk"] == "High":
    st.error("This route has high potential confusion risk.")
elif metrics["Confusion risk"] == "Medium":
    st.warning("This route may require signage improvement.")
else:
    st.success("This route appears relatively well supported.")

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
This dashboard uses simulated smart-glasses observations. The clickable heat map represents the building as a route network:

- each button is a location node
- colour icons indicate signage/wayfinding risk
- ⭐ marks nodes on the selected route
- clicking a node shows sign quality, effective coverage, critical sign, and recommendations

This version keeps the graph compact and moves detailed metrics into the inspection panel.
""")
