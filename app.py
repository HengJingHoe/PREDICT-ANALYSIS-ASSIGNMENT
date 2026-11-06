import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Cooling Load Predictor", page_icon="🏢", layout="wide")

# ------------------------------------------------------------------
# Load the trained model (cached so it is only read once)
# ------------------------------------------------------------------
@st.cache_resource
def load_artifact(path="cooling_load_model.joblib"):
    return joblib.load(path)

artifact = load_artifact()
model = artifact["model"]
FEATURES = artifact["feature_cols"]
TRAIN_RANGE = artifact["train_range"]

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
# Hard bounds used for input validation (realistic building ranges)
BOUNDS = {
    "X1": (0.5, 1.0), "X2": (450.0, 850.0), "X3": (200.0, 450.0), "X4": (100.0, 250.0),
    "X5": (3.5, 7.0), "X6": (2, 5), "X7": (0.0, 0.5), "X8": (0, 5),
}
ORIENTATION = {2: "2 (North)", 3: "3 (East)", 4: "4 (South)", 5: "5 (West)"}

# European Union energy label scale: (upper limit, grade, performance, colour name, hex)
GRADES = [
    (15.0, "A", "Excellent", "Dark Green", "#006400"),
    (20.0, "B", "Good", "Light Green", "#7ED957"),
    (28.0, "C", "Average", "Yellow", "#FFD700"),
    (35.0, "D", "Below Average", "Orange", "#FFA500"),
    (float("inf"), "E", "Poor", "Red", "#E53935"),
]
RANGE_TEXT = ["Y <= 15.0", "15.0 < Y <= 20.0", "20.0 < Y <= 28.0", "28.0 < Y <= 35.0", "Y > 35.0"]

PRESETS = {
    "Custom": None,
    "Design A": dict(X1=0.85, X2=588.0, X3=294.0, X4=147.0, X5=7.0, X6=2, X7=0.25, X8=3),
    "Design B": dict(X1=0.65, X2=784.0, X3=343.0, X4=220.5, X5=3.5, X6=4, X7=0.10, X8=1),
    "Design C": dict(X1=0.90, X2=563.5, X3=318.5, X4=122.5, X5=7.0, X6=3, X7=0.40, X8=4),
}

# ------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------
def predict_cooling_load(values):
    """Predict log(Y) with the saved pipeline, then back-transform with exp() to kWh/m2."""
    row = pd.DataFrame([[values[c] for c in FEATURES]], columns=FEATURES)
    log_pred = model.predict(row)[0]
    return float(np.exp(log_pred))

def energy_grade(y):
    for upper, grade, perf, colour_name, hex_code in GRADES:
        if y <= upper:
            return grade, perf, colour_name, hex_code

def validate(v):
    """Return (errors, warnings). Errors block the prediction; warnings do not."""
    errors, warns = [], []
    for k, (lo, hi) in BOUNDS.items():
        if not lo <= v[k] <= hi:
            errors.append(f"{k} = {v[k]} is outside the allowed range [{lo}, {hi}].")
    # Geometry: surface area = wall area + 2 * roof area
    expected = v["X3"] + 2 * v["X4"]
    if abs(v["X2"] - expected) > 1.0:
        errors.append(f"Geometry check failed: Surface Area (X2 = {v['X2']:.1f}) must equal "
                      f"Wall Area + 2 x Roof Area (X3 + 2*X4 = {expected:.1f}).")
    # Glazing consistency (as in the training data)
    if v["X7"] <= 1e-9 and v["X8"] != 0:
        errors.append("If Glazing Area (X7) is 0, Glazing Area Distribution (X8) must be 0.")
    if v["X7"] > 1e-9 and v["X8"] == 0:
        errors.append("If Glazing Area (X7) > 0, choose a Glazing Area Distribution (X8) from 1 to 5.")
    # Soft warning: outside the range the model was trained on
    for k, (lo, hi) in TRAIN_RANGE.items():
        if not lo <= v[k] <= hi:
            warns.append(f"{k} = {v[k]} is outside the training range [{lo:.2f}, {hi:.2f}]; "
                         "the prediction is an extrapolation.")
        # Soft warning: the model uses X3-X8; check that the building shape was seen in training
    shapes = artifact.get("known_shapes", [])
    if shapes and min(abs(v["X3"] - a) + abs(v["X4"] - b) + abs(v["X5"] - c) for a, b, c in shapes) > 0.5:
        warns.append("This wall / roof / height combination does not match any of the 12 building shapes "
                     "in the training data; the prediction interpolates between shapes and is less reliable.")
    return errors, warns

def draw_gauge(y):
    xmax = max(50.0, y * 1.1)
    edges = [0, 15, 20, 28, 35, xmax]
    fig, ax = plt.subplots(figsize=(8, 1.9))
    for i, (_, grade, _, _, hex_code) in enumerate(GRADES):
        lo, hi = edges[i], edges[i + 1]
        ax.barh(0, hi - lo, left=lo, height=0.5, color=hex_code, edgecolor="white")
        ax.text((lo + hi) / 2, 0, grade, ha="center", va="center", fontsize=14, fontweight="bold")
    ax.plot([y], [0.38], marker="v", markersize=14, color="black")
    ax.text(y, 0.55, f"{y:.1f} kWh/m2", ha="center", va="bottom", fontweight="bold")
    ax.set_xlim(0, xmax)
    ax.set_ylim(-0.4, 0.95)
    ax.set_yticks([])
    ax.set_xlabel("Cooling Load (kWh/m2)")
    for side in ["left", "right", "top"]:
        ax.spines[side].set_visible(False)
    return fig

def apply_preset():
    preset = PRESETS[st.session_state["preset"]]
    if preset is not None:
        for k, val in preset.items():
            st.session_state[k] = val

# Default values = Design A
for k, val in PRESETS["Design A"].items():
    st.session_state.setdefault(k, val)

# ------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------
st.sidebar.header("Example designs")
st.sidebar.selectbox("Load an example design", list(PRESETS), key="preset", on_change=apply_preset)

st.sidebar.header("Energy label scale")
st.sidebar.dataframe(pd.DataFrame({
    "Grade": [g[1] for g in GRADES],
    "Cooling Load (kWh/m2)": RANGE_TEXT,
    "Performance": [g[2] for g in GRADES],
    "Colour": [g[3] for g in GRADES],
}), hide_index=True)

# ------------------------------------------------------------------
# Main page
# ------------------------------------------------------------------
st.title("Building Cooling Load Predictor")
st.caption(f"Model: {artifact['model_name']}. It is trained on log(Y); "
           "predictions are back-transformed with exp() to kWh/m2.")
st.caption("X1 and X2 are derived from the building geometry (X2 = X3 + 2*X4, and X1 is fixed by the shape), "
           "so the model uses X3-X8; X2 is still checked for consistency.")

left, right = st.columns([1, 1.2])

with left:
    st.subheader("Building parameters")
    st.slider("Relative Compactness (X1)", 0.5, 1.0, step=0.01, key="X1")
    st.number_input("Surface Area (X2, m2)", 450.0, 850.0, step=0.5, format="%.2f", key="X2")
    st.number_input("Wall Area (X3, m2)", 200.0, 450.0, step=0.5, format="%.2f", key="X3")
    st.number_input("Roof Area (X4, m2)", 100.0, 250.0, step=0.25, format="%.2f", key="X4")
    st.selectbox("Overall Height (X5, m)", [3.5, 7.0], key="X5")
    st.selectbox("Orientation (X6)", [2, 3, 4, 5], format_func=lambda o: ORIENTATION[o], key="X6")
    st.slider("Glazing Area (X7)", 0.0, 0.5, step=0.05, key="X7")
    st.selectbox("Glazing Area Distribution (X8)", [0, 1, 2, 3, 4, 5], key="X8")
    predict_clicked = st.button("Predict Cooling Load", type="primary")

values = {k: st.session_state[k] for k in FEATURES}

with right:
    st.subheader("Result")
    if predict_clicked:
        errors, warns = validate(values)
        if errors:
            for e in errors:
                st.error(e)
        else:
            for w in warns:
                st.warning(w)
            y_pred = predict_cooling_load(values)
            grade, perf, colour_name, hex_code = energy_grade(y_pred)
            text_colour = "white" if grade in ("A", "E") else "black"

            st.metric("Predicted Cooling Load", f"{y_pred:.2f} kWh/m2")
            st.markdown(
                f"""<div style="background:{hex_code};padding:18px;border-radius:12px;text-align:center;">
                <span style="font-size:42px;font-weight:700;color:{text_colour};">Grade {grade}</span><br>
                <span style="font-size:20px;color:{text_colour};">{perf} ({colour_name})</span></div>""",
                unsafe_allow_html=True,
            )
            fig = draw_gauge(y_pred)
            st.pyplot(fig)
            plt.close(fig)
    else:
        st.info("Set the building parameters and click 'Predict Cooling Load'.")

    with st.expander("Predictions for the three example designs"):
        rows = []
        for name, p in PRESETS.items():
            if p is None:
                continue
            y_hat = predict_cooling_load(p)
            g, perf, _, _ = energy_grade(y_hat)
            rows.append({"Design": name, "Predicted Cooling Load (kWh/m2)": round(y_hat, 2),
                         "Grade": g, "Performance": perf})
        st.dataframe(pd.DataFrame(rows), hide_index=True)
