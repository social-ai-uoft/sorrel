import streamlit as st

st.set_page_config(page_title="Logging - Sorrel Builder", page_icon="📊")

st.header("Step 5: Logging Configuration")

st.markdown(
    """
Configure what data gets recorded during your experiment.

Sorrel's loggers always record **loss**, **reward**, and **epsilon** by default.
You can select additional custom metrics to track here.
"""
)

# Ensure logging config exists (normally initialized in Home.py)
if "logging" not in st.session_state:
    st.session_state["logging"] = {
        "logger_type": "TensorboardLogger",
        "measures": [],
    }

current = st.session_state["logging"]

# ── Logger Type ───────────────────────────────────────────────────────────────
st.subheader("Logger Type")

LOGGER_OPTIONS = ["TensorboardLogger", "ConsoleLogger", "JupyterLogger"]
LOGGER_DESCRIPTIONS = {
    "TensorboardLogger": "Writes scalars to a TensorBoard event file. Best for tracking long experiments.",
    "ConsoleLogger": "Prints a summary table to the terminal each epoch. Good for quick sanity checks.",
    "JupyterLogger": "Displays live updates in a Jupyter notebook cell.",
}

logger_type = st.selectbox(
    "Select Logger",
    LOGGER_OPTIONS,
    index=LOGGER_OPTIONS.index(current.get("logger_type", "TensorboardLogger")),
)
st.caption(LOGGER_DESCRIPTIONS[logger_type])

if logger_type == "TensorboardLogger":
    log_dir = st.text_input(
        "Log Directory (relative to project root)",
        value=current.get("log_dir", "data/logs"),
        help="TensorBoard event files are written here.",
    )
    st.session_state["logging"]["log_dir"] = log_dir

st.session_state["logging"]["logger_type"] = logger_type

# ── Default Measures ──────────────────────────────────────────────────────────
st.divider()
st.subheader("Default Logged Values")
st.info(
    "The following are **always** recorded by every logger and require no configuration:\n"
    "- `loss` — model training loss\n"
    "- `reward` — cumulative reward per epoch\n"
    "- `epsilon` — exploration rate (for applicable models)"
)

# ── Custom Measures ───────────────────────────────────────────────────────────
st.divider()
st.subheader("Additional Custom Measures")

st.markdown(
    "Add any extra scalar values to log each epoch. "
    "These will be passed as named keys to `logger.record_turn(...)` in the generated code."
)

entities = list(st.session_state["entities"].keys())
actions = st.session_state["agents"].get("actions", [])

# Suggestion helpers
with st.expander("💡 Suggested Measures", expanded=True):
    suggest_col1, suggest_col2 = st.columns(2)

    with suggest_col1:
        st.markdown("**Per-action counts:**")
        for action in actions:
            measure_key = f"action_count_{action}"
            if st.button(f"+ {measure_key}", key=f"sug_act_{action}"):
                if measure_key not in st.session_state["logging"]["measures"]:
                    st.session_state["logging"]["measures"].append(measure_key)
                    st.rerun()

    with suggest_col2:
        st.markdown("**Per-entity counts:**")
        for ename in entities:
            measure_key = f"entity_count_{ename.lower()}"
            if st.button(f"+ {measure_key}", key=f"sug_ent_{ename}"):
                if measure_key not in st.session_state["logging"]["measures"]:
                    st.session_state["logging"]["measures"].append(measure_key)
                    st.rerun()

# Custom free-form measure
with st.form("add_custom_measure"):
    custom_measure = st.text_input(
        "Custom measure name",
        placeholder="e.g.  collisions, deliveries_completed",
        help="Must be a valid Python identifier (no spaces; use underscores).",
    )
    add_custom = st.form_submit_button("Add Custom Measure")

if add_custom and custom_measure.strip():
    key = custom_measure.strip().replace(" ", "_")
    if key not in st.session_state["logging"]["measures"]:
        st.session_state["logging"]["measures"].append(key)
        st.success(f"Added measure: `{key}`")
    else:
        st.warning("Measure already in list.")

# ── Display configured measures ───────────────────────────────────────────────
measures = st.session_state["logging"].get("measures", [])
if measures:
    st.divider()
    st.subheader("Configured Measures")
    for idx, m in enumerate(measures):
        mc1, mc2 = st.columns([5, 1])
        with mc1:
            st.code(m)
        with mc2:
            if st.button("🗑️", key=f"del_measure_{idx}"):
                st.session_state["logging"]["measures"].pop(idx)
                st.rerun()
else:
    st.caption("No additional measures configured — only defaults will be logged.")

# ── Preview ───────────────────────────────────────────────────────────────────
st.divider()
st.subheader("Configuration Preview")
st.json(st.session_state["logging"])
