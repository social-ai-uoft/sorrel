import streamlit as st

st.set_page_config(
    page_title="Sorrel Environment Builder",
    page_icon="🍳",
    layout="wide",
)

st.title("Sorrel Environment Builder")

st.markdown(
    """
Welcome to the Sorrel Environment Builder! This tool helps you create new Sorrel environments without writing code.

### Workflow:
1. **Entities**: Define the objects in your world (walls, items, etc.).
2. **Agents**: Configure your agents (vision, actions).
3. **Layout**: Draw your world grid, set random spawning rules.
4. **Rules**: Define interaction rules for entities in the environment.
5. **Logging**: Configure what metrics to record during training.
6. **Export**: Generate the Python code for your new environment.

Use the sidebar to navigate between steps.
"""
)

# Initialize session state for data persistence across pages
if "entities" not in st.session_state:
    st.session_state["entities"] = {}
if "agents" not in st.session_state:
    st.session_state["agents"] = {
        "count": 1,
        "vision": 5,
        "actions": ["up", "down", "left", "right", "interact"],
    }
if "layout" not in st.session_state:
    st.session_state["layout"] = {}  # (row, col) -> entity_name
if "grid_size" not in st.session_state:
    st.session_state["grid_size"] = {"rows": 10, "cols": 10}
if "rules" not in st.session_state:
    st.session_state["rules"] = []
if "random_spawns" not in st.session_state:
    st.session_state["random_spawns"] = []  # [{"entity": str, "probability": float}]
if "random_starts" not in st.session_state:
    st.session_state["random_starts"] = []  # [{"entity": str, "count": int}]
if "logging" not in st.session_state:
    st.session_state["logging"] = {
        "logger_type": "TensorboardLogger",
        "measures": [],
    }
