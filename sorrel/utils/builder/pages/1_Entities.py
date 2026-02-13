import pandas as pd
import streamlit as st

st.set_page_config(page_title="Entities - Sorrel Builder", page_icon="🧱")

st.header("Step 1: Define Entities")

st.markdown(
    """
Define the objects that will populate your world.
- **Name**: The class name (e.g., `Wall`, `Tomato`).
- **Symbol**: A single character used in text representations (optional).
- **Color**: Color for the simple renderer.
- **Properties**: Basic boolean flags.
"""
)

# Input form for new entity
with st.form("new_entity_form"):
    col1, col2 = st.columns(2)
    with col1:
        name = st.text_input("Entity Name (ClassName)", placeholder="e.g. Wall")
        symbol = st.text_input("Symbol (1 char)", max_chars=1, placeholder="#")
    with col2:
        color = st.color_picker("Color", "#808080")

    st.subheader("Properties")
    p_col1, p_col2, p_col3 = st.columns(3)
    with p_col1:
        is_passable = st.checkbox("Passable (agents can walk through)", value=False)
    with p_col2:
        is_pickupable = st.checkbox("Pickupable", value=False)
    with p_col3:
        is_interactive = st.checkbox("Interactive", value=False)

    submitted = st.form_submit_button("Add Entity")

if submitted and name:
    if name in st.session_state["entities"]:
        st.warning(f"Entity '{name}' already exists! Overwriting.")

    st.session_state["entities"][name] = {
        "name": name,
        "symbol": symbol or name[0],
        "color": color,
        "passable": is_passable,
        "pickupable": is_pickupable,
        "interactive": is_interactive,
        "type": "custom",
    }
    st.success(f"Added {name}")

st.divider()
st.subheader("Import Default Entities")
with st.form("import_default"):
    def_entity = st.selectbox("Select Standard Entity", ["Wall", "Gem"])
    import_submitted = st.form_submit_button("Import")

if import_submitted:
    if def_entity == "Wall":
        st.session_state["entities"]["Wall"] = {
            "name": "Wall",
            "symbol": "W",
            "color": "#000000",
            "passable": False,
            "pickupable": False,
            "interactive": False,
            "type": "standard",
            "source": "sorrel.entities.basic_entities",
        }
        st.success("Imported Wall")
    elif def_entity == "Gem":
        st.session_state["entities"]["Gem"] = {
            "name": "Gem",
            "symbol": "G",
            "color": "#00FF00",
            "passable": True,
            "pickupable": True,
            "interactive": False,
            "type": "standard",
            "source": "sorrel.entities.basic_entities",
        }
        st.success("Imported Gem")

# Display existing entities
if st.session_state["entities"]:
    st.divider()
    st.subheader("Defined Entities")

    # Convert to DataFrame for display
    df = pd.DataFrame.from_dict(st.session_state["entities"], orient="index")
    st.dataframe(
        df[["symbol", "color", "passable", "pickupable", "type"]],
        width="stretch",
    )

    # Option to delete
    to_delete = st.selectbox(
        "Select entity to delete", [""] + list(st.session_state["entities"].keys())
    )
    if to_delete and st.button("Delete Entity"):
        del st.session_state["entities"][to_delete]
        st.rerun()
else:
    st.info("No entities defined yet. Add one above!")
