import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Layout - Sorrel Builder", page_icon="🗺️")

st.header("Step 3: Layout Editor")

st.markdown(
    """
Design your level layout.
1. Set the grid dimensions.
2. Select an entity from the palette.
3. Edit the grid cells internally.
"""
)

# Grid Configuration
col1, col2 = st.columns(2)
with col1:
    rows = st.number_input(
        "Rows", min_value=3, max_value=50, value=st.session_state["grid_size"]["rows"]
    )
with col2:
    cols = st.number_input(
        "Cols", min_value=3, max_value=50, value=st.session_state["grid_size"]["cols"]
    )

st.session_state["grid_size"]["rows"] = rows
st.session_state["grid_size"]["cols"] = cols

# Initialize grid if size changed or not exists
current_grid_state = st.session_state.get("grid_state", None)
if current_grid_state is None or current_grid_state.shape != (rows, cols):
    # Create empty grid
    st.session_state["grid_state"] = pd.DataFrame(
        [["Empty" for _ in range(cols)] for _ in range(rows)],
        columns=pd.Index([f"{i}" for i in range(cols)]),
    )

st.divider()

# Entity Palette
entities = list(st.session_state["entities"].keys())
if not entities:
    st.warning("No entities defined! Go to the 'Entities' tab first.")
    st.stop()

# Add "Agent" and "Empty" to palette
palette_options = ["Empty", "Agent"] + entities
selected_entity = st.radio("Select Tool:", palette_options, horizontal=True)

st.subheader("Map Editor")
st.info(f"Currently painting with: **{selected_entity}**")
st.caption(
    "Double click a cell to edit it manually if needed, or select a range and press delete to clear."
)

# Data Editor
# We use a dataframe where users can type the entity name.
# Ideally we'd have a click-to-paint, but Streamlit data_editor is cell-based.
# We can use a ColumnConfig to make it a dropdown!

from streamlit.column_config import SelectboxColumn

column_config = {
    col: SelectboxColumn(
        label=col,
        options=palette_options,
        required=True,
    )
    for col in st.session_state["grid_state"].columns
}

edited_df = st.data_editor(
    st.session_state["grid_state"],
    column_config=column_config,
    width="stretch",
    hide_index=True,
    height=400,
)

# Bulk Update Tools
with st.expander("Bulk Paint Tools", expanded=False):
    st.write("Paint a region with the selected tool.")
    col_a, col_b, col_c, col_d, col_e = st.columns([1, 1, 1, 1, 2])
    with col_a:
        r_start = st.number_input("Start Row", 0, rows - 1, 0)
    with col_b:
        r_end = st.number_input("End Row", 0, rows - 1, rows - 1)
    with col_c:
        c_start = st.number_input("Start Col", 0, cols - 1, 0)
    with col_d:
        c_end = st.number_input("End Col", 0, cols - 1, cols - 1)
    with col_e:
        st.write("")  # Spacer
        if st.button(f"Fill Rect with '{selected_entity}'"):
            # Update dataframe
            # Ensure indices are valid
            r1, r2 = int(min(r_start, r_end)), int(max(r_start, r_end))
            c1, c2 = int(min(c_start, c_end)), int(max(c_start, c_end))

            # Apply update
            new_df = st.session_state["grid_state"].copy()
            for r in range(r1, r2 + 1):
                for c in range(c1, c2 + 1):
                    new_df.iloc[r, c] = selected_entity
            st.session_state["grid_state"] = new_df
            st.rerun()

    if st.button(f"Fill Entire Grid with '{selected_entity}'", type="secondary"):
        st.session_state["grid_state"][:] = selected_entity
        st.rerun()

    # Paint Outer Walls Tool
    if "Wall" in entities:
        if st.button("Paint Outer Walls"):
            new_df = st.session_state["grid_state"].copy()
            # Top and Bottom rows
            new_df.iloc[0, :] = "Wall"
            new_df.iloc[rows - 1, :] = "Wall"
            # Left and Right columns
            new_df.iloc[:, 0] = "Wall"
            new_df.iloc[:, cols - 1] = "Wall"

            st.session_state["grid_state"] = new_df
            st.rerun()
    else:
        st.caption("Define a 'Wall' entity to use the perimeter tool.")

# Save state
if not edited_df.equals(st.session_state["grid_state"]):
    st.session_state["grid_state"] = edited_df
    # Extract locations logic (convert df to list of tuples)
    layout_data = []
    for r in range(rows):
        for c in range(cols):
            val = edited_df.iloc[r, c]
            if val and val != "Empty":
                layout_data.append(((r, c), val))
    st.session_state["layout"] = layout_data
    st.success("Layout updated!")
