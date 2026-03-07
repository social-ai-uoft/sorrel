import numpy as np
import pandas as pd
import streamlit as st
from streamlit.column_config import SelectboxColumn

st.set_page_config(page_title="Layout - Sorrel Builder", page_icon="🗺️")

st.header("Step 3: Layout Editor")

st.markdown(
    """
Design your level layout.
1. Set the grid dimensions.
2. Select an entity from the palette.
3. Edit the grid cells using the dropdowns.
4. Configure random spawning and random starting positions below.
"""
)

# ── Grid Configuration ─────────────────────────────────────────────────────────
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
    st.session_state["grid_state"] = pd.DataFrame(
        [["Empty" for _ in range(cols)] for _ in range(rows)],
        columns=pd.Index([f"{i}" for i in range(cols)]),
    )

st.divider()

# ── Entity Palette ────────────────────────────────────────────────────────────
entities = list(st.session_state["entities"].keys())
if not entities:
    st.warning("No entities defined! Go to the 'Entities' tab first.")
    st.stop()

palette_options = ["Empty", "Agent"] + entities
selected_entity = st.radio("Select Tool:", palette_options, horizontal=True)

# ── Map Editor ────────────────────────────────────────────────────────────────
st.subheader("Map Editor")
st.info(f"Currently painting with: **{selected_entity}**")
st.caption(
    "Double click a cell to edit it manually if needed, or select a range and press delete to clear."
)

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
    use_container_width=True,
    hide_index=True,
    height=400,
)

# ── Bulk Paint Tools ──────────────────────────────────────────────────────────
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
            r1, r2 = int(min(r_start, r_end)), int(max(r_start, r_end))
            c1, c2 = int(min(c_start, c_end)), int(max(c_start, c_end))
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
            new_df.iloc[0, :] = "Wall"
            new_df.iloc[rows - 1, :] = "Wall"
            new_df.iloc[:, 0] = "Wall"
            new_df.iloc[:, cols - 1] = "Wall"
            st.session_state["grid_state"] = new_df
            st.rerun()
    else:
        st.caption("Define a 'Wall' entity to use the perimeter tool.")

# ── Random Spawn Configuration ────────────────────────────────────────────────
st.divider()
with st.expander("🎲 Random Spawn Rules", expanded=False):
    st.markdown(
        """
Configure entities that can **randomly appear** in empty cells during the simulation.
Each turn, an `EmptyEntity` cell has a chance to transform into the specified entity type.
"""
    )
    with st.form("add_random_spawn"):
        rs_col1, rs_col2 = st.columns(2)
        with rs_col1:
            spawn_entity = st.selectbox("Entity Type", entities, key="spawn_ent_sel")
        with rs_col2:
            spawn_prob = st.slider(
                "Spawn Probability per Turn",
                min_value=0.0,
                max_value=1.0,
                value=0.01,
                step=0.001,
                format="%.3f",
                help="Probability that an empty cell spawns this entity each turn.",
            )
        add_spawn = st.form_submit_button("Add Spawn Rule")

    if add_spawn:
        new_spawn = {"entity": spawn_entity, "probability": spawn_prob}
        # Avoid duplicate entity entries — overwrite existing rule for same entity
        existing = [
            i
            for i, s in enumerate(st.session_state["random_spawns"])
            if s["entity"] == spawn_entity
        ]
        if existing:
            st.session_state["random_spawns"][existing[0]] = new_spawn
            st.success(
                f"Updated spawn rule for **{spawn_entity}** (prob={spawn_prob:.3f})"
            )
        else:
            st.session_state["random_spawns"].append(new_spawn)
            st.success(f"Added spawn rule: **{spawn_entity}** at prob={spawn_prob:.3f}")

    if st.session_state["random_spawns"]:
        st.subheader("Current Spawn Rules")
        for idx, s in enumerate(st.session_state["random_spawns"]):
            sc1, sc2 = st.columns([4, 1])
            with sc1:
                st.info(
                    f"Empty cell → **{s['entity']}** with probability **{s['probability']:.3f}** per turn"
                )
            with sc2:
                if st.button("Remove", key=f"del_spawn_{idx}"):
                    st.session_state["random_spawns"].pop(idx)
                    st.rerun()
    else:
        st.caption("No random spawn rules defined yet.")

# ── Random Starting Locations ─────────────────────────────────────────────────
with st.expander("📍 Random Starting Locations", expanded=False):
    st.markdown(
        """
Place a fixed **number** of a particular entity type at **random empty cells** at the start of each episode,
instead of manually painting them on the grid.
"""
    )
    with st.form("add_random_start"):
        rl_col1, rl_col2 = st.columns(2)
        with rl_col1:
            start_entity = st.selectbox("Entity Type", entities, key="start_ent_sel")
        with rl_col2:
            start_count = st.number_input(
                "Count",
                min_value=1,
                value=1,
                help="Number of this entity to place randomly at episode start.",
            )
        add_start = st.form_submit_button("Add Random Start")

    if add_start:
        new_start = {"entity": start_entity, "count": int(start_count)}
        existing = [
            i
            for i, s in enumerate(st.session_state["random_starts"])
            if s["entity"] == start_entity
        ]
        if existing:
            st.session_state["random_starts"][existing[0]] = new_start
            st.success(
                f"Updated random start for **{start_entity}** → count={start_count}"
            )
        else:
            st.session_state["random_starts"].append(new_start)
            st.success(f"Added random start: **{start_entity}** × {start_count}")

    if st.session_state["random_starts"]:
        st.subheader("Current Random Starts")
        for idx, s in enumerate(st.session_state["random_starts"]):
            rlc1, rlc2 = st.columns([4, 1])
            with rlc1:
                st.info(
                    f"**{s['count']}** × **{s['entity']}** placed at random empty cells"
                )
            with rlc2:
                if st.button("Remove", key=f"del_start_{idx}"):
                    st.session_state["random_starts"].pop(idx)
                    st.rerun()
    else:
        st.caption("No random starting locations defined yet.")

# ── Save state ────────────────────────────────────────────────────────────────
if not edited_df.equals(st.session_state["grid_state"]):
    st.session_state["grid_state"] = edited_df
    layout_data = []
    for r in range(rows):
        for c in range(cols):
            val = edited_df.iloc[r, c]
            if val and val != "Empty":
                layout_data.append(((r, c), val))
    st.session_state["layout"] = layout_data
    st.success("Layout updated!")
