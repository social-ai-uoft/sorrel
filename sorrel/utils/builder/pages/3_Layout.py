import base64
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

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
if "map_update_counter" not in st.session_state:
    st.session_state["map_update_counter"] = 0

current_grid_state = st.session_state.get("grid_state", None)
if current_grid_state is None:
    st.session_state["grid_state"] = pd.DataFrame(
        [["Empty" for _ in range(cols)] for _ in range(rows)],
        columns=pd.Index([f"{i}" for i in range(cols)]),
    )
    st.session_state["map_update_counter"] += 1
elif current_grid_state.shape != (rows, cols):
    # Gracefully resize by keeping existing data
    old_rows, old_cols = current_grid_state.shape
    new_df = pd.DataFrame(
        [["Empty" for _ in range(cols)] for _ in range(rows)],
        columns=pd.Index([f"{i}" for i in range(cols)]),
    )
    for r in range(min(old_rows, rows)):
        for c in range(min(old_cols, cols)):
            new_df.iloc[r, c] = current_grid_state.iloc[r, c]
    st.session_state["grid_state"] = new_df
    st.session_state["map_update_counter"] += 1

st.divider()

# ── Entity Palette ────────────────────────────────────────────────────────────
entities = list(st.session_state["entities"].keys())
if not entities:
    st.warning("No entities defined! Go to the 'Entities' tab first.")
    st.stop()

# Prepare assets dictionary (Entity Name -> Base64 Image URL)
assets_dict = {}
for ename, ent_data in st.session_state["entities"].items():
    b64_str = base64.b64encode(ent_data["asset_bytes"]).decode()
    assets_dict[ename] = f"data:image/png;base64,{b64_str}"

from pathlib import Path

sorrel_dir = Path(__file__).resolve().parents[3]
try:
    with open(sorrel_dir / "entities" / "assets" / "empty.png", "rb") as f:
        empty_b64 = base64.b64encode(f.read()).decode()
        assets_dict["Empty"] = f"data:image/png;base64,{empty_b64}"
    with open(sorrel_dir / "entities" / "assets" / "wall.png", "rb") as f:
        wall_b64 = base64.b64encode(f.read()).decode()
        assets_dict["Wall"] = f"data:image/png;base64,{wall_b64}"
    with open(sorrel_dir / "entities" / "assets" / "gem.png", "rb") as f:
        gem_b64 = base64.b64encode(f.read()).decode()
        assets_dict["Gem"] = f"data:image/png;base64,{gem_b64}"
    with open(sorrel_dir / "agents" / "assets" / "hero.png", "rb") as f:
        agent_b64 = base64.b64encode(f.read()).decode()
        assets_dict["Agent"] = f"data:image/png;base64,{agent_b64}"
except Exception as e:
    st.error(f"Failed to load standard assets: {e}")

palette_options = ["Empty", "Agent"] + entities

st.write("**Select Tool:**")
if "selected_tool" not in st.session_state:
    st.session_state["selected_tool"] = "Empty"

cols_per_row = 6
for i in range(0, len(palette_options), cols_per_row):
    row_opts = palette_options[i : i + cols_per_row]
    grid_cols = st.columns(cols_per_row)
    for col, opt in zip(grid_cols, row_opts):
        with col:
            # Render image preview
            if opt in assets_dict:
                st.markdown(
                    f'<div style="text-align:center"><img src="{assets_dict[opt]}" width="32px" style="image-rendering:pixelated; margin-bottom:5px;"></div>',
                    unsafe_allow_html=True,
                )
            elif opt == "Empty":
                st.markdown(
                    f'<div style="text-align:center"><div style="width:32px; height:32px; border:1px solid #ccc; background-color:#eee; display:inline-block; margin-bottom:5px;"></div></div>',
                    unsafe_allow_html=True,
                )
            else:
                # Absolute fallback just in case
                st.markdown(
                    f'<div style="text-align:center"><div style="width:32px; height:32px; border:1px solid #ccc; background-color:#eee; display:inline-block; margin-bottom:5px;"></div></div>',
                    unsafe_allow_html=True,
                )

            # Selection button
            is_selected = st.session_state["selected_tool"] == opt
            if st.button(
                opt,
                key=f"sel_{opt}",
                type="primary" if is_selected else "secondary",
                use_container_width=True,
            ):
                st.session_state["selected_tool"] = opt
                st.rerun()

selected_entity = st.session_state["selected_tool"]

# ── Map Editor ────────────────────────────────────────────────────────────────
st.subheader("Map Editor")
st.info(f"Currently painting with: **{selected_entity}**")
st.caption("Click and drag across the grid to paint tiles quickly.")

# Convert grid_state DataFrame to 2D list of strings for JS
grid_list = st.session_state["grid_state"].values.tolist()

from sorrel.utils.builder.components.tilemap import tilemap_editor

# Render the custom web component and get the result
new_grid = tilemap_editor(
    grid_state=grid_list,
    assets=assets_dict,
    rows=rows,
    cols=cols,
    selected_tool=selected_entity,
    key=f"tilemap_{st.session_state['map_update_counter']}",
)

if (
    new_grid is not None
    and len(new_grid) == rows
    and (rows == 0 or len(new_grid[0]) == cols)
):
    # Convert list of lists back to DataFrame
    edited_df = pd.DataFrame(new_grid, columns=pd.Index([f"{i}" for i in range(cols)]))
else:
    edited_df = st.session_state["grid_state"]

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
            st.session_state["map_update_counter"] += 1
            st.rerun()

    if st.button(f"Fill Entire Grid with '{selected_entity}'", type="secondary"):
        new_df = st.session_state["grid_state"].copy()
        new_df[:] = selected_entity
        st.session_state["grid_state"] = new_df
        st.session_state["map_update_counter"] += 1
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
            st.session_state["map_update_counter"] += 1
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
    st.rerun()
