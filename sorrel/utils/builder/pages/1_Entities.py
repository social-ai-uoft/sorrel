import base64
import io

import pandas as pd
import streamlit as st
from PIL import Image

st.set_page_config(page_title="Entities - Sorrel Builder", page_icon="🧱")

st.header("Step 1: Define Entities")

st.markdown(
    """
Define the objects that will populate your world.
- **Name**: The class name (e.g., `Wall`, `Tomato`).
- **Symbol**: A single character used in text representations (optional).
- **Color**: Color for the simple renderer, and for the auto-generated sprite.
- **Reward**: Reward (positive) or penalty (negative) value when interacted with.
- **Sprite**: Auto-generated 16×16 PNG — or upload your own.
- **Properties**: Basic boolean flags.
"""
)


def make_placeholder_png(hex_color: str) -> bytes:
    """Create a 16×16 solid-color PNG and return raw bytes."""
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    img = Image.new("RGB", (16, 16), color=(r, g, b))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def png_preview_b64(png_bytes: bytes) -> str:
    """Return a base64 data-URI for inline <img> preview."""
    return "data:image/png;base64," + base64.b64encode(png_bytes).decode()


# ── Input form for new entity ──────────────────────────────────────────────────
with st.form("new_entity_form"):
    col1, col2 = st.columns(2)
    with col1:
        name = st.text_input("Entity Name (ClassName)", placeholder="e.g. Wall")
        symbol = st.text_input("Symbol (1 char)", max_chars=1, placeholder="#")
        reward = st.number_input(
            "Reward Value",
            value=0.0,
            step=0.5,
            help="Reward given to an agent when it interacts with / steps on this entity. "
            "Use negative values for penalties.",
        )
    with col2:
        color = st.color_picker("Color (also used for auto-sprite)", "#808080")
        uploaded_file = st.file_uploader(
            "Upload PNG sprite (16×16 recommended)",
            type=["png"],
            help="Leave empty to auto-generate a solid-color placeholder.",
        )

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

    # Resolve sprite bytes
    if uploaded_file is not None:
        asset_bytes = uploaded_file.read()
    else:
        asset_bytes = make_placeholder_png(color)

    st.session_state["entities"][name] = {
        "name": name,
        "symbol": symbol or name[0],
        "color": color,
        "reward": reward,
        "passable": is_passable,
        "pickupable": is_pickupable,
        "interactive": is_interactive,
        "type": "custom",
        "asset_bytes": asset_bytes,
        # Custom entities bake value into __init__, so no constructor kwargs needed.
        "init_kwargs": {},
    }
    st.success(f"Added **{name}** with reward value `{reward}`")

# ── Import Default Entities ────────────────────────────────────────────────────
st.divider()
st.subheader("Import Default Entities")
with st.form("import_default"):
    def_entity = st.selectbox("Select Standard Entity", ["Wall", "Gem"])
    import_submitted = st.form_submit_button("Import")

if import_submitted:
    from pathlib import Path

    assets_dir = Path(__file__).resolve().parents[3] / "entities" / "assets"

    if def_entity == "Wall":
        wall_path = assets_dir / "wall.png"
        with open(wall_path, "rb") as f:
            wall_bytes = f.read()
        st.session_state["entities"]["Wall"] = {
            "name": "Wall",
            "symbol": "W",
            "color": "#000000",
            "reward": -1.0,
            "passable": False,
            "pickupable": False,
            "interactive": False,
            "type": "standard",
            "source": "sorrel.entities.basic_entities",
            "asset_bytes": wall_bytes,
            # Wall.__init__ takes no parameters.
            "init_kwargs": {},
        }
        st.success("Imported Wall (reward = -1)")
    elif def_entity == "Gem":
        gem_path = assets_dir / "gem.png"
        with open(gem_path, "rb") as f:
            gem_bytes = f.read()
        st.session_state["entities"]["Gem"] = {
            "name": "Gem",
            "symbol": "G",
            "color": "#00FF00",
            "reward": 1.0,
            "passable": True,
            "pickupable": True,
            "interactive": False,
            "type": "standard",
            "source": "sorrel.entities.basic_entities",
            "asset_bytes": gem_bytes,
            # Gem.__init__ accepts a value parameter.
            "init_kwargs": {"value": 1.0},
        }
        st.success("Imported Gem (reward = 1)")


# ── Display existing entities ──────────────────────────────────────────────────
if st.session_state["entities"]:
    st.divider()
    st.subheader("Defined Entities")

    # Show sprite previews in a grid
    entity_names = list(st.session_state["entities"].keys())
    cols_per_row = 4
    rows = [
        entity_names[i : i + cols_per_row]
        for i in range(0, len(entity_names), cols_per_row)
    ]
    for row in rows:
        grid_cols = st.columns(cols_per_row)
        for col, ename in zip(grid_cols, row):
            ent = st.session_state["entities"][ename]
            with col:
                data_uri = png_preview_b64(ent["asset_bytes"])
                st.markdown(
                    f'<img src="{data_uri}" width="32" style="image-rendering:pixelated"> '
                    f"<b>{ename}</b>",
                    unsafe_allow_html=True,
                )
                st.caption(
                    f"reward={ent.get('reward', 0.0)} | "
                    f"{'passable' if ent['passable'] else 'solid'}"
                )

    st.write("")  # spacing

    # Summary table (without asset_bytes column)
    display_cols = ["symbol", "color", "reward", "passable", "pickupable", "type"]
    df = pd.DataFrame.from_dict(st.session_state["entities"], orient="index")
    available_cols = [c for c in display_cols if c in df.columns]
    st.dataframe(df[available_cols], width="stretch")

    # Option to delete
    to_delete = st.selectbox(
        "Select entity to delete", [""] + list(st.session_state["entities"].keys())
    )
    if to_delete and st.button("Delete Entity"):
        del st.session_state["entities"][to_delete]
        st.rerun()
else:
    st.info("No entities defined yet. Add one above!")
