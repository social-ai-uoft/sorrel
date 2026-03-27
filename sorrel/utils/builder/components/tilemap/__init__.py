from pathlib import Path

import streamlit.components.v1 as components

_component_path = str(Path(__file__).parent)
tilemap_editor = components.declare_component("tilemap_editor", path=_component_path)
