import io
import os
import zipfile

import jinja2
import streamlit as st

st.set_page_config(page_title="Export - Sorrel Builder", page_icon="📦")

st.header("Step 4: Export Environment")

project_name = st.text_input("Project Name (CamelCase)", "MyNewEnv")

if st.button("Generate Code"):
    # Load templates
    template_loader = jinja2.FileSystemLoader(
        searchpath="./sorrel/utils/builder/templates"
    )
    template_env = jinja2.Environment(loader=template_loader)

    # Prepare data context
    context = {
        "project_name": project_name,
        "project_name_lower": project_name.lower(),
        "entities": list(st.session_state["entities"].values()),
        "num_agents": st.session_state["agents"]["count"],
        "vision": st.session_state["agents"]["vision"],
        "actions": st.session_state["agents"]["actions"],
        "base_class": st.session_state["agents"].get("base_class", "Agent"),
        "model": st.session_state["agents"].get("model", "RandomModel"),
        "model_params": st.session_state["agents"].get("model_params", {}),
        "height": st.session_state["grid_size"]["rows"],
        "width": st.session_state["grid_size"]["cols"],
        "layout_data": st.session_state.get("layout", []),
        "rules": st.session_state.get("rules", []),
    }

    # Render files
    files = {}
    templates = ["env.py", "world.py", "entities.py", "agents.py", "main.py"]

    for t_name in templates:
        template = template_env.get_template(f"{t_name}.jinja2")
        files[t_name] = template.render(context)

    st.session_state["generated_files"] = files
    st.session_state["generated_project_name"] = project_name
    st.success("Code generated successfully!")

if "generated_files" in st.session_state:
    files = st.session_state["generated_files"]
    p_name = st.session_state["generated_project_name"]

    # Preview
    st.subheader("Preview: env.py")
    st.code(files["env.py"], language="python")

    # Create Zip
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
        for f_name, content in files.items():
            zip_file.writestr(f"{p_name.lower()}/{f_name}", content)

        # Add basic __init__.py
        zip_file.writestr(f"{p_name.lower()}/__init__.py", "")

    col1, col2 = st.columns(2)

    with col1:
        st.download_button(
            label="Download Project (.zip)",
            data=zip_buffer.getvalue(),
            file_name=f"{p_name.lower()}.zip",
            mime="application/zip",
        )

    with col2:
        if st.button("Install to Examples"):
            try:
                example_dir = os.path.join(".", "sorrel", "examples", p_name.lower())
                if os.path.exists(example_dir):
                    st.error(f"Project '{p_name}' already exists in examples!")
                else:
                    os.makedirs(example_dir)
                    for f_name, content in files.items():
                        with open(os.path.join(example_dir, f_name), "w") as f:
                            f.write(content)
                    with open(os.path.join(example_dir, "__init__.py"), "w") as f:
                        f.write("")
                    st.success(f"Successfully installed '{p_name}' into examples!")
            except Exception as e:
                st.error(f"Error installing to examples: {e}")
