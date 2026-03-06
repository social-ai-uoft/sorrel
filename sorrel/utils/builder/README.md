# Sorrel GUI Builder Interface

A web-based interface for creating Sorrel environments without writing code.

## How to run

```bash
streamlit run sorrel/utils/builder/Home.py
```

## How to use

1. **Entities**: Define the objects in your world (walls, items, etc.).
2. **Agents**: Configure your agents (vision, actions).
3. **Layout**: Draw your world grid.
4. **Rules**: Define interaction rules for entities in the environment.
5. **Export**: Generate the Python code for your new environment.

Use the sidebar to navigate between steps.

## How to export

1. Click on the "Export" tab in the sidebar.
2. Click on the "Generate Code" button.
3. You can download the ZIP file including the project, or use the "Install to Examples" to add the project to `sorrel/examples/`.

