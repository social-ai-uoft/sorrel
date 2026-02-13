import streamlit as st

st.set_page_config(page_title="Agents - Sorrel Builder", page_icon="🤖")

st.header("Step 2: Configure Agents")

st.markdown("Configure the capabilities and observation space of your agents.")

current_config = st.session_state["agents"]

with st.expander("General Settings", expanded=True):
    num_agents = st.number_input(
        "Number of Agents", min_value=1, value=current_config["count"]
    )
    vision_radius = st.slider(
        "Vision Radius", min_value=1, max_value=20, value=current_config["vision"]
    )
    base_class = st.selectbox(
        "Base Agent Class",
        ["MovingAgent"],
        index=(
            0 if current_config.get("base_class", "MovingAgent") == "MovingAgent" else 1
        ),
        help="MovingAgent includes default movement logic and sprites.",
    )

    # Model Selection
    model_type = st.selectbox(
        "Agent Model",
        ["RandomModel", "PyTorchIQN", "HumanPlayer"],
        index=["RandomModel", "PyTorchIQN", "HumanPlayer"].index(
            current_config.get("model", "RandomModel")
        ),
        help="Select the decision-making model for the agent.",
    )

    model_params = {}
    if model_type == "PyTorchIQN":
        # Simplified IQN params
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            model_params["n_frames"] = st.number_input(
                "Frames to Stack",
                1,
                10,
                current_config.get("model_params", {}).get("n_frames", 1),
            )
            model_params["layer_size"] = st.number_input(
                "Layer Size",
                32,
                512,
                current_config.get("model_params", {}).get("layer_size", 256),
            )
        with col_m2:
            model_params["n_quantiles"] = st.number_input(
                "Quantiles",
                1,
                50,
                current_config.get("model_params", {}).get("n_quantiles", 8),
            )
            model_params["epsilon"] = st.number_input(
                "Initial Epsilon",
                0.0,
                1.0,
                current_config.get("model_params", {}).get("epsilon", 0.05),
            )
    elif model_type == "RandomModel":
        model_params["memory_size"] = 1000  # Default
    elif model_type == "HumanPlayer":
        model_params["memory_size"] = 1000

st.subheader("Action Space")
possible_actions = [
    "up",
    "down",
    "left",
    "right",
    "stay",
    "interact",
    "pick",
    "place",
    "cook",
    "serve",
    "wash",
]

selected_actions = st.multiselect(
    "Available Actions",
    options=possible_actions,
    default=current_config.get("actions", ["up", "down", "left", "right"]),
)

# Parse custom actions if needed
custom_action = st.text_input("Add Custom Action (optional)")
if st.button("Add Custom Action") and custom_action:
    if custom_action not in selected_actions:
        selected_actions.append(custom_action)
        st.success(f"Added {custom_action}")

# Save to session state
st.session_state["agents"]["count"] = num_agents
st.session_state["agents"]["vision"] = vision_radius
st.session_state["agents"]["actions"] = selected_actions
st.session_state["agents"]["base_class"] = base_class
st.session_state["agents"]["model"] = model_type
st.session_state["agents"]["model_params"] = model_params

st.success("Agent configuration saved!")

st.divider()
st.subheader("Preview")
st.json(st.session_state["agents"])
