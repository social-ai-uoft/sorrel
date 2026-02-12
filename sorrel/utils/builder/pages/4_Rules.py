import pandas as pd
import streamlit as st

st.set_page_config(page_title="Rules - Sorrel Builder", page_icon="📜")

st.header("Step 4: Interaction Rules")

st.markdown(
    """
Define what happens when an agent interacts with an entity.
Currently supporting simple transformations:
**"If Agent interacts with [Target], it becomes [Result]"**
"""
)

if "rules" not in st.session_state:
    st.session_state["rules"] = []

entities = list(st.session_state["entities"].keys())

if not entities:
    st.warning("Define entities first!")
    st.stop()

with st.form("new_rule"):
    col1, col2, col3 = st.columns([2, 1, 2])
    with col1:
        target = st.selectbox("Target Entity", entities)
    with col2:
        st.markdown("### ➡️")
    with col3:
        result = st.selectbox("Becomes", entities + ["Empty"])

    submitted = st.form_submit_button("Add Rule")

if submitted:
    rule = {"target": target, "result": result}
    # Avoid duplicates
    if rule not in st.session_state["rules"]:
        st.session_state["rules"].append(rule)
        st.success("Rule added!")
    else:
        st.warning("Rule already exists.")

# Display rules
if st.session_state["rules"]:
    st.divider()
    st.subheader("Current Rules")

    # Simple list display
    for idx, rule in enumerate(st.session_state["rules"]):
        col1, col2 = st.columns([4, 1])
        with col1:
            st.info(
                f"Interact with **{rule['target']}** ➔ Becomes **{rule['result']}**"
            )
        with col2:
            if st.button("Delete", key=f"del_{idx}"):
                st.session_state["rules"].pop(idx)
                st.rerun()
