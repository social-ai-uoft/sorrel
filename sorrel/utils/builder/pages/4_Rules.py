import streamlit as st

st.set_page_config(page_title="Rules - Sorrel Builder", page_icon="📜")

st.header("Step 4: Interaction Rules")

st.markdown(
    """
Define what happens when an agent performs an action on an entity.
Rules take the form:

**"If agent does [action] on [entity], then [outcome(s)]"**

Multiple outcomes can be stacked per rule (e.g. entity disappears *and* agent receives a reward).
"""
)

# Ensure rules list exists (normally initialized in Home.py)
if "rules" not in st.session_state:
    st.session_state["rules"] = []

entities = list(st.session_state["entities"].keys())
actions = st.session_state["agents"].get(
    "actions", ["up", "down", "left", "right", "interact"]
)

if not entities:
    st.warning("Define entities first on the Entities page!")
    st.stop()

if not actions:
    st.warning("Define agent actions first on the Agents page!")
    st.stop()

# ── Rule Builder Form ─────────────────────────────────────────────────────────
st.subheader("Add a Rule")

with st.form("new_rule_form", clear_on_submit=False):
    trig_col, target_col = st.columns(2)
    with trig_col:
        trigger_action = st.selectbox(
            "Agent performs action",
            actions,
            help="Which agent action triggers this rule.",
        )
    with target_col:
        target_entity = st.selectbox(
            "On entity",
            entities,
            help="Which entity type the action is performed on.",
        )

    st.markdown("**Outcomes** (select all that apply)")

    out_col1, out_col2, out_col3 = st.columns(3)
    with out_col1:
        change_type = st.checkbox("Entity changes type")
        new_entity_type = None
        if change_type:
            new_entity_type = st.selectbox(
                "New entity type",
                ["EmptyEntity"] + entities,
                key="new_type_sel",
            )
    with out_col2:
        give_reward = st.checkbox("Agent receives reward / penalty")
        reward_value = 0.0
        if give_reward:
            reward_value = st.number_input(
                "Reward value",
                value=1.0,
                step=0.5,
                key="outcome_reward",
            )
    with out_col3:
        entity_disappears = st.checkbox(
            "Entity disappears (becomes EmptyEntity)",
            help="Overrides 'change type' to always become EmptyEntity.",
        )

    rule_submitted = st.form_submit_button("Add Rule")

if rule_submitted:
    outcomes = []
    if entity_disappears:
        outcomes.append({"type": "disappear"})
    elif change_type and new_entity_type:
        outcomes.append({"type": "change_type", "new_type": new_entity_type})
    if give_reward:
        outcomes.append({"type": "reward", "value": reward_value})

    if not outcomes:
        st.warning("Please select at least one outcome for the rule.")
    else:
        new_rule = {
            "action": trigger_action,
            "target": target_entity,
            "outcomes": outcomes,
        }
        # Check for duplicate trigger + target combo
        duplicate_idx = None
        for i, r in enumerate(st.session_state["rules"]):
            if r["action"] == trigger_action and r["target"] == target_entity:
                duplicate_idx = i
                break

        if duplicate_idx is not None:
            st.session_state["rules"][duplicate_idx] = new_rule
            st.success(
                f"Updated existing rule for **{trigger_action}** on **{target_entity}**."
            )
        else:
            st.session_state["rules"].append(new_rule)
            st.success("Rule added!")

# ── Display Rules ─────────────────────────────────────────────────────────────
if st.session_state["rules"]:
    st.divider()
    st.subheader("Current Rules")

    for idx, rule in enumerate(st.session_state["rules"]):
        with st.expander(
            f"Rule {idx + 1}: **{rule['action']}** on **{rule['target']}**",
            expanded=True,
        ):
            # Build human-readable outcome descriptions
            outcome_strs = []
            for outcome in rule["outcomes"]:
                if outcome["type"] == "disappear":
                    outcome_strs.append("entity **disappears** (→ EmptyEntity)")
                elif outcome["type"] == "change_type":
                    outcome_strs.append(
                        f"entity **changes type** → `{outcome['new_type']}`"
                    )
                elif outcome["type"] == "reward":
                    val = outcome["value"]
                    label = "reward" if val >= 0 else "penalty"
                    outcome_strs.append(f"agent receives **{label}** of `{val:+.1f}`")

            outcome_text = (
                " & ".join(outcome_strs) if outcome_strs else "*(no outcomes)*"
            )
            st.markdown(
                f"If agent does **`{rule['action']}`** on **`{rule['target']}`** → {outcome_text}"
            )

            if st.button("🗑️ Delete Rule", key=f"del_rule_{idx}"):
                st.session_state["rules"].pop(idx)
                st.rerun()
else:
    st.info("No rules defined yet. Add one above!")
