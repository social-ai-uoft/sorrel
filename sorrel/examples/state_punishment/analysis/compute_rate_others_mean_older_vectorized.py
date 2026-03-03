"""
Vectorized "mean rate of strictly older agents" per (epoch, agent).
No Python loop over epochs — scales to many epochs.
"""
import numpy as np
import pandas as pd


def compute_rate_others_mean_older_vectorized(df):
    """
    For each row in df (epoch, agent_name, age, resource_A_rate), compute
    rate_others_mean = mean(resource_A_rate) over agents at same epoch with age > this age.

    Expects columns: epoch, agent_name, age, resource_A_rate.
    Returns Series indexed by (epoch, agent_name) with rate_others_mean (NaN when no older agents).
    """
    # 1) Per (epoch, age): sum of rates and count
    agg = df.groupby(["epoch", "age"]).agg(
        sum_rate=("resource_A_rate", "sum"),
        count=("resource_A_rate", "size"),
    ).reset_index()

    # 2) Self-merge: for each (epoch, age), join with (epoch, age_older) where age_older > age
    m = agg.merge(
        agg,
        on="epoch",
        suffixes=("", "_older"),
        how="inner",
    )
    m = m[m["age_older"] > m["age"]]

    # 3) Per (epoch, age): sum of rates and count of *older* agents (right side of merge)
    older = m.groupby(["epoch", "age"]).agg(
        sum_older=("sum_rate_older", "sum"),
        count_older=("count_older", "sum"),
    ).reset_index()

    # 4) Map back to original rows: merge df with older on (epoch, age)
    out = df[["epoch", "agent_name", "age"]].merge(
        older,
        on=["epoch", "age"],
        how="left",
    )
    out["rate_others_mean"] = np.where(
        out["count_older"].fillna(0) > 0,
        out["sum_older"] / out["count_older"],
        np.nan,
    )
    return out.set_index(["epoch", "agent_name"])["rate_others_mean"]


# Usage (replace your current compute + map):
# rate_others_series = compute_rate_others_mean_older_vectorized(
#     agent_names_df[["epoch", "agent_name", "age", "resource_A_rate"]]
# )
# df_rates["rate_others_mean"] = df_rates.set_index(["epoch", "agent_name"]).index.map(
#     rate_others_series.to_dict().get
# )
# Or if df_rates has same index as the result:
# df_rates = df_rates.merge(
#     rate_others_series,
#     left_on=["epoch", "agent_name"],
#     right_index=True,
#     how="left",
# )
