# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import marimo

__generated_with = "0.23.14"
app = marimo.App(width="columns")


@app.cell
def _():
    import polars as pl

    return (pl,)


@app.cell
def _(pl):
    import json

    with open("outputs/results_M1_S0.001_BW8.json") as f:
        data = json.load(f)

    df = pl.DataFrame([{"index": int(idx), **stats} for idx, stats in data.items()])

    df
    return (df,)


@app.cell
def _(df, pl):
    metrics = ["sens_rel_mse", "sens_rel_l2", "sens_raw_mse", "sens_raw_l2"]

    df.with_columns(
        [
            (pl.col(c) - pl.col(c).min()) / (pl.col(c).max() - pl.col(c).min()).alias(c)
            for c in metrics
        ]
    ).unpivot(
        index="index",
        on=metrics,
        variable_name="metric",
        value_name="value",
    ).plot.line(x="index", y="value", color="metric")
    return


@app.cell
def _(df):
    bws = ["bw_rel_mse", "bw_rel_l2", "bw_raw_mse", "bw_raw_l2"]

    df.unpivot(
        index="index",
        on=bws,
        variable_name="metric",
        value_name="value",
    ).plot.line(x="index", y="value", color="metric")
    return


@app.cell
def _():
    from transformers import AutoConfig

    return (AutoConfig,)


@app.cell
def _(AutoConfig):
    AutoConfig.from_pretrained("google/gemma-4-31B-it")
    return


if __name__ == "__main__":
    app.run()
