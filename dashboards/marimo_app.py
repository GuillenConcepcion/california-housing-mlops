import marimo

__generated_with = "0.8.0"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import plotly.express as px
    return mo, pd, np, px


@app.cell
def __(mo):
    mo.md(
        r"""
        # 🧠 Ajuste de Hiperparametros Vivienda California
        ### Interactive Marimo Analytics Notebook
        **Lead Architect:** Guillén Concepción | *Odysseus Framework*
        """
    )
    return


@app.cell
def __(np, pd, px):
    df_demo = pd.DataFrame({
        "x": np.linspace(0, 10, 100),
        "y": np.sin(np.linspace(0, 10, 100)) + np.random.normal(0, 0.1, 100)
    })
    fig_marimo = px.line(df_demo, x="x", y="y", title="Marimo Reactive Stream", template="plotly_dark")
    fig_marimo
    return df_demo, fig_marimo


if __name__ == "__main__":
    app.run()
