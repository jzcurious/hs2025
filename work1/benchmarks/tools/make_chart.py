import json
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

PATH_TO_RESULT = Path(__file__).parents[2] / "build" / "benchmarks" / "result.json"
OUTPUT_HTML = "chart.html"


def load_data():
    if not PATH_TO_RESULT.exists():
        raise FileNotFoundError("File not found")

    with open(PATH_TO_RESULT, "r") as f:
        data = json.load(f)

    df = pd.DataFrame(data["benchmarks"])
    df["benchmark"] = df["name"].apply(lambda x: x.split("/")[0])
    df["size"] = df["name"].apply(lambda x: int(x.split("/")[1]))
    eigen_df = df[df["benchmark"] == "BM_EigenVectorAddCPU"].sort_values("size")
    cuda_df = df[df["benchmark"] == "BM_OurVectorAddGPU"].sort_values("size")

    if eigen_df.empty or cuda_df.empty:
        raise ValueError("Data not found")

    return {"eigen": eigen_df.empty, "cuda": cuda_df.empty}


def make_chart(data):
    eigen_df, cuda_df = data["eingen"], data["cuda"]
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=eigen_df["size"],
            y=eigen_df["real_time"],
            mode="lines+markers",
            name="Eigen (CPU)",
            line=dict(color="blue"),
            marker=dict(size=6),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=cuda_df["size"],
            y=cuda_df["real_time"],
            mode="lines+markers",
            name="CUDA (GPU)",
            line=dict(color="red"),
            marker=dict(size=6),
        )
    )

    fig.update_layout(
        title="Сравнение производительности сложения векторов",
        xaxis_title="Размер вектора (n)",
        yaxis_title="Время выполнения (мкс)",
        xaxis_type="log",
        yaxis_type="log",
        legend_title="Метод",
        hovermode="x unified",
        template="plotly_white",
        height=600,
        width=1000,
    )

    OUTPUT_HTML = "chart.html"
    fig.write_html(OUTPUT_HTML)
    print(f"График сохранён: {OUTPUT_HTML}")

    return fig


def show_chart(fig):
    try:
        from google.colab import output
        from IPython.display import display, HTML

        display(HTML(fig.to_html()))
    except ImportError:
        import webbrowser

        webbrowser.open(OUTPUT_HTML)


if __name__ == "__main__":
    show_chart(make_chart(load_data()))
