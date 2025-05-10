import json
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
from argparse import ArgumentParser, ArgumentTypeError
from functools import partial

PATH_TO_RESULT = (
    Path(__file__).parents[3] / "build" / "work1" / "benchmarks" / "result.json"
)
PATH_TO_CHART = "chart.html"


def parse_data(path_to_result: Path = PATH_TO_RESULT) -> dict:
    if not path_to_result.exists():
        raise FileNotFoundError("File not found")

    with open(path_to_result, "r") as f:
        data = json.load(f)

    df = pd.DataFrame(data["benchmarks"])
    df["benchmark"] = df["name"].apply(lambda x: x.split("/")[0])
    df["size"] = df["name"].apply(lambda x: int(x.split("/")[1]))

    complexity = {}

    for benchmark in df["benchmark"]:
        if benchmark not in complexity:
            complexity[benchmark] = df[df["benchmark"] == benchmark].sort_values(
                "size"
            )  # pyright: ignore

    if len(complexity) == 0:
        raise ValueError("Data not found")

    return complexity


def make_chart(
    complexity: dict,
    path_to_chart=PATH_TO_CHART,
    width=1000,
    height=600,
    xaxis_log=True,
    yaxis_log=True,
    dark=False,
) -> go.Figure:
    fig = go.Figure()

    for benchmark, df in complexity.items():
        fig.add_trace(
            go.Scatter(
                x=df["size"],
                y=df["real_time"],
                mode="lines+markers",
                name=benchmark,
                marker=dict(size=6),
            )
        )

    fig.update_layout(
        title="Real Complexity",
        title_x=0.5,
        xaxis_title="N",
        yaxis_title=f"Time, {list(complexity.values())[0].time_unit[0]}",
        xaxis_type="log" if xaxis_log else "linear",
        yaxis_type="log" if yaxis_log else "linear",
        legend_title="Case",
        hovermode="x unified",
        template="plotly_dark" if dark else "plotly_white",
        width=width,
        height=height,
    )

    fig.write_html(path_to_chart)
    print(f"The chart file has been saved to {path_to_chart}.")

    return fig


def show_chart(fig, path_to_chart=PATH_TO_CHART):
    try:
        from google.colab import output  # noqa # pyright: ignore
        from IPython.display import display, HTML  # pyright: ignore

        display(HTML(fig.to_html()))
    except ImportError:
        import webbrowser

        webbrowser.open(path_to_chart)


def range_limited_int(min_value, max_value, value) -> int:
    ivalue = int(value)
    if not min_value <= ivalue <= max_value:
        raise ArgumentTypeError(
            f"Value must be between {min_value} and {max_value}, got {ivalue}"
        )
    return ivalue


if __name__ == "__main__":
    argparser = ArgumentParser()

    argparser.add_argument(
        "-r",
        "--result",
        type=str,
        default=str(PATH_TO_RESULT),
        help="Path to benchmark results",
    )

    argparser.add_argument(
        "-c",
        "--chart",
        type=str,
        default=str(PATH_TO_CHART),
        help="Output path for the chart file",
    )

    argparser.add_argument(
        "--xlog",
        action="store_true",
        default=True,
        help="Log scale on X axis",
    )

    argparser.add_argument(
        "--ylog",
        action="store_true",
        default=True,
        help="Log scale on Y axis",
    )

    argparser.add_argument(
        "-wx",
        "--width",
        type=partial(range_limited_int, min_value=100, max_value=1920),
        default=1000,
        help="Width of the chart in pixels",
    )

    argparser.add_argument(
        "-hy",
        "--height",
        type=partial(range_limited_int, min_value=100, max_value=1080),
        default=600,
        help="Height of the chart in pixels",
    )

    argparser.add_argument(
        "--dark",
        action="store_true",
        default=False,
        help="Use dark theme",
    )

    args = argparser.parse_args()

    show_chart(
        make_chart(
            parse_data(Path(args.result)),
            args.chart,
            args.width,
            args.height,
            args.xlog,
            args.ylog,
            args.dark,
        ),
        args.chart,
    )
