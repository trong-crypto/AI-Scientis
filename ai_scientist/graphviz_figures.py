"""Render source-attributed method/task links as a Graphviz diagram."""

import os
from pathlib import Path
import shutil
import subprocess


def _dot_executable():
    configured = os.environ.get("GRAPHVIZ_DOT", "").strip()
    candidates = [configured, shutil.which("dot")]
    if os.name == "nt":
        candidates.extend(str(Path(root) / "Graphviz" / "bin" / "dot.exe")
                          for root in (os.environ.get("ProgramFiles", r"C:\Program Files"),
                                       os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")))
    return next((path for path in candidates if path and Path(path).is_file()), None)


def render_method_task_map(entries, output_path):
    """Return True when Graphviz renders a PNG, False if its runtime is absent."""
    try:
        from graphviz import Digraph
    except ImportError:
        return False
    executable = _dot_executable()
    if not executable:
        return False

    graph = Digraph("method_task_map", format="png", engine="dot")
    graph.attr(rankdir="LR", bgcolor="white", pad="0.25", nodesep="0.35",
               ranksep="1.0", splines="polyline")
    graph.attr("node", shape="box", style="rounded,filled", fontname="Arial",
               fontsize="11", color="#2B5C8F", fillcolor="#E7F0F8")
    methods = list(dict.fromkeys(item["method"] for item in entries))
    tasks = list(dict.fromkeys(item["task"] for item in entries))
    for index, method in enumerate(methods):
        graph.node(f"m{index}", method)
    graph.attr("node", color="#327866", fillcolor="#E9F4EF")
    for index, task in enumerate(tasks):
        graph.node(f"t{index}", task)
    for item in entries:
        graph.edge(f"m{methods.index(item['method'])}",
                   f"t{tasks.index(item['task'])}", label=str(item["count"]),
                   color="#62788A", fontname="Arial", fontsize="10")
    try:
        rendered = subprocess.run([executable, "-Tpng"], input=graph.source.encode("utf-8"),
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                  check=True, timeout=30)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
        return False
    with open(output_path, "wb") as image:
        image.write(rendered.stdout)
    with open(str(output_path)[:-4] + ".gv", "w", encoding="utf-8") as source:
        source.write(graph.source)
    return True
