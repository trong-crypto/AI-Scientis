import ast
import json
import math
import os
import re
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ai_scientist.llm import get_response_from_llm


def extract_python_code(text: str) -> str:
    match = re.search(r"```(?:python)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    return (match.group(1) if match else text).strip()


def ast_security_guardrail(code_str: str) -> tuple[bool, str]:
    """Catch simple metric fabrication; this is not a security sandbox."""
    try:
        tree = ast.parse(code_str)
    except SyntaxError as exc:
        return False, f"Mã Python không hợp lệ: {exc}"
    metric_names = {"accuracy", "acc", "f1_score", "precision", "recall"}
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            value = node.value
            if any(isinstance(t, ast.Name) and t.id in metric_names for t in targets):
                if isinstance(value, ast.Constant) and isinstance(value.value, (int, float)):
                    return False, "Chỉ số không được gán bằng hằng số."
    return True, ""


def load_results(path: str) -> dict:
    if not os.path.isfile(path):
        raise ValueError("Thực nghiệm không tạo results.json.")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict) or not data:
        raise ValueError("results.json phải là một object chứa các chỉ số số học.")
    for name, value in data.items():
        if not isinstance(name, str) or not name:
            raise ValueError("Tên chỉ số trong results.json không hợp lệ.")
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"Chỉ số {name} phải là số hữu hạn.")
        if not 0 <= value <= 1:
            raise ValueError(f"Chỉ số {name} phải nằm trong khoảng 0–1.")
    return data


def plot_results(results: dict, output_path: str) -> None:
    names = list(results)
    values = list(results.values())
    fig, ax = plt.subplots(figsize=(max(6, 1.4 * len(names)), 4.5))
    ax.bar(names, values, color="#2B5C8F")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Measured experiment metrics")
    ax.tick_params(axis="x", rotation=15)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def perform_experiments(idea_data, model, workspace_dir="./workspace", experiment_cwd=None):
    if not isinstance(idea_data, dict):
        raise ValueError("Dữ liệu ý tưởng không hợp lệ.")
    workspace_dir = os.path.abspath(workspace_dir)
    os.makedirs(workspace_dir, exist_ok=True)
    exp_path = os.path.join(workspace_dir, "experiment.py")
    res_path = os.path.join(workspace_dir, "results.json")
    if os.path.isfile(exp_path):
        with open(exp_path, encoding="utf-8") as f:
            base_code = f.read()
    else:
        base_code = ""
    experiment_cwd = os.path.abspath(experiment_cwd or os.getcwd())
    prompt = f"""Sửa thí nghiệm Python dưới đây theo ý tưởng: {idea_data.get('experiment_modification', '')}
Mã gốc:
```python
{base_code}
```
Chạy trên dữ liệu thật. Tính chỉ số từ nhãn thật và dự đoán; không bịa số, không gán chỉ số cố định.
Thư mục làm việc khi chạy: {experiment_cwd!r}. Dữ liệu với đường dẫn tương đối được đọc từ đây.
Nếu dữ liệu hoặc thư viện cần thiết không có, hãy báo lỗi rõ ràng trong lúc chạy.
Ghi JSON vào đường dẫn trong biến môi trường AI_SCIENTIST_RESULTS_PATH ({res_path!r}), dạng {{"accuracy": 0.0, "f1_score": 0.0}} với các giá trị thực tính được trong khoảng 0 đến 1.
Chỉ trả về mã Python trong khối ```python ... ```.
"""
    run_env = os.environ.copy()
    run_env["AI_SCIENTIST_RESULTS_PATH"] = res_path
    last_error = ""
    for attempt in range(3):
        text, _ = get_response_from_llm(prompt, model)
        code = extract_python_code(text)
        valid, reason = ast_security_guardrail(code)
        if not valid:
            last_error = reason
        else:
            if os.path.exists(res_path):
                os.remove(res_path)
            with open(exp_path, "w", encoding="utf-8") as f:
                f.write(code)
            try:
                result = subprocess.run(
                    [sys.executable, exp_path], cwd=experiment_cwd,
                    capture_output=True, text=True, encoding="utf-8", errors="replace",
                    timeout=600, check=False, env=run_env,
                )
                if result.returncode != 0:
                    last_error = (result.stderr or result.stdout)[-3000:]
                else:
                    data = load_results(res_path)
                    plot_results(data, os.path.join(workspace_dir, "metrics_plot.png"))
                    return data
            except (subprocess.TimeoutExpired, ValueError, json.JSONDecodeError) as exc:
                last_error = str(exc)
        prompt += f"\nLần thử trước thất bại: {last_error}\nHãy sửa và trả lại toàn bộ mã Python."
    raise RuntimeError(f"Thực nghiệm không hoàn tất sau 3 lần thử: {last_error}")
