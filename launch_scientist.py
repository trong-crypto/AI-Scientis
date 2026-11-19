import os
import json
import time
import argparse
import sys
import uuid

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")
from ai_scientist.llm import create_client
from ai_scientist.generate_ideas import generate_ideas
from ai_scientist.research_evidence import collect_evidence
from ai_scientist.research_ideas import plan_and_run
from ai_scientist.perform_writeup import perform_writeup


def run_autonomous_scientific_research(topic: str, workspace_dir: str = "./workspace", *,
                                       data_path=None, target=None, task=None):
    if not topic or not topic.strip():
        raise ValueError("Chủ đề nghiên cứu không được để trống.")
    if data_path and (not target or task not in
                      {"classification", "regression", "text_classification"}):
        raise ValueError("CSV cần cột đích và loại bài toán classification, regression hoặc text_classification.")
    if not data_path and (target or task):
        raise ValueError("--target và --task cần dùng cùng --data.")
    print("=" * 75)
    print("🚀 TẠO BÀI TỔNG QUAN DỰA TRÊN TÀI LIỆU VÀ SỐ LIỆU CÔNG BỐ")
    print(f"📌 Đề tài nghiên cứu: '{topic}'")
    print("=" * 75)

    t0 = time.time()
    os.makedirs(workspace_dir, exist_ok=True)
    workspace_dir = os.path.abspath(os.path.join(workspace_dir, uuid.uuid4().hex[:12]))
    os.makedirs(workspace_dir)
    print("\n🔹 [Giai đoạn 1/5] Kết nối Gemini...")
    model, _ = create_client()

    print("\n🔹 [Giai đoạn 2/5] Lập kế hoạch tìm tài liệu và dữ liệu...")
    plan = generate_ideas(topic, model, workspace_dir)
    print(f"📌 Câu hỏi nghiên cứu: {plan['research_question']}")

    print("\n🔹 [Giai đoạn 3/5] Thu thập nguồn, đọc toàn văn mở và lập bản đồ bằng chứng...")
    evidence = collect_evidence(plan, workspace_dir, model=model)
    retrieval = evidence.get("retrieval_summary") or {}
    if retrieval:
        print(f"🔎 {retrieval['raw_hits']} lượt trả về, "
              f"{retrieval['unique_records']} hồ sơ không trùng sau gộp nguồn.")
    print(f"📚 {len(evidence['sources'])} tài liệu; "
          f"{sum((item.get('fulltext') or {}).get('status') == 'full_text' for item in evidence['sources'])} toàn văn; "
          f"{len(evidence['datasets'])} bộ dữ liệu đã phân tích; "
          f"{len(evidence['study_findings'])} kết quả định lượng đã đối chiếu; "
          f"{len(evidence['charts'])} biểu đồ.")

    print("\n🔹 [Giai đoạn 4/5] Đề xuất ý tưởng và kiểm tra khả năng thực nghiệm...")
    proposal, experiment = plan_and_run(plan, evidence, workspace_dir, model=model,
                                        data_path=data_path, target=target, task=task)
    print(f"💡 {len(proposal['ideas'])} ý tưởng đã lưu; "
          f"ý tưởng thực nghiệm: {proposal.get('selected_idea_id') or 'không có'}.")
    if experiment["status"] == "completed" and experiment["experiment_type"] == "tabular_benchmark":
        print(f"🧪 Đã chạy benchmark {experiment['dataset']['task']} trên "
              f"{experiment['dataset']['name']}.")
    elif experiment["status"] == "completed" and experiment["experiment_type"] == "text_classification":
        print(f"🧪 Đã phân loại văn bản có nhãn trên {experiment['dataset']['name']}.")
    elif experiment["status"] == "completed":
        print(f"🧪 Đã chạy backtest trên [{experiment['source_id']}], "
              f"{experiment['test_count']} năm kiểm tra.")
    else:
        print(f"📚 Giữ chế độ tổng quan: {proposal['reason']}.")

    print("\n🔹 [Giai đoạn 5/5] Soạn bài có trích dẫn và biên dịch PDF...")
    pdf_path = perform_writeup(plan, model, workspace_dir)
    with open(os.path.join(workspace_dir, "evidence.json"), encoding="utf-8") as f:
        completed_evidence = json.load(f)
    interpretation = completed_evidence.get("interpretation") or {}
    claim_count = len(interpretation.get("observations", [])) + len(interpretation.get("supported_claims", []))
    print(f"🧠 {claim_count} "
          "nhận định phân tích đã được đưa vào bài.")

    t1 = time.time()
    print("\n" + "=" * 75)
    print(f"🎉 HOÀN THÀNH QUY TRÌNH NGHIÊN CỨU TRONG {t1 - t0:.2f} GIÂY!")
    print(f"📄 Bài báo PDF xuất ra tại: {os.path.abspath(os.path.join(workspace_dir, 'paper.pdf'))}")
    print("=" * 75)
    
    return pdf_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tạo bài tổng quan dựa trên nguồn công bố.")
    parser.add_argument(
        "--topic",
        type=str,
        default="Xu hướng tuổi thọ trung bình trên thế giới",
        help="Đề tài nghiên cứu cần thực hiện."
    )
    parser.add_argument(
        "--workspace",
        type=str,
        default="./workspace",
        help="Thư mục lưu trữ sản phẩm trung gian và đầu ra."
    )
    parser.add_argument("--data", help="Đường dẫn CSV cục bộ dùng cho benchmark.")
    parser.add_argument("--target", help="Tên cột đích khi dùng CSV.")
    parser.add_argument("--task", choices=("classification", "regression", "text_classification"),
                        help="Loại bài toán khi dùng CSV.")
    args = parser.parse_args()

    run_autonomous_scientific_research(args.topic, args.workspace, data_path=args.data,
                                       target=args.target, task=args.task)
