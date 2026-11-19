import os
import json
import gradio as gr
from launch_scientist import run_autonomous_scientific_research

def execute_pipeline(topic: str):
    if not topic or not topic.strip():
        return "⚠️ Vui lòng nhập chủ đề nghiên cứu hợp lệ!", [], None, None, None, None, None, None, None
    
    workspace_dir = "./workspace"
    try:
        pdf_path = run_autonomous_scientific_research(topic.strip(), workspace_dir)
        run_dir = os.path.dirname(pdf_path)
        with open(os.path.join(run_dir, "evidence.json"), encoding="utf-8") as f:
            evidence = json.load(f)
        plots = [os.path.join(run_dir, chart["file"]) for chart in evidence["charts"]]
        experiment = evidence.get("experiment") or {}
        if experiment.get("status") == "completed":
            plots.append(os.path.join(run_dir, "experiment_plot.png"))
        csv_path = os.path.join(run_dir, "chart_data.csv")
        study_count = len(evidence.get("study_findings", []))
        interpretation = evidence.get("interpretation") or {}
        interpretation_count = (len(interpretation.get("observations", []))
                                + len(interpretation.get("supported_claims", [])))
        full_count = sum((source.get("fulltext") or {}).get("status") == "full_text"
                         for source in evidence["sources"])
        has_numbers = bool(evidence["analysis"]["series"] or study_count)
        prefix = "✅ Bài tổng quan có phân tích định lượng" if has_numbers else "✅ Bài tổng hợp tường thuật"
        if experiment.get("status") == "completed":
            prefix = "✅ Bài nghiên cứu có kết quả thực nghiệm"
        status_msg = (f"{prefix}: {len(evidence['sources'])} tài liệu phù hợp, "
                      f"{full_count} toàn văn truy cập mở, "
                      f"{len(evidence['datasets'])} chuỗi số liệu, "
                      f"{study_count} kết quả định lượng đã đối chiếu, "
                      f"{interpretation_count} nhận định phân tích có dấu vết nguồn, "
                      f"{len(plots)} biểu đồ. "
                      "Nguồn đã được ghi trong PDF và evidence.json.")
        idea_count = len((evidence.get("research_proposal") or {}).get("ideas") or [])
        status_msg += f" Đã lưu {idea_count} ý tưởng và điều kiện thử nghiệm."
        retrieval = evidence.get("retrieval_summary") or {}
        if retrieval:
            status_msg += (f" Đã sàng lọc {retrieval['unique_records']} hồ sơ không trùng "
                           f"từ {retrieval['raw_hits']} lượt trả về.")
        if has_numbers:
            status_msg += " Phân tích định lượng đã được viết trong PDF."
        else:
            status_msg += " Không tạo mục định lượng rỗng; PDF nêu giới hạn tại phần kết quả."
        if experiment.get("experiment_type") == "text_classification":
            status_msg += (" Đã chạy phân loại văn bản có nhãn bằng TF-IDF; "
                           "xem macro-F1, đối chứng và dự đoán giữ lại trong PDF/CSV.")
        elif experiment.get("experiment_type") == "tabular_benchmark":
            status_msg += (" Đã chạy benchmark phân loại hoặc hồi quy với dữ liệu phù hợp. "
                           "Xem sai số, mô hình đối chứng và các lượt kiểm tra trong PDF và JSON.")
        elif experiment.get("status") == "completed":
            status_msg += (" Đã chạy thực nghiệm backtest trên dữ liệu công bố; "
                           "xem kết quả và giới hạn trong PDF/experiment_results.json.")
        else:
            reason = (evidence.get("research_proposal") or {}).get("reason", "chưa đủ dữ liệu")
            status_msg += f" Đã đề xuất ý tưởng; chưa chạy thực nghiệm ({reason})."
        if evidence["warnings"]:
            status_msg += f"\n⚠️ {len(evidence['warnings'])} nguồn không dùng được; xem evidence.json."
        predictions = (os.path.join(run_dir, "experiment_predictions.csv")
                       if experiment.get("status") == "completed" else None)
        return (status_msg, plots, pdf_path, csv_path,
                os.path.join(run_dir, "study_findings.csv"), os.path.join(run_dir, "evidence.json"),
                os.path.join(run_dir, "research_proposal.json"),
                os.path.join(run_dir, "experiment_results.json"), predictions)
    except Exception as e:
        return f"❌ Đã xảy ra lỗi trong quá trình thực thi: {str(e)}", [], None, None, None, None, None, None, None

with gr.Blocks(title="AI-Scientist System") as demo:
    gr.Markdown(
        """
        # 📚 Tổng quan nghiên cứu từ nguồn công bố
        Nhập chủ đề để tìm tài liệu, đề xuất nhiều ý tưởng và viết bài PDF. Hệ thống thử thực nghiệm cho chuỗi năm, dữ liệu bảng hoặc văn bản có nhãn khi dữ liệu phù hợp. Nếu chưa đủ điều kiện, bài trình bày tổng quan và ý tưởng đề xuất.
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            topic_input = gr.Textbox(
                label="Chủ đề / Đề tài Nghiên cứu",
                placeholder="Ví dụ: Xu hướng tuổi thọ trung bình trên thế giới...",
                lines=3
            )
            run_button = gr.Button("🔎 Tìm nguồn, đề xuất ý tưởng và viết bài", variant="primary")
            status_box = gr.Textbox(label="Nhật ký & Trạng thái Hệ thống", interactive=False, lines=4)
            
        with gr.Column(scale=1):
            plot_display = gr.Gallery(label="Biểu đồ bằng chứng phù hợp chủ đề", columns=1)
            pdf_file_download = gr.File(label="Tải bài tổng quan (PDF)")
            csv_file_download = gr.File(label="Tải số liệu của biểu đồ (CSV)")
            study_file_download = gr.File(label="Tải bảng kết quả nghiên cứu kèm đoạn gốc (CSV)")
            evidence_file_download = gr.File(label="Tải danh sách nguồn và dữ liệu gốc (JSON)")
            proposal_file_download = gr.File(label="Tải đề xuất ý tưởng và đánh giá khả thi (JSON)")
            experiment_file_download = gr.File(label="Tải kết quả thực nghiệm hoặc lý do không chạy (JSON)")
            predictions_file_download = gr.File(label="Tải dự đoán và giá trị quan sát (CSV, nếu có)")

    run_button.click(
        fn=execute_pipeline,
        inputs=[topic_input],
        outputs=[status_box, plot_display, pdf_file_download, csv_file_download,
                 study_file_download, evidence_file_download, proposal_file_download,
                 experiment_file_download, predictions_file_download]
    )

if __name__ == "__main__":
    demo.launch(share=False, debug=True, theme=gr.themes.Soft())
