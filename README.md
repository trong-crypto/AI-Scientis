# AI Scientist — evidence review

Nhập một chủ đề để tạo bài tổng quan từ tài liệu học thuật và số liệu công khai. Ứng dụng tìm nguồn qua OpenAlex, Crossref và Semantic Scholar (khi API khả dụng), gộp bài trùng DOI, sàng lọc mức liên quan rồi thử đọc toàn văn truy cập mở từ Europe PMC hoặc PDF được chỉ mục học thuật dẫn tới. Chỉ đánh dấu "toàn văn" sau khi nội dung trích xuất được kiểm tra và ghi vị trí mục/trang. Nếu thiếu dữ liệu định lượng, bài viết chuyển sang tổng hợp tường thuật, không giữ mục định lượng rỗng. Các nguồn World Bank/Our World in Data chỉ xuất hiện khi có chuỗi số liệu liên quan đã được xác thực. Sau khi thu thập bằng chứng, hệ thống đề xuất ý tưởng, thử backtest chuỗi thời gian hoặc benchmark dữ liệu bảng khi thiết kế phù hợp.

## Chạy trên Windows

Cần Python 3.11/3.12, MiKTeX hoặc TeX Live có `pdflatex`, kết nối Internet và Gemini API key. Trong PowerShell, từ thư mục dự án:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
$env:GEMINI_API_KEY = "YOUR_KEY_HERE"
.\.venv\Scripts\python.exe app.py
```

Mở địa chỉ Gradio in ra, thường là `http://127.0.0.1:7860`. Nếu `GEMINI_API_KEY` đã được lưu làm biến môi trường người dùng của Windows, không cần nhập lại lệnh đặt key. Có thể đặt `GEMINI_MODEL` để chọn mô hình Gemini được key cho phép sử dụng.

Để dùng khóa Gemini dự phòng, thêm `GEMINI_API_KEY_BACKUP` vào **biến môi trường người dùng của Windows** (mở “Edit environment variables for your account”). Có thể thêm `GEMINI_API_KEY_2`, `GEMINI_API_KEY_3`, ... nếu cần nhiều khóa; các giá trị trùng nhau được bỏ qua. Đóng và mở lại PowerShell/ứng dụng sau khi lưu; chương trình đang chạy không tự nhận biến vừa lưu. Không ghi khóa vào mã nguồn, README hay giao diện. Hệ thống thử lại lỗi tạm thời 429/5xx với thời gian chờ tăng dần, sau đó thử tối đa hai mô hình Gemini văn bản khác khả dụng trước khi chuyển khóa. Khi một tổ hợp mô hình và khóa thành công, tổ hợp đó được dùng tiếp trong lần chạy. Khi mọi lựa chọn đều thất bại, các bước có phương án tổng quan sẽ dùng phương án đó và ghi rõ không có kết quả thực nghiệm. Lỗi 503 thường do mô hình quá tải nên đổi khóa có thể vẫn không giúp; hạn mức 429 tính theo dự án nên các khóa cùng dự án chia sẻ hạn mức.

Để xuất sơ đồ phương pháp–nhiệm vụ bằng Graphviz, cài thêm ứng dụng Graphviz trên Windows (thư viện Python đã nằm trong `requirements.txt`):

```powershell
winget install --id Graphviz.Graphviz --exact
```

Ứng dụng tự tìm `dot.exe` trong `PATH` hoặc `C:\Program Files\Graphviz\bin`. Nếu cài ở nơi khác, đặt biến môi trường `GRAPHVIZ_DOT` trỏ đến `dot.exe` rồi khởi động lại ứng dụng. Khi không có bộ vẽ Graphviz, biểu đồ nhiệt phương pháp–nhiệm vụ vẫn được xuất để việc viết bài không bị dừng.

Semantic Scholar là nguồn bổ sung, không bắt buộc. Nếu có API key, lưu `SEMANTIC_SCHOLAR_API_KEY` vào **biến môi trường người dùng của Windows** (mở “Edit environment variables for your account”), rồi đóng/mở lại PowerShell và ứng dụng. Không ghi key vào mã nguồn, README hoặc `research_plan.json`. Chương trình gửi key trong header `x-api-key`, giãn các lượt gọi Semantic Scholar của cùng một tiến trình tối thiểu **1,25 giây** (chậm hơn 1 yêu cầu/giây), và chờ thêm khi API trả lỗi 429. Hạn mức gắn với key trên mọi endpoint: nếu dùng cùng key ở nhiều ứng dụng/máy, hãy tính cả các lượt gọi đó; bộ giới hạn trong dự án không thể điều phối các tiến trình khác. Khi nguồn này lỗi hoặc hết hạn mức, OpenAlex/Crossref vẫn tiếp tục hoạt động.

Chạy bằng dòng lệnh:

```powershell
.\.venv\Scripts\python.exe launch_scientist.py --topic "Xu hướng tuổi thọ trung bình trên thế giới"
```

Trong giao diện Gradio, chỉ cần nhập chủ đề; hệ thống tự tìm dữ liệu công khai phù hợp để thử thực nghiệm. Nếu muốn dùng CSV có nhãn của riêng mình qua dòng lệnh, chỉ định tệp, tên cột đích và loại bài toán:

```powershell
.\.venv\Scripts\python.exe launch_scientist.py --topic "Dự đoán loài hoa iris từ kích thước cánh hoa" --data "C:\data\iris.csv" --target species --task classification
```

Dùng `--task regression` cho biến đích số liên tục. Gemini nhận tên và kiểu cột để đánh giá thiết kế. Giá trị từng hàng được xử lý cục bộ. Kết quả benchmark lưu một bản dữ liệu dùng để đánh giá và giá trị đích thực tế trong thư mục lần chạy.

Với văn bản có nhãn, CSV cần một cột chứa nguyên văn và một cột nhãn, ví dụ `message,label`. Chọn `text_classification` và nhập tên cột nhãn làm `--target`:

```powershell
.\.venv\Scripts\python.exe launch_scientist.py --topic "Phân loại tin nhắn spam bằng học máy" --data "C:\data\messages.csv" --target label --task text_classification
```

Quy trình văn bản dùng tối đa 6.000 dòng, 2–10 lớp và ít nhất 5 văn bản khác nhau cho mỗi lớp. Nó từ chối nhãn thiếu, văn bản trống và các văn bản giống nhau nhưng mang nhãn mâu thuẫn. TF-IDF được khớp riêng trong từng phần huấn luyện của 5 lượt đánh giá; kết quả macro-F1, balanced accuracy, accuracy và dự đoán giữ lại được lưu cùng đối chứng đoán lớp phổ biến. Đây là đánh giá thăm dò trên bộ dữ liệu hiện có; các tin nhắn từ cùng người gửi/chiến dịch có thể cần cách chia nhóm riêng trước khi dùng để kết luận rộng hơn.

## Tệp đầu ra

Mỗi lần chạy tạo một thư mục mới trong `workspace/`:

- `research_plan.json`: câu hỏi và từ khóa tìm kiếm; mã chỉ số hoặc biểu đồ được đề xuất.
- `evidence.json`: số hồ sơ đã tìm/gộp/lọc, danh sách nguồn, phạm vi đọc (toàn văn hoặc chỉ tóm tắt), URL bản mở và đoạn gốc có vị trí mục/trang, bản đồ phương pháp–nhiệm vụ, chuỗi số liệu, phép tính, các nhận định phân tích theo từng phần và từng biểu đồ (`chart_analyses`) kèm ID nguồn/căn cứ, cùng cảnh báo.
- `study_findings.csv`: bảng ước lượng từ nội dung truy được, đoạn trích nguyên văn, vị trí trong nguồn, đại lượng đo và điều kiện so sánh. Bảng rỗng nếu không xác minh được kết quả số.
- `chart_data.csv`: chính xác các giá trị đã đưa vào biểu đồ.
- `chart_*.png`: chỉ có khi có chuỗi số liệu, ước lượng hoặc cặp phương pháp–nhiệm vụ được ghi nhận; Graphviz vẽ sơ đồ mạng phương pháp–nhiệm vụ khi có `dot.exe`, còn biểu đồ số liệu tiếp tục dùng Matplotlib.
- `chart_*.gv`: mã Graphviz của sơ đồ mạng đã xuất, để có thể kiểm tra và tái tạo hình.
- `research_proposal.json`: ý tưởng, giả thuyết, nguồn gợi ý, phương pháp dự kiến và quyết định có thể thực nghiệm hay không. "Mới" ở đây nghĩa là hướng nghiên cứu đề xuất cho lần chạy này; hệ thống không chứng minh tính mới trên toàn bộ văn liệu.
- `experiment_results.json`: trạng thái, lý do, chỉ số và cấu hình đánh giá. Với benchmark, tệp còn ghi nguồn dữ liệu, cột đích, biến dự báo, tham số và kết quả từng fold.
- `experiment_predictions.csv` và `experiment_plot.png`: giá trị quan sát, dự báo trên phần giữ lại và biểu đồ kết quả.
- `experiment_dataset.csv`: bảng đầu vào đã chọn cột và loại hàng trùng để tái lập benchmark.
- `paper.tex` và `paper.pdf`: bài đặt phần diễn giải ngay trong từng mục kết quả/tổng quan và ngay dưới từng biểu đồ; dẫn nguồn `[S1]` cho tài liệu, `[D1]` cho bộ dữ liệu; cấu trúc định lượng hoặc tường thuật tùy bằng chứng. Mục tổng hợp xuyên nguồn chỉ xuất hiện khi có nhận định so sánh được kiểm tra với đoạn trích gốc. Mục thực nghiệm chỉ xuất hiện khi `experiment_results.json` có trạng thái `completed`; tên mục phụ thuộc loại phép thử (benchmark bảng, phân loại văn bản hoặc backtest chuỗi năm).

Phần viết bài yêu cầu diễn giải dài hơn khi nguồn cung cấp đủ chi tiết. Một lượt tổng hợp bổ sung chỉ đưa các đoạn so sánh vào bài khi câu chữ dẫn đúng nguồn và trích đoạn hỗ trợ tồn tại trong tóm tắt hoặc toàn văn đã thu thập. Phần thực nghiệm phân tích thêm mức cân bằng nhãn (nếu có) và kết quả của từng lượt kiểm tra. Số trang không cố định: bài có ít tài liệu đọc được hoặc không có thực nghiệm sẽ ngắn hơn để tránh kéo dài bằng nội dung thiếu căn cứ.

Với mỗi chuỗi có ít nhất hai năm quan sát, hệ thống tính giá trị đầu/cuối, thay đổi tuyệt đối và tương đối (nếu mẫu số hợp lệ), mức thay đổi ròng trung bình theo năm, cực trị, số cặp năm liên tiếp tăng/giảm và biến động trong cửa sổ năm năm gần nhất khi có dữ liệu. Các con số và nhận xét xu hướng trong mục phân tích được tạo từ dữ liệu đã truy, không do Gemini tự đặt ra. Phương pháp tính được lưu trong `evidence.json`; CSV giữ giá trị gốc để đối chiếu.

Nguồn tài liệu được tìm qua OpenAlex và Crossref, với Semantic Scholar là nguồn bổ sung khi truy cập được. Chương trình lấy thêm trang kết quả OpenAlex cho các truy vấn đầu, gộp DOI/tiêu đề, giữ tối đa 35 hồ sơ liên quan và thử đọc tối đa 22 hồ sơ (ưu tiên hồ sơ có đường dẫn toàn văn mở). Mỗi URL PDF phải qua kiểm tra HTTPS/địa chỉ công khai, giới hạn dung lượng, định dạng, khả năng trích chữ và mức khớp tiêu đề; đường dẫn không đọc được không bị tính là toàn văn. Chuỗi số liệu World Bank và Our World in Data chỉ được sử dụng nếu API của nguồn trả về dữ liệu số hợp lệ. Đề xuất mã chỉ số hoặc địa chỉ biểu đồ của Gemini không được xem là bằng chứng; chương trình kiểm tra chúng qua API của nguồn trước khi vẽ.

Với tài liệu nghiên cứu, Gemini chỉ **đề xuất** số liệu cần trích. Chương trình kiểm tra con số và đoạn nguyên văn trong phần truy cập được, đồng thời lưu vị trí mục/đoạn nếu đọc được toàn văn. Nó từ chối số không đối chiếu được, số lượng người tham gia hoặc năm xuất bản bị nhầm thành kết quả tác động. Những ước lượng cùng loại chỉ được mô tả bằng khoảng giá trị/trung vị nếu tên biến, kết quả, mức phơi nhiễm và đơn vị đều khớp. Hệ thống không tự gộp hiệu ứng theo phương pháp phân tích tổng hợp (meta-analysis) và không suy ra quan hệ nhân quả.

Bản đồ phương pháp–nhiệm vụ chỉ đếm **các tài liệu trong tập tìm được** có nhắc rõ cặp phương pháp và nhiệm vụ. Nó không chứng minh thuật toán nào hiệu quả hơn. Phần tổng hợp phê phán được viết từ các đoạn nguồn đã truy; nếu Gemini không tạo được văn bản có so sánh và dẫn nguồn hợp lệ, chương trình dùng bản dự phòng thận trọng.

## Đề xuất ý tưởng và thực nghiệm

Hệ thống chọn một trong ba quy trình có kiểm soát: backtest chuỗi năm công bố, benchmark bảng có nhãn, hoặc phân loại văn bản có nhãn. Gemini đề xuất thiết kế; nếu đề xuất dự báo thiếu biến đầu vào, chương trình có thể dựng lại thiết kế phân loại từ bộ dữ liệu UCI đã kiểm tra khi tên bộ dữ liệu khớp rõ với chủ đề, nhãn có đủ mẫu và các cột đặc trưng được UCI khai báo. Sự phù hợp của bộ dữ liệu với kết quả cần đo vẫn được kiểm tra trước khi chạy. Python chỉ chạy thuật toán đã được định nghĩa và kiểm tra dữ liệu, không thực thi mã do Gemini sinh. Với chủ đề đòi hỏi ảnh, video, can thiệp thực tế hay dữ liệu chưa tồn tại, hệ thống ghi ý tưởng là chưa thực hiện và tiếp tục viết bài tổng quan. Chương trình thử OpenML rồi UCI khi cần. UCI chỉ nhận bộ dữ liệu có CSV chuẩn, metadata khớp cột nhãn và vai trò biến, kích thước trong giới hạn; lỗi của một nguồn không chặn nguồn còn lại. Gemini được thử lại khi API tạm quá tải. `experiment_results.json` ghi trạng thái, mã lý do và kết quả kiểm tra của từng ý tưởng, tránh để lỗi tìm dữ liệu bị che bởi ý tưởng cuối cùng.

Khi backtest trên chuỗi năm đã hoàn tất, hệ thống vẫn đề xuất thêm các hướng nghiên cứu từ tài liệu; những hướng này được ghi là chưa thử nghiệm và không thay đổi kết quả backtest.

Đề cương thực nghiệm được lưu trong mục `protocol` của `research_proposal.json` và hiện trong PDF. Đề cương nêu dữ liệu cần thiết, nhóm hoặc phương pháp đối chứng, cách đo kết quả và tiêu chí đánh giá giả thuyết. Khi câu hỏi đòi hỏi một thiết kế ngoài phạm vi dự báo hiện có, Gemini có thể đề xuất nghiên cứu tiếp theo và ghi rõ điều kiện còn thiếu. Đề xuất này được lưu ngay cả khi chưa thể chạy thực nghiệm.

Phần tìm dữ liệu bảng và đánh giá mô hình được tích hợp từ PTHTTM. Ứng dụng ưu tiên backtest khi có chuỗi năm đủ điều kiện. Nếu thiếu chuỗi phù hợp, Gemini đề xuất tối đa ba ý tưởng khác nhau dựa trên tài liệu đã tìm, gồm tối đa hai phương án dự báo và một phương án nghiên cứu tiếp theo khi cần thu thập dữ liệu mới. Hệ thống kiểm tra ID nguồn, loại bài toán và trùng lặp tiêu đề với tài liệu truy được; ghi cảnh báo sàng lọc trong `research_proposal.json`. Các phương án dự báo được thử theo thứ tự cho đến khi một phương án có dữ liệu, ánh xạ biến hợp lệ và benchmark hoàn tất. Các ý tưởng chưa chạy vẫn được lưu và ghi rõ lý do. Khác biệt so với nghiên cứu trước chỉ là giả thuyết; hệ thống không khẳng định tính mới toàn cầu. Mỗi phương án tìm tối đa năm ứng viên OpenML và năm ứng viên UCI theo tên bộ dữ liệu. Khi UCI có bộ dữ liệu phù hợp với chủ đề, schema được đưa vào bước lập thiết kế để tránh yêu cầu đặc trưng không tồn tại. Khi chỉ định CSV trên CLI hoặc giao diện, ứng dụng đánh giá CSV đó. Chương trình kiểm tra bảng dữ liệu và đối chiếu các cột với giả thuyết. Với UCI, benchmark dùng bộ đặc trưng được nguồn chính thức đánh dấu là Feature sau khi thiết kế và cột nhãn được chấp nhận. Dữ liệu OpenML được giới hạn ở ARFF công khai, tối đa 25 MB và tối đa 50 biến đầu vào. Bảng có cấu trúc dùng từ 30 đến 5.000 hàng; bài toán văn bản dùng từ 30 đến 6.000 hàng. Nếu quá trình sàng lọc không tìm được ứng viên phù hợp, bài tổng quan ghi lại lý do.

Benchmark so sánh mô hình đối chứng với Logistic Regression hoặc Ridge, Random Forest và Extra Trees. Chương trình dùng 5 fold ngoài và 3 fold trong để đánh giá và chọn mô hình, đồng thời khớp tiền xử lý bên trong các fold huấn luyện. Chỉ số chính là macro-F1 cho phân loại và MAE cho hồi quy. Với phân loại bảng, chỉ loại bản ghi trùng hoàn toàn, yêu cầu ít nhất 30 mẫu đặc trưng khác nhau và giữ các hàng có cùng mẫu đặc trưng trong một fold để hạn chế rò rỉ dữ liệu. Với hồi quy, cách chia ngẫu nhiên vẫn giả định các hàng độc lập. Đánh giá mức độ phù hợp có Gemini hỗ trợ và cần được kiểm tra trước khi dùng để kết luận khoa học.

Hệ thống luôn lưu ít nhất một ý tưởng trong `research_proposal.json`. Nếu có chuỗi số liệu công khai phù hợp, nó đề xuất kiểm tra giả thuyết: dự báo xu hướng tuyến tính một bước có sai số tuyệt đối trung bình (MAE) thấp hơn dự báo giữ nguyên giá trị gần nhất hay không. Điều kiện chạy: ít nhất **12 năm liên tiếp**, không trùng năm và toàn bộ giá trị hữu hạn. Trên từng năm kiểm tra sau 8 năm đầu, hai phương pháp chỉ dùng các năm trước đó; chương trình không chia ngẫu nhiên chuỗi thời gian và không dùng giá trị tương lai để huấn luyện. Nó lưu mọi dự báo, sai số và hình so sánh. Có thể thêm ý tưởng so sánh các phương pháp cùng nhiệm vụ được ghi trong tài liệu, nhưng ý tưởng đó chưa được thực nghiệm khi không có bộ dữ liệu và nhãn đánh giá phù hợp.

Thực nghiệm này là **backtest trên dữ liệu thứ cấp**, không phải đo đạc mới, chứng minh nhân quả hay bảo đảm hiệu quả dự báo tương lai. Câu hỏi được đặt ra sau khi nhìn thấy chuỗi dữ liệu, nên kết quả chỉ mang tính khám phá; không suy ra ý nghĩa thống kê hoặc tuyên bố tính mới toàn cầu. Mã chạy là thuật toán cố định trong `ai_scientist/research_ideas.py`, không thực thi Python do LLM sinh. Nếu thiếu dữ liệu hoặc thực nghiệm lỗi, bài vẫn được tạo ở chế độ tổng quan; ý tưởng được ghi rõ là chưa kiểm chứng và không có phần kết quả thực nghiệm giả.

Ngoài bảng số và biểu đồ, chương trình luôn tạo các nhận định diễn giải có thể đối chiếu: chỉ ra khi xu hướng gần đây đảo chiều hoặc chuỗi biến động không đều, khi ước lượng giữa nghiên cứu không so sánh trực tiếp được, và khi các phương pháp, nhiệm vụ hoặc cách kiểm định khác nhau. Mỗi nhận định lưu trong `evidence.json` cùng ID nguồn và căn cứ tính/đối chiếu. Trong PDF, phân tích chuỗi được đặt tại chuỗi tương ứng, so sánh nghiên cứu ở phần ước lượng, nhận xét tài liệu ở phần tổng quan; mỗi hình có đoạn phân tích riêng về điểm nổi bật, ý nghĩa và giới hạn. Gemini có thể bổ sung so sánh tài liệu, nhưng đoạn đó chỉ được nhận khi có trích dẫn và nguyên văn hỗ trợ khớp với nguồn truy được. Với chủ đề không có dữ liệu định lượng, phần kết quả chuyển sang phân tích định tính; nếu văn bản truy được không đủ để suy luận nội dung, nó nêu rõ khoảng trống bằng chứng thay vì bịa kết luận.

## Giới hạn và cách đọc kết quả

Bài PDF hiện yêu cầu các đoạn tổng quan và bàn luận có chiều sâu; nếu Gemini chỉ trả lời vài câu hoặc không đáp ứng kiểm tra trích dẫn, hệ thống dùng bản phân tích dự phòng dài hơn dựa trên nguồn đã thu thập. Bài có thêm bảng đối chiếu nhiệm vụ, phương pháp và mức độ truy cập của từng tài liệu; khi có benchmark, bảng kết quả ghi đầy đủ chỉ số của các mô hình và biến thiên giữa các lượt kiểm tra. Độ dài phụ thuộc lượng bằng chứng kiểm chứng được, không được tăng bằng kết quả hoặc trích dẫn tự tạo.

- Bài viết là **tổng quan và phân tích dữ liệu thứ cấp**, không phải báo cáo một thí nghiệm mới hoặc tổng quan hệ thống đầy đủ.
- Phân tích chỉ mang tính mô tả trong từng chuỗi; không kiểm định ý nghĩa thống kê, chứng minh quan hệ nhân quả hoặc mặc nhiên coi các nguồn khác nhau là tương đương.
- Không phải bài nào cũng có toàn văn truy cập mở hoặc PDF trích được chữ; những bài còn lại được đánh dấu **chỉ tóm tắt**, không giả vờ đã đọc toàn văn. PDF quét ảnh, dữ liệu paywall, phụ lục/bảng hình ảnh và nội dung nằm ngoài giới hạn trích xuất có thể bị bỏ sót.
- Không phải chủ đề nào cũng có số liệu định lượng so sánh được. Khi thiếu, PDF dùng cấu trúc tổng hợp tường thuật và nêu giới hạn đúng một lần ở phần Kết quả; không có mục định lượng hoặc biểu đồ năm xuất bản rỗng nghĩa.
- Cần mở DOI và kiểm tra thủ công bối cảnh, điều chỉnh thống kê, đơn vị và cách định nghĩa khoảng tin cậy trước khi nộp hoặc xuất bản.
- DOI, tiêu đề, số liệu, đơn vị và phạm vi của nguồn nên được kiểm tra lại trước khi nộp hoặc xuất bản. Abstract và dữ liệu từ API có thể không phản ánh toàn bộ nội dung bài gốc.
- OpenAlex có hạn mức truy vấn; Gemini và tính năng tìm kiếm có thể có chi phí tùy tài khoản. Phiên bản này dùng các API công khai và không tự bật tính năng Google Search grounding trả phí.
- Một chủ đề quá rộng hoặc quá mới có thể không tìm được nguồn đủ tin cậy. Khi không có tài liệu hoặc chuỗi số liệu kiểm tra được, chương trình báo lỗi thay vì viết bài không có nguồn.
