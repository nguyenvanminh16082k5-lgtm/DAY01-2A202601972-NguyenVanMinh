# K3 — Ngày 1: Bài Tập & Phản Ánh
## Khám Phá LLM API | Phiếu Thực Hành

**Thời lượng:** 9h00–13h00
**Cách làm:** Trả lời từng câu ngay sau khi hoàn thành block tương ứng —
đừng để dồn hết về cuối buổi. Thay dòng `*Câu trả lời của bạn*` bằng câu
trả lời thật (chấm tự động sẽ đếm số câu đã trả lời).

---

## Block 1 — API Cơ Bản (trả lời sau Checkpoint 1)

### Câu 1.1 — Độ nhạy của temperature
Gọi `call_openai` với temperature 0.0, 0.5, 1.0 và 1.5 dùng prompt
**"."**

**Bạn nhận thấy quy luật gì qua bốn phản hồi?** (2–3 câu)
> *Với từng temperature, mỗi câu phản hồi của LLM sẽ khác nhau. Với temperature tăng, phản hồi có xu hướng đa dang hơn: từ một sự thật khá ổn định như cà phê hoặc hang Sơn Đòong sang các chi tiết dài, nhiều biểu tượng cảm xúc, và đôi khi chọn fact khác hẳn như Sa La. Nói ngắn gọn, temperature thấp cho câu trả lời an toàn và ổn định hơn, còn temperature cao tạo độ sáng tạo cao hơn nhưng kém nhất quán và dễ lệch chủ đề hơn.*

### Câu 1.2 — Chọn temperature cho sản phẩm
**Bạn sẽ đặt temperature bao nhiêu cho chatbot hỗ trợ khách hàng, và tại sao?**
> *Tôi sẽ đặt temperature khoảng 0.0–0.2 cho chatbot hỗ trợ khách hàng, vì cần câu trả lời ổn định, nhất quán và ít “bịa” hơn. Mức này vẫn đủ linh hoạt để diễn đạt tự nhiên, nhưng giảm rủi ro trả lời lệch ý hoặc quá sáng tạo khi xử lý câu hỏi thực tế của khách hàng.*

### Câu 1.3 — Đánh đổi chi phí
Kịch bản: 10.000 người dùng hoạt động mỗi ngày, mỗi người gọi API 3 lần,
mỗi lần trung bình ~350 token đầu ra.

**Ước tính GPT-4o đắt hơn GPT-4o-mini bao nhiêu lần cho workload này? Nêu một
trường hợp GPT-4o xứng đáng với chi phí và một trường hợp nên dùng mini:**

> Kết quả tính toán:
> - Tổng token đầu ra mỗi ngày: 10.000 × 3 × 350 = 10.500.000 token
> - GPT-4o: 10.500.000 / 1.000 × 0,010 = 105 USD/ngày
> - GPT-4o-mini: 10.500.000 / 1.000 × 0,0006 = 6,3 USD/ngày
>
> GPT-4o đắt hơn GPT-4o-mini khoảng 16,7 lần cho phần output, vì giá output là 0.010/1K token so với 0.0006/1K token. Với workload này, chi phí gần như tỷ lệ trực tiếp theo token đầu ra nên chênh lệch tổng cũng xấp xỉ như vậy.
>
> GPT-4o đáng dùng khi cần câu trả lời chất lượng cao, suy luận tốt, hoặc xử lý tình huống phức tạp, ví dụ tư vấn kỹ thuật nhiều bước. GPT-4o-mini phù hợp cho chatbot FAQ, tóm tắt đơn giản, hoặc các tác vụ lớn cần tiết kiệm chi phí và độ trễ thấp.

---

## Block 2 — System Prompt & Token (trả lời sau Checkpoint 2)

### Câu 2.1 — Sức mạnh của persona
Gọi `chat_with_system_prompt` hai lần với cùng câu hỏi
**"Giải thích blockchain là gì?"** nhưng hai system prompt khác nhau:
- "Bạn là giáo viên tiểu học, giải thích thật đơn giản cho trẻ 8 tuổi."
- "Bạn là chuyên gia tài chính, trả lời chuyên sâu bằng thuật ngữ kỹ thuật."

**Hai phản hồi khác nhau như thế nào (độ dài, từ vựng, ví dụ)? System prompt
ảnh hưởng đến hành vi model ra sao?** (3–4 câu)
> *Phản hồi cho prompt “giáo viên tiểu học” ngắn hơn, dùng từ rất đơn giản và nhiều ví dụ gần gũi như “cuốn sổ cái” hay “bản sao cho cả lớp”. Phản hồi cho prompt “chuyên gia tài chính” dài hơn rõ rệt, dùng nhiều thuật ngữ kỹ thuật như DLT, hash, P2P, nonce, Merkle root và trình bày theo kiểu chuyên sâu, có cấu trúc hơn. Điều này cho thấy system prompt có thể điều khiển mạnh mẽ giọng điệu, độ chi tiết và mức độ chuyên môn của model, tức là cùng một câu hỏi nhưng model sẽ trả lời theo đúng vai trò mà system prompt gán cho nó.*

### Câu 2.2 — tiktoken vs đếm từ
Chọn một đoạn văn tiếng Việt ~100 từ. So sánh số token theo `count_tokens`
(tiktoken) với ước lượng `số từ / 0.75` mà Part 1 đã dùng.

**Hai con số chênh nhau bao nhiêu phần trăm? Vì sao tiếng Việt thường tốn
nhiều token hơn tiếng Anh cùng độ dài?**

> Đoạn văn thử nghiệm:
> `Việt Nam là một quốc gia nằm ở khu vực Đông Nam Á, nổi tiếng với cảnh quan thiên nhiên đa dạng, nền văn hóa lâu đời và con người thân thiện. Từ những dãy núi hùng vĩ ở miền Bắc đến các bãi biển đẹp ở miền Trung và đồng bằng sông Cửu Long trù phú ở miền Nam, mỗi vùng đều có nét đặc trưng riêng. Ẩm thực Việt Nam cũng rất phong phú, với những món ăn quen thuộc như phở, bánh mì, bún chả và cà phê sữa đá.`
>
> Kết quả thu được:
> - word: 93
> - estimated: 124
> - diff: 18.25%
>
> Tiếng Việt thường tốn nhiều token hơn vì một “từ” có thể gồm nhiều âm tiết tách bằng dấu cách, nên tokenizer hay phải tách nhỏ hơn; ngoài ra dấu, ký tự Unicode và cách ghép từ cũng làm số token tăng.

---

## Block 3 — Streaming & Độ Bền (trả lời sau Checkpoint 3)

### Câu 3.1 — Trải nghiệm người dùng với streaming
**Streaming quan trọng nhất trong trường hợp nào, và khi nào thì
non-streaming lại phù hợp hơn?** (1 đoạn văn)
> *Streaming quan trọng nhất khi người dùng cần thấy kết quả ngay và phản hồi có thể dài, ví dụ chatbot, trợ lý viết nội dung, hoặc giao diện cần cảm giác “đang trả lời” để giảm thời gian chờ. Non-streaming phù hợp hơn khi câu trả lời ngắn, cần lấy toàn bộ nội dung rồi mới xử lý tiếp, hoặc khi muốn đơn giản hóa logic giao diện và kiểm tra kết quả trước khi hiển thị. Nói ngắn gọn, streaming ưu tiên trải nghiệm tức thời, còn non-streaming ưu tiên sự gọn gàng và kiểm soát toàn bộ đầu ra.*

### Câu 3.2 — Vì sao backoff theo cấp số nhân?
**So với delay cố định (ví dụ luôn chờ 1 giây), exponential backoff có lợi
thế gì khi API bị quá tải? Điều gì xảy ra nếu hàng nghìn client cùng retry
với delay cố định giống nhau?**
> *Exponential backoff giảm áp lực lên API bằng cách tăng dần thời gian chờ giữa các lần retry, nên khi hệ thống đang quá tải thì số request dồn vào sẽ ít hơn và cơ hội phục hồi cao hơn. Nếu hàng nghìn client cùng retry với delay cố định giống nhau, họ sẽ “đập” vào API đồng loạt theo nhịp, gây ra hiệu ứng bão retry, làm hệ thống càng quá tải và nhiều request tiếp tục thất bại.*

---

## Block 4 — Mini-Project (trả lời sau Checkpoint 4)

### Câu 4.1 — Thiết kế persona
**Bạn chọn persona gì cho trợ lý của mình? Viết lại system prompt đó và giải
thích 1–2 lựa chọn từ ngữ quan trọng trong prompt (ví dụ: vì sao yêu cầu
"trả lời ngắn gọn", vì sao chỉ định ngôn ngữ...):**
> Tôi chọn persona là một trợ lý, mentor có nhiều năm kinh nghiệm trong ngành AI, rõ ràng và đi thẳng vào ý chính cho sinh viên với AI. System prompt của tôi là: “Bạn là trợ lý học tập có nhiều kinh nghiệm trong việc xây dựng các hệ thống AI AI, trả lời ngắn gọn, rõ ý, bằng tiếng Việt.” Cụm “trả lời ngắn gọn” giúp model tránh lan man và giữ câu trả lời dễ đọc, còn “bằng tiếng Việt” đảm bảo đầu ra thống nhất, đúng ngôn ngữ người dùng cần.

### Câu 4.2 — Hạn chế & cải thiện
**Trợ lý của bạn hiện có hạn chế lớn nhất là gì (ví dụ: history chỉ 3 lượt,
không có bộ nhớ dài hạn, không kiểm duyệt nội dung...)? Đề xuất một cải
thiện cụ thể và mô tả ngắn cách triển khai:**
> Hạn chế lớn nhất của trợ lý hiện tại là chỉ giữ được tối đa 3 lượt hội thoại gần nhất nên rất dễ mất ngữ cảnh cũ, và không có bộ nhớ dài hạn để nhớ sở thích hay thông tin đã nói trước đó. Một cải thiện cụ thể là thêm cơ chế lưu tóm tắt hội thoại và nạp lại vào system prompt ở các lượt sau; có thể triển khai bằng cách mỗi vài lượt dùng model tóm tắt history cũ thành một đoạn ngắn, rồi ghép đoạn đó với history mới khi gọi API.

---

## Danh Sách Kiểm Tra Nộp Bài

- [ ] `python grade.py` — xem điểm tự động, mục tiêu ≥ 75/100
- [ ] Cả 4 checkpoint pytest đều pass
- [ ] Tất cả 9 câu trong file này đã được trả lời
- [ ] Đã copy bài làm vào folder `solution/` và zip theo hướng dẫn README
