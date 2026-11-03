---
tags: [minilm, fpga, muc-luc]
---

# Mô hình số nguyên MiniLM trên FPGA

Bộ ghi chú này giải thích cách đồ án tính **sentence embedding** và **score RAG** của MiniLM
(`paraphrase-multilingual-MiniLM-L12-v2`) **hoàn toàn bằng số nguyên**, theo đúng cách phần
cứng trên bo KV260 sẽ làm. Mô hình NumPy trong repo (`model/numpy/`) là mô hình tham chiếu
(golden model) cho RTL.

Nên đọc theo thứ tự. Mỗi chương dựa vào các chương trước.

## Phần I — Nền tảng

1. [[01 - Tổng quan]] — bài toán, pipeline của MiniLM, vì sao phải dùng số nguyên
2. [[02 - Số fixed-point và định dạng Q]] — biểu diễn số thực bằng số nguyên, dịch bit, làm tròn
3. [[03 - Kiểu dữ liệu phần cứng Fixed]] — class `Fixed` mô phỏng thanh ghi FPGA
4. [[04 - Scale m·2^-n]] — hệ số tỉ lệ dạng lũy thừa của 2, class `Scale`

## Phần II — Dữ liệu

5. [[05 - Cách lưu số]] — tensor rộng Q.16 và tensor hẹp INT8, trọng số, bias
6. [[06 - Lượng tử hoá]] — chọn scale cho weight và activation, scale theo chiều nào

## Phần III — Từng lớp của mô hình

7. [[07 - Linear]] — nhân ma trận INT8 và đổi kết quả về Q.16
8. [[08 - Embedding]] — tra bảng INT8 và cộng ba vector
9. [[09 - Self-attention và softmax]] — điểm attention, softmax số nguyên, context
10. [[10 - LayerNorm]] — chuẩn hoá không cần chia và căn
11. [[11 - GELU và residual]] — hàm kích hoạt bằng bảng tra, cộng residual
12. [[12 - Mean pooling và cosine]] — gộp thành một vector, so khớp RAG
13. [[13 - rsqrt và nghịch đảo]] — bảng tra cộng một bước Newton

## Phần IV — Đưa lên phần cứng

14. [[14 - Độ rộng bit và đưa lên FPGA]] — thanh ghi, bộ nhớ, những chỗ cần sửa trước khi viết RTL
15. [[15 - Kiểm chứng và độ chính xác]] — so với PyTorch, báo cáo, kiểm thử

## Phụ lục

- [[16 - Thuật ngữ]] — giải thích nhanh các từ dùng trong bộ ghi chú

> [!info] Code tương ứng
> Repo: `capstone-project/model/`. Kiểu dữ liệu ở `model/numpy/hwtypes/`, các lớp ở
> `model/numpy/layers/`, hàm số học ở `model/numpy/utils/`. Chứng minh công thức chi tiết ở
> `docs/integer_arithmetic.md`.
