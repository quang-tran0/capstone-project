---
tags: [minilm, fpga, thuat-ngu]
---

# Phụ lục — Thuật ngữ

← [[15 - Kiểm chứng và độ chính xác]] | Về: [[00 - Mục lục]]

| Thuật ngữ | Giải thích | Xem |
|---|---|---|
| **Accumulator** | Thanh ghi cộng dồn kết quả của nhiều phép nhân | [[07 - Linear]] |
| **Barrel shifter** | Mạch dịch bit với lượng dịch thay đổi được | [[14 - Độ rộng bit và đưa lên FPGA]] |
| **Bão hoà** (saturate) | Kẹp giá trị vào khoảng cho phép thay vì để tràn | [[03 - Kiểu dữ liệu phần cứng Fixed]] |
| **Bộ mã hoá ưu tiên** | Mạch tìm vị trí bit 1 cao nhất (`leading_one`) | [[13 - rsqrt và nghịch đảo]] |
| **BRAM / URAM** | Khối nhớ trên chip FPGA (36 Kb / 288 Kb mỗi khối) | [[14 - Độ rộng bit và đưa lên FPGA]] |
| **Code / mã** | Số nguyên lưu thay cho số thực sau lượng tử hoá | [[06 - Lượng tử hoá]] |
| **Cosine similarity** | Độ giống nhau của hai vector, trong $[-1, 1]$ | [[12 - Mean pooling và cosine]] |
| **DSP48E2** | Khối nhân $27 \times 18$ bit + cộng dồn 48 bit trên FPGA Xilinx | [[14 - Độ rộng bit và đưa lên FPGA]] |
| **Đỉnh** (peak) | Giá trị tuyệt đối lớn nhất của một hàng/cột | [[06 - Lượng tử hoá]] |
| **Fixed-point** | Số thực biểu diễn bằng số nguyên với dấu phẩy cố định | [[02 - Số fixed-point và định dạng Q]] |
| **Golden model** | Mô hình tham chiếu cho kết quả đúng để so với RTL | [[01 - Tổng quan]] |
| **LSB** | Bit thấp nhất; 1 LSB của Q.16 là $2^{-16}$ | [[02 - Số fixed-point và định dạng Q]] |
| **LUT / ROM** | Bảng tra giá trị tính sẵn | [[11 - GELU và residual]] |
| **Lượng tử hoá** | Đổi số thực sang mã số nguyên kèm scale | [[06 - Lượng tử hoá]] |
| **MAC** | Multiply-accumulate: nhân rồi cộng dồn | [[07 - Linear]] |
| **Mask** | Đánh dấu token thật (1) và token padding (0) | [[09 - Self-attention và softmax]] |
| **Newton** | Phương pháp lặp làm mịn nghiệm, mỗi bước gấp đôi số bit đúng | [[13 - rsqrt và nghịch đảo]] |
| **Offline** | Bước chuẩn bị trên máy tính, được dùng số thực | [[01 - Tổng quan]] |
| **Outlier** | Kênh có giá trị lớn bất thường so với các kênh khác | [[06 - Lượng tử hoá]] |
| **Q.$f$** | Số nguyên biểu diễn giá trị $X \cdot 2^{-f}$ | [[02 - Số fixed-point và định dạng Q]] |
| **RAG** | Retrieval-Augmented Generation: tìm tài liệu liên quan để trả lời | [[01 - Tổng quan]] |
| **Rescale** | Nhân với $m$ rồi dịch theo $n$ để đổi kết quả về Q.16 | [[04 - Scale m·2^-n]] |
| **Residual** | Cộng đầu vào của một khối vào đầu ra của nó | [[11 - GELU và residual]] |
| **round_shift** | Dịch phải có làm tròn nửa lên | [[02 - Số fixed-point và định dạng Q]] |
| **RTL** | Mô tả phần cứng bằng Verilog/SystemVerilog | [[14 - Độ rộng bit và đưa lên FPGA]] |
| **Scale** | Hệ số tỉ lệ $m \cdot 2^{-n}$ đi kèm mã số nguyên | [[04 - Scale m·2^-n]] |
| **Sentence embedding** | Vector 384 chiều đại diện cho nghĩa của cả câu | [[12 - Mean pooling và cosine]] |
| **Systolic array** | Lưới bộ nhân-cộng cho phép nhân ma trận | [[07 - Linear]] |
| **Tensor hẹp / rộng** | INT8 + scale / Q.16 32 bit | [[05 - Cách lưu số]] |
| **Thanh ghi** | Phần tử nhớ có độ rộng bit cố định; trong mô hình là `resize` | [[03 - Kiểu dữ liệu phần cứng Fixed]] |
| **Token** | Mảnh văn bản tokenizer tách ra (từ hoặc một phần của từ) | [[08 - Embedding]] |
| **Tràn** (overflow) | Giá trị vượt quá độ rộng thanh ghi, bit cao bị mất | [[02 - Số fixed-point và định dạng Q]] |

---

← [[15 - Kiểm chứng và độ chính xác]] | Về: [[00 - Mục lục]]
