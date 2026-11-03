---
tags: [minilm, fpga, hwtypes]
---

# Chương 3 — Kiểu dữ liệu phần cứng `Fixed`

← [[02 - Số fixed-point và định dạng Q]] | Tiếp: [[04 - Scale m·2^-n]] →

Code: `model/numpy/hwtypes/fixed.py`

## 3.1. Vì sao cần một kiểu dữ liệu riêng

Nếu viết mô hình bằng mảng `int64` của NumPy thì mọi giá trị đều có 64 bit, không ai biết một
tín hiệu thực sự cần bao nhiêu bit. Một phép nhân có thể ra 55 bit mà không ai để ý, rồi khi
viết RTL mới phát hiện thanh ghi 32 bit bị tràn.

`Fixed` giải quyết việc này: mỗi mảng mang theo **định dạng thanh ghi** của nó, và mỗi phép
tính tự tính độ rộng của kết quả theo đúng quy tắc của HDL.

## 3.2. Cấu tạo

```python
Fixed(raw, bits, frac=0, signed=True, overflow="error", name=None)
```

| Thuộc tính | Ý nghĩa                                                               |
| ---------- | --------------------------------------------------------------------- |
| `raw`      | Các giá trị nguyên (lưu bằng `np.int64` để NumPy không tràn khi tính) |
| `bits`     | Độ rộng thanh ghi, tính cả bit dấu (1 đến 63)                         |
| `signed`   | Có dấu (bù 2) hay không dấu                                           |
| `frac`     | Số bit thập phân: giá trị thật $= \text{raw} \cdot 2^{-\text{frac}}$  |

Định dạng chỉ là thông tin đi kèm. Mọi phép tính làm trên `raw`, đúng như RTL làm trên các bit.

**Cách tạo giá trị:**

| Cách tạo                          | Dùng khi                                                     |
| --------------------------------- | ------------------------------------------------------------ |
| `Fixed(raw, bits, frac, signed)`  | Đã có giá trị nguyên                                         |
| `Fixed.from_float(x, bits, frac)` | Offline: làm tròn số thực về mã gần nhất (bias, gamma, beta) |
| `Fixed.const(v)`                  | Hằng số, tự chọn độ rộng nhỏ nhất vừa giá trị                |
| `x.to_float()`                    | Chỉ để debug và in trace                                     |

**Kiểu đặt sẵn:**

| Kiểu | Định dạng |
|---|---|
| `Int8`, `Int16`, `Int32` | có dấu, 8 / 16 / 32 bit |
| `UInt8` | không dấu 8 bit |
| `Q16` | có dấu 32 bit, 16 bit thập phân |

Tạo kiểu mới bằng `fixed_type`:

```python
from model.numpy.hwtypes import Int8, Q16, fixed_type

a = Int8([100, -50])
x = Q16.from_float([1.5, -0.25])        # raw = [98304, -16384]
UInt5 = fixed_type("UInt5", 5, signed=False)
```

## 3.3. Phép tính và độ rộng kết quả

Các toán tử được overload. Kết quả **luôn đủ bit**, không bao giờ tự cắt:

| Phép | Độ rộng kết quả | `frac` của kết quả |
|---|---|---|
| `a + b`, `a - b` | rộng hơn toán hạng rộng nhất 1 bit | hai bên phải bằng nhau, nếu không báo lỗi |
| `-a` | thêm 1 bit | giữ nguyên |
| `abs(a)` | giữ nguyên, kết quả không dấu | giữ nguyên |
| `a * b` | `a.bits + b.bits` | `a.frac + b.frac` |
| `a @ b` (MAC trên $K$ phần tử) | `a.bits + b.bits` $+ \lceil \log_2 K \rceil$ | cộng lại |
| `a.sum(axis)` trên $n$ phần tử | `bits` $+ \lceil \log_2 n \rceil$ | giữ nguyên |
| `a.max()`, `a.min()` | giữ nguyên | giữ nguyên |

```python
a, b = Int8([-128, 127]), Int8([127, -128])
(a + b).bits     # 9
(a * b).bits     # 16
acc = Int8(...) @ Int8(...)   # K = 384 -> 8 + 8 + 9 = 25 bit
```

**Cộng hai số có `frac` khác nhau sẽ báo lỗi.** Đây là chủ ý: trên phần cứng, người thiết kế
phải tự căn chỉnh dấu phẩy, nên mô hình cũng bắt làm rõ.

**Trộn có dấu và không dấu:** kết quả có dấu nếu một trong hai có dấu. Số không dấu $u$ bit được
coi như số có dấu $u + 1$ bit.

**Hằng số Python** được xử lý như literal trong HDL: khi cộng thì dùng chung `frac` với toán hạng
kia, khi nhân thì là số nguyên.

## 3.4. Dịch bit

| Phép | Ý nghĩa | `frac` của kết quả |
|---|---|---|
| `a >> k`, `a << k` | Dịch bit thô (`>>` làm tròn xuống) | giữ nguyên |
| `a.round_shift(k)` | Dịch phải làm tròn nửa lên; $k < 0$ là dịch trái | `frac - k` |
| `a.floor_shift(k)` | Dịch phải làm tròn xuống; $k < 0$ là dịch trái | `frac - k` |
| `a & mask` | Giữ các bit thấp | giữ nguyên |

Hai nhóm này ứng với hai cách hiểu phép dịch ở [[02 - Số fixed-point và định dạng Q#2.3. Dịch bit]]:

- `>>`, `<<` là **chia/nhân cho $2^k$**: giữ `frac`, giá trị đổi.
- `round_shift`, `floor_shift` là **đổi định dạng**: giữ giá trị, `frac` giảm $k$.

```python
x = Q16.from_float([1.5])                    # frac 16
y = Fixed.from_float([0.5], 16, frac=13)     # frac 13
p = x * y                                    # 48 bit, frac 29
p.round_shift(13).to_float()                 # [0.75], frac 16
```

Khi lượng dịch là một mảng (mỗi phần tử dịch khác nhau), không có một `frac` chung để tự suy
ra, nên phải truyền `frac=` cho kết quả.

## 3.5. Khai báo thanh ghi bằng resize

```python
acc = (x.codes @ w.codes).resize(32, name="accumulator")
```

`resize` là chỗ **khai báo một thanh ghi** có độ rộng cố định. Nếu dữ liệu không vừa:

| `overflow` | Hành vi |
|---|---|
| `"error"` (mặc định) | Báo `OverflowError`, kèm tên thanh ghi và số bit thực sự cần |
| `"wrap"` | Bỏ bit cao (tràn kiểu bù 2, giống phần cứng thật) |
| `"saturate"` | Bão hoà về giá trị lớn nhất / nhỏ nhất |

```python
Int8([128])                            # OverflowError
Int8([128, -129], overflow="wrap")     # [-128, 127]
Int8([300, -300], overflow="saturate") # [127, -128]
```

Cách làm việc là: phép tính tự tính độ rộng đầy đủ, rồi ở những chỗ phần cứng có thanh ghi thì
gọi `resize`. Nhờ vậy mô hình bắt được mọi trường hợp tràn ngay khi chạy.

Hai hàm phụ: `with_frac(f)` đổi cách hiểu `frac` mà không đổi bit nào; `required_bits()` cho
biết dữ liệu hiện tại thực sự cần bao nhiêu bit.

## 3.6. Các khối logic nhỏ

| Hàm | Phần cứng tương ứng |
|---|---|
| `x.clip(lo, hi)` | Bộ bão hoà; độ rộng thu về vừa khoảng $[lo, hi]$ |
| `where(cond, a, b)` | Bộ chọn (multiplexer) |
| `leading_one(x)` | Bộ mã hoá ưu tiên: số bit cần để biểu diễn $x \ge 0$ |
| `lut[address]` | Đọc ROM/LUT; địa chỉ ngoài bảng báo `IndexError` |
| `x.max(axis, where=mask)` | Cây so sánh, bỏ qua phần tử bị mask |

## 3.7. Giới hạn

- Độ rộng tối đa là 63 bit (giới hạn của `int64`). Nếu quy tắc cho ra hơn 63 bit, `Fixed` dùng
  độ rộng thực tế của dữ liệu; nếu dữ liệu thật sự cần hơn 63 bit thì báo lỗi.
- Khi dịch theo một mảng lượng dịch, độ rộng kết quả được suy ra từ khoảng lượng dịch có trong
  dữ liệu, không phải từ độ rộng khai báo. Các thanh ghi đã `resize` thì luôn được kiểm tra chặt.

---

← [[02 - Số fixed-point và định dạng Q]] | Tiếp: [[04 - Scale m·2^-n]] →
