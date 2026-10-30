import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model.numpy.config import MODEL_PATH
from model.numpy.model import SentenceEmbeddingModel as NumpyModel
from model.numpy.utils.fixed import to_float
from model.numpy.utils.similarity import SCORE_FRAC_BITS, cosine_fixed
from model.pytorch.main import (
    MODEL_PATH as PYTORCH_MODEL_PATH,
    SentenceEmbeddingModel as PytorchModel,
    load_model_from_onnx,
)


MODEL_ID = "onnx-community/paraphrase-multilingual-MiniLM-L12-v2-ONNX"
REPORT_PATH = Path(__file__).resolve().parent / "report.md"
DEFAULT_TEXT = "Tôi đang làm đồ án FPGA."
DEFAULT_PASSAGES = (
    # FPGA / phần cứng
    "Đồ án dùng bo mạch Kria KV260 để tăng tốc mô hình nhúng câu MiniLM. Phần mềm trên ARM gửi "
    "token xuống phần cứng qua AXI, bộ tăng tốc tính toàn bộ 12 lớp encoder rồi trả về vector "
    "384 chiều để so khớp với cơ sở dữ liệu văn bản.",
    "Giao thức AXI4 tách kênh địa chỉ, dữ liệu và phản hồi, nên master có thể phát nhiều giao dịch "
    "outstanding mà không cần chờ giao dịch trước hoàn tất. Mỗi giao dịch mang một ID, và slave "
    "được phép trả kết quả không theo thứ tự giữa các ID khác nhau.",
    "Systolic array là lưới các phần tử xử lý, mỗi phần tử nhân hai số INT8 rồi cộng dồn vào thanh "
    "ghi 32 bit. Dữ liệu chảy qua mảng theo nhịp clock nên phép nhân ma trận lớn được chia thành "
    "các tile và tận dụng tối đa DSP trên FPGA.",
    "Sau bước place-and-route, Vivado tạo báo cáo timing cho từng đường dữ liệu. Kỹ sư cần kiểm "
    "tra worst negative slack của setup, hold và pulse width; nếu slack âm thì phải thêm tầng "
    "pipeline hoặc tối ưu lại logic thay vì giảm tần số clock.",
    "Lượng tử hóa INT8 thay số thực 32 bit bằng số nguyên 8 bit kèm hệ số tỉ lệ. Cách này giảm bốn "
    "lần dung lượng trọng số, tiết kiệm băng thông bộ nhớ và cho phép mỗi khối DSP thực hiện nhiều "
    "phép nhân hơn, đổi lại một sai số nhỏ so với mô hình gốc.",
    # Luật dân sự
    "Theo Bộ luật Dân sự, hợp đồng là sự thỏa thuận giữa các bên về việc xác lập, thay đổi hoặc "
    "chấm dứt quyền, nghĩa vụ dân sự. Hợp đồng có hiệu lực từ thời điểm giao kết, trừ khi các bên "
    "có thỏa thuận khác hoặc luật có quy định khác về thời điểm có hiệu lực.",
    "Người nào có lỗi cố ý hoặc vô ý xâm phạm tính mạng, sức khỏe, danh dự, nhân phẩm, uy tín, tài "
    "sản hoặc quyền, lợi ích hợp pháp khác của người khác mà gây thiệt hại thì phải bồi thường. "
    "Thiệt hại được bồi thường toàn bộ và kịp thời, trừ trường hợp các bên có thỏa thuận khác.",
    "Thời hiệu khởi kiện để yêu cầu Tòa án giải quyết tranh chấp hợp đồng là ba năm, kể từ ngày "
    "người có quyền yêu cầu biết hoặc phải biết quyền và lợi ích hợp pháp của mình bị xâm phạm. "
    "Hết thời hiệu, người khởi kiện có thể mất quyền yêu cầu Tòa án bảo vệ.",
    "Di sản thừa kế được chia theo di chúc nếu người chết để lại di chúc hợp pháp. Khi không có di "
    "chúc hoặc di chúc không hợp pháp, di sản được chia theo pháp luật cho những người thừa kế cùng "
    "hàng, trong đó hàng thứ nhất gồm vợ, chồng, cha mẹ và con của người chết.",
    "Giao dịch dân sự do người chưa thành niên xác lập, thực hiện mà theo quy định phải có người đại "
    "diện đồng ý nhưng lại không có sự đồng ý đó thì có thể bị Tòa án tuyên bố vô hiệu theo yêu cầu "
    "của người đại diện. Khi giao dịch vô hiệu, các bên phải hoàn trả cho nhau những gì đã nhận.",
    # Chủ đề khác
    "Hôm nay trời mưa từ sáng sớm nên tôi quyết định ở nhà đọc sách. Tôi pha một ấm trà nóng, ngồi "
    "cạnh cửa sổ nghe tiếng mưa rơi và đọc hết cuốn tiểu thuyết mà bạn tặng từ tháng trước.",
    "Phở bò Hà Nội nổi tiếng với nước dùng trong, được ninh nhiều giờ từ xương bò cùng quế, hồi, "
    "thảo quả và gừng nướng. Bát phở ngon cần bánh phở mềm, thịt bò thái mỏng và thêm hành lá, "
    "rau thơm khi ăn.",
    "Đội tuyển bóng đá Việt Nam giành chiến thắng trong trận chung kết sau khi ghi bàn quyết định ở "
    "những phút cuối. Hàng nghìn cổ động viên đổ ra đường ăn mừng, cờ đỏ sao vàng rợp khắp các "
    "tuyến phố trung tâm.",
    "Giá vàng trong nước tăng mạnh theo xu hướng của thị trường thế giới khi nhà đầu tư tìm đến "
    "kênh trú ẩn an toàn. Chênh lệch giữa giá mua và giá bán cũng bị nới rộng, khiến người mua "
    "ngắn hạn dễ chịu lỗ.",
    "Vịnh Hạ Long ở Quảng Ninh có gần hai nghìn hòn đảo đá vôi với nhiều hang động đẹp. Nơi đây "
    "được UNESCO công nhận là di sản thiên nhiên thế giới và thu hút du khách bằng các chuyến du "
    "thuyền ngủ đêm trên vịnh.",
    "Để giữ sức khỏe, mỗi người nên uống đủ nước, ngủ từ bảy đến tám tiếng mỗi đêm và tập thể dục "
    "đều đặn. Chế độ ăn nhiều rau xanh, hạn chế đồ ngọt và đồ chiên rán cũng giúp giảm nguy cơ mắc "
    "các bệnh mãn tính.",
    "Học sinh lớp mười hai đang tập trung ôn tập cho kỳ thi tốt nghiệp trung học phổ thông. Nhiều "
    "trường tổ chức thi thử, còn phụ huynh thì chuẩn bị bữa ăn đầy đủ dinh dưỡng để con có sức "
    "khỏe tốt trong mùa thi.",
    "Mèo là loài vật ngủ rất nhiều, trung bình từ mười hai đến mười sáu tiếng mỗi ngày. Chúng "
    "thường hoạt động mạnh vào lúc sáng sớm và chiều tối, phần còn lại trong ngày dành để nghỉ "
    "ngơi và chải lông.",
    "Thành phố Hồ Chí Minh thường xuyên bị ngập khi triều cường dâng cao kết hợp với mưa lớn. "
    "Nhiều tuyến đường biến thành sông, xe máy chết máy giữa đường và người dân phải chờ nước rút "
    "mới có thể đi tiếp.",
    "Python là ngôn ngữ lập trình phổ biến trong học máy nhờ cú pháp dễ đọc và hệ sinh thái thư "
    "viện phong phú. Các thư viện như NumPy, PyTorch và scikit-learn giúp xây dựng, huấn luyện và "
    "đánh giá mô hình nhanh chóng.",
)


def percentage_errors(reference, actual, eps=1e-8):
    """Mean and maximum error normalized by the full PyTorch tensor's range."""
    reference = np.asarray(reference, dtype=np.float64)
    actual = np.asarray(actual, dtype=np.float64)
    if reference.shape != actual.shape:
        raise ValueError(f"Shape mismatch: PyTorch {reference.shape}, NumPy {actual.shape}")
    denominator = max(float(reference.max() - reference.min()), eps)
    relative = 100 * np.abs(actual - reference) / denominator
    return float(relative.mean()), float(relative.max())


def traced_modules(encoder):
    """(NumPy trace name, PyTorch module) pairs, in forward order."""
    embeddings = encoder.embeddings
    modules = [
        ("embeddings.word", embeddings.word_embeddings),
        ("embeddings.position", embeddings.position_embeddings),
        ("embeddings.token_type", embeddings.token_type_embeddings),
        ("embeddings.output", embeddings.layer_norm),
    ]
    for index, layer in enumerate(encoder.layers):
        prefix = f"encoder.layer.{index}"
        modules += [
            (f"{prefix}.attention.self.query.output", layer.self_attention.query),
            (f"{prefix}.attention.self.key.output", layer.self_attention.key),
            (f"{prefix}.attention.self.value.output", layer.self_attention.value),
            (f"{prefix}.attention.self.output", layer.self_attention),
            (f"{prefix}.attention.output.dense.output", layer.attention_output.dense),
            (f"{prefix}.attention.output.LayerNorm.output", layer.attention_output.layer_norm),
            (f"{prefix}.intermediate.dense.output", layer.ffn.dense1),
            (f"{prefix}.intermediate.gelu.output", layer.ffn.gelu),
            (f"{prefix}.output.dense.output", layer.ffn.dense2),
            (f"{prefix}.output.LayerNorm.output", layer.ffn.layer_norm),
            (f"{prefix}.output", layer),
        ]
    return modules


def trace_pytorch(sentence_model, inputs):
    """Capture the same stages the NumPy model records with trace=True."""
    encoder = sentence_model.encoder
    trace = {}
    handles = []

    def save_output(name):
        def hook(module, args, output):
            trace[name] = output.detach().cpu().numpy().copy()
        return hook

    def save_sum(module, args):
        trace["embeddings.sum"] = args[0].detach().cpu().numpy().copy()

    try:
        for name, module in traced_modules(encoder):
            handles.append(module.register_forward_hook(save_output(name)))
        handles.append(encoder.embeddings.layer_norm.register_forward_pre_hook(save_sum))
        handles.append(encoder.register_forward_hook(save_output("encoder.output")))
        handles.append(sentence_model.pooling.register_forward_hook(save_output("pooling.output")))
        with torch.inference_mode():
            sentence_model(**{name: torch.from_numpy(value) for name, value in inputs.items()})
    finally:
        for handle in handles:
            handle.remove()
    return trace


def markdown_table(headers, rows, align):
    """align: one of 'l'/'r' per column."""
    def cell(value):
        return str(value).replace("|", "\\|").replace("\n", " ")

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---:" if a == "r" else ":---" for a in align) + " |",
    ]
    lines += ["| " + " | ".join(cell(value) for value in row) + " |" for row in rows]
    return "\n".join(lines)


def error_row(name, reference, actual):
    reference = np.asarray(reference, dtype=np.float64)
    actual = np.asarray(actual, dtype=np.float64)
    mean_error, max_error = percentage_errors(reference, actual)
    absolute = np.abs(actual - reference)
    cosine = cosine_similarity(actual.ravel(), reference.ravel())
    return [f"`{name}`", str(reference.shape), f"{mean_error:.4f}", f"{max_error:.4f}",
            f"{absolute.mean():.5g}", f"{absolute.max():.5g}", f"{cosine:.6f}"]


ERROR_HEADERS = ["Layer", "Shape", "Mean err (%)", "Max err (%)", "MAE", "Max abs", "Cosine"]
ERROR_ALIGN = "llrrrrr"


def ranking_table(scores, passages):
    rows = [
        [rank, index + 1, f"{scores[index]:.5f}", passages[index]]
        for rank, index in enumerate(np.argsort(-scores, kind="stable"), start=1)
    ]
    return markdown_table(["Hạng", "#", "Cosine", "Passage"], rows, "rrrl")


def read_passages():
    print("Enter passages, one per line; an empty line finishes (empty first line uses defaults):")
    passages = []
    while True:
        try:
            line = input("> ").strip()
        except EOFError:
            break
        if not line:
            break
        passages.append(line)
    return passages or list(DEFAULT_PASSAGES)


def tokenize(tokenizer, text):
    inputs = tokenizer(
        text,
        return_tensors="np",
        padding=True,
        truncation=True,
        max_length=128,
    )
    inputs["token_type_ids"] = np.zeros_like(inputs["input_ids"])
    return inputs


def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def main():
    text = input("Enter query (or press Enter to use default): ") or DEFAULT_TEXT
    passages = read_passages()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    inputs = tokenize(tokenizer, text)

    numpy_model = NumpyModel(MODEL_PATH, trace=True)
    pytorch_model = load_model_from_onnx(PYTORCH_MODEL_PATH)
    sentence_model = PytorchModel(pytorch_model).eval()

    # Run every sentence through both models; the query goes first.
    texts = [text, *passages]
    pytorch_outputs, numpy_fixed = [], []
    for sentence in texts:
        sentence_inputs = tokenize(tokenizer, sentence)
        with torch.inference_mode():
            pytorch_outputs.append(sentence_model(
                **{name: torch.from_numpy(value) for name, value in sentence_inputs.items()}
            ).numpy()[0])
        numpy_fixed.append(numpy_model(**sentence_inputs)[0])
    pytorch_outputs = np.stack(pytorch_outputs)
    numpy_fixed = np.stack(numpy_fixed)
    numpy_outputs = to_float(numpy_fixed)

    # RAG score: cosine(query, passage). NumPy uses the integer cosine in Q.15.
    pytorch_scores = np.array([cosine_similarity(pytorch_outputs[0], passage) for passage in pytorch_outputs[1:]])
    numpy_scores = cosine_fixed(numpy_fixed[:1], numpy_fixed[1:]) / (1 << SCORE_FRAC_BITS)
    pytorch_rank = np.argsort(-pytorch_scores, kind="stable")
    numpy_rank = np.argsort(-numpy_scores, kind="stable")

    # Re-run the query alone so both traces describe the same forward pass.
    numpy_model(**inputs)
    pytorch_trace = trace_pytorch(sentence_model, inputs)

    score_errors = np.abs(numpy_scores - pytorch_scores)
    score_range = max(float(pytorch_scores.max() - pytorch_scores.min()), 1e-8)
    ranking_identical = bool(np.array_equal(pytorch_rank, numpy_rank))
    embedding_cosines = [cosine_similarity(p, n) for p, n in zip(pytorch_outputs, numpy_outputs)]
    pytorch_position = {int(index): rank for rank, index in enumerate(pytorch_rank, start=1)}
    numpy_position = {int(index): rank for rank, index in enumerate(numpy_rank, start=1)}

    report = [
        "# So sánh NumPy (integer) với PyTorch (float32)",
        "",
        markdown_table(["Mục", "Giá trị"], [
            ["Query", text],
            ["Số token của query", inputs["input_ids"].shape[1]],
            ["Số passage", len(passages)],
            ["Trọng số", f"`{MODEL_PATH.name}` (cả hai model, cùng tokenizer)"],
            ["Thứ hạng giống nhau", "Có" if ranking_identical else "**Không**"],
            ["Sai số score tuyệt đối (mean / max)", f"{score_errors.mean():.5f} / {score_errors.max():.5f}"],
            ["Cosine embedding nhỏ nhất (PyTorch vs NumPy)", f"{min(embedding_cosines):.6f}"],
        ], "ll"),
        "",
        "## 1. Kết quả query",
        "",
        "### PyTorch (float32)",
        "",
        ranking_table(pytorch_scores, passages),
        "",
        "### NumPy (integer)",
        "",
        ranking_table(numpy_scores, passages),
        "",
        "## 2. Sai số chi tiết",
        "",
        "Error (%) = 100 · |NumPy − PyTorch| / max(max(PyTorch) − min(PyTorch), 1e-8); "
        "max/min lấy trên toàn bộ tensor (hoặc toàn bộ cột score) của PyTorch.",
        "",
        "### 2.1. RAG score: cosine(query, passage)",
        "",
        markdown_table(
            ["#", "PyTorch", "NumPy", "Abs error", "Error (%)", "Hạng PT", "Hạng NP", "Passage"],
            [
                [index + 1, f"{pytorch_scores[index]:.5f}", f"{numpy_scores[index]:.5f}",
                 f"{score_errors[index]:.5f}", f"{100 * score_errors[index] / score_range:.4f}",
                 pytorch_position[index],
                 numpy_position[index] if numpy_position[index] == pytorch_position[index]
                 else f"**{numpy_position[index]}**",
                 passage]
                for index, passage in enumerate(passages)
            ],
            "rrrrrrrl",
        ),
        "",
        "### 2.2. Sentence embedding của từng text",
        "",
        "Text #0 là query.",
        "",
        markdown_table(
            ["#", "Mean err (%)", "Max err (%)", "MAE", "Max abs", "Cosine", "Text"],
            [
                [index, *(f"{value:.4f}" for value in percentage_errors(pytorch_outputs[index], numpy_outputs[index])),
                 f"{np.abs(numpy_outputs[index] - pytorch_outputs[index]).mean():.5g}",
                 f"{np.abs(numpy_outputs[index] - pytorch_outputs[index]).max():.5g}",
                 f"{embedding_cosines[index]:.6f}", sentence]
                for index, sentence in enumerate(texts)
            ],
            "rrrrrrl",
        ),
        "",
        "8 giá trị đầu của embedding query:",
        "",
        markdown_table(
            ["", *(f"[{i}]" for i in range(8))],
            [["PyTorch", *(f"{v:.5f}" for v in pytorch_outputs[0, :8])],
             ["NumPy", *(f"{v:.5f}" for v in numpy_outputs[0, :8])]],
            "l" + "r" * 8,
        ),
        "",
        "### 2.3. Sai số theo layer (query)",
        "",
        "Tóm tắt output của từng khối:",
        "",
    ]
    summary = ["embeddings.output", *(f"encoder.layer.{i}.output" for i in range(len(pytorch_model.layers))),
               "pooling.output"]
    report.append(markdown_table(
        ERROR_HEADERS,
        [error_row(name, pytorch_trace[name], numpy_model.layers_output[name]) for name in summary],
        ERROR_ALIGN,
    ))

    groups = {"embeddings": []}
    for name in pytorch_trace:
        if name.startswith("encoder.layer."):
            groups.setdefault(".".join(name.split(".")[:3]), []).append(name)
        elif name.startswith("embeddings."):
            groups["embeddings"].append(name)
    for group, names in groups.items():
        report += ["", f"#### `{group}`", "", markdown_table(
            ERROR_HEADERS,
            [error_row(name.removeprefix(f"{group}."), pytorch_trace[name], numpy_model.layers_output[name])
             for name in names],
            ERROR_ALIGN,
        )]

    REPORT_PATH.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Ranking identical: {ranking_identical}; "
          f"max abs score error: {score_errors.max():.5f}")
    print(f"Report written to {REPORT_PATH}")


if __name__ == "__main__":
    main()
