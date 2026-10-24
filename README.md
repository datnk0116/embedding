# Word Embedding: Skip-gram với Negative Sampling (PyTorch)

Cài đặt từ đầu mô hình **Word2Vec Skip-gram + negative sampling** bằng PyTorch,
huấn luyện trên bộ dữ liệu **text8** (khoảng 17 triệu từ tiếng Anh từ Wikipedia),
sau đó đánh giá và trực quan hóa không gian từ bằng t-SNE.

Báo cáo chi tiết: [REPORT.md](REPORT.md)

## Cấu trúc thư mục

| File | Chức năng |
|---|---|
| `data.py` | Đọc text8, xây từ điển, subsampling, tạo cặp (từ trung tâm, từ ngữ cảnh) |
| `model.py` | Mô hình Skip-gram, hàm loss negative sampling, vòng lặp huấn luyện |
| `evaluate.py` | Từ gần nghĩa (cosine) và phép toán từ (`king - man + woman`) |
| `visualize.py` | Giảm chiều bằng t-SNE/PCA và vẽ biểu đồ |
| `word2vec.pt` | Embedding đã huấn luyện (ma trận 31.888 x 100 và danh sách từ) |
| `embedding.png` | Biểu đồ t-SNE kết quả |
| `requirements.txt` | Thư viện cần cài |

## Cài đặt

Yêu cầu Python 3.10 trở lên (code dùng cú pháp `int | None`).
Tác giả chạy trên macOS (Apple Silicon), dùng GPU Apple (MPS).

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Dữ liệu

Tải text8 (Kaggle: `gupta24789/text8-word-embedding`) và đặt ở bất kỳ đâu,
rồi truyền đường dẫn vào lệnh chạy. Hỗ trợ cả file `.zip` lẫn file text đã giải nén.
Dữ liệu không được đưa vào repo vì dung lượng lớn.

Kiểm tra xử lý dữ liệu:

```bash
python3 data.py path/to/text8.zip
```

## Huấn luyện

Cấu hình dùng cho model trong repo (toàn bộ dữ liệu, bỏ từ xuất hiện dưới 20 lần):

```bash
python3 model.py path/to/text8.zip --max_words 17000000 --min_count 20 --epochs 5
```

Kết quả sinh ra: `word2vec.pt`, `loss.json`, `loss.png`.

Các tham số chính (mặc định): `--dim 100`, `--neg 5`, `--epochs 5`,
`--batch_size 4096`, `--lr 0.003`, `--max_words 5000000`, `--min_count 20`.
Chạy thử với mặc định (5 triệu từ đầu): mỗi epoch khoảng vài chục giây.

## Đánh giá

```bash
python3 evaluate.py
python3 evaluate.py --words music water city
```

In ra bảng từ gần nghĩa và độ chính xác phép toán từ (top-1, top-5).

## Trực quan hóa

```bash
# Vẽ các từ chọn trước cùng 15 từ gần nhất, tô màu theo nhóm
python3 visualize.py --words king france computer war three

# Vẽ 300 từ phổ biến nhất (bỏ 50 từ chức năng đầu tiên)
python3 visualize.py

# Dùng PCA thay cho t-SNE, đổi tên file ảnh
python3 visualize.py --method pca --out embedding_pca.png
```

## Kết quả tóm tắt

- Từ gần nghĩa hợp lý: `three` gần `four, two, five`; `computer` gần `computers, hardware, computing`;
  `france` gần `paris, french, spain`.
- Phép toán từ trên 19 bộ kiểm tra: **top-1 = 7/19 (37%)**, **top-5 = 10/19 (53%)**.
  Các quan hệ về quốc tịch và thủ đô hoạt động tốt hơn quan hệ giới tính hoàng gia
  (`king - man + woman` chưa ra `queen`), do dữ liệu text8 nhỏ.
- Xem phân tích đầy đủ và hình minh họa trong [REPORT.md](REPORT.md).

## Tài liệu tham khảo

- Mikolov et al., *Efficient Estimation of Word Representations in Vector Space*, 2013.
- Mikolov et al., *Distributed Representations of Words and Phrases and their Compositionality*, 2013.
- van der Maaten & Hinton, *Visualizing Data using t-SNE*, 2008.