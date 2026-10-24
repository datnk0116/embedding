# Word Embedding: Skip-gram với Negative Sampling (PyTorch)

Cài đặt từ đầu mô hình Skip-gram (word2vec) với negative sampling bằng PyTorch, huấn luyện
trên ngữ liệu text8 (khoảng 17 triệu từ tiếng Anh từ Wikipedia), sau đó trực quan hóa 10 từ bằng t-SNE.

Báo cáo: [bao_cao_word_embedding_skipgram.md](bao_cao_word_embedding_skipgram.md)

## Cấu trúc thư mục

| File | Chức năng |
|---|---|
| `data.py` | Đọc text8, xây từ vựng, tạo cặp huấn luyện (từ trung tâm, từ ngữ cảnh) |
| `model.py` | Mô hình Skip-gram, hàm mất mát negative sampling, vòng lặp huấn luyện |
| `visualize.py` | Giảm chiều bằng t-SNE/PCA và vẽ 10 từ |
| `word2vec.pt` | Embedding đã huấn luyện (ma trận 31.888 x 100 và danh sách từ) |
| `embedding.png` | Hình t-SNE của 10 từ |
| `requirements.txt` | Thư viện cần cài |

## Môi trường

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Tác giả chạy trên macOS (Apple Silicon), dùng GPU Apple (MPS). Máy không có MPS sẽ tự dùng CPU.

## Ngữ liệu

Tải text8 (http://mattmahoney.net/dc/text8.zip) và truyền đường dẫn vào lệnh chạy.
Hỗ trợ file `.zip` lẫn file text đã giải nén. Ngữ liệu không nằm trong repo vì dung lượng lớn.

## Huấn luyện

Cấu hình đã dùng cho `word2vec.pt` (toàn bộ ngữ liệu, bỏ từ xuất hiện dưới 20 lần):

```bash
python3 model.py path/to/text8.zip --max_words 17000000 --min_count 20 --epochs 5
```

Tham số mặc định: `--dim 100`, `--neg 5`, `--epochs 5`, `--batch_size 4096`,
`--lr 0.003`, `--max_words 5000000`, `--min_count 20`.

Cấu hình trên tạo khoảng 98 triệu cặp huấn luyện (23.946 bước mỗi epoch). Để chạy thử
nhanh, dùng `--max_words 5000000`.

## Trực quan hóa

```bash
python3 visualize.py                          # 10 từ mặc định, t-SNE
python3 visualize.py --method pca --out embedding_pca.png            # dùng PCA
python3 visualize.py --words king queen man woman france paris italy rome three four
```

Kết quả lưu vào `embedding.png`.

## Kết quả

- Mất mát trung bình giảm từ 2,2581 (epoch 1) xuống 2,2072 (epoch 5); từ epoch 2 trở đi
  mức giảm rất nhỏ.
- Hình t-SNE 10 từ (`embedding.png`): các cặp có quan hệ như `king`–`queen`,
  `three`–`four`, `france`–`paris`, `italy`–`rome` nằm gần nhau.
- Phân tích chi tiết trong báo cáo.

## Tài liệu tham khảo

- Mikolov et al., *Efficient Estimation of Word Representations in Vector Space*, 2013.
- Mikolov et al., *Distributed Representations of Words and Phrases and their Compositionality*, 2013.
- van der Maaten & Hinton, *Visualizing Data using t-SNE*, 2008.