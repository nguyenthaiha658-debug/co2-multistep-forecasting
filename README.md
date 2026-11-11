# Dự báo nhiều bước phía trước

**Forecasting several weeks or days ahead**

## English summary

This project predicts several future values of a series, not just the next one.

I compare two ways to do that:

- Train one model that predicts the next step. Then feed each prediction back in to get further steps. In the tables this is called **recursive**.
- Predict each future step with its own model. In the tables this is called **direct**.

I use Linear Regression and Random Forest. The data are weekly CO2 at Mauna Loa and daily bike-rental counts. I also compare with two simple baselines: repeat the last value, and repeat the value from the same time last year (or last week).

On CO2, a Random Forest trained on the actual CO2 level **fails** (10-step MAE 9.67, R² −3.94). Later years are higher than anything the model saw while training. A tree model cannot invent values outside that range.

A better idea is: predict the change from last week, then add it back to the last real value. That version of Random Forest gets the best 10-step MAE, **1.40**. Linear Regression that uses a separate model for each step gets **1.75**. Predicting each step with its own model also gets worse more slowly as we look further ahead, because it does not reuse its own mistakes.

The same “predict the change, then add it back” idea does **not** help on daily bike rentals. That series goes up and down with the week. It does not keep climbing like CO2. There, Linear Regression is best (1-step MAE 758 vs 873 if we just repeat the last value).

This work is based on exercises from the Viet Nguyen AI machine learning course. The course code changed the original data table while building the input windows. The code here does not.

## Bài toán

Cho một chuỗi số theo thời gian, hãy đoán vài giá trị phía trước, không chỉ bước kế tiếp.

Tôi so sánh hai cách, với cùng cửa sổ dữ liệu cũ và cùng cách chia tập học / tập kiểm tra theo thời gian:

- **Dùng lại dự báo (recursive):** học một mô hình để đoán bước kế tiếp. Muốn đoán xa hơn thì đưa dự báo vừa có trở lại làm đầu vào. Sai số có thể cộng dồn.
- **Mỗi bước một mô hình (direct):** học một mô hình riêng cho từng bước tương lai. Cách này không dùng dự báo của chính nó làm đầu vào.

## Dữ liệu

Hai file nằm trong `data/` và lấy từ khóa học Viet Nguyen AI. Chi tiết trong [`data/README.md`](data/README.md).

| Chuỗi | Tần suất | Độ dài | Cửa sổ $w$ | Horizon $H$ | Train / test (80/20) |
|---|---|---:|---:|---:|---|
| CO2 (Mauna Loa) | tuần, 1958–2001 | 2,284 (nội suy tuyến tính 59 tuần thiếu) | 12 | 10 | 1,827 / 457 |
| Bike sharing | ngày (cộng `users` theo ngày từ chuỗi giờ) | 731 ngày, 2011–2012 | 14 | 14 | 584 / 147 |

80% đầu dùng để học, 20% sau dùng để kiểm tra. Với mỗi thời điểm trong tập kiểm tra, mô hình nhìn các giá trị thật vừa xảy ra rồi đoán sẵn một chuỗi phía trước. Tôi đo sai số ở bước 1, bước 2, … cho đến hết khoảng dự báo.

Trong bảng kết quả:

- Recursive = dùng lại dự báo để đoán bước sau
- Direct = mỗi bước tương lai có mô hình riêng
- diff = đoán mức thay đổi, rồi cộng trở lại giá trị thật cuối cùng
- Persistence = lặp lại giá trị vừa thấy
- Seasonal naive = lặp lại giá trị cùng kỳ năm trước (CO2) hoặc tuần trước (xe đạp)

## Cách làm

- Cửa sổ dữ liệu được tạo trên bản sao. Chuỗi gốc **không bị sửa**. Script gốc của khóa học có lỗi này. Có bài kiểm tra trong `tests/test_windows.py`.
- Mô hình: Linear Regression và Random Forest (200 cây).
- Hai cách đơn giản để so sánh: lặp giá trị cuối, và lặp giá trị cùng kỳ (52 tuần với CO2, 7 ngày với xe đạp).
- Thêm một thử nghiệm: đoán mức thay đổi so với tuần (hoặc ngày) trước, rồi cộng trở lại giá trị thật cuối cùng. Ý tưởng này dùng để kiểm tra việc cây không đoán được số lớn hơn những gì nó đã thấy lúc học.
- Với mỗi bước dự báo, tôi ghi MAE, RMSE và R². MAE là sai số trung bình, càng nhỏ càng tốt.

## Kết quả

Các số dưới đây lấy từ lần chạy thật, lưu tại `results/metrics.csv`.

### CO2 (đơn vị: ppm)

MAE tại một số bước đại diện:

| Phương pháp | h=1 | h=5 | h=10 |
|---|---:|---:|---:|
| **Direct-diff / Random Forest** | **0.338** | **0.781** | **1.397** |
| Recursive-diff / Random Forest | 0.338 | 0.782 | 1.689 |
| Direct / Linear Regression | 0.352 | 0.880 | 1.745 |
| Recursive / Linear Regression | 0.352 | 0.922 | 1.926 |
| Persistence | 0.407 | 1.343 | 2.422 |
| Seasonal naive (chu kỳ 52) | 1.655 | 1.662 | 1.672 |
| Recursive / Random Forest (mức gốc) | 5.512 | 5.661 | 6.089 |
| Direct / Random Forest (mức gốc) | 5.532 | 8.330 | 9.673 |

R² tại h=10: Direct-diff RF **0.869**, Direct LR 0.810, Recursive LR 0.788, Direct RF trên mức gốc **−3.94**.

![MAE theo horizon trên CO2, đã ẩn hai đường RF mức gốc](results/co2_mae_vs_horizon_zoom.png)

![Dự báo 10 bước từ một gốc nằm sâu trong tập test, khi CO2 đã vượt khỏi miền giá trị lúc train](results/co2_forecast_late.png)

Ý nghĩa chính:

- Linear Regression đã khá tốt. Chuỗi CO2 vừa tăng dần, vừa có nhịp theo mùa, nên dễ học từ 12 tuần vừa rồi. Cách “mỗi bước một mô hình” kém đi chậm hơn cách “dùng lại dự báo” khi nhìn xa hơn (MAE ở bước 10: 1.75 so với 1.93).
- Random Forest học trên **mức CO2 gốc** thì thất bại. Tập kiểm tra (1993–2001) vẫn tăng và cao hơn mức lớn nhất lúc học. Mô hình dạng cây không đoán được số nằm ngoài khoảng đó. Đường dự báo từ một thời điểm năm 1999 kẹt quanh 355–359 ppm, trong khi thực tế khoảng 365 ppm.
- Cùng mô hình đó, nếu đoán mức thay đổi rồi cộng trở lại, lại thành cách tốt nhất. Direct-diff RF hơn Direct LR ở mọi bước đã đo. Recursive-diff RF gần bằng ở vài bước đầu, rồi kém hơn vì nó đưa sai số của chính mình trở lại. Đây đúng như kỳ vọng khi so hai cách dự báo.
- Cách lặp giá trị cùng kỳ năm trước gần như không đổi theo số bước (MAE ≈ 1.66). Từ khoảng bước 7, nó hơn cách chỉ lặp giá trị cuối. Vẫn kém các mô hình đã xử lý được xu hướng tăng.

Biểu đồ đầy đủ, gồm cả hai đường RF học trên mức gốc, nằm ở `results/co2_mae_vs_horizon.png`.

### Bike sharing (tổng lượt thuê mỗi ngày)

| Phương pháp | h=1 | h=7 | h=14 |
|---|---:|---:|---:|
| **Recursive / Linear Regression** | **758.0** | 994.9 | 1165.8 |
| **Direct / Linear Regression** | 762.5 | 987.9 | **1156.3** |
| Recursive / Random Forest | 826.4 | 1128.8 | 1279.7 |
| Direct / Random Forest | 849.2 | 1062.7 | 1240.0 |
| Recursive-diff / Random Forest | 819.4 | 1212.8 | 1419.9 |
| Direct-diff / Random Forest | 808.1 | 1271.3 | 1491.1 |
| Persistence | 872.7 | 1169.9 | 1271.1 |
| Seasonal naive (chu kỳ 7) | 1100.6 | 1169.9 | 1230.2 |

![MAE theo horizon trên nhu cầu thuê xe đạp theo ngày](results/bike_sharing_daily_mae_vs_horizon.png)

Ở đây Linear Regression là tốt nhất. Hai cách dự báo gần như trùng nhau ở vài bước đầu. Cách “mỗi bước một mô hình” nhỉnh hơn một chút khi nhìn xa. Đoán mức thay đổi rồi cộng trở lại **không giúp** Random Forest: lượt thuê theo ngày lên xuống theo tuần, không phải một đường tăng dài hạn như CO2. Vì vậy nên thử cùng một ý tưởng trên hơn một chuỗi, chứ không kết luận từ mỗi CO2.

## Khác gì so với bài tập gốc

- Viết lại phần tạo cửa sổ dữ liệu để không sửa chuỗi gốc. Có bài kiểm tra.
- So sánh hai cách dự báo trên cùng một quy trình. Báo cáo sai số ở từng bước, không chỉ một bước cố định.
- Thêm Random Forest (script gốc có import nhưng không dùng), cùng hai cách đoán đơn giản ở trên.
- Chạy thêm trên dữ liệu thuê xe đạp (cộng theo ngày), không chỉ CO2.
- Thêm thử nghiệm: đoán mức thay đổi rồi cộng trở lại, sau khi thấy Random Forest học trên mức gốc bị thất bại.
- Repo có cấu trúc rõ, có biểu đồ và số liệu chạy lại được.

## Cách chạy

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python tests/test_windows.py
python scripts/run_experiments.py
```

Kết quả được ghi vào `results/metrics.csv` và các file ảnh cùng thư mục. Lần chạy trên máy này mất khoảng 10 phút, chủ yếu vì Random Forest phải đoán nhiều thời điểm, từng điểm một.

## Nguồn

This work is based on exercises from the Viet Nguyen AI machine learning course.

Dựa trên bài tập của khóa học machine learning của Viet Nguyen AI (các script `recursive_ts.py` và `direct_ts.py`). Dữ liệu xe đạp gốc là UCI Bike Sharing Dataset. Phần so sánh hai cách dự báo, sửa lỗi, thử nghiệm đoán mức thay đổi và phân tích kết quả được làm thêm trên nền bài tập đó.
