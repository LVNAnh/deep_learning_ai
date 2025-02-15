# Resize Hình Ảnh và Phân Loại với MobileNet

## Mô Tả
Kho lưu trữ này minh họa cách tiền xử lý hình ảnh, thay đổi kích thước ảnh và phân loại ảnh sử dụng mô hình MobileNetV2 từ TensorFlow Hub. Hình ảnh sẽ được thay đổi kích thước về `224x224` pixel, kích thước yêu cầu của mô hình MobileNetV2, và sau đó được đưa qua mô hình để phân loại. Nhãn dự đoán sẽ được hiển thị trên hình ảnh.

## Các Bước Tiền Xử Lý và Phân Loại Hình Ảnh

### 1. **Thay Đổi Kích Thước Hình Ảnh**
   - **Mục Tiêu**: Thay đổi kích thước hình ảnh đầu vào thành `224x224` pixel, vì đây là kích thước yêu cầu cho mô hình MobileNetV2.
   - **Tại Sao Cần Thay Đổi Kích Thước?**: Mô hình MobileNetV2 được huấn luyện trên bộ dữ liệu ImageNet, nơi các hình ảnh đầu vào được thay đổi kích thước thành `224x224` pixel.
   - **Phương Pháp**: Hình ảnh được tải lên từ URL hoặc đường dẫn tệp và sau đó thay đổi kích thước sử dụng `PIL` (Python Imaging Library).

### 2. **Tiền Xử Lý Hình Ảnh**
   - **Mục Tiêu**: Chuẩn hóa giá trị pixel của hình ảnh về khoảng `[0, 1]`.
   - **Phương Pháp**: Chuyển đổi hình ảnh thành mảng NumPy và chia mỗi giá trị pixel cho `255` để chuẩn hóa. Điều này đảm bảo mô hình nhận đầu vào trong phạm vi mong đợi.
   - **Batch Dimension**: Vì MobileNetV2 yêu cầu đầu vào là một batch (nhóm) hình ảnh, mảng hình ảnh được mở rộng để thêm một chiều batch (`1, 224, 224, 3`).

### 3. **Tải Mô Hình MobileNetV2 Đã Huấn Luyện Sẵn**
   - **Mục Tiêu**: Sử dụng mô hình MobileNetV2 đã được huấn luyện trên ImageNet từ TensorFlow Hub.
   - **Tại Sao Chọn Mô Hình Đã Huấn Luyện Sẵn?**: Mô hình đã được huấn luyện trên ImageNet và có khả năng phân loại hình ảnh thành một trong 1.000 lớp trong bộ dữ liệu ImageNet.
   - **Phương Pháp**: Tải mô hình từ TensorFlow Hub thông qua URL của nó.

### 4. **Thực Hiện Dự Đoán**
   - **Mục Tiêu**: Sử dụng mô hình MobileNetV2 để dự đoán lớp của hình ảnh.
   - **Phương Pháp**: Đưa hình ảnh đã được tiền xử lý vào mô hình và trích xuất chỉ số lớp có xác suất cao nhất bằng `np.argmax`.

### 5. **Hiển Thị Nhãn Dự Đoán**
   - **Mục Tiêu**: Hiển thị nhãn dự đoán lên trên hình ảnh.
   - **Phương Pháp**: Tải danh sách các nhãn của ImageNet, đọc nhãn tương ứng với chỉ số lớp dự đoán và hiển thị hình ảnh với nhãn dự đoán.

## Yêu Cầu
- `tensorflow`
- `tensorflow_hub`
- `PIL`
- `numpy`
- `matplotlib`

## Tạo Môi Trường Ảo

Trước khi chạy dự án, bạn nên tạo môi trường ảo để giữ các phụ thuộc của dự án. Dưới đây là các bước để tạo và kích hoạt môi trường ảo:

### 1. **Tạo Môi Trường Ảo**
Chạy lệnh sau trong thư mục dự án để tạo môi trường ảo:

```bash
python -m venv tf_env
.\tf_env\Scripts\activate
pip install tensorflow tensorflow_hub pillow numpy matplotlib
