import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Bước 2: Tải mô hình MobileNet từ TensorFlow Hub
model_url = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4"
model = hub.load(model_url)
print("Model loaded successfully!")

# Bước 3: Chuẩn bị dữ liệu đầu vào
# Tải và hiển thị hình ảnh mẫu
image_path = "https://tropicpet.vn/wp-content/uploads/2024/12/cho-corgi.jpg"  # URL hình ảnh
image_file = tf.keras.utils.get_file("corgi.jpg", image_path)
image = Image.open(image_file).resize((224, 224))

plt.imshow(image)
plt.axis('off')
plt.show()

# ✅ Cập nhật: Tiền xử lý hình ảnh (Fix lỗi kiểu dữ liệu)
def preprocess_image(image):
    image = np.array(image).astype(np.float32) / 255.0  # Đảm bảo float32
    return np.expand_dims(image, axis=0)  # Thêm batch dimension (1, 224, 224, 3)

processed_image = preprocess_image(image)
print("Image preprocessed successfully!")

# Bước 4: Chạy dự đoán với mô hình
predictions = model(processed_image).numpy()  # Chuyển kết quả về numpy array
predicted_class = np.argmax(predictions, axis=-1)  # Lấy chỉ số của lớp có xác suất cao nhất
print("Predicted class index:", predicted_class[0])

# ✅ Cập nhật: Đọc danh sách nhãn lớp từ ImageNet
labels_path = "https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt"
labels_file = tf.keras.utils.get_file("ImageNetLabels.txt", labels_path)

with open(labels_file, "r") as f:
    labels = f.read().splitlines()

predicted_label = labels[predicted_class[0]]
print("Predicted label:", predicted_label)

# ✅ Cập nhật: Hiển thị dự đoán dưới hình ảnh
plt.figure(figsize=(6, 6))
plt.imshow(image)
plt.title(f"Prediction: {predicted_label}", fontsize=14, color="blue")  # Fix lỗi không hiển thị tiêu đề
plt.axis('off')
plt.show()
