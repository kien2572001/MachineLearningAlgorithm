import numpy as np
import matplotlib.pyplot as plt

# Tập dữ liệu về số giờ học và điểm số bài kiểm tra
hours_studied = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
test_scores = np.array([60, 70, 80, 85, 90, 92, 95, 97, 98, 99])

# Khởi tạo ngẫu nhiên các giá trị m và b
m = np.random.rand()
b = np.random.rand()

# Learning rate
alpha = 0.01

# Số lần lặp để tối ưu hóa
num_iterations = 1000

# Gradient Descent
for _ in range(num_iterations):
    # Tính giá trị dự đoán
    predicted_scores = m * hours_studied + b
    
    # Tính đạo hàm của hàm mất mát theo m và b
    dm = (-2/len(hours_studied)) * np.sum(hours_studied * (test_scores - predicted_scores))
    db = (-2/len(hours_studied)) * np.sum(test_scores - predicted_scores)
    
    # Cập nhật các giá trị m và b
    m -= alpha * dm
    b -= alpha * db

# Đánh giá mô hình
print("Hệ số góc m:", m)
print("Hệ số chặn b:", b)

# Vẽ đồ thị
plt.scatter(hours_studied, test_scores, label='Data')
plt.plot(hours_studied, m * hours_studied + b, color='red', label='Linear Regression')
plt.xlabel('Số giờ học')
plt.ylabel('Điểm số')
plt.legend()
plt.show()
