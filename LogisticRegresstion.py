import numpy as np
from sklearn.linear_model import LogisticRegression

# Tạo dữ liệu mẫu
X = np.array([[5], [10], [2], [8], [3], [7], [1], [6]])
y = np.array([1, 1, 0, 1, 0, 1, 0, 1])

# Xây dựng mô hình Logistic Regression
model = LogisticRegression()
model.fit(X, y)

# Dự đoán xác suất đỗ kỳ thi cho số giờ học là 9
hours_of_study = np.array([[9]])
predicted_prob = model.predict_proba(hours_of_study)[:, 1]
print(f"Xác suất đỗ kỳ thi với {hours_of_study[0][0]} giờ học: {predicted_prob[0]}")
