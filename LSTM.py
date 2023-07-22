import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Chuẩn bị dữ liệu chuỗi thời gian
time_series_data = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
X, y = [], []
for i in range(len(time_series_data) - 3):
    X.append(time_series_data[i:i+3])
    y.append(time_series_data[i+3])

X = tf.reshape(X, (len(X), 3, 1))
y = tf.reshape(y, (len(y), 1))

# Xây dựng mô hình LSTM
model = Sequential()
model.add(LSTM(50, input_shape=(3, 1)))
model.add(Dense(1, activation='linear'))

# Biên dịch mô hình
model.compile(optimizer='adam', loss='mse')

# Huấn luyện mô hình
model.fit(X, y, epochs=1000)

# Dự đoán giá trị tiếp theo
next_sequence = [7, 8, 9]
next_sequence = tf.reshape(next_sequence, (1, 3, 1))
predicted_value = model.predict(next_sequence)
print(predicted_value)
