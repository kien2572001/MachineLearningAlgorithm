import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Embedding, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Chuẩn bị dữ liệu văn bản
text_data = ["Hello world",
             "How are you",
             "I am fine",
             "Thank you"]

tokenizer = Tokenizer()
tokenizer.fit_on_texts(text_data)
sequences = tokenizer.texts_to_sequences(text_data)

vocab_size = len(tokenizer.word_index) + 1
max_length = max([len(seq) for seq in sequences])
padded_sequences = pad_sequences(sequences, maxlen=max_length)

# Xây dựng mô hình RNN
model = Sequential()
model.add(Embedding(vocab_size, 8, input_length=max_length))
model.add(SimpleRNN(32))
model.add(Dense(vocab_size, activation='softmax'))

# Biên dịch mô hình
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Huấn luyện mô hình
model.fit(padded_sequences, tf.keras.utils.to_categorical(padded_sequences, num_classes=vocab_size), epochs=100)

# Dự đoán từ tiếp theo trong chuỗi
test_data = ["How"]
test_sequence = tokenizer.texts_to_sequences(test_data)
test_padded = pad_sequences(test_sequence, maxlen=max_length)
predicted_output = model.predict(test_padded)
print(predicted_output)