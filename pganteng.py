import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Path untuk dataset yang sudah dinormalisasi
DATA_PATH = os.path.join('Head_Gesture_Data')

# Kelas gerakan kepala
actions = np.array(["Kanan", "Maju", "Stop", "Mundur", "Kiri"])
no_sequences = 10
sequence_length = 7

# Landmark yang relevan untuk mendeteksi gerakan kepala
# Mengambil landmark pada dahi, dagu, dan area lainnya yang relevan
relevant_landmarks = [10, 152, 234, 454, 1, 9, 200, 421, 361, 397]  # Contoh landmark untuk area wajah yang relevan

# Fungsi untuk memuat dataset yang sudah dinormalisasi
def load_normalized_data(actions, data_path, no_sequences, sequence_length):
    data, labels = [], []
    for action_idx, action in enumerate(actions):
        for sequence in range(no_sequences):
            sequence_data = []
            for frame_num in range(sequence_length):
                norm_npy_path = os.path.join(data_path, action, str(sequence), f"{frame_num}-norm.npy")
                if os.path.exists(norm_npy_path):
                    keypoints = np.load(norm_npy_path)
                    # Mengambil hanya landmark yang relevan
                    filtered_keypoints = keypoints.reshape(468, 3)[relevant_landmarks].flatten()
                    sequence_data.append(filtered_keypoints)
            if len(sequence_data) == sequence_length:
                data.append(sequence_data)
                labels.append(action_idx)
    return np.array(data, dtype=np.float32), np.array(labels, dtype=np.int32)

# Memuat data dan label
X, y = load_normalized_data(actions, DATA_PATH, no_sequences, sequence_length)

# Membagi data menjadi training dan testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Membuat model LSTM
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(sequence_length, len(relevant_landmarks) * 3)))
model.add(Dropout(0.3))
model.add(LSTM(128, return_sequences=False, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(len(actions), activation='softmax'))

# Kompilasi model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Training model dengan EarlyStopping
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test), callbacks=[early_stopping])

# Menyimpan model yang sudah dilatih
model.save('head_gesture_lstm_model.h5')

# Plot Loss and Accuracy
def plot_history(history):
    plt.figure(figsize=(12, 4))

    # Plot training & validation loss values
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Plot training & validation accuracy values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

plot_history(history)

# Confusion Matrix
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
conf_matrix = confusion_matrix(y_test, y_pred_classes)

# Plot confusion matrix
def plot_confusion_matrix(cm, classes):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks(ticks=np.arange(len(classes)) + 0.5, labels=classes, rotation=45)
    plt.yticks(ticks=np.arange(len(classes)) + 0.5, labels=classes, rotation=45)
    plt.show()

plot_confusion_matrix(conf_matrix, actions)