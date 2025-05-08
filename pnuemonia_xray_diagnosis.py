import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.optimizers import Adam

# --- Function to Load Data ---
def load_data_from_folder(folder_path, image_size=(224, 224), max_images=None):
    X, y = [], []
    labels = sorted([label for label in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, label))])
    label_map = {label: idx for idx, label in enumerate(labels)}
    
    for label in labels:
        label_folder = os.path.join(folder_path, label)
        count = 0
        for filename in os.listdir(label_folder):
            if max_images and count >= max_images:
                break
            if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                img_path = os.path.join(label_folder, filename)
                img = cv2.imread(img_path)
                if img is None:
                    continue
                img = cv2.resize(img, image_size)
                img = img / 255.0
                X.append(img)
                y.append(label_map[label])
                count += 1
    return np.array(X), np.array(y), label_map

# --- Custom CNN Model ---
def build_cnn_model(input_shape):
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2,2)),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D((2,2)),
        Conv2D(128, (3,3), activation='relu'),
        MaxPooling2D((2,2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    return model

# --- Paths ---
base_path = "/Users/nyaradzomararanje/Desktop/Spring 2025 - Madrid/AI - UCM/Final Project/chest_xray"
train_path = os.path.join(base_path, "train")
val_path = os.path.join(base_path, "val")
test_path = os.path.join(base_path, "test")

# --- Load a smaller dataset ---
X_train_full, y_train, label_map = load_data_from_folder(train_path, (224, 224), max_images=100)
X_val_full, y_val, _ = load_data_from_folder(val_path, (224, 224), max_images=50)
X_test_full, y_test, _ = load_data_from_folder(test_path, (224, 224), max_images=50)

# --- Resize for CNN (100x100) ---
X_train_cnn = np.array([cv2.resize(img, (100, 100)) for img in X_train_full])
X_val_cnn = np.array([cv2.resize(img, (100, 100)) for img in X_val_full])
X_test_cnn = np.array([cv2.resize(img, (100, 100)) for img in X_test_full])

# --- Train Custom CNN ---
model_cnn = build_cnn_model((100, 100, 3))
model_cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history_cnn = model_cnn.fit(X_train_cnn, y_train, validation_data=(X_val_cnn, y_val), epochs=10, batch_size=32, verbose=1)

# --- Evaluate Custom CNN ---
y_pred_probs = model_cnn.predict(X_test_cnn)
y_pred = (y_pred_probs > 0.5).astype(int).flatten()
cm = confusion_matrix(y_test, y_pred)
acc_custom = accuracy_score(y_test, y_pred)

# --- Prepare Data for VGG16 (224x224 and preprocess) ---
X_train_vgg = preprocess_input(X_train_full)
X_val_vgg = preprocess_input(X_val_full)
X_test_vgg = preprocess_input(X_test_full)

# --- Build VGG16 Model ---
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
for layer in base_model.layers:
    layer.trainable = False

x = Flatten()(base_model.output)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(1, activation='sigmoid')(x)
model_vgg = Model(inputs=base_model.input, outputs=output)

model_vgg.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
history_vgg = model_vgg.fit(X_train_vgg, y_train, validation_data=(X_val_vgg, y_val), epochs=10, batch_size=32, verbose=1)

# --- Evaluate VGG16 ---
y_pred_vgg_probs = model_vgg.predict(X_test_vgg)
y_pred_vgg = (y_pred_vgg_probs > 0.5).astype(int).flatten()
cm_vgg = confusion_matrix(y_test, y_pred_vgg)
acc_vgg = accuracy_score(y_test, y_pred_vgg)

# --- Print Accuracy Scores ---
print(f"\nCustom CNN Test Accuracy: {acc_custom:.2f}")
print(f"VGG16 Test Accuracy: {acc_vgg:.2f}")

# --- Plot Side-by-Side Confusion Matrices ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

disp_custom = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["NORMAL", "PNEUMONIA"])
disp_custom.plot(ax=axes[0], cmap='Blues', values_format='d')
axes[0].set_title("Custom CNN")

disp_vgg = ConfusionMatrixDisplay(confusion_matrix=cm_vgg, display_labels=["NORMAL", "PNEUMONIA"])
disp_vgg.plot(ax=axes[1], cmap='Greens', values_format='d')
axes[1].set_title("VGG16")

plt.tight_layout()
plt.show()

# --- Compute dynamic metrics for bar plot ---
TN_cnn, FP_cnn, FN_cnn, TP_cnn = cm.ravel()
TN_vgg, FP_vgg, FN_vgg, TP_vgg = cm_vgg.ravel()

cnn_scores = [
    acc_custom,
    TN_cnn / (TN_cnn + FP_cnn) if (TN_cnn + FP_cnn) != 0 else 0,
    TP_cnn / (TP_cnn + FN_cnn) if (TP_cnn + FN_cnn) != 0 else 0
]

vgg_scores = [
    acc_vgg,
    TN_vgg / (TN_vgg + FP_vgg) if (TN_vgg + FP_vgg) != 0 else 0,
    TP_vgg / (TP_vgg + FN_vgg) if (TP_vgg + FN_vgg) != 0 else 0
]

# --- Plot bar graph comparing models ---
metrics = ['Test Accuracy', 'Normal Accuracy', 'Pneumonia Recall']
x = np.arange(len(metrics))
width = 0.35

plt.bar(x - width/2, cnn_scores, width, label='CNN', color='skyblue')
plt.bar(x + width/2, vgg_scores, width, label='VGG16', color='mediumseagreen')

plt.xticks(x, metrics)
plt.ylim(0, 1.1)
plt.ylabel('Score')
plt.title('Model Comparison by Metric')
plt.legend()
plt.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.show()

# --- Plot Loss/Accuracy Curves ---
def plot_history(history, title):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,4))
    
    ax1.plot(history.history['loss'], label='Train Loss')
    ax1.plot(history.history['val_loss'], label='Val Loss')
    ax1.set_title(f'{title} - Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(history.history['accuracy'], label='Train Accuracy')
    ax2.plot(history.history['val_accuracy'], label='Val Accuracy')
    ax2.set_title(f'{title} - Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

# --- Plot them for each model ---
plot_history(history_cnn, "Custom CNN")
plot_history(history_vgg, "VGG16")
