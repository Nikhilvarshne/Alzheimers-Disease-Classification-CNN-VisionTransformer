from google.colab import drive
drive.mount('/content/drive')

!pip install tensorflow
import tensorflow as tf
from tensorflow.keras import layers, models
import pandas as pd
import numpy as np
import os

csv_path = "/content/drive/MyDrive/alzheimer_dataset.csv"
image_root = "/content/drive/MyDrive/NEW DATASET"

df = pd.read_csv(csv_path)
print(df.head())
print("Total samples:", len(df))

import os
import pandas as pd
DATASET_ROOT = "/content/drive/MyDrive/NEW DATASET"
classes = os.listdir(DATASET_ROOT)
print("Classes found:", classes)

data = []
for cls in classes:
    class_path = os.path.join(DATASET_ROOT, cls)
    if not os.path.isdir(class_path):
        continue
    for img in os.listdir(class_path):
        if img.lower().endswith((".jpg", ".png", ".jpeg")):
            data.append({
                "image_path": os.path.join(class_path, img),
                "label_text": cls
            })
df = pd.DataFrame(data)
print(df.head())
print("Total images:", len(df))

df["label_clean"] = (
    df["label_text"]
    .str.lower()
    .str.replace(" ", "")
)

label_map = {
    "nondemented": 0,
    "verymilddementia": 1,
    "verymilddemented": 1,
    "verymilddimentia": 1,
    "verymilddimented": 1,
    "milddemented": 2,
    "milddimented": 2,
    "moderatedemented": 3,
    "modratedemented": 3,
    "moderatedimented": 3
}

df["label"] = df["label_clean"].map(label_map)
print("NaN labels:", df["label"].isna().sum())
print(df["label"].unique())
df["label"] = df["label"].astype("int32")
df.drop(columns=["label_text", "label_clean"], inplace=True)


from sklearn.model_selection import train_test_split
train_df, temp_df = train_test_split(
    df, test_size=0.2, stratify=df["label"], random_state=42
)
val_df, test_df = train_test_split(
    temp_df, test_size=0.5, stratify=temp_df["label"], random_state=42
)

import tensorflow as tf
IMG_SIZE = 224
BATCH_SIZE = 16
def make_dataset(dataframe, batch_size=BATCH_SIZE, shuffle=True):
    paths = dataframe["image_path"].values
    labels = dataframe["label"].values

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))

    def load_image(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
        img = tf.cast(img, tf.float32) / 255.0
        return img, label
    ds = ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

train_ds = make_dataset(train_df)
val_ds   = make_dataset(val_df, shuffle=False)
test_ds  = make_dataset(test_df, shuffle=False)


import numpy as np
import cv2
import matplotlib.pyplot as plt
# Collect pixel intensities from a subset (to save memory)
pixel_values = []
sample_paths = df["image_path"].sample(300, random_state=42)

for path in sample_paths:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is not None:
        pixel_values.extend(img.flatten())

pixel_values = np.array(pixel_values)
# Plot intensity distribution
plt.figure(figsize=(8,5))
plt.hist(pixel_values, bins=50, density=True)
plt.xlabel("Pixel Intensity Value")
plt.ylabel("Normalized Frequency")
plt.title("Typical MRI Intensity Distribution Curve")
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()

IMG_SIZE = 224
PATCH_SIZE = 16
NUM_CLASSES = 4
EMBED_DIM = 128
NUM_HEADS = 4
MLP_DIM = 256
NUM_LAYERS = 6
class PatchExtractor(layers.Layer):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size

    def call(self, images):
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1,1,1,1],
            padding='VALID'
        )
        batch = tf.shape(images)[0]
        patch_dim = patches.shape[-1]
        return tf.reshape(patches, [batch, -1, patch_dim])
class PatchEmbedding(layers.Layer):
    def __init__(self, num_patches, embed_dim):
        super().__init__()
        self.projection = layers.Dense(embed_dim)
        self.position_embedding = layers.Embedding(num_patches, embed_dim)

    def call(self, patches):
        positions = tf.range(start=0, limit=tf.shape(patches)[1], delta=1)
        return self.projection(patches) + self.position_embedding(positions)
def transformer_encoder(x):
    x1 = layers.LayerNormalization(epsilon=1e-6)(x)
    attention = layers.MultiHeadAttention(
        num_heads=NUM_HEADS, key_dim=EMBED_DIM
    )(x1, x1)
    x2 = layers.Add()([x, attention])

    x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
    ffn = layers.Dense(MLP_DIM, activation="gelu")(x3)
    ffn = layers.Dense(EMBED_DIM)(ffn)
    return layers.Add()([x2, ffn])
def build_vit():
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))

    patches = PatchExtractor(PATCH_SIZE)(inputs)
    num_patches = (IMG_SIZE // PATCH_SIZE) ** 2

    x = PatchEmbedding(num_patches, EMBED_DIM)(patches)

    for _ in range(NUM_LAYERS):
        x = transformer_encoder(x)

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(128, activation="gelu")(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)
    return models.Model(inputs, outputs)

model = build_vit()
model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

import matplotlib.pyplot as plt

plt.figure(figsize=(8,5))
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Training vs Validation Accuracy")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(8,5))
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid(True)
plt.show()

test_loss, test_acc = model.evaluate(test_ds)
print("Test Accuracy:", test_acc)
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

y_true, y_pred = [], []

for images, labels in test_ds:
    preds = model.predict(images)
    y_true.extend(labels.numpy())
    y_pred.extend(np.argmax(preds, axis=1))

print(classification_report(
    y_true, y_pred,
    target_names=[
        "NonDemented",
        "VeryMildDementia",
        "MildDemented",
        "ModerateDemented"
    ]
))

cm = confusion_matrix(y_true, y_pred)
print(cm)
model.save("/content/drive/MyDrive/vit_alzheimer_paper_model.h5")

