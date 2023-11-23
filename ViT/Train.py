import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import cv2
from glob import glob
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from patchify import patchify
import tensorflow as tf
from ViT import Vit
import keras
from keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping

hp = {}
hp["image_size"] = 200
hp["num_channels"] = 3
hp["patch_size"] = 25
hp["num_patches"] = (hp["image_size"] ** 2) // (hp["patch_size"] ** 2)
hp["flat_patches_shape"] = (hp["num_patches"], hp["patch_size"] * hp["patch_size"] * hp["num_channels"])

hp["batch_size"] = 32
# learning rate
hp["lr"] = 1e-4
# stop point
hp["num_epochs"] = 500
hp["num_classes"] = 5
hp["class_names"] = ["daisy", "dandelion", "roses", "sunflowers", "tulips"]

hp["num_layers"] = 12
hp["hidden_dim"] = 768
hp["mlp_dim"] = 3072
hp["num_heads"] = 12
hp["dropout_rate"] = 0.1


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def load_data(path, split=0.1):
    images = glob(os.path.join(path, "*", "*.jpg"))

    split_size = int(len(images) * split)
    train_x, valid_x = train_test_split(images, test_size=split_size, random_state=42);
    train_x, test_x = train_test_split(train_x, test_size=split_size, random_state=42);

    return train_x, valid_x, test_x


def process_image_label(path):
    path = path.decode()
    # Reading image
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (hp["image_size"], hp["image_size"]))
    image = image / 255.0
    # print(image.shape)
    # Pre processing to patches
    patch_shape = (hp["patch_size"], hp["patch_size"], hp["num_channels"])
    patches = patchify(image, patch_shape, hp["patch_size"])

    patches = np.reshape(patches, hp["flat_patches_shape"])
    patches = patches.astype(np.float32)
    # print(patches.shape)
    # Label
    # print(path)
    class_name = path.split("\\")[-2]
    # print(class_name)
    class_idx = hp["class_names"].index(class_name)
    class_idx = np.array(class_idx, dtype=np.int32)

    # print(class_idx)
    return patches, class_idx

# Convert path to patches image and label vector
def parse(path):
    patches, labels = tf.numpy_function(process_image_label, [path], [tf.float32, tf.int32])
    # convert number to vector (2 -> [0,0,1,0,0])
    labels = tf.one_hot(labels, hp["num_classes"])

    patches.set_shape(hp["flat_patches_shape"])
    labels.set_shape(hp["num_classes"])
    return patches, labels

# Hàm tf_dataset có ý nghĩa là tạo một đối tượng dataset từ một danh sách các đường dẫn ảnh.
# Dataset là một cấu trúc dữ liệu của TensorFlow cho phép bạn xử lý, biến đổi và trộn các dữ liệu
# một cách hiệu quả và song song. Hàm tf_dataset sẽ làm các việc sau:

# Tạo một đối tượng tf.data.Dataset từ một mảng tensor chứa các đường dẫn ảnh.
# Áp dụng hàm parse cho mỗi phần tử của dataset để trả về patches và nhãn.
# Gom nhóm các phần tử thành các batch có kích thước bằng tham số batch và sử dụng phương pháp prefetch để tăng tốc độ xử lý.
# Trả về dataset.
def tf_dataset(images, batch=32):
    ds = tf.data.Dataset.from_tensor_slices((images))
    ds = ds.map(parse).batch(batch).prefetch(8)
    return ds


if __name__ == "__main__":
    # Seeding
    np.random.seed(42)
    tf.random.set_seed(42)

    # Directory for storing file
    create_dir("files")

    # Paths
    dataset_path = "H:\LearnBasicPython\\ViT\\flower_photos"
    model_path = os.path.join("files", "model.h5")
    csv_path = os.path.join("files", "log.csv")

    # Dataset
    train_x, valid_x, test_x = load_data(dataset_path)
    # print(f"Train:{len(train_x)} - Valid: {len(valid_x)} - Test: {len(test_x)}")
    # process_image_label(train_x[0])

    train_ds = tf_dataset(train_x, batch=hp["batch_size"])
    valid_ds = tf_dataset(valid_x, batch=hp["batch_size"])
    # Model
    model = Vit(hp)
    model.compile(
        loss="categorical_crossentropy",
        optimizer=keras.optimizers.Adam(hp["lr"], clipvalue=1.0),
        metrics=["acc"]
    )
    callbacks = [
        ModelCheckpoint(model_path, monitor='val_loss', verbose=1, save_best_only=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, min_lr=1e-10, verbose=1),
        CSVLogger(csv_path),
        EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=False),
    ]

    model.fit(
        train_ds,
        epochs=hp["num_epochs"],
        validation_data=valid_ds,
        callbacks=callbacks
    )