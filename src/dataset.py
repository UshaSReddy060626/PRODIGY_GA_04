import os
import pathlib
import tensorflow as tf
MAPS_URL = "http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/maps.tar.gz"
IMG_WIDTH = 256
IMG_HEIGHT = 256
def download_maps_dataset(cache_dir="~/.keras"):
    """Downloads and extracts the maps dataset, returning its local path."""
    path_to_zip = tf.keras.utils.get_file(fname="maps.tar.gz",
        origin=MAPS_URL,
        extract=True,
        cache_dir=os.path.expanduser(cache_dir),)
    path_to_zip = pathlib.Path(path_to_zip)
    extracted_dir = path_to_zip.parent / "maps_extracted" / "maps"
    if extracted_dir.exists():
        return extracted_dir
    return path_to_zip.parent / "maps"
def load_image_pair(image_file):
    """Reads a single side-by-side image file and splits it into
    (input_image, real_image) float32 tensors in range [-1, 1]."""
    image = tf.io.read_file(image_file)
    image = tf.io.decode_jpeg(image)
    w = tf.shape(image)[1]
    w = w // 2
    input_image = image[:, w:, :]   # right half = target/map in the maps dataset
    real_image = image[:, :w, :]    # left half = input/satellite photo
    input_image = tf.cast(input_image, tf.float32)
    real_image = tf.cast(real_image, tf.float32)
    return input_image, real_image
def resize(input_image, real_image, height, width):
    input_image = tf.image.resize(
        input_image, [height, width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    real_image = tf.image.resize(
        real_image, [height, width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return input_image, real_image
def random_crop(input_image, real_image):
    stacked_image = tf.stack([input_image, real_image], axis=0)
    cropped_image = tf.image.random_crop(
        stacked_image, size=[2, IMG_HEIGHT, IMG_WIDTH, 3])
    return cropped_image[0], cropped_image[1]
def normalize(input_image, real_image):
    """Scale pixel values from [0, 255] to [-1, 1] (matches the tanh
    output activation of the generator)."""
    input_image = (input_image / 127.5) - 1
    real_image = (real_image / 127.5) - 1
    return input_image, real_image
@tf.function()
def random_jitter(input_image, real_image):
    """Standard pix2pix data augmentation: resize up, random crop back
    down, then random horizontal flip."""
    input_image, real_image = resize(input_image, real_image, 286, 286)
    input_image, real_image = random_crop(input_image, real_image)
    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.flip_left_right(input_image)
        real_image = tf.image.flip_left_right(real_image)
    return input_image, real_image
def load_train_image(image_file):
    input_image, real_image = load_image_pair(image_file)
    input_image, real_image = random_jitter(input_image, real_image)
    input_image, real_image = normalize(input_image, real_image)
    return input_image, real_image
def load_test_image(image_file):
    input_image, real_image = load_image_pair(image_file)
    input_image, real_image = resize(input_image, real_image, IMG_HEIGHT, IMG_WIDTH)
    input_image, real_image = normalize(input_image, real_image)
    return input_image, real_image
def build_datasets(data_dir, batch_size=1, buffer_size=400):
    """Returns (train_dataset, test_dataset) tf.data.Dataset objects.
    `data_dir` should contain `train/` and (`val/` or `test/`)
    subfolders of side-by-side paired images, e.g. the layout produced
    by download_maps_dataset().
    """
    data_dir = pathlib.Path(data_dir)
    train_dataset = tf.data.Dataset.list_files(str(data_dir / "train" / "*.jpg"))
    train_dataset = train_dataset.map(
        load_train_image, num_parallel_calls=tf.data.AUTOTUNE)
    train_dataset = train_dataset.shuffle(buffer_size)
    train_dataset = train_dataset.batch(batch_size)
    test_dir = data_dir / "val"
    if not test_dir.exists():
        test_dir = data_dir / "test"
    test_dataset = tf.data.Dataset.list_files(str(test_dir / "*.jpg"))
    test_dataset = test_dataset.map(load_test_image)
    test_dataset = test_dataset.batch(batch_size)
    return train_dataset, test_dataset
