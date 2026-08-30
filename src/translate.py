import argparse
import os
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from models import Generator
def load_single_image(path, img_height=256, img_width=256):
    image = tf.io.read_file(path)
    image = tf.io.decode_image(image, channels=3, expand_animations=False)
    image = tf.image.resize(image, [img_height, img_width])
    image = tf.cast(image, tf.float32)
    image = (image / 127.5) - 1  # normalize to [-1, 1]
    return image[tf.newaxis, ...]  # add batch dimension
def restore_generator(checkpoint_dir):
    generator = Generator()
    checkpoint = tf.train.Checkpoint(generator=generator)
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    if latest is None:
        raise FileNotFoundError(
            f"No checkpoint found in '{checkpoint_dir}'. Train the model first with train.py.")
    checkpoint.restore(latest).expect_partial()
    print(f"Restored generator from checkpoint: {latest}")
    return generator
def translate_image(generator, input_path, output_path):
    input_image = load_single_image(input_path)
    prediction = generator(input_image, training=True)  # pix2pix uses training=True at inference too (see paper note below)
    plt.figure(figsize=(8, 8))
    plt.imshow(prediction[0] * 0.5 + 0.5)
    plt.axis("off")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0)
    plt.close()
    print(f"Saved translated image to: {output_path}")
def main():
    parser = argparse.ArgumentParser(description="Run a trained pix2pix generator on an image")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--input", required=True, help="Path to input image")
    parser.add_argument("--output", default="outputs/translated.png")
    args = parser.parse_args()
    generator = restore_generator(args.checkpoint_dir)
    translate_image(generator, args.input, args.output)
if __name__ == "__main__":
    main()
