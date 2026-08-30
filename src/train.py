import argparse
import datetime
import os
import time
import tensorflow as tf
from dataset import build_datasets, download_maps_dataset
from models import Discriminator, Generator
from losses import discriminator_loss, generator_loss
from utils import generate_and_save_images
def get_datasets(args):
    if args.data_dir:
        data_dir = args.data_dir
    elif args.dataset == "maps":
        print("Downloading/locating the 'maps' dataset (this may take a while the first time)...")
        data_dir = download_maps_dataset()
    else:
        raise ValueError("Provide either --dataset maps or --data-dir <path>")
    print(f"Using dataset directory: {data_dir}")
    return build_datasets(data_dir, batch_size=args.batch_size)
@tf.function
def train_step(generator, discriminator, generator_optimizer, discriminator_optimizer,
                input_image, target, summary_writer, step):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator(input_image, training=True)
        disc_real_output = discriminator([input_image, target], training=True)
        disc_generated_output = discriminator([input_image, gen_output], training=True)
        gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(
            disc_generated_output, gen_output, target)
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)
    generator_gradients = gen_tape.gradient(gen_total_loss, generator.trainable_variables)
    discriminator_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    generator_optimizer.apply_gradients(
        zip(generator_gradients, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(
        zip(discriminator_gradients, discriminator.trainable_variables))
    if summary_writer is not None:
        with summary_writer.as_default():
            tf.summary.scalar("gen_total_loss", gen_total_loss, step=step)
            tf.summary.scalar("gen_gan_loss", gen_gan_loss, step=step)
            tf.summary.scalar("gen_l1_loss", gen_l1_loss, step=step)
            tf.summary.scalar("disc_loss", disc_loss, step=step)
    return gen_total_loss, disc_loss
def fit(generator, discriminator, generator_optimizer, discriminator_optimizer,
        train_ds, test_ds, epochs, checkpoint, checkpoint_prefix,
        samples_dir, log_dir, checkpoint_every=5):
    summary_writer = tf.summary.create_file_writer(
        os.path.join(log_dir, datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))
    example_input, example_target = next(iter(test_ds.take(1)))
    step = tf.Variable(0, dtype=tf.int64)
    start = time.time()
    for epoch in range(epochs):
        epoch_start = time.time()
        print(f"\nEpoch {epoch + 1}/{epochs}")
        for n, (input_image, target) in train_ds.enumerate():
            gen_loss, disc_loss = train_step(
                generator, discriminator, generator_optimizer, discriminator_optimizer,
                input_image, target, summary_writer, step)
            step.assign_add(1)
            if n % 100 == 0:
                print(f"  step {int(n)}: gen_loss={float(gen_loss):.3f}  disc_loss={float(disc_loss):.3f}")
        print(f"Epoch {epoch + 1} took {time.time() - epoch_start:.1f}s")
        generate_and_save_images(
            generator, example_input, example_target,
            os.path.join(samples_dir, f"epoch_{epoch + 1:03d}.png"))
        if (epoch + 1) % checkpoint_every == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)
            print(f"Saved checkpoint at epoch {epoch + 1}")
    checkpoint.save(file_prefix=checkpoint_prefix)
    print(f"\nTraining complete. Total time: {time.time() - start:.1f}s")
def main():
    parser = argparse.ArgumentParser(description="Train pix2pix (conditional GAN)")
    parser.add_argument("--dataset", choices=["maps"], default="maps",
                         help="Named dataset to auto-download")
    parser.add_argument("--data-dir", default=None,
                         help="Path to a custom dataset dir (overrides --dataset)")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=1,
                         help="pix2pix paper uses batch size 1 (per-image instance norm behavior via batchnorm)")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--samples-dir", default="outputs/training_samples")
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--checkpoint-every", type=int, default=5)
    args = parser.parse_args()
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.samples_dir, exist_ok=True)
    train_ds, test_ds = get_datasets(args)
    generator = Generator()
    discriminator = Discriminator()
    generator_optimizer = tf.keras.optimizers.Adam(args.lr, beta_1=0.5)
    discriminator_optimizer = tf.keras.optimizers.Adam(args.lr, beta_1=0.5)
    checkpoint_prefix = os.path.join(args.checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(
        generator_optimizer=generator_optimizer,
        discriminator_optimizer=discriminator_optimizer,
        generator=generator,
        discriminator=discriminator,)
    fit(generator, discriminator, generator_optimizer, discriminator_optimizer,
        train_ds, test_ds, args.epochs, checkpoint, checkpoint_prefix,
        args.samples_dir, args.log_dir, args.checkpoint_every,)
if __name__ == "__main__":
    main()
