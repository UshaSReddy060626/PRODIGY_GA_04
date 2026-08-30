# 🖼️ Image-to-Image Translation with cGAN

> Translate images from one visual domain to another using a Conditional Generative Adversarial Network (cGAN).

This project implements **Image-to-Image Translation with a Conditional Generative Adversarial Network (cGAN)** based on the **pix2pix architecture** introduced by **Isola et al.** It is designed for **paired image-to-image translation**.

Unlike a traditional GAN that generates images from random noise, this model learns a mapping between corresponding input and target images:

```text
Input Image
     ↓
Conditional GAN
     ↓
Translated Image
```

Examples include:

```text
Satellite Image → Map
Building Photo  → Segmentation Map
Sketch           → Realistic Photo
Day Image        → Night Image
```

The implementation contains the complete training pipeline, including dataset loading, augmentation, U-Net generator, PatchGAN discriminator, adversarial training, L1 reconstruction loss, checkpointing, and image translation.

---

# ✨ What is Image-to-Image Translation with cGAN?

A **Conditional Generative Adversarial Network (cGAN)** can generate an output image based on a given input condition.

This project uses the **pix2pix approach** for paired image-to-image translation.

The model learns from pairs of corresponding images:

```text
Input Image  →  Target Image
```

For example:

```text
Satellite Photo  →  Geographic Map
```

During training, the generator learns to produce an image that:

1. Looks realistic.
2. Matches the structure of the target image.
3. Preserves important information from the input image.

---

# 🧠 How Does a cGAN Work?

A standard GAN contains two main components:

* **Generator** — creates generated images.
* **Discriminator** — determines whether an image is real or generated.

A **Conditional GAN** additionally provides an input condition to guide the generation process.

Instead of:

```text
Random Noise
      ↓
  Generator
      ↓
    Image
```

this project uses:

```text
Input Image
      ↓
  Generator
      ↓
Generated Image
```

The discriminator receives both the input image and the target/generated image.

```text
                 ┌───────────────┐
Input Image ───► │   Generator   │ ───► Generated Image
                 └───────────────┘
                         │
                         ▼
                   ┌─────────────┐
Input Image ─────► │ Discriminator│
Target Image ────► │             │ ───► Real / Fake
                   └─────────────┘
```

The two networks learn together.

### Generator

The generator tries to:

* Fool the discriminator.
* Produce realistic outputs.
* Stay close to the expected target.

### Discriminator

The discriminator tries to distinguish between:

```text
Real Pair:
(Input Image, Target Image)

Fake Pair:
(Input Image, Generated Image)
```

As training progresses, both networks improve.

---

# 🏗️ Architecture

The project uses two major neural network components:

| Component               | Architecture | Purpose                                               |
| ----------------------- | ------------ | ----------------------------------------------------- |
| 🎨 Generator            | U-Net        | Converts input images into translated images          |
| 🔍 Discriminator        | PatchGAN     | Determines whether local image patches look realistic |
| 📉 Reconstruction       | L1 Loss      | Keeps generated image close to target                 |
| ⚔️ Adversarial Training | cGAN Loss    | Encourages realistic outputs                          |

---

# 🎨 Generator — U-Net

The generator uses a **U-Net encoder-decoder architecture**.

```text
Input Image
     ↓
  Encoder
     ↓
Feature Representation
     ↓
  Decoder
     ↓
Output Image
```

The important feature of U-Net is its **skip connections**.

```text
Encoder ───────────────► Decoder
   │                       │
   ├───────────────────────┤
   ├───────────────────────┤
   └───────────────────────┘
```

Skip connections allow low-level spatial information to bypass the bottleneck and reach the decoder.

This helps preserve:

* Edges
* Shapes
* Fine details
* Spatial structure

Without skip connections, the generator can lose important information during encoding.

---

# 🔍 Discriminator — PatchGAN

The discriminator uses **PatchGAN** rather than classifying the entire image as a single unit.

Instead of asking:

```text
Is this entire image real or fake?
```

PatchGAN evaluates local regions:

```text
┌────┬────┬────┐
│ P1 │ P2 │ P3 │
├────┼────┼────┤
│ P4 │ P5 │ P6 │
├────┼────┼────┤
│ P7 │ P8 │ P9 │
└────┴────┴────┘
```

Each patch receives a real/fake prediction.

This encourages the generator to produce realistic:

* Textures
* Edges
* Local patterns
* Fine details

The implementation follows the **70×70 PatchGAN** design described in the original pix2pix work.

---

# 📉 Loss Functions

The generator is trained using two main objectives:

```text
Generator Loss
      =
Adversarial Loss
      +
λ × L1 Loss
```

The implementation uses:

```text
λ = 100
```

---

## ⚔️ Adversarial Loss

The adversarial component encourages the generator to create images that the discriminator considers realistic.

```text
Generated Image
      ↓
Discriminator
      ↓
Should look REAL
```

Without adversarial loss, the generated images may become overly smooth.

---

## 📐 L1 Reconstruction Loss

The L1 loss directly compares the generated image with the target image.

```text
L1 Loss = |Target Image - Generated Image|
```

It encourages the output to remain structurally similar to the target.

This is particularly important for paired image translation.

---

## 🎯 Why Combine Both?

Adversarial loss and L1 loss solve different problems.

```text
Adversarial Loss
       ↓
Realistic appearance

       +

L1 Loss
       ↓
Correct structure and content

       ↓

Better translated image
```

Adversarial loss makes the image look realistic, while L1 loss helps ensure that it corresponds correctly to the target.

---

# 🔄 Training Pipeline

The complete training process can be summarized as:

```text
Paired Dataset
      ↓
Load Input + Target
      ↓
Data Augmentation
      ↓
Normalization
      ↓
U-Net Generator
      ↓
Generated Image
      ↓
PatchGAN Discriminator
      ↓
Calculate Losses
      ↓
Backpropagation
      ↓
Update Generator
      ↓
Update Discriminator
      ↓
Save Checkpoint
      ↓
Generate Sample
      ↓
Repeat
```

---

# 🧹 Data Preprocessing

The dataset pipeline supports common preprocessing operations:

* Random jitter
* Random cropping
* Random horizontal flipping
* Image resizing
* Normalization

These transformations help the model generalize better and provide additional variation during training.

---

# 📦 Dataset Format

The model expects **paired images**.

A training example contains:

```text
[ Input Image | Target Image ]
```

For example:

```text
┌──────────────────┬──────────────────┐
│   Satellite      │       Map        │
│     Image        │                  │
└──────────────────┴──────────────────┘
```

The dataset can be organized as:

```text
dataset/
├── train/
│   ├── image_001.jpg
│   ├── image_002.jpg
│   └── image_003.jpg
│
└── val/
    ├── image_001.jpg
    ├── image_002.jpg
    └── image_003.jpg
```

Each image should contain the input and target side-by-side.

---

# 🗂️ Project Structure

```text
image-to-image-translation-cgan/
│
├── src/
│   ├── dataset.py
│   ├── models.py
│   ├── losses.py
│   ├── train.py
│   ├── translate.py
│   └── utils.py
│
├── examples/
│   └── my_input.jpg
│
├── outputs/
│   └── training_samples/
│
├── checkpoints/
│
├── requirements.txt
└── README.md
```

### File Overview

| File           | Purpose                                      |
| -------------- | -------------------------------------------- |
| `dataset.py`   | Dataset loading, augmentation, preprocessing |
| `models.py`    | U-Net Generator and PatchGAN Discriminator   |
| `losses.py`    | Generator and Discriminator losses           |
| `train.py`     | Training loop and checkpoint management      |
| `translate.py` | Translation using a trained generator        |
| `utils.py`     | Image saving and visualization utilities     |
| `examples/`    | Test images                                  |
| `outputs/`     | Generated samples                            |
| `checkpoints/` | Saved model weights                          |

---

# 🛠️ Technologies Used

* **Python**
* **TensorFlow**
* **NumPy**
* **Pillow**
* **Deep Learning**
* **Generative Adversarial Networks**
* **Conditional GANs**
* **Convolutional Neural Networks**
* **U-Net**
* **PatchGAN**
* **Computer Vision**

---

# ⚙️ Installation

## Windows — PowerShell

```powershell
git clone https://github.com/<your-username>/image-to-image-translation-cgan.git
cd image-to-image-translation-cgan
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## macOS / Linux

```bash
git clone https://github.com/<your-username>/image-to-image-translation-cgan.git
cd image-to-image-translation-cgan
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

# 🚀 Usage

## 1. Train on the Maps Dataset

The project supports the **maps** dataset by default.

```powershell
python src\train.py --dataset maps --epochs 40
```

The dataset is downloaded automatically on the first run.

---

## 2. Train on Your Own Dataset

For a custom paired dataset:

```powershell
python src\train.py --data-dir path\to\your\dataset --epochs 40
```

Expected structure:

```text
dataset/
├── train/
│   ├── train_001.jpg
│   ├── train_002.jpg
│   └── ...
│
└── val/
    ├── val_001.jpg
    ├── val_002.jpg
    └── ...
```

---

# 🎛️ Training Parameters

| Parameter            | Default | Description                       |
| -------------------- | ------: | --------------------------------- |
| `--epochs`           |    `40` | Number of training epochs         |
| `--batch-size`       |     `1` | Training batch size               |
| `--lr`               |  `2e-4` | Learning rate for Adam optimizers |
| `--checkpoint-every` |     `5` | Save checkpoint every N epochs    |

### Batch Size

The implementation uses:

```text
batch-size = 1
```

which is consistent with the original pix2pix setup.

### Learning Rate

The default learning rate is:

```text
2e-4
```

for both the generator and discriminator optimizers.

---

# 💾 Checkpoints

Training checkpoints are saved inside:

```text
checkpoints/
```

For example:

```text
checkpoints/
├── epoch_005/
├── epoch_010/
├── epoch_015/
└── ...
```

Checkpoints allow training to be resumed and provide saved versions of the trained model.

---

# 🖼️ Training Samples

After each epoch, the training process generates a sample comparison containing:

```text
Input
  +
Ground Truth
  +
Prediction
```

Conceptually:

```text
┌──────────────┬──────────────┬──────────────┐
│    Input     │ Ground Truth │  Prediction  │
└──────────────┴──────────────┴──────────────┘
```

These samples are stored in:

```text
outputs/training_samples/
```

This makes it possible to monitor how the generator improves throughout training.

---

# 🔮 Translate a New Image

Once a trained checkpoint is available, a new image can be translated using:

```powershell
python src\translate.py --input examples\my_input.jpg --output outputs\translated.png
```

The trained generator processes the input image and produces the translated result.

---

# ⏱️ Training Performance

The model is trained **from scratch**, so training is significantly more computationally expensive than using a pretrained model.

Approximate experience:

```text
CPU
 ↓
Long training time

GPU
 ↓
Much faster training
```

The exact training time depends on:

* GPU/CPU hardware
* Image resolution
* Dataset size
* Number of epochs
* Batch size

For experimentation, training for fewer epochs can be useful to verify that the pipeline works correctly before starting a full run.

---

# 🔬 What This Project Demonstrates

This implementation provides practical experience with:

* Conditional GANs
* GAN training
* Generator vs Discriminator optimization
* U-Net architecture
* Skip connections
* PatchGAN
* Adversarial loss
* L1 reconstruction loss
* Paired image datasets
* Data augmentation
* Image normalization
* Checkpointing
* Image-to-image translation
* Computer Vision
* Deep Learning

---

# 💡 Key Learning

Image-to-image translation with cGANs demonstrates how GANs can be conditioned on an existing image instead of generating an image from random noise.

The central idea is:

```text
Input Image
     ↓
Conditional Information
     ↓
Generator
     ↓
Translated Image
     ↓
Discriminator
     ↓
Realistic + Correct Output
```

The project also demonstrates why combining different objectives is useful:

```text
Adversarial Loss
       ↓
Realistic appearance

L1 Loss
       ↓
Correct correspondence

       ↓

High-quality image translation
```

---

# ⚠️ Limitations

The current implementation has several limitations:

* Requires paired training data.
* Training from scratch can be computationally expensive.
* GAN training can be unstable.
* Results depend heavily on dataset quality.
* A model trained for one translation task may not generalize to another domain.
* High-resolution image translation requires significantly more computational resources.

---

# 🔮 Future Improvements

Possible extensions include:

* Add GPU/mixed-precision training
* Add training resume functionality
* Add TensorBoard training visualization
* Add validation metrics
* Add side-by-side result visualization
* Support unpaired image translation
* Implement CycleGAN
* Add a Streamlit interface
* Add a Flask API
* Support higher-resolution images
* Add experiment tracking
* Compare different GAN architectures
* Add video-to-video translation

---

# 📚 References

* [TensorFlow — pix2pix Tutorial](https://www.tensorflow.org/tutorials/generative/pix2pix)
* Isola, Zhu, Zhou & Efros — [Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/abs/1611.07004)
* [Conditional Generative Adversarial Network — GeeksforGeeks](https://www.geeksforgeeks.org/conditional-generative-adversarial-network/)
* [cGAN: How to Gain Control Over GAN Outputs](https://scribe.rip/cgan-conditional-generative-adversarial-network-how-to-gain-control-over-gan-outputs-b30620bd0cc8)

---

# ⭐ Conclusion

This project implements **Image-to-Image Translation with a Conditional Generative Adversarial Network (cGAN)** for paired image-to-image translation.

It combines a **U-Net Generator**, **PatchGAN Discriminator**, **adversarial loss**, and **L1 reconstruction loss** to learn mappings between visual domains.

The project provides both a practical implementation and a deeper understanding of how GANs can learn structured transformations such as:

```text
Satellite Image → Map
Sketch → Photo
Photo → Segmentation
```

Overall, the project demonstrates how **conditional generative models can learn meaningful visual transformations from paired examples**.

---

# 👩‍💻 Author

**Usha S. Reddy**

If you found this project useful, consider giving the repository a ⭐.
