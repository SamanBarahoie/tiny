

# Tiny-ImageNet Trainer 🧠📦

This project is a class-based PyTorch trainer for the **Tiny-ImageNet-200** dataset using a **heavyweight ResNet-152** model. It's suitable for training on image classification tasks or quickly **benchmarking your GPU** performance.

---

## 🚀 Features

- Downloads and prepares the Tiny-ImageNet dataset automatically
- Uses a deep ResNet-152 model (can switch to pretrained or custom models)
- Monitors GPU memory usage
- Simple training & evaluation loop
- Great for quick **stress tests** on your hardware

---

## 📦 Requirements

Install the required packages:

```bash
pip install torch torchvision tqdm requests gputil
```

---

## 📁 How to Use

Run the trainer:

```bash
python tiny.py
```

By default, it trains for a very large number of epochs (`epochs=500000`) so you can observe GPU usage over time. You can reduce this value for shorter tests.

---

## ⚡ Quick GPU Benchmarking

This project is useful to **test your GPU load** using a heavy model (`ResNet152`) and a relatively large dataset. You can monitor GPU memory usage through:

- The **printed report** at the end:
  ```
  ====== Final GPU Usage Report ======
  GPU Memory: XXXX MB / YYYY MB
  ```

- Or live with tools like:
  - `nvidia-smi` (terminal)
  - `watch -n 1 nvidia-smi`
  - `gpustat`
  - `htop` + `nvtop` (Linux)

---

## ⚙️ Adjusting Hyperparameters

You can modify the training parameters in `tiny.py`:

```python
trainer = TinyImageNetTrainer(
    batch_size=64,
    image_size=256,
    num_workers=4,
    epochs=500000,
    lr=1e-3
)
```

For faster testing:
- Set `epochs=1` or `5`
- Reduce `batch_size` if you're getting OOM errors

---

## 📌 Notes

- This code does **not use pretrained weights** by default.
  You can enable them with:  
  `models.resnet152(pretrained=True)`

- Make sure your GPU has enough memory (recommended ≥ 8GB)

---

## 🧠 Dataset Info

Tiny-ImageNet-200:
- 200 classes
- 64x64 image resolution
- ~500 training images per class
- https://cs231n.stanford.edu/tiny-imagenet-200.zip

---

## 🛠 TODOs

- Add support for different architectures (e.g., ViT, EfficientNet)
- Implement learning rate scheduler
- Add checkpointing and resume

---

## 🧑‍💻 Author

Built with 💻 + ❤️ for educational and benchmarking purposes.

