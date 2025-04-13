import os
import zipfile
import shutil
import requests
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import GPUtil

class TinyImageNetTrainer:
    def __init__(self, dataset_path='tiny-imagenet-200', batch_size=64, image_size=224,
                 num_workers=4, epochs=1, lr=1e-3):
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_workers = num_workers
        self.epochs = epochs
        self.lr = lr
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def download_and_prepare_dataset(self):
        if os.path.exists(self.dataset_path):
            print("Dataset already exists. Skipping download.")
            return

        print("Downloading Tiny-ImageNet dataset...")
        url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
        try:
            r = requests.get(url, stream=True)
            r.raise_for_status()
            with open("tiny-imagenet-200.zip", "wb") as f:
                total = int(r.headers.get('content-length', 0))
                for chunk in tqdm(r.iter_content(chunk_size=1024), total=total // 1024, unit='KB'):
                    f.write(chunk)
        except requests.exceptions.RequestException as e:
            print(f"Download error: {e}")
            return

        print("Extracting file...")
        try:
            with zipfile.ZipFile("tiny-imagenet-200.zip", 'r') as zip_ref:
                zip_ref.extractall()
        except zipfile.BadZipFile:
            print("Extraction error: Invalid zip file.")
            return

        val_dir = os.path.join(self.dataset_path, "val")
        val_images_dir = os.path.join(val_dir, "images_by_class")

        if not os.path.exists(val_images_dir):
            print("Organizing validation images by class...")
            os.makedirs(val_images_dir, exist_ok=True)
            annotations_file = os.path.join(val_dir, "val_annotations.txt")

            if not os.path.exists(annotations_file):
                print("Annotations file not found.")
                return

            with open(annotations_file, 'r') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    img_name, class_name = parts[0], parts[1]
                    class_dir = os.path.join(val_images_dir, class_name)
                    os.makedirs(class_dir, exist_ok=True)
                    src = os.path.join(val_dir, "images", img_name)
                    dst = os.path.join(class_dir, img_name)
                    if os.path.exists(src):
                        shutil.move(src, dst)
                    else:
                        print(f"Image not found: {src}")

            print("Validation images organized.")

    def get_dataloaders(self):
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        train_dir = os.path.join(self.dataset_path, "train")
        val_dir = os.path.join(self.dataset_path, "val", "images_by_class")

        if not os.path.exists(train_dir) or not os.path.exists(val_dir):
            print("Dataset path not found. Please check the path.")
            return None, None

        train_dataset = ImageFolder(train_dir, transform=transform)
        val_dataset = ImageFolder(val_dir, transform=transform)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True,
                                  num_workers=self.num_workers, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False,
                                num_workers=self.num_workers, pin_memory=True)

        return train_loader, val_loader

    def train_one_epoch(self, model, loader, optimizer, criterion):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for inputs, targets in tqdm(loader, desc="Training", leave=False):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)

        avg_loss = total_loss / len(loader)
        acc = 100. * correct / total
        return avg_loss, acc

    def evaluate(self, model, loader, criterion):
        model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in tqdm(loader, desc="Validating", leave=False):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                total_loss += loss.item()
                _, predicted = outputs.max(1)
                correct += predicted.eq(targets).sum().item()
                total += targets.size(0)

        avg_loss = total_loss / len(loader)
        acc = 100. * correct / total
        return avg_loss, acc

    def report_gpu_usage(self):
        if not torch.cuda.is_available():
            return 0, 0
        gpu = GPUtil.getGPUs()[0]
        return gpu.memoryUsed, gpu.memoryTotal

    def run(self):
        self.download_and_prepare_dataset()
        train_loader, val_loader = self.get_dataloaders()

        if train_loader is None or val_loader is None:
            print("Error creating dataloaders.")
            return

        print("Loading heavy model (ResNet152)...")
        model = models.resnet152(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 200)
        model.to(self.device)

        optimizer = optim.Adam(model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(1, self.epochs + 1):
            print(f"\nEpoch {epoch}/{self.epochs}")
            train_loss, train_acc = self.train_one_epoch(model, train_loader, optimizer, criterion)
            val_loss, val_acc = self.evaluate(model, val_loader, criterion)
            print(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
            print(f"Val   Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")

        mem_used, mem_total = self.report_gpu_usage()
        print("\n====== Final GPU Usage Report ======")
        print(f"GPU Memory: {mem_used} MB / {mem_total} MB")

if __name__ == "__main__":
    trainer = TinyImageNetTrainer(
        batch_size=64,
        image_size=224,
        num_workers=4,
        epochs=1000,
        lr=1e-3
    )
    trainer.run()