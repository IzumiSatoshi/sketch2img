import torch
from torch.utils.data import Dataset
import torchvision
from torchvision import transforms
import cv2
import pathlib


class CelebaDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir):
        super().__init__()

        self.data_dir = pathlib.Path(data_dir)
        self.images_path = [str(p) for p in self.data_dir.iterdir()]

        self.to_normalized_tensor = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        )

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, index):
        image_path = self.images_path[index]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        sketch = self.create_sketch(image)

        image_tensor = self.to_normalized_tensor(image)
        sketch_tensor = self.to_normalized_tensor(sketch)

        return image_tensor, sketch_tensor

    def create_sketch(self, image):
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_canny = cv2.Canny(image_gray, 100, 200)
        return image_canny


class FashionMNISTDataset(Dataset):
    def __init__(self, data_root):
        self.fmnist_dataset = torchvision.datasets.FashionMNIST(data_root)
        self.classes = self.fmnist_dataset.classes

    def __len__(self):
        return len(self.fmnist_dataset)

    def __getitem__(self, index):
        img = self.fmnist_dataset.data[index]
        sketch = torch.from_numpy(cv2.Canny(img.numpy(), 200, 200))

        # to shape: [1, h, w]
        img = img.unsqueeze(0).float()
        sketch = sketch.unsqueeze(0).float()

        img = self.normalize(img)
        sketch = self.normalize(sketch)

        return img, sketch

    def normalize(self, x):
        # map x to -1 < x < 1
        # I'm doing normalization with zero understanding :o
        x /= 255
        x = torchvision.transforms.Normalize(0.5, 0.5)(x)
        return x
