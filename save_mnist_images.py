from torchvision import datasets, transforms
from PIL import Image
import os

output_dir = "ml/mnist_images"
os.makedirs(output_dir, exist_ok=True)

mnist = datasets.MNIST(root=".", train=True, download=True, transform=transforms.ToTensor())

NUM_IMAGES = 20

for i in range(NUM_IMAGES):
    img, label = mnist[i]
    img = transforms.ToPILImage()(img)
    img.save(f"{output_dir}/{label}_{i}.png")
