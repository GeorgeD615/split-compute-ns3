import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = models.mobilenet_v2(pretrained=False).to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor()
])
dataset = MNIST(root='./data', train=False, transform=transform, download=True)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

tensor_sizes = {}

def register_hooks(module, name):
    def hook(module, input, output):
        if isinstance(output, torch.Tensor):
            size_bytes = output.element_size() * output.nelement()
            tensor_sizes[name] = {
                'shape': list(output.shape),
                'num_elements': output.nelement(),
                'bytes': size_bytes
            }
    module.register_forward_hook(hook)


for name, module in model.named_modules():
    if not isinstance(module, nn.Sequential) and name != "":
        register_hooks(module, name)


with torch.no_grad():
    for img, _ in dataloader:
        img = img.to(device)
        _ = model(img)
        break

print(f"{'Layer':<40} {'Shape':<25} {'Elements':<10} {'Bytes':<10}")
print('-' * 90)
for name, info in tensor_sizes.items():
    shape = str(info['shape'])
    print(f"{name:<40} {shape:<25} {info['num_elements']:<10} {info['bytes']:<10}")
