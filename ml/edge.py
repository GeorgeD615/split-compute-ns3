import argparse
import pickle
import time

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

# Загрузка модели MobileNetV2
model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(model.last_channel, 10)
model.load_state_dict(torch.load("/home/george/ns-allinone-3.41/ns-3.41/ml/mobilenet_mnist.pth", map_location=torch.device('cpu')))
model.eval()

# Предобработка изображения
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307, 0.1307, 0.1307), (0.3081, 0.3081, 0.3081)),
    ])
    image = Image.open(image_path)
    return transform(image).unsqueeze(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", help="Path to input image")
    parser.add_argument("--split_layer", type=int, default=13, help="Layer index to split the model at")
    args = parser.parse_args()

    if not (0 <= args.split_layer <= 18):
        raise ValueError("split_layer must be between 0 and 18")

    input_tensor = preprocess_image(args.image_path)

    edge_part = nn.Sequential(*list(model.features)[:args.split_layer])

    start_edge = time.time()
    with torch.no_grad():
        edge_output = edge_part(input_tensor)
    latency_edge = time.time() - start_edge

    #Логгируем задержку
    with open("files/latency_edge.log", "w") as f:
        f.write(f"{latency_edge}\n")

    if edge_output.numel() == 0:
        print("[EDGE] Warning: Edge output is empty, skipping serialization.")
        open("files/input_tensor_on_edge.bin", "wb").close()
    else:
        with open("files/input_tensor_on_edge.bin", "wb") as f:
            pickle.dump((edge_output, args.split_layer, args.image_path), f)
        print(f"[EDGE] Inference complete. Split layer: {args.split_layer}")
