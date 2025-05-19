import argparse
import os
import pickle
import time
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

# Коэффициенты энергии (в мкДж)
E_MUL = 3.1e-3
E_ADD = 0.9e-3 
E_MEM = 0.5e-3  

# Профили слоёв MobileNetV2
LAYER_TO_PROFILE = {
    1: {"size": 1606127, "M": 1.20e+07, "A": 1.20e+07, "Gamma": 1.86e+03},
    2: {"size": 803313, "M": 2.37e+07, "A": 2.37e+07, "Gamma": 3.65e+03},
    3: {"size": 301553, "M": 5.76e+07, "A": 5.76e+07, "Gamma": 1.39e+04},
    4: {"size": 301553, "M": 8.62e+07, "A": 8.62e+07, "Gamma": 3.16e+04},
    5: {"size": 100843, "M": 1.03e+08, "A": 1.03e+08, "Gamma": 5.16e+04},
    6: {"size": 100843, "M": 1.15e+08, "A": 1.15e+08, "Gamma": 8.13e+04},
    7: {"size": 100843, "M": 1.27e+08, "A": 1.27e+08, "Gamma": 1.11e+05},
    8: {"size": 50657, "M": 1.35e+08, "A": 1.35e+08, "Gamma": 1.53e+05},
    9: {"size": 50657, "M": 1.46e+08, "A": 1.46e+08, "Gamma": 2.62e+05},
    10: {"size": 50657, "M": 1.57e+08, "A": 1.57e+08, "Gamma": 3.70e+05},
    11: {"size": 50657, "M": 1.68e+08, "A": 1.68e+08, "Gamma": 4.79e+05},
    12: {"size": 75756, "M": 1.81e+08, "A": 1.81e+08, "Gamma": 6.12e+05},
    13: {"size": 75756, "M": 2.04e+08, "A": 2.04e+08, "Gamma": 8.49e+05},
    14: {"size": 75756, "M": 2.28e+08, "A": 2.28e+08, "Gamma": 1.09e+06},
    15: {"size": 31844, "M": 2.44e+08, "A": 2.44e+08, "Gamma": 1.40e+06},
    16: {"size": 31844, "M": 2.60e+08, "A": 2.60e+08, "Gamma": 2.04e+06},
    17: {"size": 31844, "M": 2.75e+08, "A": 2.75e+08, "Gamma": 2.68e+06},
}

# Энергия на операции
def compute_compute_energy(M, A):
    return M * E_MUL + A * E_ADD

def compute_memory_energy(Gamma):
    return Gamma * E_MEM

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
    args = parser.parse_args()

    # Загрузка модели
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, 10)
    model.load_state_dict(torch.load("/home/george/ns-allinone-3.41/ns-3.41/ml/mobilenet_mnist.pth", map_location=torch.device('cpu')))
    model.eval()

    # Предобработка
    input_tensor = preprocess_image(args.image_path)

    # Инференс и замер времени
    start_time = time.time()
    with torch.no_grad():
        output = model(input_tensor)
    latency_edge = time.time() - start_time

    # Получаем результат инференса
    probabilities = torch.nn.functional.softmax(output, dim=1)
    top1_prob, top1_class = torch.max(probabilities, dim=1)


    # Считаем энергию по слоям (с 1 по 18)
    E_c = 0
    E_m = 0
    for i in range(1, 19):
        profile = LAYER_TO_PROFILE[i]
        M, A, Gamma = profile["M"], profile["A"], profile["Gamma"]
        E_c += compute_compute_energy(M, A)
        E_m += compute_memory_energy(Gamma)

    total_E = E_c + E_m

    # Логируем задержку
    log_path = "files/latency_full_edge_log.csv"
    with open(log_path, "a") as log_file:
        log_file.write(f"{latency_edge:.3f},{0},{0},{latency_edge:.3f}\n")

    # Логируем затраченную энергию
    log_path = "files/energy_full_edge_log.csv"
    with open(log_path, "a") as log_file:
        log_file.write(f"0,0,0,{E_c:.2f},{E_m:.2f},0.00,{total_E:.2f}\n")


    image_name = os.path.basename(args.image_path)
    true_label = int(image_name.split("_")[0])

    with open("files/prediction_results_full_edge.csv", "a") as f:
        f.write(f"{true_label},{top1_class.item()},{top1_class.item() == true_label}\n")

    print("[EDGE] Full inference complete.")
    print(f"[EDGE] Final Prediction: Class {top1_class.item()}, Probability {top1_prob.item():.4f}")
    print(f"[EDGE] Latency: {latency_edge:.4f} sec | Compute Energy: {E_c:.2f} µJ | Memory Energy: {E_m:.2f} µJ | Total: {total_E:.2f} µJ")
