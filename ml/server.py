import os
import time
import pickle
import torch
import torch.nn as nn
import torchvision.models


model = torchvision.models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(model.last_channel, 10)
model.load_state_dict(torch.load("/home/george/ns-allinone-3.41/ns-3.41/ml/mobilenet_mnist.pth", map_location="cpu"))
model.eval()

def build_cloud_part(split_layer):
    return nn.Sequential(
        *list(model.features)[split_layer:],
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        model.classifier
    )

def wait_for_file(filename, timeout=30):
    start_time = time.time()
    while not os.path.exists(filename):
        if time.time() - start_time > timeout:
            raise TimeoutError(f"Timeout: File {filename} not received within {timeout} seconds")
        time.sleep(0.5)

if __name__ == "__main__":
    input_path = "files/input_tensor_on_server.bin"
    output_path = "files/result_tensor_on_server.bin"

    wait_for_file(input_path)

    if os.path.getsize(input_path) == 0:
        raise ValueError(f"[SERVER] Error: Received empty tensor file at {input_path}")

    with open(input_path, "rb") as f:
        try:
            edge_tensor, split_layer, image_path = pickle.load(f)
        except EOFError:
            raise ValueError(f"[SERVER] Error: Failed to unpickle tensor. File appears to be corrupt or incomplete: {input_path}")

    if not (0 <= split_layer <= 18):
        raise ValueError("split_layer must be between 0 and 18")

    if edge_tensor.numel() == 0:
        print("[SERVER] Warning: Edge tensor is empty. Skipping inference.")
        open(output_path, "wb").close()
        exit(0)

    cloud_part = build_cloud_part(split_layer)

    start_cloud = time.time()
    with torch.no_grad():
        result = cloud_part(edge_tensor)
    latency_cloud = time.time() - start_cloud

    with open("files/latency_cloud.log", "w") as f:
        f.write(f"{latency_cloud}\n")

    with open(output_path, "wb") as f:
        pickle.dump((result, image_path), f)

    print("[SERVER] Cloud inference complete. Saved to result_tensor.bin")
