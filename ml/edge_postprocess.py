import os
import time
import pickle
import torch
import torch.nn.functional as F
import argparse

def wait_for_file(filename, timeout=30):
    #print(f"[EDGE] Waiting for {filename} ...")
    start_time = time.time()
    while not os.path.exists(filename):
        if time.time() - start_time > timeout:
            raise TimeoutError(f"Timeout: File {filename} not received within {timeout} seconds")
        time.sleep(0.5)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("latency_log_file", help="Path to latency log file")
    parser.add_argument("prediction_log_file", help="Path to prediction log file")
    args = parser.parse_args()
    result_file = "files/result_tensor_on_edge.bin"
    wait_for_file(result_file)

    if os.path.getsize(result_file) == 0:
        print(f"[EDGE] Warning: Received empty result file at {result_file}. Skipping post-processing.")
        exit(0)

    with open(result_file, "rb") as f:
        try:
            output_tensor, image_path = pickle.load(f)
        except EOFError:
            print(f"[EDGE] Error: Failed to unpickle tensor. File appears to be corrupt or incomplete: {result_file}")
            exit(1)

    if output_tensor.numel() == 0:
        print("[EDGE] Warning: Output tensor is empty. Skipping post-processing.")
        exit(0)

    probabilities = F.softmax(output_tensor[0], dim=0)
    top1_class = torch.argmax(probabilities).item()
    top1_prob = probabilities[top1_class].item()

    print(f"[EDGE] Final Prediction: Class {top1_class}, Probability {top1_prob:.4f}")

    image_name = os.path.basename(image_path)
    true_label = int(image_name.split("_")[0])

    with open(args.prediction_log_file, "a") as f:
        f.write(f"{true_label},{top1_class},{top1_class == true_label}\n")

    print(f"[EDGE] Predicted: {top1_class}, True: {true_label}")

    try:
        latency_edge = float(open("files/latency_edge.log").read().strip())
        latency_transfer = float(open("files/latency_transfer.log").read().strip())
        latency_cloud = float(open("files/latency_cloud.log").read().strip())

        latency_total = latency_edge + latency_transfer + latency_cloud

        print(f"[LATENCY] Edge: {latency_edge:.3f}s, Transfer: {latency_transfer:.3f}s, Cloud: {latency_cloud:.3f}s")
        print(f"[LATENCY] Total latency: {latency_total:.3f}s")

        # Логгируем в CSV
        log_path = args.latency_log_file
        with open(log_path, "a") as log_file:
            log_file.write(f"{latency_edge:.3f},{latency_transfer:.3f},{latency_cloud:.3f},{latency_total:.3f}\n")

    except FileNotFoundError as e:
        print(f"[LATENCY] Missing latency file: {e.filename}")
    except ValueError:
        print("[LATENCY] Failed to parse latency values. Check log file formats.")
