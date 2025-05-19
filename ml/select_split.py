import re
import math

# === Параметры канала ===
W = 20e6
N0 = 1e-9

def snr_db_to_linear(snr_db):
    return 10 ** (snr_db / 10)

def compute_throughput_from_snr(snr_db):
    snr_linear = snr_db_to_linear(snr_db)
    return W * math.log2(1 + snr_linear) / 1e6 

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


# === Энергия на операцию ===
E_MUL = 3.1e-3
E_ADD = 0.9e-3
E_MEM = 0.5e-3

TX_ENERGY_PER_BYTE = 0.5

def compute_compute_energy(M, A):
    return E_MUL * M + E_ADD * A

def compute_memory_energy(Gamma):
    return Gamma * E_MEM

def compute_transfer_energy(size_bytes, snr_db=None):
    if snr_db:
        throughput = compute_throughput_from_snr(snr_db)
        transfer_time = (size_bytes * 8) / (throughput * 1e6)
        return 10 ** ((17 - 30) / 10) * transfer_time * 1e6
    return size_bytes * TX_ENERGY_PER_BYTE

def select_split_layer(snr, throughput):
    if snr is None or throughput is None:
        return 13

    min_energy = float('inf')
    best_layer = 13

    current_energy = 0

    for layer, profile in LAYER_TO_PROFILE.items():
        size, M, A, Gamma = profile["size"], profile["M"], profile["A"], profile["Gamma"]
        E_c = compute_compute_energy(M, A)
        E_m = compute_memory_energy(Gamma)
        E_tr = compute_transfer_energy(size, snr)

        current_energy += E_c + E_m

        total_E = current_energy + E_tr
        if total_E < min_energy:
            min_energy = total_E
            best_layer = layer

    return best_layer

def parse_last_snr_rssi(log_path):
    with open(log_path, "r") as f:
        lines = f.readlines()
        if not lines:
            return None, None
        last_line = lines[-1]
        match = re.search(r"RSSI=(-?\d+\.?\d*),SNR=(\d+\.?\d*)", last_line)
        if match:
            return float(match.group(2)), float(match.group(1))  # snr, rssi
    return None, None

if __name__ == "__main__":
    snr, rssi = parse_last_snr_rssi("files/channel_metrics.log")

    if snr is not None:
        throughput = compute_throughput_from_snr(snr)
    else:
        throughput = 0.0

    selected_layer = select_split_layer(snr, throughput)

    with open("files/selected_split_layer.txt", "w") as f:
        f.write(str(selected_layer))

    E_c = 0
    E_m = 0
    for i in range(1, selected_layer + 1):
        profile = LAYER_TO_PROFILE[i]
        M, A, Gamma = profile["M"], profile["A"], profile["Gamma"]
        E_c += compute_compute_energy(M, A)
        E_m += compute_memory_energy(Gamma)

    profile = LAYER_TO_PROFILE[selected_layer]
    size = profile["size"]
    E_tr = compute_transfer_energy(size, snr)
    total_E = E_c + E_m + E_tr

    log_path = "files/energy_log.csv"
    with open(log_path, "a") as log_file:
        log_file.write(f"{selected_layer},{snr:.2f},{throughput:.2f},{E_c:.2f},{E_m:.2f},{E_tr:.2f},{total_E:.2f}\n")

    print(f"[SELECT_SPLIT] SNR={snr}, computed Throughput={throughput:.2f} Mbps → split_layer={selected_layer}")
    print(f"[ENERGY_LOG] E_c={E_c:.2f} µJ, E_m={E_m:.2f} µJ, E_tr={E_tr:.2f} µJ → Total={total_E:.2f} µJ")