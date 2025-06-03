import matplotlib.pyplot as plt
import numpy as np
import os

files = {
    "Dynamic": "size.csv",
    "Random": "rand_size.csv"
}

size_data = {}

for label, path in files.items():
    if os.path.exists(path):
        with open(path, "r") as f:
            sizes = [int(line.strip()) for line in f if line.strip().isdigit()]
            if sizes:
                size_data[label] = sizes
            else:
                print(f"Файл {path} пуст или не содержит корректных чисел.")
    else:
        print(f"Файл {path} не найден.")

if not size_data:
    print("Нет данных для отображения.")
else:
    for label, sizes in size_data.items():
        print(f"Минимальный size: {min(sizes)} байт")
        print(f"Максимальный size: {max(sizes)} байт")
        print(f"Средний size: {np.mean(sizes):.2f} байт")
        print(f"Медианный size: {np.median(sizes)} байт")

    plt.figure(figsize=(12, 6))
    for label, sizes in size_data.items():
        x = list(range(1, len(sizes) + 1)) 
        plt.plot(x, sizes, marker='o', linestyle='-', label=label)

    plt.title("Зависимость размера передаваемых данных от индекса симуляции")
    plt.xlabel("Simulation index")
    plt.ylabel("Size")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()

    output_path = "size_line_plot.png"
    plt.savefig(output_path)

    plt.show()
