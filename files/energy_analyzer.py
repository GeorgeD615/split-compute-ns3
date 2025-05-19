import pandas as pd
import matplotlib.pyplot as plt

# Пути к CSV-файлам
file_edge = "energy_full_edge_log.csv"
file_ours = "energy_log.csv"
file_random = "energy_rand_log.csv"

# Загрузка данных
df_edge = pd.read_csv(file_edge, header=None, names=["layer", "snr", "throughput", "E_c", "E_m", "E_tr", "total_E"])
df_ours = pd.read_csv(file_ours, header=None, names=["layer", "snr", "throughput", "E_c", "E_m", "E_tr", "total_E"])
df_random = pd.read_csv(file_random, header=None, names=["layer", "snr", "throughput", "E_c", "E_m", "E_tr", "total_E"])

# Убеждаемся, что все по 100 записей
assert len(df_edge) == len(df_ours) == len(df_random) == 100

# Метки и цвета
labels = ["Full Edge", "Our Method", "Random Split"]
colors = ["blue", "green", "red"]

# Список метрик
metrics = ["E_c", "E_m", "E_tr", "total_E"]

# Создаем графики
for metric in metrics:
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, 101), df_edge[metric], label=labels[0], color=colors[0])
    plt.plot(range(1, 101), df_ours[metric], label=labels[1], color=colors[1])
    plt.plot(range(1, 101), df_random[metric], label=labels[2], color=colors[2])
    plt.xlabel("Simulation Index")
    plt.ylabel(f"{metric} (µJ)")
    plt.title(f"Comparison of {metric}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"plot_{metric}.png")  # Сохраняем графики в PNG
    plt.show()
