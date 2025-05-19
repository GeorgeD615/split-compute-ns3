import pandas as pd
import matplotlib.pyplot as plt

# Пути к CSV-файлам
file_edge = "latency_full_edge_log.csv"
file_ours = "latency_log.csv"
file_random = "latency_rand_log.csv"

# Загрузка данных
df_edge = pd.read_csv(file_edge, header=None, names=["latency_edge", "latency_transfer", "latency_cloud", "latency_total"])
df_ours = pd.read_csv(file_ours, header=None, names=["latency_edge", "latency_transfer", "latency_cloud", "latency_total"])
df_random = pd.read_csv(file_random, header=None, names=["latency_edge", "latency_transfer", "latency_cloud", "latency_total"])

# Проверка, что все по 100 строк
assert len(df_edge) == len(df_ours) == len(df_random) == 100

# Метки и цвета
labels = ["Full Edge", "Our Method", "Random Split"]
colors = ["blue", "green", "red"]

# Список параметров задержки
metrics = ["latency_edge", "latency_transfer", "latency_cloud", "latency_total"]

# Строим графики
for metric in metrics:
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, 101), df_edge[metric], label=labels[0], color=colors[0])
    plt.plot(range(1, 101), df_ours[metric], label=labels[1], color=colors[1])
    plt.plot(range(1, 101), df_random[metric], label=labels[2], color=colors[2])
    plt.xlabel("Simulation Index")
    plt.ylabel(f"{metric} (s)")
    plt.title(f"Comparison of {metric}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"plot_{metric}.png")
    plt.show()
