import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("layer_log.csv", header=None, names=["Layer"])

df["Simulation Index"] = df.index

plt.figure(figsize=(10, 6))
plt.plot(df["Simulation Index"], df["Layer"], marker='o', linestyle='-', color='royalblue')

plt.title("Выбранные слои по симуляциям")
plt.xlabel("Simulation Index")
plt.ylabel("Выбранный слой")
plt.xticks(range(0, len(df), max(1, len(df)//10)))
plt.yticks(range(1, 19))
plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig("layer_vs_simulation.png")
plt.show()