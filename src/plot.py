
import os
import matplotlib.pyplot as plt

def plot_fold_mae(avg_mae_hist, save_path="graphs/avg_mae_hist.jpg"):
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(avg_mae_hist) + 1), avg_mae_hist, marker='o')
    plt.title('Validation MAE per fold')
    plt.xlabel('Epochs')
    plt.ylabel('Validation MAE')
    plt.grid(True)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    plt.plot()
