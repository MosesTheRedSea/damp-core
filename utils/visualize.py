import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_ir_single(ir, ch, save_path=None):
    plt.figure(figsize=(12, 4))
    plt.plot(ir, color='black', linewidth=1)
    plt.title(f"Impulse Response - Mic {ch}")
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200)
        plt.close()
    else:
        plt.show()

def plot_ir_grid(responses, start=5000, end=5250, save_path=None):
    fig, axs = plt.subplots(4, 4, figsize=(16, 12))
    fig.suptitle(f"Impulse Response Comparison (Samples {start}–{end})", fontsize=16)

    for ch in range(16):
        ax = axs[ch // 4][ch % 4]
        for label, ir_list in responses.items():
            ir = ir_list[ch]
            ax.plot(np.arange(start, end), ir[start:end], label=label)
        ax.set_title(f"Mic {ch + 1}")
        ax.set_xlabel("Sample")
        ax.set_ylabel("Amplitude")
        ax.grid(True)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    if save_path:
        plt.savefig(save_path, dpi=200)
        plt.close()
    else:
        plt.show()

def plot_ir_zoom(responses, start=5000, end=5300, save_path=None):
    for ch in range(16):
        plt.figure(figsize=(10, 4))
        for label, ir_list in responses.items():
            plt.plot(np.arange(start, end + 1), ir_list[ch][start:end + 1], label=label)
        plt.title(f"IR Zoomed - Mic {ch + 1} [{start}:{end}]")
        plt.xlabel("Sample Index")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=200)
            plt.close()
        else:
            plt.show()

def plot_ir_aligned(aligned_segment, save_path=None):
    fig, axes = plt.subplots(4, 4, figsize=(16, 10))
    fig.suptitle("Aligned IR Segments", fontsize=16)

    for ch in range(16):
        ax = axes[ch // 4, ch % 4]
        for label, segments in aligned_segment.items():
            ax.plot(segments[ch], label=label)
        ax.set_title(f"Mic {ch + 1}")
        ax.set_xlabel("Sample Index")
        ax.set_ylabel("Amplitude")
        ax.grid(True)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.92, 0.96))
    plt.tight_layout(rect=[0, 0.03, 0.9, 0.95])
    if save_path:
        plt.savefig(save_path, dpi=200)
        plt.close()
    else:
        plt.show()

def plot_spectrogram_grid(title, spectrograms, f, t, cmap='magma', save_path=None):
    fig, axes = plt.subplots(4, 4, figsize=(16, 12))
    fig.subplots_adjust(hspace=0.4, wspace=0.3, top=0.90, right=0.88)
    fig.suptitle(title, fontsize=18)

    for i in range(16):
        ax = axes[i // 4, i % 4]
        im = ax.pcolormesh(t, f, spectrograms[i], shading='gouraud', cmap=cmap)
        ax.set_title(f"Mic {i + 1}", fontsize=10)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Freq [Hz]")

    cbar_ax = fig.add_axes([0.90, 0.15, 0.015, 0.7])
    fig.colorbar(im, cax=cbar_ax, label="Power [dB]")

    if save_path:
        plt.savefig(save_path, dpi=200)
        plt.close()
    else:
        plt.show()


def plot_spectrogram_diff(spec_ref, spec_target, f, t, save_path=None):
    spec_diff = [np.abs(s1 - s2) for s1, s2 in zip(spec_ref, spec_target)]
    plot_spectrogram_grid("Spectrogram Difference (abs dB)", spec_diff, f, t,
                          cmap='inferno', save_path=save_path)

def plot_loss_curve(train_losses, val_losses, save_path=None):
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.title("Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_accuracy_curve(accuracies, labels=None, save_path=None):
    plt.figure()
    if isinstance(accuracies[0], (list, np.ndarray)):
        for acc, label in zip(accuracies, labels or [f"Task {i}" for i in range(len(accuracies))]):
            plt.plot([a * 100 for a in acc], marker='o', label=label)
    else:
        plt.plot([a * 100 for a in accuracies], marker='o', label="Accuracy")
    plt.title("Accuracy over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_rmse_curve(rmse_dict, save_path=None):
    
    plt.figure()

    for label, values in rmse_dict.items():
        plt.plot(values, label=label)

    plt.title("Regression RMSE over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("RMSE")

    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_confusion_matrix(y_true, y_pred, labels, title, save_path=None):
    cm = confusion_matrix(y_true, y_pred, labels=range(len(labels)))
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

