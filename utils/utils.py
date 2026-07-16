import re
import torch
from torch import nn
import json
import seaborn as sns
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from configs.config import OBJECT_TO_MATERIAL
from sklearn.metrics import classification_report
from torch.utils.data import Dataset
from scipy.signal import fftconvolve, spectrogram as compute_spec, correlate
from sklearn.metrics import confusion_matrix, mean_squared_error, mean_absolute_error

def load_excitation(excitation_path):
    excitation, fs = sf.read(excitation_path)
    if excitation.ndim > 1:
        excitation = excitation[:, 0]
    inv_filter = excitation[::-1]
    N = len(excitation)
    return inv_filter, N, fs

def compute_ir(data, inv_filter, N, start_sample, end_sample):
    if data.ndim == 1:
        data = data[:, np.newaxis]
    irs = []
    for ch in range(data.shape[1]):
        ir_full = fftconvolve(data[:, ch], inv_filter, mode='full')
        ir_full = ir_full[N:N * 2]
        irs.append(ir_full[start_sample:end_sample])
    return irs

def align_signals(ref_ir, target_ir):
    ref_norm = ref_ir / (np.max(np.abs(ref_ir)) + 1e-8)
    target_norm = target_ir / (np.max(np.abs(target_ir)) + 1e-8)
    corr = correlate(target_norm, ref_norm, mode='full')
    lag = np.argmax(corr) - (len(ref_norm) - 1)
    aligned = np.roll(target_ir, -lag)
    return aligned, lag
    
def compute_spectrogram(ir, fs, nperseg=256, noverlap=128):
    _, _, Sxx = compute_spec(ir, fs=fs, nperseg=nperseg, noverlap=noverlap)
    return Sxx

def run_evaluation(model, loader, device, det_classes, mat_classes, save_dir, history=None):

    model.eval()

    all_det_p, all_det_t = [], []
    all_mat_p, all_mat_t = [], []
    all_dist_p, all_dist_t = [], []

    OBJ_CLASSES = [
        "no_object",
        "cardboard_box",
        "speaker",
        "pot",
        "strainer",
        "pitcher",
        "ladder",
        "sandbag",
        "ceramic_mug",
        "glass_mug",
        "plate",
        "ceramic_bowl",
        "trash_bin"
    ]

    MAT_CLASSES = [
        "none",
        "paper_cardboard",
        "plastic",
        "metal",
        "sand",
        "ceramic",
        "glass"
    ]

    with torch.no_grad():
        for ir, spec, t_det, t_dist, t_mat in loader:
            ir, spec = ir.to(device), spec.to(device)
            p_det, p_dist, p_mat = model(ir, spec)
            all_det_p.extend(torch.argmax(p_det, dim=1).cpu().numpy())
            all_det_t.extend(t_det.numpy())
            all_mat_p.extend(torch.argmax(p_mat, dim=1).cpu().numpy())
            all_mat_t.extend(t_mat.numpy())
            all_dist_p.extend(p_dist.squeeze(-1).cpu().numpy())
            all_dist_t.extend(t_dist.numpy())

    acc_det   = np.mean(np.array(all_det_p) == np.array(all_det_t)) * 100
    rmse_dist = np.sqrt(mean_squared_error(all_dist_t, all_dist_p))
    mae_dist  = mean_absolute_error(all_dist_t, all_dist_p)

    print(f"\nDetection Accuracy : {acc_det:.2f}%")
    print(f"Distance RMSE      : {rmse_dist:.2f} m")
    print(f"Distance MAE       : {mae_dist:.2f} m")

    print(classification_report(all_det_t, all_det_p, target_names=det_classes))
    print(classification_report(all_mat_t, all_mat_p, target_names=mat_classes))

    # Object Detection Confusion Matrix 
    n_det = len(det_classes)
    fig, ax = plt.subplots(figsize=(n_det * 1.4 + 2, n_det * 1.4 + 1.5))  # <-- this was missing

    cm_det = confusion_matrix(all_det_t, all_det_p, normalize='true')
    sns.heatmap(
        cm_det,
        annot=True,
        fmt='.2f',
        cmap='rocket_r',
        linewidths=0.3,
        linecolor='#1a1a1a',
        xticklabels=det_classes,
        yticklabels=det_classes,
        ax=ax,
        annot_kws={"size": 9},
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title("Detection Confusion Matrix", fontsize=14, pad=14)

    ax.set_xlabel("Predicted", fontsize=12, labelpad=10)
    ax.set_ylabel("True", fontsize=12, labelpad=10)
    ax.tick_params(axis='x', labelsize=9, rotation=45)
    ax.tick_params(axis='y', labelsize=9, rotation=0)
    plt.tight_layout()
    plt.savefig(save_dir / "detection_cm.png", dpi=150, bbox_inches='tight')
    plt.close()

    n_mat = len(mat_classes)
    fig, ax = plt.subplots(figsize=(n_mat * 1.6 + 2, n_mat * 1.6 + 1.5))

    # Material Detection Confusion Matrix

    cm_mat = confusion_matrix(all_mat_t, all_mat_p, normalize='true')
    sns.heatmap(
        cm_mat,
        annot=True,
        fmt='.2f',
        cmap='rocket_r',
        linewidths=0.4,
        linecolor='#1a1a1a',
        xticklabels=mat_classes,
        yticklabels=mat_classes,
        ax=ax,
        annot_kws={"size": 13},
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title("Material Confusion Matrix", fontsize=15, pad=14)
    ax.set_xlabel("Predicted", fontsize=13, labelpad=10)
    ax.set_ylabel("True", fontsize=13, labelpad=10)
    ax.tick_params(axis='x', labelsize=12, rotation=45)
    ax.tick_params(axis='y', labelsize=12, rotation=0)
    plt.tight_layout()
    plt.savefig(save_dir / "material_cm.png", dpi=150, bbox_inches='tight')
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(all_dist_t, all_dist_p, alpha=0.4, color='teal', s=18)
    lim = [min(all_dist_t) - 0.02, max(all_dist_t) + 0.02]
    ax.plot(lim, lim, 'r--', linewidth=1.5, label='Perfect prediction')
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("True distance (m)", fontsize=12)
    ax.set_ylabel("Predicted distance (m)", fontsize=12)
    ax.set_title(f"Distance  RMSE={rmse_dist:.3f}m  MAE={mae_dist:.3f}m", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / "distance_scatter.png", dpi=150, bbox_inches='tight')
    plt.close()

    with open(save_dir / "results.txt", "w") as f:
        f.write(f"Detection Accuracy : {acc_det:.2f}%\n")
        f.write(f"Distance RMSE      : {rmse_dist:.2f} m\n")
        f.write(f"Distance MAE       : {mae_dist:.2f} m\n")

    if history is not None:

        epochs = np.arange(
            1,
            len(history["train_loss"])+1
        )
        
        # Loss curve
        plt.figure(figsize=(8,5))
        plt.plot(epochs, history["train_loss"], label="Train")
        plt.plot(epochs, history["val_loss"], label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss")
        plt.legend()
        plt.grid()
        plt.savefig(save_dir/"loss_curve.png", dpi=300)
        plt.close()

        # Individual task losses
        fig, ax = plt.subplots(1,3,figsize=(15,4))

        ax[0].plot(epochs, history["train_det_loss"])
        ax[0].plot(epochs, history["val_det_loss"])
        ax[0].set_title("Detection Loss")

        ax[1].plot(epochs, history["train_dist_loss"])
        ax[1].plot(epochs, history["val_dist_loss"])
        ax[1].set_title("Distance Loss")

        ax[2].plot(epochs, history["train_mat_loss"])
        ax[2].plot(epochs, history["val_mat_loss"])
        ax[2].set_title("Material Loss")

        plt.tight_layout()
        plt.savefig(save_dir/"task_losses.png",dpi=300)
        plt.close()

        # Accuracy

        plt.figure(figsize=(8,5))

        plt.plot(
            epochs,
            history["det_acc"],
            label="Object Detection"
        )

        plt.plot(
            epochs,
            history["mat_acc"],
            label="Material"
        )

        plt.xlabel("Epoch")
        plt.ylabel("Accuracy (%)")
        plt.legend()
        plt.grid()

        plt.savefig(
            save_dir/"accuracy_curve.png",
            dpi=300
        )

        plt.close()



        # Distance regression

        plt.figure(figsize=(8,5))

        plt.plot(
            epochs,
            history["rmse"],
            label="RMSE"
        )

        plt.plot(
            epochs,
            history["mae"],
            label="MAE"
        )

        plt.xlabel("Epoch")
        plt.ylabel("Meters")
        plt.legend()
        plt.grid()

        plt.savefig(
            save_dir/"distance_error.png",
            dpi=300
        )

        plt.close()



        # Learning rate

        plt.figure(figsize=(8,5))

        plt.plot(
            epochs,
            history["lr"]
        )

        plt.xlabel("Epoch")
        plt.ylabel("Learning Rate")
        plt.yscale("log")

        plt.grid()

        plt.savefig(
            save_dir/"learning_rate.png",
            dpi=300
        )

        plt.close()



        # Runtime

        plt.figure(figsize=(8,5))

        plt.plot(
            epochs,
            history["epoch_time"]
        )

        plt.xlabel("Epoch")
        plt.ylabel("Seconds")

        plt.title("Training Time Per Epoch")

        plt.grid()

        plt.savefig(
            save_dir/"epoch_runtime.png",
            dpi=300
        )

        plt.close()

        # ICASSP summary figure

        fig, axs = plt.subplots(
            2,
            2,
            figsize=(12,9)
        )


        axs[0,0].plot(
            epochs,
            history["train_loss"]
        )

        axs[0,0].plot(
            epochs,
            history["val_loss"]
        )

        axs[0,0].set_title("Loss")


        axs[0,1].plot(
            epochs,
            history["det_acc"]
        )

        axs[0,1].plot(
            epochs,
            history["mat_acc"]
        )

        axs[0,1].set_title("Accuracy")


        axs[1,0].plot(
            epochs,
            history["rmse"]
        )

        axs[1,0].plot(
            epochs,
            history["mae"]
        )

        axs[1,0].set_title("Distance")


        axs[1,1].plot(
            epochs,
            history["epoch_time"]
        )

        axs[1,1].set_title("Runtime")


        plt.tight_layout()

        plt.savefig(
            save_dir/"training_summary.png",
            dpi=300
        )

        plt.close()

    # ── Detection confusion matrix ────────────────────────────────────────────
    # n_det = len(det_classes)
    # cell  = 1.1                              # inches per cell
    # fig, ax = plt.subplots(figsize=(n_det * cell + 3, n_det * cell + 2))

    # cm_det = confusion_matrix(all_det_t, all_det_p, normalize='true')
    # sns.heatmap(
    #     cm_det,
    #     annot=True,
    #     fmt='.2f',
    #     cmap='rocket_r',
    #     linewidths=0.3,
    #     linecolor='#222',
    #     xticklabels=det_classes,
    #     yticklabels=det_classes,
    #     ax=ax,
    #     annot_kws={"size": 10},
    #     cbar_kws={"shrink": 0.8},
    # )
    # ax.set_title("Detection Confusion Matrix", fontsize=15, pad=14)
    # ax.set_xlabel("Predicted", fontsize=12, labelpad=10)
    # ax.set_ylabel("True",      fontsize=12, labelpad=10)
    # ax.tick_params(axis='x', labelsize=10, rotation=45)
    # ax.tick_params(axis='y', labelsize=10, rotation=0)
    # plt.tight_layout()
    # plt.savefig(save_dir / "detection_cm.png", dpi=150, bbox_inches='tight')
    # plt.close()

    # # ── Material confusion matrix ─────────────────────────────────────────────
    # n_mat = len(mat_classes)
    # fig, ax = plt.subplots(figsize=(n_mat * 1.4 + 2, n_mat * 1.4 + 1.5))

    # cm_mat = confusion_matrix(all_mat_t, all_mat_p, normalize='true')
    # sns.heatmap(
    #     cm_mat,
    #     annot=True,
    #     fmt='.2f',
    #     cmap='rocket_r',
    #     linewidths=0.4,
    #     linecolor='#222',
    #     xticklabels=mat_classes,
    #     yticklabels=mat_classes,
    #     ax=ax,
    #     annot_kws={"size": 12},
    #     cbar_kws={"shrink": 0.8},
    # )
    # ax.set_title("Material Confusion Matrix", fontsize=15, pad=14)
    # ax.set_xlabel("Predicted", fontsize=12, labelpad=10)
    # ax.set_ylabel("True",      fontsize=12, labelpad=10)
    # ax.tick_params(axis='x', labelsize=11, rotation=45)
    # ax.tick_params(axis='y', labelsize=11, rotation=0)
    # plt.tight_layout()
    # plt.savefig(save_dir / "material_cm.png", dpi=150, bbox_inches='tight')
    # plt.close()

    # # ── Distance scatter ──────────────────────────────────────────────────────
    # fig, ax = plt.subplots(figsize=(8, 7))
    # ax.scatter(all_dist_t, all_dist_p, alpha=0.4, color='teal', s=18)
    # lim = [min(all_dist_t) - 0.02, max(all_dist_t) + 0.02]
    # ax.plot(lim, lim, 'r--', linewidth=1.5, label='Perfect prediction')
    # ax.set_xlim(lim); ax.set_ylim(lim)
    # ax.set_xlabel("True distance (m)",      fontsize=12)
    # ax.set_ylabel("Predicted distance (m)", fontsize=12)
    # ax.set_title(f"Distance  RMSE={rmse_dist:.3f}m  MAE={mae_dist:.3f}m", fontsize=13)
    # ax.legend(fontsize=11)
    # ax.grid(True, alpha=0.3)
    # plt.tight_layout()
    # plt.savefig(save_dir / "distance_scatter.png", dpi=150, bbox_inches='tight')
    # plt.close()

    # # ── Results text ──────────────────────────────────────────────────────────
    # with open(save_dir / "results.txt", "w") as f:
    #     f.write(f"Detection Accuracy : {acc_det:.2f}%\n")
    #     f.write(f"Distance RMSE      : {rmse_dist:.2f} m\n")
    #     f.write(f"Distance MAE       : {mae_dist:.2f} m\n")

# def run_evaluation(model, loader, device, det_classes, mat_classes, save_dir):
#     model.eval()
#     all_det_p, all_det_t = [], []
#     all_mat_p, all_mat_t = [], []
#     all_dist_p, all_dist_t = [], []
    
#     OBJ_CLASSES = [
#         "no_object",
#         "cardboard_box",
#         "speaker",
#         "pot",
#         "strainer",
#         "pitcher",
#         "ladder",
#         "ceramic_mug",
#         "glass_mug",
#         "plate",
#         "ceramic_bowl",
#         "trash_bin",
#         "metal_cup",
#         "plastic_bottle",
#         "plastic_bowl",
#         "plastic_container",
#         "plastic_sport",
#         "plastic_shaker",
#         "monitor",
#         "teapot",
#         "glass_vodka", 
#         "glass_shooter"
#     ]

#     MAT_CLASSES = [
#         "none",
#         "paper_cardboard",
#         "plastic",
#         "metal",
#         "ceramic",
#         "glass"
#     ]

#     with torch.no_grad():
#         for ir, spec, t_det, t_dist, t_mat in loader:
#             ir, spec = ir.to(device), spec.to(device)
#             p_det, p_dist, p_mat = model(ir, spec)
#             all_det_p.extend(torch.argmax(p_det, dim=1).cpu().numpy())
#             all_det_t.extend(t_det.numpy())
#             all_mat_p.extend(torch.argmax(p_mat, dim=1).cpu().numpy())
#             all_mat_t.extend(t_mat.numpy())
#             all_dist_p.extend(p_dist.squeeze(-1).cpu().numpy())
#             all_dist_t.extend(t_dist.numpy())

#     acc_det  = np.mean(np.array(all_det_p) == np.array(all_det_t)) * 100
#     rmse_dist = np.sqrt(mean_squared_error(all_dist_t, all_dist_p))
#     mae_dist  = mean_absolute_error(all_dist_t, all_dist_p)

#     print(f"\nDetection Accuracy : {acc_det:.2f}%")
#     print(f"Distance RMSE      : {rmse_dist:.2f} m")
#     print(f"Distance MAE       : {mae_dist:.2f} m")

#     print(classification_report(all_det_t, all_det_p, target_names=OBJ_CLASSES))
#     print(classification_report(all_mat_t, all_mat_p, target_names=MAT_CLASSES))

#     fig, ax = plt.subplots(figsize=(8, 6))
#     cm_det = confusion_matrix(all_det_t, all_det_p, normalize='true')
#     sns.heatmap(cm_det, annot=True, fmt='.2f', ax=ax,
#                 xticklabels=det_classes, yticklabels=det_classes)
#     ax.set_title("Detection Confusion Matrix")
#     ax.set_xlabel("Predicted")
#     ax.set_ylabel("True")
#     plt.tight_layout()
#     plt.savefig(save_dir / "detection_cm.png", dpi=150)
#     plt.close()

#     fig, ax = plt.subplots(figsize=(8, 6))
#     cm_mat = confusion_matrix(all_mat_t, all_mat_p, normalize='true')
#     sns.heatmap(cm_mat, annot=True, fmt='.2f', ax=ax,
#                 xticklabels=mat_classes, yticklabels=mat_classes)
#     ax.set_title("Material Confusion Matrix")
#     ax.set_xlabel("Predicted")
#     ax.set_ylabel("True")
#     plt.tight_layout()
#     plt.savefig(save_dir / "material_cm.png", dpi=150)
#     plt.close()

#     fig, ax = plt.subplots(figsize=(8, 6))
#     ax.scatter(all_dist_t, all_dist_p, alpha=0.5, color='teal')
#     ax.plot([min(all_dist_t), max(all_dist_t)],
#             [min(all_dist_t), max(all_dist_t)], 'r--', label='Perfect prediction')
#     ax.set_xlabel("True distance (m)")
#     ax.set_ylabel("Predicted distance (m)")
#     ax.set_title(f"Distance  RMSE={rmse_dist:.3f}m  MAE={mae_dist:.3f}m")
#     ax.legend()
#     plt.tight_layout()
#     plt.savefig(save_dir / "distance_scatter.png", dpi=150)
#     plt.close()

#     with open(save_dir / "results.txt", "w") as f:
#         f.write(f"Detection Accuracy : {acc_det:.2f}%\n")
#         f.write(f"Distance RMSE      : {rmse_dist:.2f} m\n")
#         f.write(f"Distance MAE       : {mae_dist:.2f} m\n")

def epoch_metrics(model, loader, device):

    model.eval()
    det_preds, det_targets = [], []
    mat_preds, mat_targets = [], []
    dist_preds, dist_targets = [], []

    with torch.no_grad():
        for ir, spec, t_det, t_dist, t_mat in loader:
            ir = ir.to(device)
            spec = spec.to(device)

            p_det, p_dist, p_mat = model(ir, spec)

            det_preds.extend(torch.argmax(p_det, dim=1).cpu().numpy())
            det_targets.extend(t_det.numpy())

            mat_preds.extend(torch.argmax(p_mat, dim=1).cpu().numpy())
            mat_targets.extend(t_mat.numpy())

            dist_preds.extend(p_dist.squeeze(-1).cpu().numpy())
            dist_targets.extend(t_dist.numpy())

    det_acc = np.mean(np.array(det_preds) == np.array(det_targets)) * 100
    mat_acc = np.mean(np.array(mat_preds) == np.array(mat_targets)) * 100
    rmse = np.sqrt(mean_squared_error(dist_targets, dist_preds))
    mae = mean_absolute_error(dist_targets, dist_preds)

    return det_acc, mat_acc, rmse, mae

def get_next_run_folder(base_path):
    base_path = Path(base_path)
    base_path.mkdir(exist_ok=True)

    runs = []

    for folder in base_path.iterdir():
        if folder.is_dir():
            match = re.match(r"run_(\d+)", folder.name)
            if match:
                runs.append(int(match.group(1)))

    next_run = max(runs, default=0) + 1

    run_path = base_path / f"run_{next_run}"
    run_path.mkdir()

    return run_path