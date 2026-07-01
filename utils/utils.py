import re
import torch
import seaborn as sns
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
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

def run_evaluation(model, loader, device, det_classes, mat_classes, save_dir):
    model.eval()
    all_det_p, all_det_t = [], []
    all_mat_p, all_mat_t = [], []
    all_dist_p, all_dist_t = [], []

    with torch.no_grad():
        for ir, spec, t_det, t_dist, t_mat in loader:
            ir, spec = ir.to(device), spec.to(device)
            p_det, p_dist, p_mat = model(ir, spec)

            all_det_p.extend(torch.argmax(p_det, dim=1).cpu().numpy())
            all_det_t.extend(t_det.numpy())
            all_mat_p.extend(torch.argmax(p_mat, dim=1).cpu().numpy())
            all_mat_t.extend(t_mat.numpy())
            all_dist_p.extend(p_dist.squeeze().cpu().numpy())
            all_dist_t.extend(t_dist.numpy())

    acc_det = np.mean(np.array(all_det_p) == np.array(all_det_t)) * 100
    rmse_dist = np.sqrt(mean_squared_error(all_dist_t, all_dist_p))
    mae_dist = mean_absolute_error(all_dist_t, all_dist_p)
    
    print(f"\nDetection Accuracy: {acc_det:.2f}%")
    print(f"Distance RMSE: {rmse_dist:.2f} cm")
    print(f"Distance MAE:  {mae_dist:.2f} cm")

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    cm_det = confusion_matrix(all_det_t, all_det_p, normalize='true')
    sns.heatmap(cm_det, annot=True, fmt='.2f', ax=axes[0], xticklabels=det_classes, yticklabels=det_classes)
    axes[0].set_title("Detection Accuracy")

    cm_mat = confusion_matrix(all_mat_t, all_mat_p, normalize='true')
    sns.heatmap(cm_mat, annot=True, fmt='.2f', ax=axes[1], xticklabels=mat_classes, yticklabels=mat_classes)
    axes[1].set_title("Material Accuracy")

    axes[2].scatter(all_dist_t, all_dist_p, alpha=0.5, color='teal')
    axes[2].plot([min(all_dist_t), max(all_dist_t)], [min(all_dist_t), max(all_dist_t)], 'r--')
    axes[2].set_title(f"Distance RMSE: {rmse_dist:.2f}cm")
    
    plt.tight_layout()
    plt.savefig(save_dir / "results.png")
    plt.show()

    with open(save_dir / "results.txt", "w") as f:
        f.write(f"Detection Accuracy: {acc_det:.2f}%\n")
        f.write(f"Distance RMSE: {rmse_dist:.2f} cm\n")
        f.write(f"Distance MAE: {mae_dist:.2f} cm\n")

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

            dist_preds.extend(p_dist.squeeze().cpu().numpy())
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
