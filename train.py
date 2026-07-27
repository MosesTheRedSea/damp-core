import os
import json
import torch
import math
import time
import argparse
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm.auto import tqdm
from torch.utils.data import DataLoader, Subset, Dataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from src.damp import DampNet
from collections import Counter
from utils.utils import get_next_run_folder
from utils.utils import epoch_metrics
from utils.utils import run_evaluation
from collections import defaultdict
from data.data import run_extraction

from data.augment import (
    random_polarity_flip,
    interchannel_delay,
    random_decay_scaling,
    random_room_augment,
    colored_noise,
    channel_dropout,
    distance_jitter,
    spec_augment,
    save_augmented
)


class ImpulseData(Dataset):

    def __init__(self, folders, det_mapping, mat_mapping, obj_to_mat):

        self.folders = folders
        self.det_mapping = det_mapping
        self.mat_mapping = mat_mapping
        self.obj_to_mat  = obj_to_mat

    def __len__(self):
        return len(self.folders)

    def __getitem__(self, idx):
        folder = self.folders[idx]

        with open(folder / "metadata.json", "r") as f:
            meta = json.load(f)

        ir_list   = []
        spec_list = []

        for ch in range(1, 17):
            ir_list.append(np.load(folder / f"ir_mic_{ch}.npy"))
            spec_list.append(np.load(folder / f"spec_mic_{ch}.npy"))

        ir_tensor   = torch.from_numpy(np.stack(ir_list)).float()
        spec_tensor = torch.from_numpy(np.stack(spec_list)).float()

        t_det  = torch.tensor(self.det_mapping[meta["object_type"]]).long() 

        # Estimate the Distance to the Object - not just the occlusion distance
        t_dist = torch.tensor(meta["object_distance"] + meta["occlusion_distance"], dtype=torch.float32)

        mat_key = self.obj_to_mat[meta["object_type"]]
        t_mat   = torch.tensor(self.mat_mapping[mat_key]).long()

        return ir_tensor, spec_tensor, t_det, t_dist, t_mat

class MultiTaskLoss(nn.Module):

    def __init__(self, w_det=1.0, w_dist=1.0, w_mat=1.0, 
                ortho_lambda=0.01, num_det_classes=8, num_mat_classes=5, max_dist=1.5):

        super().__init__()

        self.w_det = w_det
        self.w_dist = w_dist
        self.w_mat = w_mat
        self.ortho_lambda = ortho_lambda
        self.num_det_classes = num_det_classes
        self.num_mat_classes = num_mat_classes
        self.max_dist = max_dist

    def forward(self, p_det, t_det, p_dist, t_dist, p_mat, t_mat, ortho_loss=None):

        loss_det = F.cross_entropy(p_det, t_det, label_smoothing=0.1)
        loss_mat = F.cross_entropy(p_mat, t_mat, label_smoothing=0.1)

        loss_dist = F.l1_loss(p_dist.squeeze(-1), t_dist)

        # Normalise each loss to roughly [0, 1]
        loss_det_norm  = loss_det  / math.log(self.num_det_classes)
        loss_mat_norm  = loss_mat  / math.log(self.num_mat_classes)
        loss_dist_norm = loss_dist / self.max_dist

        total = (
            self.w_det  * loss_det_norm  +
            self.w_mat  * loss_mat_norm  +
            self.w_dist * loss_dist_norm
        )

        if ortho_loss is not None:
            total = total + self.ortho_lambda * torch.clamp(ortho_loss, 0, 10)

        return total, (loss_det_norm.item(), loss_dist_norm.item(), loss_mat_norm.item())

def get_object_groups(folders):
    groups = defaultdict(list)
    for folder in folders:
        with open(folder / "metadata.json") as f:
            meta = json.load(f)
        groups[meta["object_type"]].append(folder)
    return groups

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Disentangled Acoustic Multi-task Perception (DAMP)")
    parser.add_argument('--pr', type=str, default="", help="Processed data directory")
    parser.add_argument('--ar', type=str, default="", help="Augmented data directory")
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=32, help='Size of batches')
    parser.add_argument('--num_epochs', type=int, default=1000, help='Number of batch iterations')
    parser.add_argument('--weight_decay', type=int, default=None, help="Wegiht Decay")
    parser.add_argument("--augment", action="store_true", help="Use augmented training data")
    parser.add_argument("--regen", action="store_true", help="Regenerate augmentations")
    
    args = parser.parse_args()

    AUDIO_DATA_ROOT = "/home/3/um07293/data/audio"
    EXCITATION_PATH = Path('/home/3/um07293/research/occul-net/excitation.wav')
    FORCE_AUGMENT = args.augment
    FORCE_REGEN = args.regen
    PROCESSED_ROOT = Path("./data/processed")
    AUGMENTED_ROOT = Path("./data/augmented")
    AUGMENTATIONS = ["polarity", "delay", "scale", "room", "noise", "dropout", "jitter", "spec_aug"]

    # 25 Objects -> 6 Indvidual Classes
    OBJECT_TO_MATERIAL = {

        "no_object": "none",

        "cardboard_box_large": "paper_cardboard",
        "cardboard_box_small": "paper_cardboard",
        "hardcover_textbook": "paper_cardboard",
        "printer_paper": "paper_cardboard",
        
        "pot": "metal",
        "strainer": "metal",
        "pitcher": "metal",
        "ladder": "metal",
        "metal_cup": "metal",

        "ceramic_mug": "ceramic",
        "plate": "ceramic",
        "ceramic_bowl": "ceramic",
        "teapot": "ceramic",

        "trash_bin": "plastic",
        "plastic_bottle": "plastic",
        "plastic_bowl": "plastic",
        "plastic_container": "plastic",
        "plastic_sport": "plastic",
        "plastic_shaker": "plastic",
        "monitor":"plastic",
        "speaker": "plastic",

        "glass_vodka": "glass", 
        "glass_shooter": "glass",
        "glass_mug": "glass"
    }

    # 25 Individual Objects
    OBJ_CLASSES = [
        "no_object",
        "cardboard_box",
        "speaker",
        "pot",
        "strainer",
        "pitcher",
        "ladder",
        "ceramic_mug",
        "glass_mug",
        "plate",
        "ceramic_bowl",
        "trash_bin",
        "metal_cup",
        "plastic_bottle",
        "plastic_bowl",
        "plastic_container",
        "plastic_sport",
        "plastic_shaker",
        "monitor",
        "teapot",
        "glass_vodka", 
        "glass_shooter",
        "cardboard_box_small",
        "hardcover_textbook",
        "printer_paper"
    ]

    # 6 Material Classes
    MAT_CLASSES = [

        "none",
        "paper_cardboard",
        "plastic",
        "metal",
        "ceramic",
        "glass"
        
    ]

    det_map = {name: i for i, name in enumerate(OBJ_CLASSES)}
    mat_map = {name: i for i, name in enumerate(MAT_CLASSES)}

    run_extraction(
        AUDIO_DATA_ROOT, EXCITATION_PATH, PROCESSED_ROOT,
        skip_if_exists=True,  # won't redo work if already processed
    )

    original_folders = sorted([
        f for f in PROCESSED_ROOT.iterdir()
        if f.is_dir() and (f / "metadata.json").exists()
    ])

    print(f"Found {len(original_folders)} processed recordings...")

    all_distances = []

    for folder in original_folders:
        with open(folder / "metadata.json") as f:
            meta = json.load(f)

        all_distances.append(
            meta["object_distance"] +
            meta["occlusion_distance"]
        )

    MAX_DIST = max(all_distances)
    print(f"Max occlusion distance: {MAX_DIST}m")

    groups = get_object_groups(original_folders)

    print("\nRecordings per object:")
    for obj, folders in sorted(groups.items()):
        print(f"  {obj:20s}: {len(folders)}")

    train_folders = []
    val_folders = []

    for obj, folders in groups.items():

        train, val = train_test_split(
            folders,
            test_size=0.2,
            random_state=42,
            shuffle=True,
        )

        print(
            f"{obj:20s} -> "
            f"Train: {len(train)} | "
            f"Val: {len(val)}"
        )

        train_folders.extend(train)
        val_folders.extend(val)

    train_full = []

    if FORCE_AUGMENT:

        print(f"\nGenerating augmentations for {len(train_folders)} training recordings...")
    
        augmentations = {
            "polarity": random_polarity_flip,
            "delay": interchannel_delay,
            "scale": random_decay_scaling,
            "room": random_room_augment,
            "noise": colored_noise,
            "dropout": channel_dropout,
            "jitter": distance_jitter,
            "spec_aug": spec_augment,
        }

        for source_folder in train_folders:
            for aug_name, aug_fn in augmentations.items():
                output_folder = AUGMENTED_ROOT / f"{aug_name}_{source_folder.name}"
                if (output_folder / "metadata.json").exists() and not FORCE_REGEN:
                    continue

                save_augmented(
                    source_folder,
                    output_folder,
                    aug_fn,
                    aug_name,
                    FORCE_REGEN
                )

        # Add augmentations to training set only
        for folder in train_folders:
            # Keep the original recording
            train_full.append(folder)

            # Add every augmented version
            for aug in AUGMENTATIONS:
                aug_folder = AUGMENTED_ROOT / f"{aug}_{folder.name}"

                if aug_folder.exists():
                    train_full.append(aug_folder)
        
        print(f"\nDone. {len(train_folders) * len(augmentations)} augmented recordings saved to {AUGMENTED_ROOT}")
    
    else:

        print("\nTraining without augmentations.")
        train_full = train_folders.copy()

    # Validation contains originals only
    val_full = val_folders

    print(f"Original recordings: {len(original_folders)}")
    print(f"Training originals: {len(train_folders)}")
    print(f"Validation originals: {len(val_folders)}")

    print(f"Training samples (+aug): {len(train_full)}")
    print(f"Augmented samples added: {len(train_full) - len(train_folders)}")

    train_dataset = ImpulseData(
        train_full,
        det_mapping=det_map,
        mat_mapping=mat_map,
        obj_to_mat=OBJECT_TO_MATERIAL
    )

    val_dataset = ImpulseData(
        val_full,
        det_mapping=det_map,
        mat_mapping=mat_map,
        obj_to_mat=OBJECT_TO_MATERIAL
    )

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=128, shuffle=False)

    print(f"Original recordings: {len(original_folders)}")
    print(f"Training originals: {len(train_folders)}")
    print(f"Validation originals: {len(val_folders)}")

    print(f"Training samples (w/ augment): {len(train_full)}")
    print(f"Augmented samples added: {len(train_full) - len(train_folders)}")

    EPOCHS = 100
    LR = 1e-3
    WEIGHT_DECAY = 5e-4
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    PATIENCE = 25
    VAL_SMOOTH = 0.9

    model_run_dir  = get_next_run_folder("./models")
    result_run_dir = get_next_run_folder("./results")

    print(f"Saving model to:   {model_run_dir}")
    print(f"Saving results to: {result_run_dir}")

    model = DampNet(len(OBJ_CLASSES), len(MAT_CLASSES)).to(DEVICE)

    # fine tuning model 
    criterion = MultiTaskLoss(
        w_det=1.0,
        w_dist=5.0,
        w_mat=1.0,
        ortho_lambda=0.01,
        num_det_classes=len(OBJ_CLASSES),
        num_mat_classes=len(MAT_CLASSES),
        max_dist=MAX_DIST
    )

    # criterion has no learnable params now — only model params needed
    optimizer = optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-5
    )

    best_val_loss  = float("inf")
    early_stop_cnt = 0

    history = {

        # losses
        "train_loss": [],
        "val_loss": [],

        "train_det_loss": [],
        "train_dist_loss": [],
        "train_mat_loss": [],

        "val_det_loss": [],
        "val_dist_loss": [],
        "val_mat_loss": [],

        # performance
        "det_acc": [],
        "mat_acc": [],

        "rmse": [],
        "mae": [],

        # optimization
        "lr": [],
        "ortho_loss": [],

        # runtime
        "epoch_time": []
    }

    for epoch in range(EPOCHS):

        epoch_start = time.time()
        
        model.train()

        train_losses, train_det, train_dist, train_mat = [], [], [], []
        for ir, spec, t_det, t_dist, t_mat in train_loader:

            ir, spec  = ir.to(DEVICE), spec.to(DEVICE)
            t_det, t_dist, t_mat = t_det.to(DEVICE), t_dist.to(DEVICE), t_mat.to(DEVICE)

            optimizer.zero_grad()

            p_det, p_dist, p_mat = model(ir, spec)
            loss, (l_det, l_dist, l_mat) = criterion(
                p_det, t_det,
                p_dist, t_dist,
                p_mat, t_mat,
                ortho_loss=model.orthogonality_loss
            )

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_losses.append(loss.item())
            train_det.append(l_det)
            train_dist.append(l_dist)
            train_mat.append(l_mat)

        model.eval()
        val_losses, val_det, val_dist, val_mat = [], [], [], []

        with torch.no_grad():
            for ir, spec, t_det, t_dist, t_mat in val_loader:
                ir, spec       = ir.to(DEVICE), spec.to(DEVICE)
                t_det, t_dist, t_mat = t_det.to(DEVICE), t_dist.to(DEVICE), t_mat.to(DEVICE)

                p_det, p_dist, p_mat = model(ir, spec)

                v_loss, (vl_det, vl_dist, vl_mat) = criterion(
                    p_det, t_det,
                    p_dist, t_dist,
                    p_mat, t_mat,
                    ortho_loss=model.orthogonality_loss
                )

                val_losses.append(v_loss.item())
                val_det.append(vl_det)
                val_dist.append(vl_dist)
                val_mat.append(vl_mat)

        avg_train = np.mean(train_losses)
        avg_val = np.mean(val_losses)
        smoothed_val = avg_val 
        smoothed_val = VAL_SMOOTH * smoothed_val + (1 - VAL_SMOOTH) * avg_val
        avg_train_det = np.mean(train_det)
        avg_train_dist = np.mean(train_dist)
        avg_train_mat = np.mean(train_mat)
        avg_val_det = np.mean(val_det)
        avg_val_dist = np.mean(val_dist)
        avg_val_mat = np.mean(val_mat)

        det_acc, mat_acc, rmse, mae = epoch_metrics(model, val_loader, DEVICE)

        epoch_time = time.time() - epoch_start

        history["train_loss"].append(avg_train)
        history["val_loss"].append(avg_val)

        history["train_det_loss"].append(avg_train_det)
        history["train_dist_loss"].append(avg_train_dist)
        history["train_mat_loss"].append(avg_train_mat)

        history["val_det_loss"].append(avg_val_det)
        history["val_dist_loss"].append(avg_val_dist)
        history["val_mat_loss"].append(avg_val_mat)

        history["det_acc"].append(det_acc)
        history["mat_acc"].append(mat_acc)

        history["rmse"].append(rmse)
        history["mae"].append(mae)

        history["lr"].append(
            scheduler.get_last_lr()[0]
        )

        history["epoch_time"].append(epoch_time)

        if hasattr(model, "orthogonality_loss"):
            history["ortho_loss"].append(
                model.orthogonality_loss.item()
            )
        else:
            history["ortho_loss"].append(0)

        scheduler.step()

        print(
            f"Epoch [{epoch+1:3d}/{EPOCHS}] "
            f"Train: {avg_train:.4f} | Val: {avg_val:.4f} | "
            f"Det: {det_acc:.1f}% | Mat: {mat_acc:.1f}% | "
            f"RMSE: {rmse:.3f}m | MAE: {mae:.3f}m | "
            f"TrainLoss=({avg_train_det:.3f}, {avg_train_dist:.3f}, {avg_train_mat:.3f}) | "
            f"ValLoss=({avg_val_det:.3f}, {avg_val_dist:.3f}, {avg_val_mat:.3f}) | "
            f"LR: {scheduler.get_last_lr()[0]:.2e}"
        )

        if smoothed_val < best_val_loss:
            best_val_loss  = smoothed_val
            early_stop_cnt = 0
            torch.save(model.state_dict(), model_run_dir / "damp_best.pth")
        else:
            early_stop_cnt += 1
            if early_stop_cnt >= PATIENCE:
                print(f"Early stopping at epoch {epoch+1}")
                break

    torch.save(model.state_dict(), model_run_dir / "damp.pth")
    model.load_state_dict(torch.load(model_run_dir / "damp.pth"))

    with open(result_run_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=4)

    run_evaluation(model, val_loader, DEVICE, result_run_dir, history)