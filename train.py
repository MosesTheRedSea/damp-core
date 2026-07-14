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
from sklearn.metrics import confusion_matrix, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from src.damp import DampNet
from torchmetrics import ConfusionMatrix
from collections import Counter
from utils.utils import get_next_run_folder
from utils.utils import epoch_metrics
from utils.utils import run_evaluation

from data.augment import (add_white_noise, random_time_shift, amplitude_scaling, random_bandpass, random_eq, random_dropout, save_augmented)


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
        t_dist = torch.tensor(  meta["occlusion_distance"] + meta["object_distance"]).float()

        mat_key = self.obj_to_mat[meta["object_type"]]
        t_mat   = torch.tensor(self.mat_mapping[mat_key]).long()

        return ir_tensor, spec_tensor, t_det, t_dist, t_mat


class MultiTaskLoss(nn.Module):

    """
    Combines:
      - Detection classification loss
      - Distance regression loss
      - Material classification loss
      - Optional orthogonality regularization

    Each loss is weighted before summation.
    """

    def __init__(self, w_det=1.0, w_dist=3.0, w_mat=1.0, ortho_lambda=0.01, num_det_classes=22, num_mat_classes=5, max_dist=1.5):
        
        super().__init__()

        self.w_det = w_det
        self.w_dist = w_dist
        self.w_mat = w_mat

        self.ortho_lambda = ortho_lambda
        self.num_det_classes = num_det_classes
        self.num_mat_classes = num_mat_classes
        self.max_dist = max_dist

    def forward(self, p_det, t_det, p_dist, t_dist, p_mat, t_mat, ortho_loss=None):

        loss_det  = F.cross_entropy(p_det, t_det)
        loss_mat  = F.cross_entropy(p_mat, t_mat)
        loss_dist = F.l1_loss(p_dist.squeeze(-1), t_dist)  # |prediciton - target|

        # Normalise each loss to roughly [0, 1]
        loss_det_norm  = loss_det  / math.log(self.num_det_classes)
        loss_mat_norm  = loss_mat  / math.log(self.num_mat_classes)
        loss_dist_norm = loss_dist / self.max_dist

        total = (self.w_det  * loss_det_norm  + self.w_mat  * loss_mat_norm  + self.w_dist * loss_dist_norm)

        if ortho_loss is not None:
            total = total + self.ortho_lambda * torch.clamp(ortho_loss, 0, 10)

        return total, (loss_det_norm.item(), loss_dist_norm.item(), loss_mat_norm.item())

def get_object_groups(root):

    objects = {}

    for meta_path in root.rglob("metadata.json"):

        with open(meta_path) as f:
            meta=json.load(f)

        obj = meta["object_type"]

        folder = meta_path.parent

        if obj not in objects:
            objects[obj]=[]

        objects[obj].append(folder)

    return objects


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Disentangled Acoustic Multi-task Perception (DAMP)")
    parser.add_argument('--pr', type=str, default="", help="Processed data directory")
    parser.add_argument('--ar', type=str, default="", help="Augmented data directory")
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=32, help='Size of batches')
    parser.add_argument('--num_epochs', type=int, default=1000, help='Number of batch iterations')
    parser.add_argument('--weight_decay', type=int, default=None, help="Wegiht Decay")


    OBJECT_TO_MATERIAL = {

        "no_object": "none",
        "cardboard_box": "paper_cardboard",
        "speaker": "plastic",
        "pot": "metal",
        "strainer": "metal",
        "pitcher": "metal",
        "ladder": "metal",
        "ceramic_mug": "ceramic",
        "glass_mug": "glass",
        "plate": "ceramic",
        "ceramic_bowl": "ceramic",
        "trash_bin": "plastic",
        "metal_cup": "metal",
        "plastic_bottle": "plastic",
        "plastic_bowl": "plastic",
        "plastic_container": "plastic",
        "plastic_sport": "plastic",
        "plastic_shaker": "plastic",
        "monitor":"plastic",
        "teapot": "ceramic",
        "glass_vodka": "glass", 
        "glass_shooter": "glass"
    }

    
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
        "plaastic_container",
        "plastic_sport",
        "plastic_shaker",
        "monitor",
        "teapot",
        "glass_vodka", 
        "glass_shooter"
    ]

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

    processed_root = Path("./data/processed/")
    augmented_root = Path("./data/augmented/")

    check = set()

    # Original recordings only
    original_folders = sorted([
        f for f in processed_root.iterdir()
        if f.is_dir() and (f / "metadata.json").exists()
    ])

    all_distances = []

    for f in original_folders:
        with open(f / "metadata.json", "r") as file:
            meta = json.load(file)

        total_distance = (
            meta["occlusion_distance"] +
            meta["object_distance"]
        )

        all_distances.append(total_distance)

    MAX_DIST = max(all_distances)
    print(f"Max occlusion distance: {MAX_DIST}m")

    # Temporary dataset for labels
    temp_dataset = ImpulseData(
        original_folders,
        det_mapping=det_map,
        mat_mapping=mat_map,
        obj_to_mat=OBJECT_TO_MATERIAL
    )

    labels = [temp_dataset[i][2].item() for i in range(len(temp_dataset))]

    processed_root = Path("./data/processed")
    object_groups = get_object_groups(processed_root)
    
    train_folders = []
    val_folders = []


    for obj, folders in object_groups.items():

        train_obj, val_obj = train_test_split(
            folders,
            test_size=0.20,
            random_state=42
        )

        train_folders.extend(train_obj)
        val_folders.extend(val_obj)


    # Split ORIGINAL recordings first
    # train_folders, val_folders = train_test_split(
    #     original_folders,
    #     test_size=0.2,
    #     stratify=labels,
    #     random_state=42
    # )

    # Validation remains ORIGINAL ONLY
    val_full = val_folders

    augmentations = {
        "noise": add_white_noise,
        "shift": random_time_shift,
        "scale": amplitude_scaling,
        "bandpass": random_bandpass,
        "eq": random_eq,
        "dropout": random_dropout
    }


    for folder in train_folders:
        for name, fn in augmentations.items():
            aug_folder = augmented_root / f"{name}_{folder.name}"
            if not aug_folder.exists():
                save_augmented(
                    folder,
                    aug_folder,
                    fn,
                    name
                )
    

     # Add augmentations ONLY to training set
    train_full = []
    for folder in train_folders:
        train_full.append(folder)
        name = folder.name
        for aug in ["noise", "shift", "scale", "bandpass", "eq", "dropout"]:
            aug_folder = augmented_root / f"{aug}_{name}"
            if aug_folder.exists():
                train_full.append(aug_folder)

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

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False
    )

    print(f"Original recordings: {len(original_folders)}")
    print(f"Training originals: {len(train_folders)}")
    print(f"Validation originals: {len(val_folders)}")
    print(f"Training samples after augmentation: {len(train_full)}")

    aug_count = len(train_full) - len(train_folders)
    print(f"Augmented samples added: {aug_count}")

    train_objects = set()

    for f in train_full:
        with open(f/"metadata.json") as file:
            meta=json.load(file)
        train_objects.add(meta["object_type"])

    test_objects = set()

    for f in val_full:
        with open(f/"metadata.json") as file:
            meta=json.load(file)
        test_objects.add(meta["object_type"])


    print("TRAIN OBJECTS:", train_objects)
    print("TEST OBJECTS:", test_objects)

    print("OVERLAP:", train_objects.intersection(test_objects))


    BATCH_SIZE = 32
    EPOCHS = 100
    LR = 1e-3
    WEIGHT_DECAY = 1e-4
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    PATIENCE = 20
    ORTHO_LAMBDA = 0.01
    VAL_SMOOTH = 0.9

    MODEL_DIR = (
        "./models"
    )

    RESULT_DIR = (
        "./results"
    )

    model_run_dir = get_next_run_folder(MODEL_DIR)
    result_run_dir = get_next_run_folder(RESULT_DIR)

    model = DampNet(len(OBJ_CLASSES), len(MAT_CLASSES)).to(DEVICE)
    criterion = MultiTaskLoss(ortho_lambda=ORTHO_LAMBDA)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS,
        eta_min=1e-5,
    )

    best_val_loss   = float('inf')
    early_stop_cnt  = 0
    best_score = -float("inf")

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

            ir, spec = ir.to(DEVICE), spec.to(DEVICE)
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

        avg_train      = np.mean(train_losses)
        avg_val        = np.mean(val_losses)
        smoothed_val = avg_val 
        smoothed_val = VAL_SMOOTH * smoothed_val + (1 - VAL_SMOOTH) * avg_val
        avg_train_det  = np.mean(train_det)
        avg_train_dist = np.mean(train_dist)
        avg_train_mat  = np.mean(train_mat)
        avg_val_det    = np.mean(val_det)
        avg_val_dist   = np.mean(val_dist)
        avg_val_mat    = np.mean(val_mat)

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

        score = (
            0.4 * (det_acc / 100.0) +
            0.4 * (mat_acc / 100.0) -
            0.2 * mae
        )

        print(
            f"Epoch [{epoch+1:3d}/{EPOCHS}] "
            f"Train: {avg_train:.4f} | Val: {avg_val:.4f} | "
            f"Det: {det_acc:.1f}% | Mat: {mat_acc:.1f}% | "
            f"RMSE: {rmse:.3f}m | MAE: {mae:.3f}m | "
            f"TrainLoss=({avg_train_det:.3f}, {avg_train_dist:.3f}, {avg_train_mat:.3f}) | "
            f"ValLoss=({avg_val_det:.3f}, {avg_val_dist:.3f}, {avg_val_mat:.3f}) | "
            f"LR: {scheduler.get_last_lr()[0]:.2e}"
        )


        if score > best_score:
            best_score = score
            early_stop_cnt = 0
            torch.save(model.state_dict(), model_run_dir / "damp_best.pth")
        else:
            early_stop_cnt += 1
            if early_stop_cnt >= PATIENCE:
                print(f"Early stopping at epoch {epoch+1}")
                break

        # if smoothed_val < best_val_loss:
        #     best_val_loss  = smoothed_val
        #     early_stop_cnt = 0
        #     torch.save(model.state_dict(), model_run_dir / "best_model.pth")
        # else:
        #     early_stop_cnt += 1
        #     if early_stop_cnt >= PATIENCE:
        #         print(f"Early stopping at epoch {epoch+1}")
        #         break

    torch.save(model.state_dict(), model_run_dir / "damp_final.pth")
    model.load_state_dict(torch.load(model_run_dir / "damp_final.pth"))

    with open(result_run_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=4)

    run_evaluation(model, val_loader, DEVICE, OBJ_CLASSES, MAT_CLASSES, result_run_dir, history)
