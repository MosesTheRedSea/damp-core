import os
import json
import torch
import re
import torch.nn as nn
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

        ir_list = []
        spec_list = []

        for ch in range(1, 17):
            ir_list.append(np.load(folder / f"ir_mic_{ch}.npy"))
            spec_list.append(np.load(folder / f"spec_mic_{ch}.npy"))

        ir_tensor = torch.from_numpy(np.stack(ir_list)).float()
        spec_tensor = torch.from_numpy(np.stack(spec_list)).float()

        2det = torch.tensor(self.det_mapping[meta["object_type"]]).long()
        t_dist = torch.tensor(meta["occlusion_distance"]).float()
        mat_key = OBJECT_TO_MATERIAL[meta["object_type"]]

        t_mat   = torch.tensor(self.mat_mapping[mat_key]).long()
        return ir_tensor, spec_tensor, t_det, t_dist, t_mat
    
class MultiTaskLoss(nn.Module):

    def __init__(self, ortho_lambda=0.01):
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(3))
        self.ortho_lambda = ortho_lambda

    def forward(self, p_det, t_det,
                      p_dist, t_dist,
                      p_mat, t_mat,
                      ortho_loss=None):

        # classification (logits -> CrossEntropyLoss)
        loss_det = nn.functional.cross_entropy(p_det, t_det)
        loss_mat = nn.functional.cross_entropy(p_mat, t_mat)

        # regression
        loss_dist = nn.functional.mse_loss(p_dist.squeeze(), t_dist)

        # uncertainty weighting
        prec_det  = torch.exp(-self.log_vars[0])
        prec_dist = torch.exp(-self.log_vars[1])
        prec_mat  = torch.exp(-self.log_vars[2])

        total = (
            prec_det  * loss_det  + self.log_vars[0] +
            prec_dist * loss_dist + self.log_vars[1] +
            prec_mat  * loss_mat  + self.log_vars[2]
        )

        if ortho_loss is not None:
            total = total + self.ortho_lambda * ortho_loss

        return total, (loss_det.item(), loss_dist.item(), loss_mat.item())

if __name__ == '__main__':
    
    OBJECT_TO_MATERIAL = {

        "no_object": "none",

        # fabric
        "bookbag": "fabric",
        "rags": "fabric",

        # paper_cardboard
        "textbooks": "paper_cardboard",
        "paper": "paper_cardboard",
        "cardboard_box": "paper_cardboard",

        # plastic
        "plastic_bottle": "plastic",
        "container": "plastic",
        "trash_bin": "plastic",
        "plastic_cup": "plastic",
        "plastic_bowl": "plastic",

        # metal
        "pot": "metal",
        "strainer": "metal",
        "sign": "metal",
        "ladder": "metal",
        "rolling_chair": "metal",

        # glass
        "glass_bottle": "glass",
        "glass_cup": "glass",
        "glass_bowl": "glass",

        # ceramic
        "mug": "ceramic",
        "ceramic_bowl": "ceramic",
        "plate": "ceramic",

        # sand
        "sandbag": "sand"
    }

    obj_classes = [

        "no_object",

        # fabric
        "bookbag",
        "rags",

        # wood
        ""

        # paper_cardboard
        "textbooks",
        "paper",
        "cardboard_box",

        # plastic
        "plastic_bottle",
        "container",
        "trash_bin",
        "plastic_cup",
        "plastic_bowl",

        # metal
        "pot",
        "strainer",
        "sign",
        "ladder",
        "rolling_chair",

        # glass
        "glass_bottle",
        "glass_cup",
        "glass_bowl"

        # ceramic
        "mug",
        "ceramic_bowl",
        "plate",

        #sand
        "sandbag"
    ]

    # 9 Materials
    mat_classes = [
        "none",
        "wood",
        "metal",
        "plastic",
        "fabric",
        "glass",
        "ceramic",
        "paper_cardboard",
        "sand",
    ]

    det_map = {name: i for i, name in enumerate(obj_classes)}
    mat_map = {name: i for i, name in enumerate(mat_classes)}

    print(mat_classes)
    print(obj_classes)
    print(det_map)
    print(mat_map)

    processed_root = Path("/home/moses/Moses/Research/Current/Institute of Science Tokyo/acoustic-robotics/Multi-Task Acoustic Perception for Occluded Object Detection, Distance Estimation, and Material Classification/data/processed/")
    augmented_root = Path("/home/moses/Moses/Research/Current/Institute of Science Tokyo/acoustic-robotics/Multi-Task Acoustic Perception for Occluded Object Detection, Distance Estimation, and Material Classification/data/augmented/")

    check = set()

    # Original recordings only
    original_folders = sorted([
        f for f in processed_root.iterdir()
        if f.is_dir() and (f / "metadata.json").exists()
    ])

    
    # Temporary dataset for labels
    temp_dataset = ImpulseData(
        original_folders,
        det_mapping=det_map,
        mat_mapping=mat_map,
        obj_to_mat=OBJECT_TO_MATERIAL
    )

    labels = [
        temp_dataset[i][2].item()
        for i in range(len(temp_dataset))
    ]

    # Split ORIGINAL recordings first
    train_folders, val_folders = train_test_split(
        original_folders,
        test_size=0.15,
        stratify=labels,
        random_state=42
    )

    # Add augmentations ONLY to training set
    train_full = []
    for folder in train_folders:
        train_full.append(folder)
        name = folder.name
        for aug in ["noise", "shift", "scale"]:
            aug_folder = augmented_root / f"{aug}_{name}"
            if aug_folder.exists():
                train_full.append(aug_folder)

    # Validation remains ORIGINAL ONLY
    val_full = val_folders

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

    BATCH_SIZE = 32
    EPOCHS = 100
    LR = 1e-3
    WEIGHT_DECAY = 1e-4
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    PATIENCE = 15
    ORTHO_LAMBDA = 0.01

    MODEL_DIR = (
        "/home/moses/Moses/Research/Current/"
        "Institute of Science Tokyo/acoustic-robotics/"
        "Multi-Task Acoustic Perception for Occluded Object Detection, "
        "Distance Estimation, and Material Classification/models"
    )

    RESULT_DIR = (
        "/home/moses/Moses/Research/Current/"
        "Institute of Science Tokyo/acoustic-robotics/"
        "Multi-Task Acoustic Perception for Occluded Object Detection, "
        "Distance Estimation, and Material Classification/results"
    )

    model_run_dir = get_next_run_folder(MODEL_DIR)
    result_run_dir = get_next_run_folder(RESULT_DIR)

    print(f"Saving model to: {model_run_dir}")
    print(f"Saving results to: {result_run_dir}")


    model = DampNet(len(obj_classes), len(mat_classes)).to(DEVICE)
    criterion = MultiTaskLoss(ortho_lambda=ORTHO_LAMBDA).to(DEVICE)
 

    optimizer = optim.AdamW(
        [
            {'params': model.parameters()},
            {'params': criterion.parameters(), 'lr': LR},
        ],
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )
 
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-5
    )

    best_val_loss   = float('inf')
    early_stop_cnt  = 0

    for epoch in range(EPOCHS):
        model.train()
        train_losses = []
        
        for ir, spec, t_det, t_dist, t_mat in train_loader:

            ir, spec = ir.to(DEVICE), spec.to(DEVICE)

            t_det, t_dist, t_mat = t_det.to(DEVICE), t_dist.to(DEVICE), t_mat.to(DEVICE)

            optimizer.zero_grad()
            p_det, p_dist, p_mat = model(ir, spec)

            loss, _ = criterion(
                p_det, t_det,
                p_dist, t_dist,
                p_mat, t_mat,
                ortho_loss=model.orthogonality_loss
            )

            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        
        with torch.no_grad():
            for ir, spec, t_det, t_dist, t_mat in val_loader:
                ir, spec = ir.to(DEVICE), spec.to(DEVICE)
                t_det, t_dist, t_mat = (
                    t_det.to(DEVICE), t_dist.to(DEVICE), t_mat.to(DEVICE)
                )
                p_det, p_dist, p_mat = model(ir, spec)
                v_loss, _ = criterion(p_det, t_det, p_dist, t_dist, p_mat, t_mat)
                val_losses.append(v_loss.item())
 
        avg_train = np.mean(train_losses)
        avg_val   = np.mean(val_losses)
        det_acc, mat_acc, rmse, mae = epoch_metrics(model, val_loader, DEVICE)

        print(
            f"Epoch [{epoch+1:3d}/{EPOCHS}] "
            f"Train: {avg_train:.4f} | Val: {avg_val:.4f} | "
            f"Det: {det_acc:.1f}% | Mat: {mat_acc:.1f}% | "
            f"RMSE: {rmse:.3f}m | MAE: {mae:.3f}m | "
            f"LR: {scheduler.get_last_lr()[0]:.2e}"
        )

        scheduler.step()
 
        if avg_val < best_val_loss:
            best_val_loss  = avg_val
            early_stop_cnt = 0
            torch.save(model.state_dict(), model_run_dir / "best_model.pth")
        else:
            early_stop_cnt += 1
            if early_stop_cnt >= PATIENCE:
                print(f"Early stopping at epoch {epoch+1}")
                break
 
    # Always save the final checkpoint too
    torch.save(model.state_dict(), model_run_dir / "final_model.pth")
 
    # Load best weights for evaluation
    model.load_state_dict(torch.load(model_run_dir / "best_model.pth"))
    run_evaluation(model, val_loader, DEVICE, obj_classes, mat_classes, result_run_dir)
 