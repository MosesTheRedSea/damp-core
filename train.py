import json
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import numpy as np
from pathlib import Path
from src.damp import DampNet
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

        ir_list   = []
        spec_list = []

        for ch in range(1, 17):
            ir_list.append(np.load(folder / f"ir_mic_{ch}.npy"))
            spec_list.append(np.load(folder / f"spec_mic_{ch}.npy"))

        ir_tensor   = torch.from_numpy(np.stack(ir_list)).float()
        spec_tensor = torch.from_numpy(np.stack(spec_list)).float()

        t_det  = torch.tensor(self.det_mapping[meta["object_type"]]).long()
        t_dist = torch.tensor(meta["occlusion_distance"]).float()

        mat_key = self.obj_to_mat[meta["object_type"]]
        t_mat   = torch.tensor(self.mat_mapping[mat_key]).long()

        return ir_tensor, spec_tensor, t_det, t_dist, t_mat


class MultiTaskLoss(nn.Module):

    def __init__(self, w_det=1.0, w_dist=1.0, w_mat=1.0, ortho_lambda=0.01,
                 num_det_classes=8, num_mat_classes=5, max_dist=1.5):
        super().__init__()
        self.w_det            = w_det
        self.w_dist           = w_dist
        self.w_mat            = w_mat
        self.ortho_lambda     = ortho_lambda
        self.num_det_classes  = num_det_classes
        self.num_mat_classes  = num_mat_classes
        self.max_dist         = max_dist

    def forward(self, p_det, t_det,
                      p_dist, t_dist,
                      p_mat, t_mat,
                      ortho_loss=None):

        loss_det  = F.cross_entropy(p_det, t_det)
        loss_mat  = F.cross_entropy(p_mat, t_mat)
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


if __name__ == '__main__':

    OBJECT_TO_MATERIAL = {
        "no_object":     "none",
        "cardboard_box": "paper_cardboard",
        "speaker":       "plastic",
        "pot":           "metal",
        "strainer":      "metal",
        "pitcher":       "metal",
        "ladder":        "metal",
        "sandbag":       "sand",
    }

    OBJ_CLASSES = [
        "no_object",
        "cardboard_box",
        "speaker",
        "pot",
        "strainer",
        "pitcher",
        "ladder",
        "sandbag",
    ]

    MAT_CLASSES = [
        "none",
        "paper_cardboard",
        "plastic",
        "metal",
        "sand",
    ]

    det_map = {name: i for i, name in enumerate(OBJ_CLASSES)}
    mat_map = {name: i for i, name in enumerate(MAT_CLASSES)}

    print(MAT_CLASSES)
    print(OBJ_CLASSES)
    print(det_map)
    print(mat_map)

    processed_root = Path("./data/processed/")
    augmented_root = Path("./data/augmented/")

    original_folders = sorted([
        f for f in processed_root.iterdir()
        if f.is_dir() and (f / "metadata.json").exists()
    ])

    # Auto-compute max distance from dataset
    all_distances = [
        json.load(open(f / "metadata.json"))["occlusion_distance"]
        for f in original_folders
    ]
    MAX_DIST = max(all_distances)
    print(f"Max occlusion distance: {MAX_DIST}m")

    # Temporary dataset for stratified split labels
    temp_dataset = ImpulseData(
        original_folders,
        det_mapping=det_map,
        mat_mapping=mat_map,
        obj_to_mat=OBJECT_TO_MATERIAL
    )

    labels = [temp_dataset[i][2].item() for i in range(len(temp_dataset))]

    train_folders, val_folders = train_test_split(
        original_folders,
        test_size=0.2,
        stratify=labels,
        random_state=42
    )

    # Add augmentations to training set only
    train_full = []
    for folder in train_folders:
        train_full.append(folder)
        name = folder.name
        for aug in ["noise", "shift", "scale"]:
            aug_folder = augmented_root / f"{aug}_{name}"
            if aug_folder.exists():
                train_full.append(aug_folder)

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

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=32, shuffle=False)

    print(f"Original recordings:              {len(original_folders)}")
    print(f"Training originals:               {len(train_folders)}")
    print(f"Validation originals:             {len(val_folders)}")
    print(f"Training samples (w/ augment):    {len(train_full)}")
    print(f"Augmented samples added:          {len(train_full) - len(train_folders)}")

    EPOCHS       = 100
    LR           = 5e-4
    WEIGHT_DECAY = 1e-4
    DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    PATIENCE     = 25

    VAL_SMOOTH = 0.9

    model_run_dir  = get_next_run_folder("./models")
    result_run_dir = get_next_run_folder("./results")

    print(f"Saving model to:   {model_run_dir}")
    print(f"Saving results to: {result_run_dir}")

    model = DampNet(len(OBJ_CLASSES), len(MAT_CLASSES)).to(DEVICE)

    criterion = MultiTaskLoss(
        w_det=1.0,
        w_dist=1.0,
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

    for epoch in range(EPOCHS):

        # ── Training ──────────────────────────────────────────────────────
        model.train()
        train_losses, train_det, train_dist, train_mat = [], [], [], []

        for ir, spec, t_det, t_dist, t_mat in train_loader:
            ir, spec       = ir.to(DEVICE), spec.to(DEVICE)
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

        # ── Validation ────────────────────────────────────────────────────
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

        # Early stopping on val loss (stable — no log_vars involved)
        if smoothed_val < best_val_loss:
            best_val_loss  = smoothed_val
            early_stop_cnt = 0
            torch.save(model.state_dict(), model_run_dir / "best_model.pth")
        else:
            early_stop_cnt += 1
            if early_stop_cnt >= PATIENCE:
                print(f"Early stopping at epoch {epoch+1}")
                break

    torch.save(model.state_dict(), model_run_dir / "final_model.pth")

    model.load_state_dict(torch.load(model_run_dir / "best_model.pth"))
    run_evaluation(model, val_loader, DEVICE, OBJ_CLASSES, MAT_CLASSES, result_run_dir)

# import json
# import torch
# import argparse
# import re
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from tqdm.auto import tqdm
# from torch.utils.data import DataLoader, Subset, Dataset
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import confusion_matrix, mean_squared_error, mean_absolute_error
# import matplotlib.pyplot as plt
# import seaborn as sns
# import numpy as np
# from pathlib import Path
# from src.damp import DampNet
# from torchmetrics import ConfusionMatrix
# from collections import Counter
# import math

# # from configs.config import OBJECT_TO_MATERIAL
# # from configs.config import OBJ_CLASSES
# # from configs.config import MAT_CLASSES

# from utils.utils import get_next_run_folder
# from utils.utils import epoch_metrics
# from utils.utils import run_evaluation

# class ImpulseData(Dataset):

#     def __init__(self, folders, det_mapping, mat_mapping, obj_to_mat):

#         self.folders = folders
#         self.det_mapping = det_mapping
#         self.mat_mapping = mat_mapping
#         self.obj_to_mat  = obj_to_mat

#     def __len__(self):
#         return len(self.folders)

#     def __getitem__(self, idx):
#         folder = self.folders[idx]

#         with open(folder / "metadata.json", "r") as f:
#             meta = json.load(f)

#         ir_list = []
#         spec_list = []

#         for ch in range(1, 17):
#             ir_list.append(np.load(folder / f"ir_mic_{ch}.npy"))
#             spec_list.append(np.load(folder / f"spec_mic_{ch}.npy"))

#         ir_tensor = torch.from_numpy(np.stack(ir_list)).float()
#         spec_tensor = torch.from_numpy(np.stack(spec_list)).float()

#         t_det = torch.tensor(self.det_mapping[meta["object_type"]]).long()
#         t_dist = torch.tensor(meta["occlusion_distance"]).float()
#         mat_key = OBJECT_TO_MATERIAL[meta["object_type"]]

#         t_mat   = torch.tensor(self.mat_mapping[mat_key]).long()
#         return ir_tensor, spec_tensor, t_det, t_dist, t_mat

# class MultiTaskLoss(nn.Module):
#     def __init__(self, w_det=1.0, w_dist=1.0, w_mat=1.0, ortho_lambda=0.01,
#                  num_det_classes=8, num_mat_classes=5, max_dist=1.5):

#         super().__init__()
#         self.w_det = w_det
#         self.w_dist = w_dist
#         self.w_mat = w_mat
#         self.ortho_lambda = ortho_lambda
#         self.num_det_classes = num_det_classes
#         self.num_mat_classes = num_mat_classes
#         self.max_dist = max_dist

#     def forward(self, p_det, t_det,
#                       p_dist, t_dist,
#                       p_mat, t_mat,
#                       ortho_loss=None):

#         loss_det  = F.cross_entropy(p_det, t_det)
#         loss_mat  = F.cross_entropy(p_mat, t_mat)
#         loss_dist = F.l1_loss(p_dist.squeeze(-1), t_dist)

#         loss_det_norm  = loss_det  / math.log(self.num_det_classes)
#         loss_mat_norm  = loss_mat  / math.log(self.num_mat_classes)
#         loss_dist_norm = loss_dist / self.max_dist

#         total = (
#             self.w_det  * loss_det_norm  +
#             self.w_mat  * loss_mat_norm  +
#             self.w_dist * loss_dist_norm
#         )

#         if ortho_loss is not None:
#             total = total + self.ortho_lambda * torch.clamp(ortho_loss, 0, 10)

#         return total, (loss_det_norm.item(), loss_dist_norm.item(), loss_mat_norm.item())

# if __name__ == '__main__':
    
#     OBJECT_TO_MATERIAL = {
#         "no_object":     "none",
#         "cardboard_box": "paper_cardboard",
#         "speaker":       "plastic",
#         "pot":           "metal",
#         "strainer":      "metal",
#         "pitcher":       "metal",
#         "ladder":        "metal",
#         "sandbag":       "sand",
#     }

#     OBJ_CLASSES = [
#         "no_object",
#         "cardboard_box",
#         "speaker",
#         "pot",
#         "strainer",
#         "pitcher",
#         "ladder",
#         "sandbag",
#     ]

#     MAT_CLASSES = [
#         "none",
#         "paper_cardboard",
#         "plastic",
#         "metal",
#         "sand",
#     ]

#     det_map = {name: i for i, name in enumerate(OBJ_CLASSES)}
#     mat_map = {name: i for i, name in enumerate(MAT_CLASSES)}

#     print(MAT_CLASSES)
#     print(OBJ_CLASSES)
#     print(det_map)
#     print(mat_map)

#     processed_root = Path("./data/processed/")
#     augmented_root = Path("./data/augmented/")

#     check = set()

#     # Original recordings only
#     original_folders = sorted([
#         f for f in processed_root.iterdir()
#         if f.is_dir() and (f / "metadata.json").exists()
#     ])

#     # Temporary dataset for labels
#     temp_dataset = ImpulseData(
#         original_folders,
#         det_mapping=det_map,
#         mat_mapping=mat_map,
#         obj_to_mat=OBJECT_TO_MATERIAL
#     )

#     labels = [
#         temp_dataset[i][2].item()
#         for i in range(len(temp_dataset))
#     ]

#     # Split ORIGINAL recordings first
#     train_folders, val_folders = train_test_split(
#         original_folders,
#         test_size=0.15,
#         stratify=labels,
#         random_state=42
#     )

#     # Add augmentations ONLY to training set
#     train_full = []
#     for folder in train_folders:
#         train_full.append(folder)
#         name = folder.name
#         for aug in ["noise", "shift", "scale"]:
#             aug_folder = augmented_root / f"{aug}_{name}"
#             if aug_folder.exists():
#                 train_full.append(aug_folder)

#     # Validation remains ORIGINAL ONLY
#     val_full = val_folders

#     train_dataset = ImpulseData(
#         train_full,
#         det_mapping=det_map,
#         mat_mapping=mat_map,
#         obj_to_mat=OBJECT_TO_MATERIAL
#     )

#     val_dataset = ImpulseData(
#         val_full,
#         det_mapping=det_map,
#         mat_mapping=mat_map,
#         obj_to_mat=OBJECT_TO_MATERIAL
#     )

#     train_loader = DataLoader(
#         train_dataset,
#         batch_size=32,
#         shuffle=True
#     )

#     val_loader = DataLoader(
#         val_dataset,
#         batch_size=32,
#         shuffle=False,
#     )

#     print(f"Original recordings: {len(original_folders)}")
#     print(f"Training originals: {len(train_folders)}")
#     print(f"Validation originals: {len(val_folders)}")
#     print(f"Training samples after augmentation: {len(train_full)}")

#     aug_count = len(train_full) - len(train_folders)
#     print(f"Augmented samples added: {aug_count}")

#     BATCH_SIZE = 32
#     EPOCHS = 100
#     LR = 1e-3
#     WEIGHT_DECAY = 1e-4
#     DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     PATIENCE = 15
#     ORTHO_LAMBDA = 0.01

#     MODEL_DIR = (
#         "./models"
#     )

#     RESULT_DIR = (
#         "./results"
#     )

#     model_run_dir = get_next_run_folder(MODEL_DIR)
#     result_run_dir = get_next_run_folder(RESULT_DIR)

#     print(f"Saving model to: {model_run_dir}")
#     print(f"Saving results to: {result_run_dir}")


#     model = DampNet(len(OBJ_CLASSES), len(MAT_CLASSES)).to(DEVICE)

#     criterion = MultiTaskLoss(
#         w_det=1.0,
#         w_dist=1.0,
#         w_mat=1.0,
#         ortho_lambda=0.01,
#         num_det_classes=len(OBJ_CLASSES),
#         num_mat_classes=len(MAT_CLASSES),
#         max_dist=1.5
#     )
 
#     optimizer = optim.AdamW(
#         [
#             {'params': model.parameters()},
#             {'params': criterion.parameters(), 'lr': LR},
#         ],
#         lr=LR,
#         weight_decay=WEIGHT_DECAY
#     )
 
#     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
#         optimizer, T_max=EPOCHS, eta_min=1e-5
#     )

#     best_score = -float("inf")
#     early_stop_cnt  = 0

#     for epoch in range(EPOCHS):
#         model.train()
#         train_losses = []
#         train_det_losses = []
#         train_dist_losses = []
#         train_mat_losses = []
        
#         for ir, spec, t_det, t_dist, t_mat in train_loader:

#             ir, spec = ir.to(DEVICE), spec.to(DEVICE)

#             t_det, t_dist, t_mat = t_det.to(DEVICE), t_dist.to(DEVICE), t_mat.to(DEVICE)

#             optimizer.zero_grad()
#             p_det, p_dist, p_mat = model(ir, spec)

#             loss, (det_loss, dist_loss, mat_loss) = criterion(
#                 p_det, t_det,
#                 p_dist, t_dist,
#                 p_mat, t_mat,
#                 ortho_loss=model.orthogonality_loss
#             )

#             loss.backward()

#             nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

#             optimizer.step()
#             train_losses.append(loss.item())
#             train_det_losses.append(det_loss)
#             train_dist_losses.append(dist_loss)
#             train_mat_losses.append(mat_loss)

#         model.eval()
#         val_losses = []
        
#         with torch.no_grad():
#             for ir, spec, t_det, t_dist, t_mat in val_loader:
#                 ir, spec = ir.to(DEVICE), spec.to(DEVICE)
#                 t_det, t_dist, t_mat = (
#                     t_det.to(DEVICE), t_dist.to(DEVICE), t_mat.to(DEVICE)
#                 )
#                 p_det, p_dist, p_mat = model(ir, spec)
#                 v_loss, _ = criterion(
#                     p_det,
#                     t_det,
#                     p_dist,
#                     t_dist,
#                     p_mat,
#                     t_mat,
#                     ortho_loss=model.orthogonality_loss
#                 )
#                 val_losses.append(v_loss.item())
 
#         avg_train = np.mean(train_losses)
#         avg_val = np.mean(val_losses)
#         avg_train_det = np.mean(train_det_losses)
#         avg_train_dist = np.mean(train_dist_losses)
#         avg_train_mat = np.mean(train_mat_losses)

#         det_acc, mat_acc, rmse, mae = epoch_metrics(model, val_loader, DEVICE)

#         sigmas = (
#             torch.exp(0.5 * criterion.log_vars)
#             .detach()
#             .cpu()
#             .numpy()
#         )

#         print(
#             f"Epoch [{epoch+1:3d}/{EPOCHS}] "
#             f"Train: {avg_train:.4f} | Val: {avg_val:.4f} | "
#             f"Det: {det_acc:.1f}% | Mat: {mat_acc:.1f}% | "
#             f"RMSE: {rmse:.3f}m | MAE: {mae:.3f}m | "
#             f"DetLoss: {avg_train_det:.3f} | "
#             f"DistLoss: {avg_train_dist:.3f} | "
#             f"MatLoss: {avg_train_mat:.3f} | "
#             f"σ=[{sigmas[0]:.2f}, {sigmas[1]:.2f}, {sigmas[2]:.2f}] | "
#             f"LR: {scheduler.get_last_lr()[0]:.2e}"
#         )

#         scheduler.step()
 
#         score = (
#             0.4 * (det_acc / 100.0) +
#             0.4 * (mat_acc / 100.0) -
#             0.2 * rmse
#         )

#         if score > best_score:
#             best_score = score
#             early_stop_cnt = 0

#             torch.save(
#                 model.state_dict(),
#                 model_run_dir / "best_model.pth"
#             )
 
#         else:
#             early_stop_cnt += 1

#             if early_stop_cnt >= PATIENCE:
#                 print(f"Early stopping at epoch {epoch+1}")
#                 break
 
#     # Always save the final checkpoint too
#     torch.save(model.state_dict(), model_run_dir / "final_model.pth")
 
#     # Load best weights for evaluation
#     model.load_state_dict(torch.load(model_run_dir / "best_model.pth"))
#     run_evaluation(model, val_loader, DEVICE, OBJ_CLASSES, MAT_CLASSES, result_run_dir)
 