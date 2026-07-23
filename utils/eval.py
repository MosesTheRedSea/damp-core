import json
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from collections import defaultdict

from src.damp import DampNet
from utils.utils import run_evaluation, get_next_run_folder


class ImpulseData(Dataset):

    def __init__(self, folders, det_mapping, mat_mapping, obj_to_mat):
        self.folders = folders
        self.det_mapping = det_mapping
        self.mat_mapping = mat_mapping
        self.obj_to_mat = obj_to_mat

    def __len__(self):
        return len(self.folders)

    def __getitem__(self, idx):
        folder = self.folders[idx]

        with open(folder / "metadata.json", "r") as f:
            meta = json.load(f)

        ir_list, spec_list = [], []
        for ch in range(1, 17):
            ir_list.append(np.load(folder / f"ir_mic_{ch}.npy"))
            spec_list.append(np.load(folder / f"spec_mic_{ch}.npy"))

        ir_tensor = torch.from_numpy(np.stack(ir_list)).float()
        spec_tensor = torch.from_numpy(np.stack(spec_list)).float()

        t_det = torch.tensor(self.det_mapping[meta["object_type"]]).long()
        t_dist = torch.tensor(
            meta["object_distance"] + meta["occlusion_distance"],
            dtype=torch.float32
        )

        mat_key = self.obj_to_mat[meta["object_type"]]
        t_mat = torch.tensor(self.mat_mapping[mat_key]).long()

        return ir_tensor, spec_tensor, t_det, t_dist, t_mat


def get_object_groups(folders):
    groups = defaultdict(list)
    for folder in folders:
        with open(folder / "metadata.json") as f:
            meta = json.load(f)
        groups[meta["object_type"]].append(folder)
    return groups


if __name__ == '__main__':

    CHECKPOINT_PATH = Path("/home/3/um07293/research/occul-net/models/run_1/best_model.pth")
    PROCESSED_ROOT = Path("./data/processed")

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
        "monitor": "plastic",
        "teapot": "ceramic",
        "glass_vodka": "glass",
        "glass_shooter": "glass"
    }

    OBJ_CLASSES = [
        "no_object", "cardboard_box", "speaker", "pot", "strainer",
        "pitcher", "ladder", "ceramic_mug", "glass_mug", "plate",
        "ceramic_bowl", "trash_bin", "metal_cup", "plastic_bottle",
        "plastic_bowl", "plastic_container", "plastic_sport",
        "plastic_shaker", "monitor", "teapot", "glass_vodka", "glass_shooter"
    ]

    MAT_CLASSES = ["none", "paper_cardboard", "plastic", "metal", "ceramic", "glass"]

    det_map = {name: i for i, name in enumerate(OBJ_CLASSES)}
    mat_map = {name: i for i, name in enumerate(MAT_CLASSES)}

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Rebuild the SAME val split used at train time ──────────────────────
    original_folders = sorted([
        f for f in PROCESSED_ROOT.iterdir()
        if f.is_dir() and (f / "metadata.json").exists()
    ])

    print(f"Found {len(original_folders)} processed recordings.")

    groups = get_object_groups(original_folders)

    val_folders = []
    for obj, folders in groups.items():
        _, val = train_test_split(
            folders,
            test_size=0.2,
            random_state=42,   # must match training exactly
            shuffle=True,
        )
        val_folders.extend(val)

    print(f"Validation recordings: {len(val_folders)}")

    val_dataset = ImpulseData(
        val_folders,
        det_mapping=det_map,
        mat_mapping=mat_map,
        obj_to_mat=OBJECT_TO_MATERIAL
    )

    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    # ── Load model ───────────────────────────────────────────────────────
    model = DampNet(len(OBJ_CLASSES), len(MAT_CLASSES)).to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    model.eval()

    # ── Optionally load training history if it exists next to the run ────
    # Adjust this path if your results/run_N folder differs from models/run_N
    history_path = Path("./results/run_1/training_history.json")
    history = None
    if history_path.exists():
        with open(history_path, "r") as f:
            history = json.load(f)
        print(f"Loaded training history from {history_path}")
    else:
        print("No training_history.json found — skipping loss/accuracy curves.")

    result_run_dir = get_next_run_folder("./results")
    print(f"Saving eval results to: {result_run_dir}")

    run_evaluation(model, val_loader, DEVICE, OBJ_CLASSES, MAT_CLASSES, result_run_dir, history)