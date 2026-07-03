import os
import yaml 
import json
import numpy as np
import pandas as pd
from pathlib import Path
import soundfile as sf
from scipy.signal import fftconvolve, spectrogram as compute_spectogram
from configs.config import PROJECT
 
PROCESSED_DATA_ROOT = ""
EXCITATION_PATH = ""
AUGMENTED_ROOT = ""

WHITE_NOISE_STD = 0.005
MAX_SHIFT = 10
AMPLITUDE_RANGE = (0.9, 1.1)
FS = 16000
NUM_CHANNELS = 16
NPERSEG = 256
NOVERLAP = 128

def add_white_noise(ir):
    return ir + np.random.normal(0, WHITE_NOISE_STD, size=ir.shape)
 
def random_time_shift(ir):
    shift = np.random.randint(-MAX_SHIFT, MAX_SHIFT + 1)
    return np.roll(ir, shift)
 
def amplitude_scaling(ir):
    scale = np.random.uniform(*AMPLITUDE_RANGE)
    return ir * scale
 
def compute_spectrogram(ir):
    _, _, Sxx = compute_spectogram(ir, fs=FS, nperseg=NPERSEG, noverlap=NOVERLAP)
    return Sxx

def save_augmented(source_folder, output_folder, aug_fn, aug_name):
    os.makedirs(output_folder, exist_ok=True)
 
    for ch in range(1, NUM_CHANNELS + 1):
        ir_path = source_folder / f"ir_mic_{ch}.npy"
        if not ir_path.exists():
            continue
 
        ir = np.load(ir_path)
        ir_aug = aug_fn(ir)
        Sxx = compute_spectrogram(ir_aug)
 
        np.save(output_folder / f"ir_mic_{ch}.npy", ir_aug)
        np.save(output_folder / f"spec_mic_{ch}.npy", Sxx)
 
    # Copy metadata.json unchanged — labels don't change with augmentation
    src_meta = source_folder / "metadata.json"
    if src_meta.exists():
        with open(src_meta) as f:
            meta = json.load(f)
        meta["augmentation"] = aug_name
        with open(output_folder / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)
 
    print(f"  [{aug_name}] → {output_folder.name}")
 

if __name__ == "__main__":

    project = PROJECT

    PROCESSED_DATA_ROOT = Path(f'{Path(__file__).parent}/processed')
    EXCITATION_PATH = Path(f'{project}/excitation.wav')
    AUGMENTED_ROOT = Path(f'{Path(__file__).parent}/augmented')

    os.makedirs(AUGMENTED_ROOT, exist_ok=True)

    augmentations = {
        "noise": add_white_noise,
        "shift": random_time_shift,
        "scale": amplitude_scaling,
    }
 
    recording_folders = [
        p for p in sorted(PROCESSED_DATA_ROOT.iterdir())
        if p.is_dir() and (p / "metadata.json").exists()
    ]
 
    print(f"Found {len(recording_folders)} recordings to augment.\n")
 
    for source_folder in recording_folders:
        print(f"Augmenting: {source_folder.name}")
        for aug_name, aug_fn in augmentations.items():
            output_folder = AUGMENTED_ROOT / f"{aug_name}_{source_folder.name}"
            save_augmented(source_folder, output_folder, aug_fn, aug_name)
 
    print(f"\nDone. {len(recording_folders) * len(augmentations)} augmented recordings saved to {AUGMENTED_ROOT}")
 