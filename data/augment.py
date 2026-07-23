import os
import yaml 
import json
import numpy as np
import pandas as pd
from pathlib import Path
import soundfile as sf
from scipy.signal import (
    fftconvolve,
    spectrogram as compute_spectrogram,
    butter,
    sosfilt
)
from configs.config import PROJECT


WHITE_NOISE_STD = 0.01
MAX_SHIFT = 30
AMPLITUDE_RANGE = (0.7, 1.3)

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

def random_bandpass(ir):
    low = np.random.uniform(100, 1000)
    high = np.random.uniform(3000, 7000)

    if high <= low:
        high = low + 500

    sos = butter(
        4,
        [low, high],
        btype="bandpass",
        fs=FS,
        output="sos"
    )

    return sosfilt(sos, ir)

def random_eq(ir):
    fft = np.fft.rfft(ir)

    gains = np.random.uniform(0.7, 1.3, len(fft))

    window = np.hanning(101)
    window /= window.sum()

    gains = np.convolve(gains, window, mode="same")

    fft *= gains

    return np.fft.irfft(fft, len(ir))

def random_dropout(ir):
    ir = ir.copy()
    length = np.random.randint(5, 40)
    start = np.random.randint(0, len(ir) - length)
    ir[start:start+length] = 0

    return ir
 
def compute_spectro(ir):
    _, _, Sxx = compute_spectrogram(ir, fs=FS, nperseg=NPERSEG, noverlap=NOVERLAP)
    return Sxx

# def save_augmented(source_folder, output_folder, aug_fn, aug_name):
#     os.makedirs(output_folder, exist_ok=True)
 
#     for ch in range(1, NUM_CHANNELS + 1):
#         ir_path = source_folder / f"ir_mic_{ch}.npy"
#         if not ir_path.exists():
#             continue
 
#         ir = np.load(ir_path)
#         ir_aug = aug_fn(ir)
#         Sxx = compute_spectro(ir_aug)
 
#         np.save(output_folder / f"ir_mic_{ch}.npy", ir_aug)
#         np.save(output_folder / f"spec_mic_{ch}.npy", Sxx)
 
#     # Copy metadata.json unchanged — labels don't change with augmentation
#     src_meta = source_folder / "metadata.json"
#     if src_meta.exists():
#         with open(src_meta) as f:
#             meta = json.load(f)
#         meta["augmentation"] = aug_name
#         with open(output_folder / "metadata.json", "w") as f:
#             json.dump(meta, f, indent=2)
 
#     print(f"  [{aug_name}] → {output_folder.name}")

def save_augmented(source_folder, output_folder, aug_fn, aug_name, FORCE_REGEN=False):

    if output_folder.exists() and FORCE_REGEN:
        import shutil
        shutil.rmtree(output_folder)

    os.makedirs(output_folder, exist_ok=True)

    for ch in range(1, NUM_CHANNELS + 1):

        ir_path = source_folder / f"ir_mic_{ch}.npy"

        if not ir_path.exists():
            continue

        ir = np.load(ir_path)

        ir_aug = aug_fn(ir)

        Sxx = compute_spectro(ir_aug)

        np.save(
            output_folder / f"ir_mic_{ch}.npy",
            ir_aug
        )

        np.save(
            output_folder / f"spec_mic_{ch}.npy",
            Sxx
        )

        src_meta = source_folder / "metadata.json"
        if src_meta.exists():
            with open(src_meta) as f:
                meta = json.load(f)

            meta["augmentation"] = aug_name

            with open(output_folder / "metadata.json", "w") as f:
                json.dump(meta, f, indent=2)

        print(f"  [{aug_name}] → {output_folder.name}")


if __name__ == "__main__":

    project = "/home/3/um07293/research/occul-net"

    PROCESSED_DATA_ROOT = Path(f'{Path(__file__).parent}/processed')
    EXCITATION_PATH = Path(f'{project}/excitation.wav')
    AUGMENTED_ROOT = Path(f'{Path(__file__).parent}/augmented')

    os.makedirs(AUGMENTED_ROOT, exist_ok=True)

    augmentations = {
        "noise": add_white_noise,
        "shift": random_time_shift,
        "scale": amplitude_scaling,
        "bandpass": random_bandpass,
        "eq": random_eq,
        "dropout": random_dropout,
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
 