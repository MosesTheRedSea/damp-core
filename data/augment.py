import os
import yaml 
import json
import numpy as np
import pandas as pd
from pathlib import Path
import soundfile as sf
import pyroomacoustics
from pyroomacoustics import ShoeBox

from scipy.signal import (
    fftconvolve,
    spectrogram as compute_spectrogram,
    butter,
    sosfilt
)

WHITE_NOISE_STD = 0.01
MAX_SHIFT = 30
AMPLITUDE_RANGE = (0.7, 1.3)

FS = 16000
NUM_CHANNELS = 16
NPERSEG = 256
NOVERLAP = 128

# Augmentation Methods Data Manipulation
def random_polarity_flip(ir):
    return ir * (-1.0 if np.random.rand() > 0.5 else 1.0)

def interchannel_delay(ir, max_mic_shift=2):
    ir_aug = ir.copy()
    for c in range(ir_aug.shape[0]):
        shift = np.random.randint(-max_mic_shift, max_mic_shift + 1)
        ir_aug[c] = np.roll(ir_aug[c], shift)
    return ir_aug

def random_decay_scaling(ir):
    length = ir.shape[-1]
    decay_factor = np.random.uniform(-0.001, 0.001)
    envelope = np.exp(decay_factor * np.arange(length))
    return ir * envelope

def random_room_augment(ir):
    room_dim = [
        np.random.uniform(3, 8),
        np.random.uniform(3, 8), 
        np.random.uniform(2.5, 4)
    ]
    absorption = np.random.uniform(0.1, 0.6)
    
    room = ShoeBox(
        room_dim,
        fs=FS,
        materials=pyroomacoustics.Material(absorption),
        max_order=10
    )
    
    room.add_source([1.0, 1.0, 1.0])
    room.add_microphone([2.0, 2.0, 1.5])
    room.compute_rir()
    
    room_rir = room.rir[0][0]
    room_rir = room_rir[:min(len(room_rir), 512)]
    room_rir = room_rir / (np.max(np.abs(room_rir)) + 1e-8)
    
    augmented = np.convolve(ir, room_rir * 0.15, mode='same')
    return ir + augmented

def colored_noise(ir, noise_type='None'):
    if noise_type is None:
        noise_type = np.random.choice(['pink', 'brown', 'blue'])

    n = len(ir)
    freqs = np.fft.rfftfreq(n, 1/FS)
    freqs[0] = 1
    
    if noise_type == 'pink':
        power = 1 / freqs          
    elif noise_type == 'brown':
        power = 1 / freqs**2       
    elif noise_type == 'blue':
        power = freqs              
    
    power = power / np.max(power)
    
    noise_fft = np.random.randn(len(freqs)) + 1j * np.random.randn(len(freqs))
    noise_fft *= np.sqrt(power)
    noise = np.fft.irfft(noise_fft, n=n)
    noise = noise / (np.max(np.abs(noise)) + 1e-8)
    
    snr_db = np.random.uniform(20, 40)
    signal_power = np.mean(ir**2)
    noise_power = 10**(-snr_db/10) * signal_power
    noise *= np.sqrt(noise_power / (np.mean(noise**2) + 1e-8))
    
    return ir + noise

def channel_dropout(ir_tensor, p=0.15):
    # ir_tensor: (16, L)
    mask = np.random.rand(16) > p
    while mask.sum() < 8: 
        mask = np.random.rand(16) > p
    return ir_tensor * mask[:, np.newaxis]


def distance_jitter(ir, max_jitter_cm=5):
    speed_of_sound = 343
    max_jitter_samples = int((max_jitter_cm / 100) * FS / speed_of_sound * 2)
    jitter = np.random.randint(-max_jitter_samples, max_jitter_samples)
    return np.roll(ir, jitter)


def spec_augment(spec, num_freq_masks=2, num_time_masks=2, freq_width=10, time_width=20):
    H, W = spec.shape
    augmented = spec.copy()
    
    for _ in range(num_freq_masks):
        f = np.random.randint(0, freq_width)
        f0 = np.random.randint(0, H - f)
        augmented[f0:f0+f, :] = 0
    
    # time masking
    for _ in range(num_time_masks):
        t = np.random.randint(0, time_width)
        t0 = np.random.randint(0, W - t)
        augmented[:, t0:t0+t] = 0
    
    return augmented

 
def compute_spectro(ir):
    _, _, Sxx = compute_spectrogram(ir, fs=FS, nperseg=NPERSEG, noverlap=NOVERLAP)
    return Sxx

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

        np.save(output_folder / f"ir_mic_{ch}.npy", ir_aug)

        np.save(output_folder / f"spec_mic_{ch}.npy", Sxx)

        src_meta = source_folder / "metadata.json"
        if src_meta.exists():
            with open(src_meta) as f:
                meta = json.load(f)

            meta["augmentation"] = aug_name

            with open(output_folder / "metadata.json", "w") as f:
                json.dump(meta, f, indent=2)

        print(f"  [{aug_name}] → {output_folder.name}")
