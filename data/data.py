
import os
import json
import yaml
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve, spectrogram as compute_spec

AUDIO_DATA_ROOT = ""
EXCITATION_PATH = ""
SAVE_ROOT = ""
START_SAMPLE = 4900
END_SAMPLE = 6000
FS = 16000
NUM_CHANNELS = 16

def parse_dist(s):
    try:
        return float(s)
    except ValueError:
        return 0.0

def build_metadata_from_paths(audio_data_root):
    
    records = []

    for room in sorted(os.listdir(audio_data_root)):
        room_path = os.path.join(audio_data_root, room)
        if not os.path.isdir(room_path):
            continue
 
        for occ_type in sorted(os.listdir(room_path)):
            occ_path = os.path.join(room_path, occ_type)
            if not os.path.isdir(occ_path):
                continue
 
            if occ_type == "no_object":
                for fname in sorted(os.listdir(occ_path)):
                    if not fname.endswith(".wav"):
                        continue
                    records.append({
                        "filepath": os.path.join(occ_path, fname),
                        "room": room,
                        "occ_type": "no_object",
                        "occ_dist": 0.0,
                        "occld_type": "no_object",
                        "occld_dist": 0.0,
                    })
                continue
 
            for occld_type in sorted(os.listdir(occ_path)):
                occld_path = os.path.join(occ_path, occld_type)
                if not os.path.isdir(occld_path):
                    continue
 
                for dist_folder in sorted(os.listdir(occld_path)):
                    dist_path = os.path.join(occld_path, dist_folder)
                    if not os.path.isdir(dist_path):
                        continue
 
                    parts = dist_folder.split("-")
                    occ_dist = parse_dist(parts[0])
                    occld_dist = parse_dist(parts[1]) if len(parts) == 2 else 0.0
 
                    effective_occld = occld_type if occld_type != occ_type else "no_object"
 
                    for fname in sorted(os.listdir(dist_path)):
                        if not fname.endswith(".wav"):
                            continue
                        records.append({
                            "filepath": os.path.join(dist_path, fname),
                            "room": room,
                            "occ_type": occ_type.replace("-", "+"),
                            "occ_dist": occ_dist,
                            "occld_type": effective_occld,
                            "occld_dist": occld_dist,
                        })
 
    print(f"Found {len(records)} WAV files across all rooms.")
    return records

def load_excitation(excitation_path):
    excitation, fs = sf.read(excitation_path)
    if excitation.ndim > 1:
        excitation = excitation[:, 0]
    inv_filter = excitation[::-1]
    N = len(excitation)
    return inv_filter, N, fs

def extract_and_save(record, inv_filter, N, save_root,
                     start_sample, end_sample, fs, num_channels):

    filepath = record["filepath"]
 
    rel = os.path.relpath(filepath, start=os.path.dirname(os.path.dirname(filepath)))
    safe_name = rel.replace(os.sep, "_").replace(" ", "_").rstrip(".wav")
    save_folder = os.path.join(save_root, safe_name)
    os.makedirs(save_folder, exist_ok=True)
 
    try:
        data, file_fs = sf.read(filepath)
    except Exception as e:
        print(f"  ERROR reading {filepath}: {e}")
        return
 
    if file_fs != fs:
        print(f"  WARNING: {filepath} has fs={file_fs}, expected {fs}. Skipping.")
        return
 
    if data.ndim == 1:
        data = data[:, np.newaxis]
 
    n_ch = min(data.shape[1], num_channels)
 
    for ch in range(n_ch):
        ir_full = fftconvolve(data[:, ch], inv_filter, mode='full')
        ir_full = ir_full[N:N * 2]
        ir = ir_full[start_sample:end_sample]
 
        np.save(os.path.join(save_folder, f"ir_mic_{ch + 1}.npy"), ir)
 
        _, _, Sxx = compute_spec(ir, fs=fs, nperseg=256, noverlap=128)
        np.save(os.path.join(save_folder, f"spec_mic_{ch + 1}.npy"), Sxx)
 
    meta = {
        "filename": os.path.basename(filepath),
        "filepath": filepath,
        "room": record["room"],
        "occlusion_type": record["occ_type"],
        "occlusion_distance": record["occ_dist"],
        "object_type": record["occld_type"],
        "object_distance": record["occld_dist"],
        "num_channels": n_ch,
    }
    with open(os.path.join(save_folder, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
 
    print(f"  Saved {n_ch} channels → {save_folder}")
 

def print_summary(records):
    from collections import Counter
    print("\n=== Dataset Summary ===")
    occ_counts = Counter(r["occ_type"] for r in records)
    occld_counts = Counter(r["occld_type"] for r in records)
    room_counts = Counter(r["room"] for r in records)
 
    print("\nBy occlusion type:")
    for k, v in sorted(occ_counts.items()):
        print(f"  {k}: {v}")
 
    print("\nBy occluder type:")
    for k, v in sorted(occld_counts.items()):
        print(f"  {k}: {v}")
 
    print("\nBy room:")
    for k, v in sorted(room_counts.items()):
        print(f"  {k}: {v}")
 
    print(f"\nTotal: {len(records)} recordings\n")
 
def run_extraction(AUDIO_DATA_ROOT, EXCITATION_PATH, PROCESSED_ROOT, skip_if_exists=False):

    if skip_if_exists == True:
        return
    
    project = "/home/3/um07293/research/occul-net"

    AUDIO_DATA_ROOT = "/home/3/um07293/data/audio"
    EXCITATION_PATH = f'{project}/excitation.wav'
    SAVE_ROOT = f'{Path(__file__).parent}/processed'

    os.makedirs(SAVE_ROOT, exist_ok=True)

    records = build_metadata_from_paths(AUDIO_DATA_ROOT)
    print_summary(records)
 
    inv_filter, N, fs_read = load_excitation(EXCITATION_PATH)
    if fs_read != FS:
        print(f"WARNING: excitation fs={fs_read} doesn't match configured FS={FS}")

    for i, record in enumerate(records):
        print(f"[{i + 1}/{len(records)}] {record['filepath']}")
        extract_and_save(
            record, inv_filter, N, SAVE_ROOT,
            START_SAMPLE, END_SAMPLE, FS, NUM_CHANNELS
        )
 
    print("\nDone. All files processed.")