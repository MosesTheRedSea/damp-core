import os
import time
import pathlib
import argparse
import numpy as np
import sounddevice as sd
import soundfile as sf
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import fftconvolve
from datetime import datetime

def record_speaker_data(dataset_directory, room_directory, excitation_path, occlusion_type, occlusion_distance, occluded_type, occluded_distance, base_filename, channels=16, repeat=8, sleep_duration=3):
    
    distance_dir = f"{occlusion_distance}-{occluded_distance}".strip()

    target_directory = (
        f"{dataset_directory}/"
        f"{room_directory}/"
        f"{occlusion_type}/"
        f"{occluded_type}/"
        f"{distance_dir}"
    )

    Path(target_directory).mkdir(parents=True, exist_ok=True)

    excitation, fs = sf.read(excitation_path)

    if excitation.ndim > 1:
        excitation = excitation[:, 0] 

    excitation = excitation / np.max(np.abs(excitation))
    duration = len(excitation) / fs

    for i in range(repeat):

        print(f"\n--- Recording {i + 1}/{repeat} ---")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        recorded_filename = f"{timestamp}_{base_filename}_{i + 1}.wav"

        print(f"Playing excitation ({duration:.2f}s) and recording {channels}-channel response...")
        recorded = sd.playrec(excitation, samplerate=fs, channels=channels)
        
        sd.wait()
        print("Recording complete.")
        
        recorded_path = os.path.join(f"{dataset_directory}/{room_directory}/{occlusion_type}/{occluded_type}/{distance_dir}", recorded_filename)

        sf.write(recorded_path, recorded, fs)
        print(f"Saved {channels}-channel recording to: {recorded_path}")

        print("Computing impulse response for microphone 1 (channel 0)...")
        inv_filter = excitation[::-1]
        rir = fftconvolve(recorded[:, 0], inv_filter)

        start_sample = 21300
        end_sample = 22000
        rir_cropped = rir[start_sample:end_sample]

        # start_sample = 21300
        # end_sample = 22000
        # rir_cropped = rir[start_sample:end_sample]

        plot_ir(start_sample, end_sample, rir_cropped, i)

        if i < repeat - 1:
            print(f"Waiting {sleep_duration} seconds before next recording...")
            time.sleep(sleep_duration)

    return excitation


def plot_ir(start_sample, end_sample, rir_cropped, i):
    plt.figure(figsize=(10, 4))
    plt.plot(np.arange(start_sample, end_sample), rir_cropped)
    plt.title(f"IR Segment (Mic 1) - Samples {start_sample} to {end_sample} - Recording {i + 1}")
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Record multi-channel speaker response data"
    )

    parser.add_argument("--dataset-directory", required=True)
    parser.add_argument("--room-directory", required=True)
    parser.add_argument("--excitation-path", required=True)
    parser.add_argument("--occlusion-type", required=True)
    parser.add_argument("--occlusion-distance", required=True)
    parser.add_argument("--occluded-type", required=True)
    parser.add_argument("--occluded-distance", required=True)
    parser.add_argument("--base-filename", required=True)
    parser.add_argument("--channels", type=int, default=16)
    parser.add_argument("--repeat", type=int, default=8)
    parser.add_argument("--sleep-duration", type=int, default=3)

    args = parser.parse_args()

    record_speaker_data(
        dataset_directory=args.dataset_directory,
        room_directory=args.room_directory,
        excitation_path=args.excitation_path,
        occlusion_type=args.occlusion_type,
        occlusion_distance=args.occlusion_distance,
        occluded_type=args.occluded_type,
        occluded_distance=args.occluded_distance,
        base_filename=args.base_filename,
        channels=args.channels,
        repeat=args.repeat,
        sleep_duration=args.sleep_duration,
    )

"""
python record.py \
    --dataset-directory "/home/moses/Moses/Research/Current/Institute of Science Tokyo/acoustic-robotics/IST-AUDN20/audio" \
    --room-directory room_test \
    --excitation-path "/home/moses/Moses/Research/Current/Institute of Science Tokyo/acoustic-robotics/Multi-Task Acoustic Perception for Occluded Object Detection, Distance Estimation, and Material Classification/excitation.wav" \
    --occlusion-type wood+foam \
    --occlusion-distance 1.6 \
    --occluded-type speaker \
    --occluded-distance 0.5 \
    --base-filename speaker_test \
    --channels 16 \
    --repeat 8 \
    --sleep-duration 3
"""


"""
python record.py \
    --dataset-directory "/home/moses/Moses/Research/Current/Institute of Science Tokyo/IST-AUDN20/audio" \
    --room-directory room_A \
    --excitation-path "/home/moses/Moses/Research/Current/Institute of Science Tokyo/acoustic-robotics/Multi-Task Acoustic Perception for Occluded Object Detection, Distance Estimation, and Material Classification/excitation.wav" \
    --occlusion-type wood \
    --occlusion-distance 0.1 \
    --occluded-type ladder \
    --occluded-distance 0.5 \
    --base-filename ladder \
    --channels 16 \
    --repeat 8 \
    --sleep-duration 3
"""

