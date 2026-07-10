import sys
from pathlib import Path

def count_wav_files(root_dir):

    root = Path(root_dir)
    
    if not root.is_dir():
        print(f"Error: '{root_dir}' is not a valid directory.")
        sys.exit(1)

    wav_files = list(root.rglob("*.wav")) + list(root.rglob("*.WAV"))
    wav_files = list({f.resolve() for f in wav_files})

    print(f"Total .wav files found: {len(wav_files)}")
    return len(wav_files)

if __name__ == "__main__":

    directory = "/home/moses/Moses/Research/Current/Institute of Science Tokyo/IST-AUDN25/audio/room_A"

    count_wav_files(directory)
