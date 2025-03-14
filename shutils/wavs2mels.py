import os
import argparse
import numpy as np
import soundfile as sf
from joblib import Parallel, delayed

import torch
import bigvgan
import librosa
from meldataset import get_mel_spectrogram

sr = 44100
#device = 'cpu'

model = bigvgan.BigVGAN.from_pretrained('BigVGAN/bigvgan_v2_44khz_128band_512x', use_cuda_kernel=False)
#model = bigvgan.BigVGAN.from_pretrained('nvidia/bigvgan_v2_44khz_128band_512x', use_cuda_kernel=False)

model = model.eval()
# Remove weight norm in the model and set to eval mode
model.remove_weight_norm()



# Number of parallel jobs (adjust based on CPU cores)
NUM_WORKERS = max(1, os.cpu_count() - 1)

def wav2mel(wav_data):
    """
    Converts a WAV signal to a mel spectrogram using BigVGAN.
    """
    wav = torch.FloatTensor(wav_data).unsqueeze(0)  # Shape [1, T_time]
    #mel = get_mel_spectrogram(wav, model.h).to(device)  # Shape [1, C_mel, T_frame]
    mel = get_mel_spectrogram(wav, model.h)  # Shape [1, C_mel, T_frame]
    return mel

def process_single_wav(wav_path, destination_dir):
    """
    Processes a single .wav file, converts it using wav2mel(),
    and saves the output as a .npy file in the destination directory.
    """
    try:
        # Read WAV file
        wav_data, sample_rate = librosa.load(wav_path, sr=model.h.sampling_rate, mono=True)

        # Extract filename without extension
        filename = os.path.splitext(os.path.basename(wav_path))[0]
        mel_path = os.path.join(destination_dir, filename + ".npy")

        # Check if WAV file meets conditions
        if len(wav_data.shape) > 1:
            raise ValueError("File is not single-channel (mono).")

        if sample_rate != sr:
            raise ValueError(f"Unexpected sample rate {sample_rate}, expected 44100 Hz.")

        # Process with wav2mel()
        processed_data = wav2mel(wav_data)
        print(f'wav2mel {wav_path}')

        # Save as .npy file
        np.save(mel_path, processed_data)

        return f"Processed: {filename}.wav → {filename}.npy"

    except Exception as e:
        return f"Error processing {wav_path}: {e}"

def process_wav_files(source_dir, destination_dir):
    """
    Reads all .wav files in source_dir, processes them in parallel with wav2mel(),
    and saves the output to destination_dir in .npy format.
    """

    global model

    # Ensure destination directory exists
    os.makedirs(destination_dir, exist_ok=True)

    # List all .wav files in the source directory
    wav_files = [os.path.join(source_dir, f) for f in os.listdir(source_dir) if f.endswith(".wav")]

    if not wav_files:
        print("No .wav files found in the source directory.")
        return


    # Run parallel processing
    results = Parallel(n_jobs=NUM_WORKERS)(
        delayed(process_single_wav)(wav_path, destination_dir) for wav_path in wav_files
    )

    # Print results (success or errors)
    for result in results:
        print(result)

# Command-line argument parsing
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert .wav files to .npy using wav2mel() with parallel processing.")
    parser.add_argument("source_dir", type=str, help="Directory containing .wav files")
    parser.add_argument("destination_dir", type=str, help="Directory to save .npy files")
    #parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Device to use ('cpu' or 'cuda')")

    args = parser.parse_args()
    process_wav_files(args.source_dir, args.destination_dir)
