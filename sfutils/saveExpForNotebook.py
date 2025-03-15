import os
import shutil
import librosa
import librosa.display
import matplotlib.pyplot as plt
import soundfile as sf
import numpy as np

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def plot_tensorboard_loss(log_dir, output_path):
    """
    Extracts loss values from a TensorBoard log directory and saves a log-scale loss plot.

    Args:
        log_dir (str): Path to the TensorBoard log directory.
        output_path (str): Full path to save the output JPG file.
    """
    # Find event files in the log directory
    event_files = [os.path.join(log_dir, f) for f in os.listdir(log_dir) if "events" in f]

    if not event_files:
        print("No event files found in the directory.")
        return

    # Load TensorBoard logs
    event_acc = EventAccumulator(event_files[0])
    event_acc.Reload()

    # Get available scalar tags (loss values)
    tags = event_acc.Tags().get("scalars", [])

    # Filter tags related to loss
    loss_tags = [tag for tag in tags if "loss" in tag.lower()]

    if not loss_tags:
        print("No loss metrics found in TensorBoard logs.")
        return

    # Create figure without displaying it
    plt.ioff()  # Turn off interactive mode to prevent display
    fig, ax = plt.subplots(figsize=(8, 5))

    for tag in loss_tags:
        loss_events = event_acc.Scalars(tag)  # Extract data
        steps = [e.step for e in loss_events]
        values = [e.value for e in loss_events]

        ax.plot(steps, values, label=tag, linewidth=2)

    # Apply logarithmic scaling to the y-axis
    ax.set_yscale("log")
    ax.set_xlabel("Steps")
    ax.set_ylabel("Loss (log scale)")
    ax.set_title("Loss Over Time (Log Scale)")
    ax.legend()
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)

    # Save the plot as a JPG without displaying it
    fig.savefig(output_path)
    plt.close(fig)  # Close the figure to free memory

    print(f"Saved loss plot as {output_path}")

# Example usage:
# plot_tensorboard_loss("path/to/tensorboard/logs", "output/loss_plot_log.jpg")

# ===========================================================================================================================
# adata  - audio data array used to save as wave file and to generate spectrogram, both of which go to pathname
# param_jpg - the file name of the jpeg that plots the conditioning parameters used to generate the audio data 
# logdir - path to directory where tensorboard loss data is stored
# pathname, - pathname to where all audio and plots will be stored
# fname,    - a ¨base" file name (usually incoporating parameters) used to store all files before their extensions are concatenated
# dfile - name of the config file to copy to the pathname folder and provide as a link in the html output
# sr=44100
# usage: md_text, displaypath = savefordiary(adata, paramplotfname, diarydir, experiment_name, paramfile) 
def savefordiary(adata, param_jpg, log_dir, pathname, fname, dfile, sr=44100):
    # Ensure output directory exists
    os.makedirs(pathname, exist_ok=True)
    
    # Define file paths
    spectrogram_path = os.path.join(pathname, f"{fname}.jpg")
    wav_path = os.path.join(pathname, f"{fname}.wav")
    param_fig_path = os.path.join(pathname, f"{fname}_params.jpg")
    lossplot_path = os.path.join(pathname, f"{fname}_loss.jpg")
    
    # Copy the provided parameter JPEG file
    shutil.copy(param_jpg, param_fig_path)
    
    # Compute and save the spectrogram
    plt.figure(figsize=(10, 4))
    
    if 0 : # plot linear spectrogram
        D = librosa.amplitude_to_db(librosa.stft(adata), ref=np.max)
        librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
    else: # plot mel spectrogram
        mel_spec = librosa.feature.melspectrogram(y=adata, sr=sr, n_mels=128)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        librosa.display.specshow(mel_spec_db, sr=sr, x_axis='time', y_axis='mel')

    plt.colorbar(format='%+2.0f dB')
    plt.title(f"{pathname}/{fname}")
    plt.savefig(spectrogram_path, bbox_inches='tight')
    plt.close()
    
    # Save the audio file
    sf.write(wav_path, adata, sr)

    # compute loss plot and save
    plot_tensorboard_loss(log_dir, lossplot_path)
    
    # Copy the data file to the same directory
    shutil.copy(dfile, pathname)
    dfile_copy_path = os.path.join(pathname, os.path.basename(dfile))
    
    # Generate the Markdown text for Jupyter
    notebook_text = f"""
{pathname}/{fname}<br>
<img width="600" height="500" src="{param_fig_path}">  <br>
<img width="600" height="500" src="{spectrogram_path}">  <br>
<audio src="{wav_path}" controls>alternative text</audio>  <br>
<img width="600" height="500" src="{lossplot_path}">  <br>
<a href="{dfile_copy_path}">paramfile link</a>
"""
    
    # Save the notebook cell text as a file
    notebook_text_path = os.path.join(pathname, f"{fname}_notebook_cell.txt")
    with open(notebook_text_path, "w") as f:
        f.write(notebook_text)
    
#    print(f"Processed files saved in: {pathname}")
#    print(f"- Spectrogram: {spectrogram_path}")
#    print(f"- Audio: {wav_path}")
#    print(f"- Parameter Figure: {param_fig_path}")
#    print(f"- Copied data file: {dfile_copy_path}")
#    print(f"- Notebook cell text saved in: {notebook_text_path}:")
    print(f"{notebook_text}")
    return notebook_text, notebook_text_path
