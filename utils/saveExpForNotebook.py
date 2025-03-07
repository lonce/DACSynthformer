import os
import shutil
import librosa
import librosa.display
import matplotlib.pyplot as plt
import soundfile as sf
import numpy as np

def savefordiary(adata, param_jpg, pathname, fname, dfile, sr=44100):
    # Ensure output directory exists
    os.makedirs(pathname, exist_ok=True)
    
    # Define file paths
    spectrogram_path = os.path.join(pathname, f"{fname}.jpg")
    wav_path = os.path.join(pathname, f"{fname}.wav")
    param_fig_path = os.path.join(pathname, f"{fname}_params.jpg")
    
    # Copy the provided parameter JPEG file
    shutil.copy(param_jpg, param_fig_path)
    
    # Compute and save the spectrogram
    plt.figure(figsize=(10, 4))
    D = librosa.amplitude_to_db(librosa.stft(adata), ref=np.max)
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f"{pathname}/{fname}")
    plt.savefig(spectrogram_path, bbox_inches='tight')
    plt.close()
    
    # Save the audio file
    sf.write(wav_path, adata, sr)
    
    # Copy the data file to the same directory
    shutil.copy(dfile, pathname)
    dfile_copy_path = os.path.join(pathname, os.path.basename(dfile))
    
    # Generate the Markdown text for Jupyter
    notebook_text = f"""
{pathname}/{fname}<br>
<img width="600" height="500" src="{param_fig_path}">  <br>
<img width="600" height="500" src="{spectrogram_path}">  <br>
<audio src="{wav_path}" controls>alternative text</audio>  <br>
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
    return notebook_text_path
