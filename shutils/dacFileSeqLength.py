import argparse
import dac
import torch

def get_sequence_length(filename):
    """Load a DAC file and return the sequence length (T)."""
    try:
        dacfile = dac.DACFile.load(filename)  # Load the DAC file
        data = dacfile.codes  # Extract the data
        data = data.squeeze(0)  # Remove the first dimension if it's 1
        T = data.shape[-1]  # Get the sequence length (last dimension)
        return T
    except Exception as e:
        print(f"Error loading file {filename}: {e}")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get sequence length from a DAC file.")
    parser.add_argument("filename", type=str, help="Path to the DAC file.")
    args = parser.parse_args()
    
    T = get_sequence_length(args.filename)
    if T is not None:
        print(f"Sequence length (T) in {args.filename}: {T}")
