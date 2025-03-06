import os
import csv
import argparse
import re

def extract_info_from_filename(filename):
    # Updated regular expression to correctly match the given filenames
    pattern = r"^(.+?)_(\d+)-(\d+)-a(\.\d+)\.dac$"
    match = re.match(pattern, filename)
    
    if not match:
        return None  # Skip files that do not match the pattern
    
    class_name = match.group(1)  # Extract the class name
    note_number = int(match.group(3))  # Extract the note number
    note = round((note_number - 64) / 12, 2)  # Compute note value
    amp = round(float(match.group(4)) / 0.9, 2)  # Normalize amp value and round to two decimal places
    
    return filename, class_name, note, amp

def generate_csv(directory, output_csv):
    with open(output_csv, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Full File Name", "Class Name", "note", "amp"])  # Write headers
        
        for file in os.listdir(directory):
            if file.endswith(".dac"):
                extracted_data = extract_info_from_filename(file)
                if extracted_data:
                    writer.writerow(extracted_data)
    
    print(f"CSV file '{output_csv}' has been generated successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate CSV from DAC file names.")
    parser.add_argument("folder", help="Path to the folder containing .dac files")
    parser.add_argument("output_csv", help="Path to the output CSV file")
    args = parser.parse_args()
    
    generate_csv(args.folder, args.output_csv)
