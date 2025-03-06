import os
import pandas as pd
from glob import glob
import argparse

def parse_filename(filename):
    """
    Given a filename like:
      TokWotalDuet--wmratio-01.00--c-02--x-98.dac
    This function extracts:
      - Class Name: 'TokWotalDuet'
      - Parameter values (ignoring their names), in order:
           Param1: 1.00, Param2: 2.0, Param3: 98.0
    Returns a dictionary containing the class name and each parameter value under keys 'Param1', 'Param2', etc.
    """
    # Remove directory path and file extension
    base = os.path.basename(filename)
    if base.endswith('.dac'):
        base = base[:-4]
    
    # Split the filename using '--' as the delimiter
    parts = base.split('--')
    if len(parts) < 2:
        return None  # File does not match the expected format
    
    info = {}
    # The first part is the class name
    info['Class Name'] = parts[0]
    
    # Process each subsequent token to extract the value (ignoring the parameter name)
    param_values = []
    for token in parts[1:]:
        tokens = token.split('-')
        if len(tokens) >= 2:
            # Join the remaining tokens to get the full value string
            value_str = '-'.join(tokens[1:])
            try:
                # Convert to float (e.g., "01.00" -> 1.0)
                value = float(value_str)
            except ValueError:
                value = value_str
            param_values.append(value)
        else:
            print(f"Skipping token (unexpected format): {token} in file {filename}")
    
    # Store each value in keys as 'Param1', 'Param2', etc.
    for i, value in enumerate(param_values):
        info[f"Param{i+1}"] = value
    
    return info

def create_excel_from_files(directory, output_excel):
    """
    Searches for .dac files in the specified directory, parses each file,
    and writes an Excel file with columns for:
      - Full File Name
      - Class Name
      - Param1, Param2, ..., ParamN (the parameter values in order)
    """
    data = []
    file_list = glob(os.path.join(directory, '*.dac'))
    
    for file in file_list:
        full_name = os.path.basename(file)
        file_info = parse_filename(full_name)
        if file_info:
            # Include the full file name in the dictionary
            file_info['Full File Name'] = full_name
            data.append(file_info)
        else:
            print(f"Skipping file (unexpected format): {full_name}")
    
    # Create a DataFrame from the list of dictionaries.
    df = pd.DataFrame(data)
    
    # Optional: Rearrange columns so 'Full File Name' and 'Class Name' come first.
    cols = list(df.columns)
    if 'Full File Name' in cols:
        cols.insert(0, cols.pop(cols.index('Full File Name')))
    if 'Class Name' in cols:
        cols.insert(1, cols.pop(cols.index('Class Name')))
    df = df[cols]
    
    # Write the DataFrame to an Excel file
    df.to_excel(output_excel, index=False)
    print(f"Excel file '{output_excel}' has been created with {len(df)} records.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process .dac files to create an Excel summary with parameters labeled as 'Param1', 'Param2', etc."
    )
    parser.add_argument('directory', type=str, help="Directory where .dac files are located")
    parser.add_argument('output_excel', type=str, help="Output Excel file path and filename")
    args = parser.parse_args()

    create_excel_from_files(args.directory, args.output_excel)
