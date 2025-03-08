import pandas as pd
import argparse

def transform_excel(input_file, output_file, column_name):
    # Load the Excel file
    xls = pd.ExcelFile(input_file)
    
    # Dictionary to store transformed dataframes
    transformed_dfs = {}
    
    for sheet_name in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name)
        
        if column_name not in df.columns:
            print(f"Skipping sheet '{sheet_name}' (no '{column_name}' column found).")
            transformed_dfs[sheet_name] = df
            continue
        
        # Get unique values in sorted order
        unique_values = sorted(df[column_name].unique())
        
        # Create new columns for each unique value with 0s
        for i, value in enumerate(unique_values, start=1):
            col_name = f"{column_name} {i}"
            df[col_name] = (df[column_name] == value).astype(int)
        
        # Drop the original column
        df.drop(columns=[column_name], inplace=True)
        
        # Store transformed dataframe
        transformed_dfs[sheet_name] = df
    
    # Save transformed dataframes back to a new Excel file
    with pd.ExcelWriter(output_file) as writer:
        for sheet, transformed_df in transformed_dfs.items():
            transformed_df.to_excel(writer, sheet_name=sheet, index=False)
    
    print(f"Transformed Excel file saved as: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transform an Excel file by encoding a specified column into multiple binary columns.")
    parser.add_argument("input_file", help="Path to the input Excel file")
    parser.add_argument("output_file", help="Path to the output transformed Excel file")
    parser.add_argument("column_name", help="Name of the column to expand into multiple binary columns")
    args = parser.parse_args()
    
    transform_excel(args.input_file, args.output_file, args.column_name)
