import torch
from torch.utils.data import Dataset, DataLoader
import os
import dac
import numpy as np  # for mel files
import torch.nn.functional as F  # for the integer to one-hot
import pandas as pd


class CustomDACDataset(Dataset):
    def __init__(self, data_dir, metadata_excel, sub_seq_len, ftype='dac', transforms=None):
        """
        Args:
            data_dir (string): Directory with all the data files.
            metadata_excel (string): Path to the Excel file containing file metadata.
            sub_seq_len (int): Length of each subsequence to extract from full sequences.
            transforms (callable, optional): Optional transforms to be applied on a sample.
        """
        self.data_dir = data_dir
        self.metadata_df = pd.read_excel(metadata_excel)
        self.file_names = self.metadata_df["Full File Name"].tolist()
        self.metadata_dict = self.metadata_df.set_index("Full File Name").to_dict(orient="index")
        self.transforms = transforms
        self.sub_seq_len = sub_seq_len
        
        unique_classes = self.metadata_df["Class Name"].unique().tolist()
        unique_classes.sort()
        self.class_name_to_int = {cls: i for i, cls in enumerate(unique_classes)}
        self.int2classname = {i: cls for cls, i in self.class_name_to_int.items()}
        
        class_name_index = self.metadata_df.columns.get_loc("Class Name")
        self.param_columns = self.metadata_df.columns[class_name_index + 1:].tolist()
        self.ftype = ftype
        
        # Compute total available subsequences
        self.subsequences = []  # (file_idx, start_idx) pairs

        # just for reporting the count
        total_files = len(self.file_names)
        total_subsequences = 0  # Counter for total subsequences

        for file_idx, filename in enumerate(self.file_names):
            file_path = os.path.join(self.data_dir, filename)
            full_seq_len = self.get_sequence_length(file_path)
            num_subsequences = (full_seq_len - 1) // self.sub_seq_len  # Adjusted to require one extra token
            total_subsequences += num_subsequences  # Count subsequences
            for i in range(num_subsequences):
                self.subsequences.append((file_idx, i * self.sub_seq_len))

        # Print dataset statistics
        print(f"Total files in dataset: {total_files}")
        print(f"Total subsequences available: {total_subsequences}")

    def get_num_classes(self):
        return len(self.class_name_to_int)
    
    def get_num_params(self):
        return len(self.param_columns)

    def get_class_names(self):
        return list(self.class_name_to_int.keys())
    
    def get_param_names(self):
        return self.param_columns
    
    def get_sequence_length(self, filename):
        try:
            if self.ftype == 'dac':
                dacfile = dac.DACFile.load(filename)
                data = dacfile.codes
            elif self.ftype == 'mel':
                datain = np.load(filename)
                data = torch.tensor(datain)
            data = data.squeeze(0)
            return data.shape[-1]
        except Exception as e:
            raise ValueError(f"Error loading file {filename}: {e}")

    def onehot(self, class_name):
        class_num = self.class_name_to_int.get(class_name, -1)
        if class_num == -1:
            print(f'class_name not found: {class_name}')
        return F.one_hot(torch.tensor(class_num), num_classes=self.get_num_classes()).to(torch.float)

    def extract_conditioning_vector(self, filename):
        metadata = self.metadata_dict.get(filename, None)
        if metadata is None:
            raise ValueError(f"Metadata for file {filename} not found in the Excel file")
        class_name = metadata["Class Name"]
        param_values = [metadata[param] for param in self.param_columns]
        one_hot_fvector = self.onehot(class_name)
        return torch.cat((one_hot_fvector, torch.tensor(param_values, dtype=torch.float)))

    def __len__(self):
        return len(self.subsequences)
    
    def __getitem__(self, idx):
        file_idx, start_idx = self.subsequences[idx]
        filename = self.file_names[file_idx]
        fpath = os.path.join(self.data_dir, filename)
        
        if self.ftype == 'dac':
            dacfile = dac.DACFile.load(fpath)
            data = dacfile.codes
        elif self.ftype == 'mel':
            datain = np.load(fpath)
            data = torch.tensor(datain)
        
        data = data.squeeze(0)
        input_data = data[:, start_idx : start_idx + self.sub_seq_len]
        target_data = data[:, start_idx + 1 : start_idx + self.sub_seq_len + 1]  # Keep shift
        
        condvect = self.extract_conditioning_vector(filename)
        
        id = input_data.transpose(0, 1)
        td = target_data.transpose(0, 1)

        
        return id, td, condvect
