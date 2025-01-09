import torch.utils.data as data
import os
import os.path as osp
import numpy as np
from glob import glob
import csv


class DATAReader(data.Dataset):
    def __init__(self, split=None, labels=None):
        # self.data_root = data_root
        self.split = split
        # self.partial_root = '/media/mufarooq/SSD_SMILES/Umar/UMFlint/Research/Audio_Dataset/database' #/segment_labels/train_seglab_0.16.npy
        self.root = '/media/mufarooq/SSD_SMILES/Umar/UMFlint/Research/Audio_Dataset'  # /media/mufarooq/SSD_SMILES/Umar/UMFlint/Research/AA_Audio, /data/Umar/A_Datasets
        if self.split in 'TRAIN':
            # load corresponding labels
            # train_partial_labels = os.path.join(self.partial_root,'segment_labels/train_seglab_0.16.npy')
            # partial_labels = get_partial_labels(train_partial_labels)
            train_labels_file = os.path.join(self.root,'ASV_2019/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt')
            labels = get_labels(train_labels_file)
            self.data_root = os.path.join(self.root,'ASV_2019/ASVspoof2019_LA_train/flac')
        elif self.split in 'TEST':
            test_labels_file = os.path.join(self.root,'ASV_2019/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt')
            labels = get_labels(test_labels_file)
            self.data_root = os.path.join(self.root,'ASV_2019/ASVspoof2019_LA_eval/flac')
                    # self.split = split
        elif self.split in 'DEV':
            dev_labels_file = os.path.join(self.root,'ASV_2019/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt')
            labels = get_labels(dev_labels_file)
            self.data_root = os.path.join(self.root,'ASV_2019/ASVspoof2019_LA_dev/flac')
        elif self.split in 'In_The_Wild':
            dev_labels_file = '/data/Umar/A_Datasets/release_in_the_wild/meta.csv'
            labels = get_in_the_wild_labels(dev_labels_file)
            self.data_root = '/data/Umar/A_Datasets/release_in_the_wild'

        else:
            print("Invalid selection in data loader!")
            return

        # self.partial_labels = partial_labels  # Provided labels for real/fake classification
        self.labels = labels  # Provided labels for real/fake classification

        # List real and fake files
        # self.real_files = self.list_files(label=0)  # 0 for real
        self.fake_files = self.list_files(label=1)  # 1 for fake

        # print(f"{split}, Real files: {len(self.real_files)}, Fake files: {len(self.fake_files)}")

        # self.n = min(len(self.real_files), len(self.fake_files))  # Balance dataset
        self.n = len(self.fake_files)
        
        print(f"Balanced dataset size: {self.n}")


    def __len__(self):
        return self.n

    def __getitem__(self, index):
        # Get a sample from real and fake categories
        # real_audio = self.real_files[index]
        fake_audio = self.fake_files[index]

        # Preprocess the audio files
        # real_data = load_preprocess_AASIST(real_audio)
        fake_data = load_preprocess_AASIST(fake_audio)

        # Get file ID (for label lookup)
        real_file_id = os.path.splitext(os.path.basename(fake_audio))[0]
        # fake_file_id = os.path.splitext(os.path.basename(fake_audio))[0]

        # Get the corresponding labels (real=0, fake=1)
        real_label = self.labels.get(real_file_id, (real_file_id, "unknown", 0))  # Default to real if not found
        # fake_label = self.labels.get(fake_file_id, 1)  # Default to fake if not found

        return fake_audio, fake_data, real_label

    def list_files(self, label):
        file_list = []
        dataset_path = self.data_root  # Assuming split is 'TRAIN' or 'TEST'

        for file_name in os.listdir(dataset_path):
            if file_name.endswith('.flac') or file_name.endswith('.wav'):  # Audio files of interest
                file_path = os.path.join(dataset_path, file_name)

                # Check label (real=0, fake=1)
                file_id = os.path.splitext(file_name)[0]
                label_data = self.labels.get(file_id)
                if label_data is not None:
                    file_id, speaker_id, file_label = label_data  # Get (speaker_id, label) tuple
                    if file_label == label: 
                        file_list.append(file_path)
                # file_list.append(file_path)


        return file_list

def get_partial_labels(labels_file):
    labels = np.load(labels_file, allow_pickle=True)  # Use allow_pickle=True if the file contains objects
    return labels



def get_labels(labels_file):
    labels = {}
    with open(labels_file, 'r') as file:
        for line in file:
            parts = line.strip().split(' ')
            if len(parts) >= 5:  # Assuming the line structure is consistent
                speaker_id = parts[0].strip()  # Extract the speaker_id
                file_id = parts[1].strip()  # Extract the file_id
                label = parts[-1].strip()   # Extract the label (e.g., "spoof" or "bonafide")
                labels[file_id] = (file_id, speaker_id, 1 if label == 'spoof' else 0)  # Store as (speaker_id, label)
    return labels


def get_in_the_wild_labels(labels_file):
    labels = {}
    with open(labels_file, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file, delimiter=',')  # Assuming tab-separated CSV, adjust delimiter if needed
        # print("CSV Headers:", reader.fieldnames)  # This will show you the headers of the CSV

        for row in reader:
            file_id = os.path.splitext(row['file'].strip())[0]  # Strip extension from 'file' (e.g., '0.wav' -> '0')
            label = row['label'].strip()   # Extract the label (e.g., 'spoof' or 'bona-fide')
            
            # Map the label to 1 (spoof) or 0 (bona-fide)
            labels[file_id] = 1 if label == 'spoof' else 0
    
    return labels


def load_preprocess_AASIST(path, cut=96000):   # 96000, 64600
    from torch import Tensor
    import librosa


    # X, _ = sf.read(path)
    X, _ = librosa.load(path, sr=16000, mono=True)  # Record_1.mp3    Derek_orig_1.wav
    X_pad = pad(X,'zero', cut)
    x_inp = Tensor(X_pad)
    if len(x_inp.shape) != 1:
        if x_inp.shape[-1] != 1:
            x_inp = x_inp.mean(dim=-1, keepdim=False)
    return x_inp


def pad(x, padding_type, max_len=64600):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]

    # Apply zero padding
    if padding_type == 'zero':
        padded_x = np.zeros(max_len, dtype=x.dtype)
        padded_x[:x_len] = x    
    # Apply repeat padding
    else:     
        num_repeats = int(max_len / x_len) + 1
        padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]

    return padded_x


if __name__ == '__main__':
    # dev_labels_file = '/data/Umar/A_Datasets/release_in_the_wild/meta.csv'
    # labels = get_in_the_wild_labels(dev_labels_file)
    # print(labels)

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', '-sp', type=str, default='TRAIN', help='Split of the dataset (e.g., TRAIN, TEST)')
    parser.add_argument("--gpu_devices", type=int, nargs='+', default=[0, 1], help='GPU devices to use')
    
    args = parser.parse_args()


    # Initialize the dataset and dataloader
    train_dataset = DATAReader(split=args.split, labels=None)
    train_loader = data.DataLoader(train_dataset, batch_size=24, shuffle=True)

    print('Total train files: ', len(train_dataset))

    # Example: Iterate through the dataset
    for batch_idx, (real_data, real_label) in enumerate(train_loader):
        print(f"Batch {batch_idx+1}")
        print(f"Real data shape: {real_data.shape}, Real label: {real_label}")
        break  # Only for demonstration purposes, break after the first batch
