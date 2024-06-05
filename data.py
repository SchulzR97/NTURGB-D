import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
from numpy.random import default_rng
from tqdm import tqdm

CS_SUBJECT_IDS_TRAIN = [1,  2,  4,  5,  8,  9,  13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38]
CS_SUBJECT_IDS_VAL = [3,  6,  7,  10, 11, 12, 20, 21, 22, 23, 24, 26, 29, 30, 32, 33, 36, 37, 39, 40]

CV_CAMERA_IDS_TRAIN = [2, 3]
CV_CAMERA_IDS_VAL = 1

def get_datasets_3d_skeletons_full(in_dir:str, split_prop = 0.7, data_cnt = None, shuffle = False):
    files = []

    in_files = sorted(os.listdir(in_dir))
    data_cnt = len(in_files) if data_cnt is None else data_cnt
    i = 0
    progress = tqdm(total=data_cnt, desc='Loading file names')
    while len(files) < data_cnt:
        if i >= len(in_files):
            break
        entry = in_files[i]
        i += 1
        if entry.startswith('.'):
            continue

        # extract subject and camera from filename
        # subject, camera = int(entry[1:4]), int(entry[5:8])
        files.append(entry)
        progress.update(len(files))
    progress.close()

    if shuffle:
        rng = default_rng()
        indices = rng.choice(len(files), size=len(files), replace=False)
    else:
        indices = np.arange(len(files), dtype=np.int32)

    train_cnt = int(split_prop * len(files))  
    idx_train = indices[:train_cnt]
    idx_val = indices[train_cnt:]

    files = np.asarray(files)

    ds_train = NTURGBD_Skeletons_DS(in_dir, files[idx_train])
    ds_val = NTURGBD_Skeletons_DS(in_dir, files[idx_val])

    return ds_train, ds_val

def get_datasets_3d_skeletons_CV():
    pass

def get_datasets_3d_skeletons_CS():
    pass

class NTURGBD_Base_DS(Dataset):
    def __init__(self, in_dir:str, files:list = None, num_classes:int = 64, sequence_length:int = 64):
        super().__init__()

        self.__in_dir__ = in_dir
        self.__files__ = files
        self.__num_classes__ = num_classes
        self.__sequence_length__ = sequence_length

    def __len__(self):
        raise NotImplementedError()
    
    def __getitem__(self, index):
        raise NotImplementedError()

class NTURGBD_Skeletons_DS(NTURGBD_Base_DS):
    def __init__(self, in_dir:str, files:list, num_classes:int = 64, sequence_length:int = 64):
        super().__init__(in_dir, files, num_classes, sequence_length)

    def __len__(self):
        return len(self.__files__)
    
    def __getitem__(self, index):
        skeleton_file = self.__files__[index]
        skeleton_seq = self.__load_skeleton_sequence__(f'{self.__in_dir__}/{skeleton_file}')
        action = int(skeleton_file[17:20]) - 1

        if skeleton_seq.shape[0] <= self.__sequence_length__:
            X = torch.zeros((self.__sequence_length__, skeleton_seq.shape[1], skeleton_seq.shape[2]))
            X[0:skeleton_seq.shape[0]] = skeleton_seq
        else:
            start_idx = np.random.randint(0, skeleton_seq.shape[0]-self.__sequence_length__)
            end_idx = start_idx + self.__sequence_length__
            X = skeleton_seq[start_idx:end_idx]

        T = torch.zeros((self.__num_classes__))
        T[action] = 1
        
        return X, T

    def __load_skeleton_sequence__(self, filename:str):
        with open(filename, 'r') as f:
            lines = f.readlines()
        num_frames = int(lines[0].replace('\n', ''))

        if filename in [
            '/Volumes/EXTERN 500G/Studium/Neurorobotik/Forschungspraktikum/NTU RGB+D Skeletons/nturgb+d_skeletons_original/S001C002P008R002A060.skeleton',
            '/Volumes/EXTERN 500G/Studium/Neurorobotik/Forschungspraktikum/NTU RGB+D Skeletons/nturgb+d_skeletons_original/S002C003P011R002A011.skeleton',
            '/Volumes/EXTERN 500G/Studium/Neurorobotik/Forschungspraktikum/NTU RGB+D Skeletons/nturgb+d_skeletons_original/S001C002P004R002A055.skeleton']:
            pass
        
        next_idx = 1
        # shape: num_frames, num_subjects, num_joints, coordinates_xyz
        skeleton_seq = torch.zeros((num_frames, 75, 3))
        for i in range(num_frames):
            if next_idx >= len(lines):
                i = i if i > self.__sequence_length__ else self.__sequence_length__
                skeleton_seq = skeleton_seq[:i]
                return skeleton_seq
            next_idx, joints = self.__get_skeleton_joints__(lines, next_idx)

            if joints is None:
                # drop frames if no skeleton was detected within last frames
                i = i if i > self.__sequence_length__ else self.__sequence_length__
                skeleton_seq = skeleton_seq[:i]
                break
            else:
                skeleton_seq[i] = joints
            pass
        return skeleton_seq

    def __get_skeleton_joints__(self, lines, start_idx:int):
        i = start_idx
        joints = np.zeros((75, 3))
        while i < len(lines) and lines[i] == '0\n':
            i += 1
        if i >= len(lines):
            return i, None
        
        num_subjects = int(lines[i].replace('\n', ''))
        if num_subjects > 1:
            pass
        i += 2

        for s in range(num_subjects):
            if s >= 1:
                i += 1
            num_joints = int(lines[i].replace('\n', ''))
            i += 1
            if i >= len(lines) - 1:
                return i, torch.tensor(joints)
            subject_joints = np.zeros((25, 3))
            for j in range(num_joints):
                subject_joints[j] = np.array(lines[i].split(' ')[0:3], dtype=np.float32)

                i += 1
            joints[s*25:s*25+25] = subject_joints
        return i, torch.tensor(joints)

class NTURGBD_DepthMasked_DS(NTURGBD_Base_DS):
    def __init__(self, in_dir, files:list, num_classes:int = 64, sequence_length:int = 64):
        super().__init__(in_dir, files, num_classes, sequence_length)

        for file in sorted(files):
            pass