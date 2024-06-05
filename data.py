import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
from numpy.random import default_rng
from tqdm import tqdm
import cv2 as cv

CS_SUBJECT_IDS_TRAIN = [1,  2,  4,  5,  8,  9,  13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38]
CS_SUBJECT_IDS_VAL = [3,  6,  7,  10, 11, 12, 20, 21, 22, 23, 24, 26, 29, 30, 32, 33, 36, 37, 39, 40]

CV_CAMERA_IDS_TRAIN = [2, 3]
CV_CAMERA_IDS_VAL = [1]

#region "NTU RGB+D" - 3D Skeletons
def __get_skeleton_files__(in_dir:str, data_cnt:int = None):
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

        files.append(entry)
        progress.update(len(files))
    progress.close()

    return files

def get_datasets_skeletons_full(in_dir:str, split_prop = 0.7, data_cnt:int = None, shuffle = False):
    files = __get_skeleton_files__(in_dir, data_cnt)

    if shuffle:
        rng = default_rng()
        indices = rng.choice(len(files), size=len(files), replace=False)
    else:
        indices = np.arange(len(files), dtype=np.int32)

    train_cnt = int(split_prop * len(files))  
    idx_train = indices[:train_cnt]
    idx_val = indices[train_cnt:]

    files = np.asarray(files)

    ds_train = skeletons_dataset(in_dir, files[idx_train])
    ds_val = skeletons_dataset(in_dir, files[idx_val])

    return ds_train, ds_val

def get_datasets_skeletons_CV(in_dir:str):
    files = __get_skeleton_files__(in_dir, None)

    files_train = []
    files_val = []

    for file in tqdm(files, desc='Creating CV split'):
        camera = int(file[5:8])
        
        if camera in CV_CAMERA_IDS_TRAIN:
            files_train.append(file)
        elif camera in CV_CAMERA_IDS_VAL:
            files_val.append(file)
        else:
            raise Exception(f'Camera ID {camera} not defined for split CV.')

    ds_train = skeletons_dataset(in_dir, files_train)
    ds_val = skeletons_dataset(in_dir, files_val)
    
    return ds_train, ds_val

def get_datasets_skeletons_CS(in_dir:str):
    files = __get_skeleton_files__(in_dir, None)

    files_train = []
    files_val = []

    for file in tqdm(files, desc='Creating CV split'):
        subject = int(file[1:4])
        
        if subject in CS_SUBJECT_IDS_TRAIN:
            files_train.append(file)
        elif subject in CS_SUBJECT_IDS_VAL:
            files_val.append(file)
        else:
            raise Exception(f'Subject ID {subject} not defined for split CV.')

    ds_train = skeletons_dataset(in_dir, files_train)
    ds_val = skeletons_dataset(in_dir, files_val)
    
    return ds_train, ds_val
#endregion

#region "NTU RGB+D" - Masked Depth Maps
def __get_depth_masked_sequences__(in_dir:str, data_cnt:int = None):
    sequences = []
    for dir in tqdm(sorted(os.listdir(in_dir)), desc='Loading files'):
        if dir.startswith('.'):
            continue
        sequences.append(f'{in_dir}/{dir}')
        pass
    return sequences

def get_datasets_depth_masked_full(in_dir:str, split_prop = 0.7, data_cnt:int = None, shuffle = False):
    sequence_dirs = __get_depth_masked_sequences__(in_dir, data_cnt)

    if shuffle:
        rng = default_rng()
        indices = rng.choice(len(sequence_dirs), size=len(sequence_dirs), replace=False)
    else:
        indices = np.arange(len(sequence_dirs), dtype=np.int32)

    train_cnt = int(split_prop * len(sequence_dirs))  
    idx_train = indices[:train_cnt]
    idx_val = indices[train_cnt:]

    sequence_dirs = np.asarray(sequence_dirs)

    ds_train = depth_masked_dataset(in_dir, sequence_dirs[idx_train])
    ds_val = depth_masked_dataset(in_dir, sequence_dirs[idx_val])

    return ds_train, ds_val

def get_datasets_depth_masked_CV(in_dir:str):
    pass

def get_datasets_depth_masked_CS(in_dir:str):
    pass
#endregion

#region Datasets
class base_dataset(Dataset):
    def __init__(self, in_dir:str, files:list = None, num_classes:int = 64, sequence_length:int = 64):
        super().__init__()

        self.__in_dir__ = in_dir
        self.__files__ = files
        self.__num_classes__ = num_classes
        self.__sequence_length__ = sequence_length

    def __len__(self):
        return len(self.__files__)
    
    def __getitem__(self, index):
        raise NotImplementedError()

class skeletons_dataset(base_dataset):
    def __init__(self, in_dir:str, files:list, num_classes:int = 64, sequence_length:int = 64):
        super().__init__(in_dir, files, num_classes, sequence_length)
    
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
                #i = i if i > self.__sequence_length__ else self.__sequence_length__
                #skeleton_seq = skeleton_seq[:i]
                return skeleton_seq
            next_idx, joints = self.__get_skeleton_joints__(lines, next_idx)

            if joints is None:
                # drop frames if no skeleton was detected within last frames
                #i = i if i > self.__sequence_length__ else self.__sequence_length__
                #skeleton_seq = skeleton_seq[:i]
                #return skeleton_seq
                continue
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

class depth_masked_dataset(base_dataset):
    def __init__(self, in_dir, files:list, num_classes:int = 64, sequence_length:int = 64):
        super().__init__(in_dir, files, num_classes, sequence_length)

    def __getitem__(self, index):
        sequence_dir = self.__files__[index]

        sequence_name = sequence_dir.split('/')[-1]
        action = int(sequence_name[-3:])-1

        files = os.listdir(sequence_dir)
        img_sequence = []
        for entry in sorted(files):
            file = str(entry).replace('b\'', '').replace('\'', '')
            img = cv.imread(f'{sequence_dir}/{file}')[:, :, 0] / 19
            #img[img == 0] = 1
            #img = 1 - img
            img_sequence.append(img)

        img_sequence = np.array(img_sequence)

        X = torch.zeros((self.__sequence_length__, img.shape[0], img.shape[1]))

        if img_sequence.shape[0] <= self.__sequence_length__:
            X[0:len(img_sequence)] = torch.tensor(img_sequence, dtype=torch.float32)
        else:
            start_idx = np.random.randint(0, img_sequence.shape[0]-self.__sequence_length__)
            end_idx = start_idx + self.__sequence_length__
            img_sequence = img_sequence[start_idx:end_idx]
            X = torch.tensor(img_sequence, dtype=torch.float32)

        #X = X.permute(0, 3, 1, 2)
        T = torch.zeros((self.__num_classes__))
        T[action] = 1

        return X, T
#endregion