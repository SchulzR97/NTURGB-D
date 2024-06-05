from args import get_args_demo
import data
from torch.utils.data import DataLoader
from datetime import datetime
import visualization as vis
import cv2 as cv

DATASET_SKELETONS = 'skeletons'
DATASET_DEPTH_MASKED = 'depth_masked'

if __name__ == '__main__':
    args = get_args_demo()

    if args.dataset == DATASET_SKELETONS:
        if args.split == 'full':
            ds_train, ds_val = data.get_datasets_skeletons_full(args.in_dir, data_cnt = None)
        elif args.split == 'CV':
            ds_train, ds_val = data.get_datasets_skeletons_CV(args.in_dir)
        elif args.split == 'CS':
            ds_train, ds_val = data.get_datasets_skeletons_CS(args.in_dir)
        else:
            raise Exception(f'Split {args.split} is not supported. Please privide one of the following values: [full, CV, CS]. See the documentation for more information.')
    elif args.dataset == DATASET_DEPTH_MASKED:
        if args.split == 'full':
            ds_train, ds_val = data.get_datasets_depth_masked_full(args.in_dir, data_cnt = None)
        elif args.split == 'CV':
            ds_train, ds_val = data.get_datasets_depth_masked_CV(args.in_dir)
        elif args.split == 'CS':
            ds_train, ds_val = data.get_datasets_depth_masked_CS(args.in_dir)
        else:
            raise Exception(f'Split {args.split} is not supported. Please privide one of the following values: [full, CV, CS]. See the documentation for more information.')
    
    test = len(ds_train)
    dl_train = DataLoader(dataset = ds_train,
                          batch_size = args.batch_size,
                          shuffle = True,
                          num_workers = args.num_workers)
    dl_val = DataLoader(dataset = ds_val,
                        batch_size = args.batch_size,
                        shuffle = True,
                        num_workers = args.num_workers)
    
    start_time = datetime.now()
    for i, (X, T) in enumerate(dl_train):
        if args.visualize:
            if args.dataset == DATASET_SKELETONS:
                vis.show_skeleton_sequences(X, T)
            elif args.dataset == DATASET_DEPTH_MASKED:
                vis.show_image_sequences(X, T, colormap=cv.COLORMAP_HOT)
            print(f'sample {i}')
        else:
            print(f'sample {i} time per sample: {(datetime.now() - start_time) / (i + 1)}')
