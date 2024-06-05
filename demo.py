from args import get_args
import data
from torch.utils.data import DataLoader
from datetime import datetime

if __name__ == '__main__':
    args = get_args()

    if args.dataset == '3d_skeletons':
        if args.split == 'full':
            ds_train, ds_val = data.get_datasets_3d_skeletons_full(args.in_dir, data_cnt = None)
        elif args.split == 'CV':
            ds_train, ds_val = data.get_datasets_3d_skeletons_CV(args.in_dir)
        elif args.split == 'CS':
            ds_train, ds_val = data.get_datasets_3d_skeletons_CS(args.in_dir)
        else:
            raise Exception(f'split {args.split} is not supported. Please privide one of the following values: [full, CV, CS]. See the documentation for more information.')
    
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
        print(f'sample {i} time per sample: {(datetime.now() - start_time) / (i + 1)}')
