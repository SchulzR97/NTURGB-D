import argparse

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str)
    parser.add_argument('--split', type=str, default='full')
    parser.add_argument('--in_dir', type=str)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--visualize', type=bool, default='True')

    args = parser.parse_args()
    return args