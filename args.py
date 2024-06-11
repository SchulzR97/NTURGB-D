import argparse

def get_args_demo():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str)
    parser.add_argument('--split', type=str, default='full')
    parser.add_argument('--in_dir', type=str)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--visualize', type=bool, default='True')

    args = parser.parse_args()
    return args

def get_args_demo_train():
    parser = argparse.ArgumentParser()

    parser.add_argument('--split', type=str, default='CS')
    parser.add_argument('--in_dir', type=str)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_epochs', type=int, default=500000)
    parser.add_argument('--num_classes', type=int, default=60)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--samples_per_epoch', type=int, default=10)
    parser.add_argument('--sequence_length', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--mavg_epochs', type=int, default=50)
    parser.add_argument('--load_run', type=str, default=None)
    parser.add_argument('--mode', type=str, default='train')

    args = parser.parse_args()
    return args