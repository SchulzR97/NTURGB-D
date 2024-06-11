import torch

def accuracy(Y, T):
    Y = torch.argmax(Y, dim=1)
    T = torch.argmax(T, dim=1)

    acc = ((Y == T).sum() / T.numel()).item()

    return acc

def top_k_accuracy(Y, T, k:int):
    Y_top_k = torch.argsort(Y, dim=1, descending=True)[:, 0:k]
    T_indices = torch.argmax(T, dim=1)

    tp = 0
    for y, t in zip(Y_top_k, T_indices):
        if torch.isin(t.cpu(), y.cpu()):
            tp += 1
    top_k_acc = tp / T.shape[0]
    return top_k_acc

def top_5_accuracy(Y, T):
    return top_k_accuracy(Y, T, 5)