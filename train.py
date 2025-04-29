import sys
# 修改路径以符合新的工作环境
sys.path.append('/home/MBNet')

import numpy as np
import torch
import torch.nn as nn
import os
import argparse
import pandas as pd
from sklearn.metrics import roc_curve, auc
from model import MBNet
from trainer import training_classing, evaluate_test_scores
# from dgllife.utils import EarlyStopping
from dataset import TestbedDataset
from torch_geometric.loader import DataLoader
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import random
from metric import accuracy, precision, recall, f1_score, bacc_score, roc_auc, mcc_score, kappa, ap_score
import xgboost as xgb

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# 设置随机数种子
def save_AUCs(AUCs, filename):
    with open(filename, 'a') as f:
        f.write('\t'.join(map(str, AUCs)) + '\n')


def metric_df(tp, y_pred, y_test):
    acc = accuracy(y_pred, y_test)
    prec = precision(y_pred, y_test)
    rec = recall(y_pred, y_test)
    f1 = f1_score(y_pred, y_test)
    bacc = bacc_score(y_pred, y_test)
    auc_roc = roc_auc(y_pred, y_test)
    mcc = mcc_score(y_pred, y_test)
    kap = kappa(y_pred, y_test)
    ap = ap_score(y_pred, y_test)
    return [tp, acc, prec, rec, f1, bacc, auc_roc, mcc, kap, ap]

class CustomDataset(InMemoryDataset):
    def __init__(self, root, data, slices):
        super(CustomDataset, self).__init__(root)
        self.data = data
        self.slices = slices

def load_data_from_file(file_path):
    try:
        # 加载数据文件
        data, slices = torch.load(file_path)
    except Exception as e:
        print(f"Error loading data from {file_path}: {e}")
        raise e

    # 使用自定义的数据集类来创建 InMemoryDataset 对象
    dataset = CustomDataset(root=file_path, data=data, slices=slices)

    return dataset

def train_model(modeling, train_batch, test_batch, criterion, lr, epoch_num, cuda_name, i):
    model_st = modeling.__name__
    print(model_st)

    # 定义数据路径
    root_path = '/home/MBNet/data/processed'
    train_data_path = os.path.join(root_path, f'train_set_{i}.pt')
    val_data_path = os.path.join(root_path, f'val_set_{i}.pt')
    test_data_path = os.path.join(root_path, f'test_set_{i}.pt')

    # 加载数据
    train_data = load_data_from_file(train_data_path)
    val_data = load_data_from_file(val_data_path)
    test_data = load_data_from_file(test_data_path)

    train_loader = DataLoader(train_data, batch_size=train_batch, shuffle=False)
    val_loader = DataLoader(val_data, batch_size=train_batch, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=test_batch, shuffle=False)
    
    device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
    print(device)
    model = modeling()
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 创建目录确保存在
    results_directory = '/home/MBNet/results/'
    create_directory_if_not_exists(results_directory)

    file_AUCs = os.path.join(results_directory, f'MultiViewNet_matrix{i}.txt')
    AUCs = 'Epoch\tACC\tPrec\tRec\tF1\tBACC\troc_auc\tmcc\tkappa\tap'
    with open(file_AUCs, 'w') as f:
        f.write(AUCs + '\n')

    best_auc = 0
    not_improved_count = 0
    for epoch in range(epoch_num):
        train_loss = training_classing(model, train_loader, optimizer, criterion, device)
        val_loss, y_true, y_pred = evaluate_test_scores(model, val_loader, criterion, device)
        print('Train epoch: {} \ttrain_loss: {:.6f}'.format(epoch, train_loss))
        print('Train epoch: {} \tval_loss: {:.6f}'.format(epoch, val_loss))
        AUC = roc_auc(y_pred, y_true)
        if best_auc < AUC:
            best_auc = AUC
            not_improved_count = 0
            type = epoch
            score = metric_df(type, y_pred, y_true)
            save_AUCs(score, file_AUCs)
            torch.save(model.state_dict(), '/home/MBNet/results/model_{num}.model'.format(num=i))
        else:
            not_improved_count += 1
        print('best_auc', best_auc)
        if not_improved_count > 50:
            break

    best_model = modeling()
    best_model = best_model.to(device)
    best_model.load_state_dict(torch.load('/home/MBNet/results/model_{num}.model'.format(num=i)))
    best_model.eval()
    test_loss, y_true, y_pred = evaluate_test_scores(best_model, test_loader, criterion, device)
    type = 'test'
    score = metric_df(type, y_pred, y_true)
    save_AUCs(score, file_AUCs)
    print('Train epoch: {} \ttest_loss: {:.6f}'.format(epoch, test_loss))

    # 在开始写入文件之前，先创建结果目录
    results_directory = '/home/MBNet/results/'
    create_directory_if_not_exists(results_directory)

    if __name__ == "__main__":
        setup_seed(42)

    # 判断是否在 Jupyter 环境下
    if "ipykernel_launcher" in sys.argv[0]:
        train_batch = 256
        val_batch = 256
        test_batch = 256
        lr = 1e-5
        num_epoch = 1000
        cuda_name = 'cuda:0'
    else:
        parser = argparse.ArgumentParser(description='train model')
        parser.add_argument('--train_batch', type=int, required=False, default=128, help='Batch size training set')
        parser.add_argument('--val_batch', type=int, required=False, default=128, help='Batch size validation set')
        parser.add_argument('--test_batch', type=int, required=False, default=128, help='Batch size test set')
        parser.add_argument('--lr', type=float, required=False, default=1e-5, help='Learning rate')
        parser.add_argument('--num_epoch', type=int, required=False, default=1000, help='Number of epoch')
        parser.add_argument('--cuda_name', type=str, required=False, default='cuda:0', help='Cuda')
        args = parser.parse_args()

        train_batch = args.train_batch
        val_batch = args.val_batch
        test_batch = args.test_batch
        lr = args.lr
        num_epoch = args.num_epoch
        cuda_name = args.cuda_name

    modeling = MBNet
    criterion = nn.BCEWithLogitsLoss()

    for i in range(0, 5):
        train_model(modeling, train_batch, test_batch, criterion, lr, num_epoch, cuda_name, i)
