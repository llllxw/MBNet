# %%


# %%
import sys
from typing import List, Tuple, Union
import torch
from torch_geometric import data as DATA
from torch_geometric.data import InMemoryDataset
import os
import numpy as np
import pickle
import pandas as pd
from rdkit import Chem
from rdkit.Chem import MACCSkeys, AllChem, RDKFingerprint
from smiles2graph import smile_to_graph
import os.path as osp
import torch
from torch_geometric.data import InMemoryDataset, Data


# 修改路径以符合新的工作环境
sys.path.append('/home/xwl/MBNet/data')

# %%

class TestbedDataset(InMemoryDataset):
    def __init__(self, root='/home/xwl/MBNet/data', dataset='set', xds=None, xmixfp=None, xbert=None, y=None, smile_graph=None, compound_index_dict=None, compound_smile_dict=None, transform=None, pre_transform=None):
        # root is required for saving preprocessed data, default is '/home/xwl/MBNet/data'
        self.dataset = dataset
        self.xds = xds
        self.xmixfp = xmixfp
        self.xbert = xbert
        self.y = y
        self.smile_graph = smile_graph
        self.compound_index_dict = compound_index_dict
        self.compound_smile_dict = compound_smile_dict

        # Ensure all input data is non-null and consistent in length
        if any(v is None for v in [xds, xmixfp, xbert, y, smile_graph, compound_index_dict, compound_smile_dict]):
            raise ValueError("One or more inputs are None. Please provide all necessary data.")

        assert len(xds) == len(y), "Length of xds and y must be the same."

        super(TestbedDataset, self).__init__(root, transform, pre_transform)

        if osp.exists(self.processed_paths[0]):
            print('Pre-processed data found: {}, loading...'.format(self.processed_paths[0]))
            self.data, self.slices = torch.load(self.processed_paths[0])
        else:
            print('Pre-processed data {} not found, doing pre-processing...'.format(self.processed_paths[0]))
            self.process()
            if osp.exists(self.processed_paths[0]):
                self.data, self.slices = torch.load(self.processed_paths[0])
            else:
                print("Error: Failed to save processed data.")

    @property
    def raw_file_names(self):
        # 必须提供 raw_file_names，以便 torch_geometric 正确地构建路径
        return []

    @property
    def processed_file_names(self):
        # 必须提供 processed_file_names，以便 torch_geometric 正确地构建路径
        return [f'{self.dataset}.pt']

    def download(self):
        # No download needed for the dataset
        pass

    def process(self):
    # Check that all data attributes are non-null
        xds, xmixfp, xbert, y, smile_graph, compound_index_dict, compound_smile_dict = self.xds, self.xmixfp, self.xbert, self.y, self.smile_graph, self.compound_index_dict, self.compound_smile_dict

        if any(v is None for v in [xds, xmixfp, xbert, y, smile_graph, compound_index_dict, compound_smile_dict]):
            print("Error: Missing input data, skipping processing.")
            return

        data_list = []

        for i in range(len(xds)):
            compound_id = xds[i]
            mixfp = xmixfp[i]
            bert = xbert[i]
            labels = y[i]

            # 获取分子图数据
            c_size, features, edge_index = smile_graph[compound_id]

            # 将 features 从列表转换为 NumPy 数组，再转为 PyTorch 张量
            features = np.array(features, dtype=np.float32)
            features = torch.tensor(features, dtype=torch.float)

            # 确保 edge_index 的形状为 (2, num_edges)
            edge_index = np.array(edge_index, dtype=np.int64)
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

            # 创建 PyTorch Geometric Data 对象
            processedData = DATA.Data(
                x=features,
                edge_index=edge_index,
                y=torch.FloatTensor([labels])
            )

            # 将 mixfp 转换为适当的 PyTorch 张量类型
            # if isinstance(mixfp, np.ndarray) and mixfp.dtype == np.object_:
            #     mixfp = np.concatenate([np.array(m, dtype=np.float32) for m in mixfp])
            # mixfp = torch.tensor(mixfp, dtype=torch.float)

            # # 将 bert 转换为适当的 PyTorch 张量类型
            # if isinstance(bert, np.ndarray) and bert.dtype == np.object_:
            #     bert = np.concatenate([np.array(b, dtype=np.float32) for b in bert])
            # bert = torch.tensor(bert, dtype=torch.float)

            # 将 mixfp 和 bert 添加到 Data 对象中
            processedData.mixfp = torch.tensor(mixfp, dtype=torch.float).view(-1)
            processedData.bert = torch.tensor(bert, dtype=torch.float).view(-1)
            # processedData.mixfp = mixfp  # 直接使用 mixfp，因为它已经是一个 Tensor
            # processedData.bert = torch.from_numpy(bert)  # 使用 torch.from_numpy 将 bert 转为 Tensor

            # 打印每个 Data 对象的摘要信息用于调试
            print(f"Processed Data {i}: x shape {processedData.x.shape}, edge_index shape {processedData.edge_index.shape}, mixfp shape {processedData.mixfp.shape}, bert shape {processedData.bert.shape}")

            processedData.__setitem__('c_size', torch.LongTensor([c_size]))
            # append graph, label and target sequence to data list
            data_list.append(processedData)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        print('Graph construction done. Saving to file.')
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

   


# %%
# Function to get BERT embeddings for SMILES

def get_drug_bert(filename, drug_smile_dict, drug_index_dict):
    bert_dict = pickle.load(open(filename, 'rb'))
    drug_bert = {}
    for drug, smile in drug_smile_dict.items():
        drug_bert[drug_index_dict[drug]] = bert_dict[smile]
    return drug_bert






# %%
