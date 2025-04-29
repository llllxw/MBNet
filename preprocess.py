import sys
import csv
import torch
import numpy as np
import random
import pandas as pd
from rdkit import Chem
from rdkit.Chem import MACCSkeys, AllChem, RDKFingerprint
from smiles2graph import smile_to_graph

# 修改路径以符合新的工作环境
sys.path.append('/home/MBNet')

# 处理化合物特征，包括SMILES、分子图以及分子指纹

def get_drug_mixfp(compound_smile_dict):
    """
    生成复合指纹特征，包括MACCS、ECFP4和RDKit指纹。
    """
    mixfp = {}
    for compound, smile in compound_smile_dict.items():
        mol = Chem.MolFromSmiles(smile)
        if mol is None:
            continue
        # 获取MACCS指纹
        fp_maccs = MACCSkeys.GenMACCSKeys(mol)
        # 获取ECFP4指纹
        fp_ecfp4 = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
        # 获取RDKit指纹
        fp_rdkit = RDKFingerprint(mol)
        # 合并特征
        mixfp[compound] = np.concatenate((np.array(fp_maccs), np.array(fp_ecfp4), np.array(fp_rdkit)))
    return mixfp

def get_all_graph(compound_smile_dict, compound_index_dict):
    """
    将所有化合物的SMILES序列转换为分子图结构。
    """
    smile_graph = {}
    for compound in compound_smile_dict:
        graph = smile_to_graph(compound_smile_dict[compound])
        smile_graph[compound_index_dict[compound]] = graph
    return smile_graph

def read_response_data_and_process(filename):
    # 加载化合物信息
    compound_info = pd.read_csv('/home/MBNet/data/compound_info.csv')
    compound_index_dict = dict(zip(compound_info['Name'], compound_info['ID']))
    compound_smile_dict = dict(zip(compound_info['Name'], compound_info['SMILES']))
    compound_labels = dict(zip(compound_info['Name'], compound_info['Label']))

    # 获取复合指纹特征（包括MACCS、ECFP4和RDKit）
    compound_mixfp = get_drug_mixfp(compound_smile_dict)
    # 获取分子图结构
    smile_graph = get_all_graph(compound_smile_dict, compound_index_dict)

    # 读取化合物数据
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        reader.__next__()
        data = []
        for line in reader:
            compound = line[0]
            label = int(line[1])  # 化合物标签
            data.append((compound, label))
    random.shuffle(data)

    # 匹配特征与标签
    compound_mixfp_list = []
    compound_smile_list = []
    compound_smiles_vector_list = []
    label_list = []

    for item in data:
        compound, label_value = item
        if compound not in compound_mixfp:
            continue
        compound_mixfp_list.append(compound_mixfp[compound])
        compound_smile_list.append(compound_index_dict[compound])
        compound_smiles_vector_list.append(compound_smile_dict[compound])
        label_list.append(label_value)

    # 转换为NumPy数组
    compound_mixfp_list = np.asarray(compound_mixfp_list)
    compound_smile_list = np.asarray(compound_smile_list)
    compound_smiles_vector_list = np.asarray(compound_smiles_vector_list)
    label_list = np.asarray(label_list)

    # 数据集划分
    for i in range(5):
        total_size = compound_smile_list.shape[0]
        size_0 = int(total_size * 0.2 * i)
        size_1 = size_0 + int(total_size * 0.1)
        size_2 = int(total_size * 0.2 * (i + 1))

        # 划分训练、验证和测试集
        compound_train = np.concatenate((compound_smile_list[:size_0], compound_smile_list[size_2:]), axis=0)
        compound_mixfp_train = np.concatenate((compound_mixfp_list[:size_0], compound_mixfp_list[size_2:]), axis=0)
        compound_smiles_train = np.concatenate((compound_smiles_vector_list[:size_0], compound_smiles_vector_list[size_2:]), axis=0)
        label_train = np.concatenate((label_list[:size_0], label_list[size_2:]), axis=0)

        compound_val = compound_smile_list[size_1:size_2]
        compound_mixfp_val = compound_mixfp_list[size_1:size_2]
        compound_smiles_val = compound_smiles_vector_list[size_1:size_2]
        label_val = label_list[size_1:size_2]

        compound_test = compound_smile_list[size_0:size_1]
        compound_mixfp_test = compound_mixfp_list[size_0:size_1]
        compound_smiles_test = compound_smiles_vector_list[size_0:size_1]
        label_test = label_list[size_0:size_1]

        # 保存数据集对象
        TestbedDataset(root='/home/data', dataset='train_set_{num}'.format(num=i),
                       xds=compound_train, xmixfp=compound_mixfp_train, xsmiles=compound_smiles_train, y=label_train, smile_graph=smile_graph)
        TestbedDataset(root='/home/data', dataset='val_set_{num}'.format(num=i),
                       xds=compound_val, xmixfp=compound_mixfp_val, xsmiles=compound_smiles_val, y=label_val, smile_graph=smile_graph)
        TestbedDataset(root='/home/data', dataset='test_set_{num}'.format(num=i),
                       xds=compound_test, xmixfp=compound_mixfp_test, xsmiles=compound_smiles_test, y=label_test, smile_graph=smile_graph)
        
if __name__ == '__main__':
    read_response_data_and_process('/home/MBNet/data/compound_property.csv')