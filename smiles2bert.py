import torch
from transformers import BertTokenizer, BertModel
import pandas as pd
import pickle
from transformers import BertTokenizer, BertModel

def generate_bert_embeddings(smiles_list, model_path):
    """
    生成每个SMILES的BERT嵌入
    """
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertModel.from_pretrained(model_path)

    bert_embeddings = {}
    for smile in smiles_list:
        inputs = tokenizer(smile, return_tensors='pt')
        with torch.no_grad():
            outputs = model(**inputs)
        # 获取最后一层隐藏状态的平均值，作为句子的嵌入
        embedding = torch.mean(outputs.last_hidden_state, dim=1).squeeze().numpy()
        bert_embeddings[smile] = embedding

    return bert_embeddings

def save_bert_embeddings_to_file(smiles_list, output_file, model_path):
    """
    为每个SMILES生成BERT嵌入并保存到文件中
    """
    # 使用本地模型路径
    bert_embeddings = generate_bert_embeddings(smiles_list, model_path)
    with open(output_file, 'wb') as f:
        pickle.dump(bert_embeddings, f)
    print(f"BERT embeddings saved to {output_file}")


# 指定本地的模型路径
local_model_path = "/home/MBNet/bert-base-uncased"
# 加载分词器
tokenizer = BertTokenizer.from_pretrained(local_model_path)
# 加载模型
model = BertModel.from_pretrained(local_model_path)
# 打印模型和分词器信息，确认是否加载成功
print("Tokenizer and model loaded successfully.")

if __name__ == "__main__":
    model_path = "/home/MBNet/bert-base-uncased"  # 本地模型路径

    # 从CSV文件中加载化合物信息
    compound_info_file = '/home/MBNet/data/compound_info.csv'
    compound_info = pd.read_csv(compound_info_file)

    # 提取SMILES列表
    smiles_list = compound_info['SMILES'].tolist()

    # 指定输出文件名
    output_file = '/home/MBNet/data/bert_embeddings.pkl'

    # 生成BERT嵌入并保存到文件
    save_bert_embeddings_to_file(smiles_list, output_file, model_path)