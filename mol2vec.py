import os
import numpy as np
import pandas as pd
from rdkit import Chem
from gensim.models import word2vec
from tqdm import tqdm
import features

# 加载预训练的 Word2Vec 模型
MODEL_PATH = "../models/model_300dim.pkl"  # 请确保此模型文件存在
word2vec_model = word2vec.Word2Vec.load(MODEL_PATH)
RADIUS = 1  # Morgan 指纹半径
UNCOMMON = "UNK"  # 未知词汇的处理


# 从 SMILES 转换为分子句子
def smiles_to_sentence(smiles, radius):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    sentence = features.mol2alt_sentence(mol, radius)
    return features.MolSentence(sentence)


# 将分子句子转换为向量
def sentence_to_vector(sentence, model, unseen=None):
    if sentence is None:
        return np.zeros(model.vector_size)
    return sum(model.wv[word] if word in model.wv else model.wv[unseen] for word in sentence.sentence)


# 为数据集提取特征
def extract_features_with_mol2vec(df, model, radius, unseen):
    tqdm.pandas(desc="Processing SMILES")  # 添加进度条显示

    # 将 SMILES 转换为分子句子
    df['mol_sentence_1'] = df['smiles_1'].progress_apply(lambda x: smiles_to_sentence(x, radius))
    df['mol_sentence_2'] = df['smiles_2'].progress_apply(lambda x: smiles_to_sentence(x, radius))

    # 将分子句子转换为向量
    vectors_1 = df['mol_sentence_1'].apply(lambda x: sentence_to_vector(x, model, unseen))
    vectors_2 = df['mol_sentence_2'].apply(lambda x: sentence_to_vector(x, model, unseen))

    # 转换为矩阵并合并
    features_1 = np.vstack(vectors_1.values)
    features_2 = np.vstack(vectors_2.values)
    labels = df['label'].values
    combined_features = np.hstack([features_1, features_2])  # 合并药物1和药物2的特征

    # 清理临时列
    df.drop(columns=['mol_sentence_1', 'mol_sentence_2'], inplace=True)

    return combined_features, labels


# 保存特征到 CSV
def save_features(features, labels, original_data, output_path):
    # 将特征和标签拼接成 DataFrame
    feature_data = pd.DataFrame(features)
    feature_data['label'] = labels  # 添加标签列

    # 获取原始数据的前四列
    original_fixed_columns = original_data.iloc[:, :4]

    # 合并数据：原始前四列 + 特征数据 + 原始剩余列
    combined_data = pd.concat([original_fixed_columns.reset_index(drop=True),
                               feature_data.reset_index(drop=True)], axis=1)

    # 保存到 CSV 文件
    os.makedirs(os.path.dirname(output_path), exist_ok=True)  # 确保目录存在
    combined_data.to_csv(output_path, index=False)
    print(f"特征已保存到: {output_path}")


# 批量处理多个 CSV 文件
def process_multiple_files(input_root, output_root, model, radius, unseen):
    for root, _, files in os.walk(input_root):
        for file in files:
            if file.endswith('.csv'):
                input_path = os.path.join(root, file)
                relative_path = os.path.relpath(input_path, input_root)  # 相对路径
                output_path = os.path.join(output_root, relative_path)  # 对应的输出路径

                # 加载数据
                df = pd.read_csv(input_path)

                # 提取特征
                features, labels = extract_features_with_mol2vec(df, model, radius, unseen)

                # 保存特征到 CSV 文件
                save_features(features, labels, df, output_path)


# 主函数
if __name__ == "__main__":
    input_root = '../dataset/trans/miner'  # 输入文件夹
    output_root = '../data_set/features_data'  # 输出文件夹

    # 批量处理所有 CSV 文件
    process_multiple_files(input_root, output_root, word2vec_model, RADIUS, UNCOMMON)
