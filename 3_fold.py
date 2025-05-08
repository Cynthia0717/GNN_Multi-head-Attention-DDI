import os
import glob
import json
import logging
import numpy as np
import pandas as pd
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_recall_curve, f1_score, confusion_matrix, auc
)
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# 导入模型
from LSTM import LSTM
from MLP import model_MLP
from LSTM_Attention import AttentionLSTM
from BiLSTM import BiLSTM
from DNN import DNN
from GRU import GRU
from RNN import RNN
from TextCNN import TextCNN
from TextRCNN import TextRCNN
from VDCNN import VDCNN

from GNN import GNN
from GNN_Attention import GNN_Attention
from GNN_Attention_MH1 import GNN_Attention_MH1
from GNN_Attention_MH2 import GNN_Attention_MH2
from GNN_Attention_MH3 import GNN_Attention_MH3


# 初始化日志系统
def setup_logging(log_file='training.log'):
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='a'),
            logging.StreamHandler()
        ]
    )


setup_logging()


# ==================== 数据 & 模型相关工具函数 ====================


# 数据加载函数
def load_data(file_path):
    data = pd.read_csv(file_path)
    X = data.iloc[:, 4:-1].values.astype('float32')
    y = data['label'].values.astype('float32')
    X = torch.tensor(X)
    y = torch.tensor(y)
    return TensorDataset(X, y)


# 模型评估函数
def evaluate_model(y_true, y_pred_prob, threshold=0.5):
    """计算各类评估指标"""
    y_pred_prob = np.array(y_pred_prob)  # 转换为 NumPy 数组
    y_true = np.array(y_true)
    y_pred = (y_pred_prob >= threshold).astype(int)
    precision, recall, _ = precision_recall_curve(y_true, y_pred_prob)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    return {
        "ACC": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_pred_prob),
        "AUPR": auc(recall, precision),
        "F1": f1_score(y_true, y_pred),
        "Sn": tp / (tp + fn),  # 灵敏度
        "Sp": tn / (tn + fp)  # 特异性
    }


# 保存结果函数
def save_results(results, output_path):
    columns = ['Sampling Method', 'Model', 'Fold', 'Test Set', 'ACC', 'AUC', 'AUPR', 'F1', 'Sn', 'Sp']
    results_df = pd.DataFrame(results, columns=columns)
    results_df.to_csv(output_path, index=False)
    logging.info(f"[Info] Results saved to {output_path}")
    # print(f"Results saved to {output_path}")


# 模型训练函数
def train_model(model, train_loader, device, epochs=20, lr=0.001):
    model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            y_pred_logits = model(X_batch)  # 原始输出 (logits)
            y_pred_logits = y_pred_logits.squeeze()  # 去掉多余维度
            loss = criterion(y_pred_logits, y_batch.float())
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        logging.info(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(train_loader):.4f}")
        # print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(train_loader):.4f}")


# 模型测试函数
def test_model(model, test_loader, device):
    model.eval()
    sigmoid = nn.Sigmoid()
    y_true, y_pred_prob = [], []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred_logits = model(X_batch).squeeze()  # 原始输出 (logits)
            y_pred = sigmoid(y_pred_logits).cpu().numpy()  # 应用 Sigmoid 转为概率
            y_true.extend(y_batch.cpu().numpy())
            y_pred_prob.extend(y_pred)

    return y_true, y_pred_prob


# 模型字典
def get_models():
    """返回所有模型的字典"""
    return {
        # 'VDCNN': VDCNN,
        # 'LSTM': LSTM,
        # 'BiLSTM': BiLSTM,
        # 'GRU': GRU,
        # 'DNN': DNN,
        # 'RNN': RNN,
        # 'TextCNN': TextCNN,
        # 'TextRCNN': TextRCNN,
        # 'MLP': model_MLP,
        # 'AttentionLSTM': AttentionLSTM,
        # 'GNN': GNN,
        # 'GNN_Attention': GNN_Attention,
        # 'GNN_Attention_MH1': GNN_Attention_MH1,
        # 'GNN_Attention_MH2': GNN_Attention_MH2,
        'GNN_Attention_MH3': GNN_Attention_MH3,
    }


def save_temp_result(temp_dir, model_name, fold_idx, test_name, metrics):
    os.makedirs(temp_dir, exist_ok=True)
    temp_file = os.path.join(temp_dir, f"{model_name}_fold{fold_idx}_{test_name}.json")
    with open(temp_file, 'w') as f:
        json.dump(metrics, f)


def load_temp_result(temp_dir, model_name, fold_idx, test_name):
    temp_file = os.path.join(temp_dir, f"{model_name}_fold{fold_idx}_{test_name}.json")
    if os.path.exists(temp_file):
        with open(temp_file, 'r') as f:
            return json.load(f)
    return None


# ==================== 训练流程主逻辑 ====================


# 处理单个采样方法
def process_sampling_method(sampling_method_path, sampling_method, models, device, epochs, batch_size, temp_dir, clean_temp=True):
    """对某个采样方法下的所有模型进行交叉验证"""
    results = []
    metrics_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))  # 自动创建嵌套字典

    for fold_idx in tqdm(range(1), desc=f"{sampling_method} Cross-Validation"):  # 3折交叉验证
        fold_path = os.path.join(sampling_method_path, f'fold{fold_idx}')

        fold_train_file = next((f for f in os.listdir(fold_path) if f.endswith('downsample.csv') or f.startswith('train')), None)
        if not fold_train_file:
            raise FileNotFoundError(f"No training data found for fold {fold_idx} in {fold_path}")
        fold_train_path = os.path.join(fold_path, fold_train_file)

        fold_test_paths = {
            "S1": os.path.join(fold_path, next(f for f in os.listdir(fold_path) if f.startswith('test_S1'))),
            "S2": os.path.join(fold_path, next(f for f in os.listdir(fold_path) if f.startswith('test_S2')))
        }

        logging.info(f"[INFO] Processing {sampling_method} - Fold {fold_idx}...")
        # print(f"[INFO] Processing {sampling_method} - Fold {fold_idx}...")

        train_loader = DataLoader(load_data(fold_train_path), batch_size=batch_size, shuffle=True)
        test_loaders = {name: DataLoader(load_data(path), batch_size=batch_size, shuffle=False)
                        for name, path in fold_test_paths.items()}

        # 获取特征维度
        input_size = train_loader.dataset.tensors[0].shape[1]  # 获取特征维度（X的列数）
        logging.info(f"[INFO] input_size: {input_size}")
        # print(f"[INFO] input_size: {input_size}")

        tasks = []

        # 定义任务函数（每个模型一个任务）
        def train_and_eval(model_name, ModelClass, graph_type=None):
            model_specific_name = f"{model_name}_{graph_type}" if graph_type else model_name
            model = ModelClass(input_size, output_size=1, graph_type=graph_type).to(
                device) if graph_type else ModelClass(input_size, output_size=1).to(device)

            # 训练模型
            train_model(model, train_loader, device, epochs)

            for test_name, test_loader in test_loaders.items():
                # 检查是否已存在结果文件
                existing_result = load_temp_result(temp_dir, model_specific_name, fold_idx, test_name)
                if existing_result:
                    logging.info(
                        f"[RESUME] Skipping {model_specific_name} Fold {fold_idx} Test {test_name}, using saved result.")
                    # print(f"[INFO] Resume: Skip {model_specific_name} Fold {fold_idx} Test {test_name}, using saved
                    # result.")
                    metrics = existing_result
                    if clean_temp:
                        temp_file = os.path.join(temp_dir, f"{model_specific_name}_fold{fold_idx}_{test_name}.json")
                        os.remove(temp_file)
                else:
                    y_true, y_pred_prob = test_model(model, test_loader, device)
                    metrics = evaluate_model(y_true, y_pred_prob)
                    save_temp_result(temp_dir, model_specific_name, fold_idx, test_name, metrics)

                # 存储结果
                for metric, value in metrics.items():
                    metrics_dict[model_specific_name][test_name][metric].append(value)

                results.append({
                    'Sampling Method': sampling_method,
                    'Model': model_specific_name,
                    'Fold': fold_idx,
                    'Test Set': test_name,
                    **metrics
                })
                logging.info(f"[RESULT] {model_specific_name} - {test_name} - {metrics}")
                # print(f"{model_specific_name} - {test_name} - {metrics}")

        for model_name, ModelClass in models.items():
            logging.info(f"[Train] Training and testing model: {model_name}")
            # print(f"[INFO] Training and testing model: {model_name}")

            if model_name.startswith('GNN'):
                # 遍历每种图类型并训练不同模型
                for graph_type in [
                    # 'fully_connected',
                    # 'knn',
                    # 'similarity',
                    # 'cos_sim',
                    'hybrid'
                ]:
                    tasks.append((model_name, ModelClass, graph_type))
            else:
                tasks.append((model_name, ModelClass, None))

        # 并行执行任务
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = [executor.submit(train_and_eval, *task) for task in tasks]
            for _ in tqdm(as_completed(futures), total=len(futures), desc=f"{sampling_method} Fold {fold_idx}"):
                pass

    # 计算每个模型的平均值并存入结果
    for model_name, test_results in metrics_dict.items():
        for test_name, metric_values in test_results.items():
            avg_metrics = {metric: np.mean(values) for metric, values in metric_values.items()}
            avg_metrics.update({'Sampling Method': sampling_method, 'Model': model_name, 'Fold': 'Average', 'Test Set': test_name})
            results.append(avg_metrics)

    return results


# ==================== 主函数 ====================


# 主函数
def main(data_roots, output_dir, target_folders, epochs=20, batch_size=64, clean_temp=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"[INFO] Using device: {device}")
    # print(f"[INFO] Using device: {device}")
    os.makedirs(output_dir, exist_ok=True)

    all_results = []  # 存储所有采样方法的结果

    # 获取模型字典
    models_to_use = get_models()

    for data_root in data_roots:
        data_root_name = os.path.basename(data_root.rstrip('/'))

        # 遍历采样方法（每个采样方法是data_root下的子文件夹）
        for sampling_method in target_folders:
            # 匹配所有以sampling_method开头的文件夹（例如：ClusterCentroids_100、ClusterCentroids_200等）
            sampling_method_pattern = os.path.join(data_root, f'{sampling_method}*')
            matching_folders = glob.glob(sampling_method_pattern)

            if not matching_folders:
                logging.warning(f"No folders found for sampling method: {sampling_method}")
                # print(f"[INFO] No folders found for sampling method: {sampling_method}")
                continue

            for sampling_method_path in matching_folders:
                logging.info(f"[INFO] Processing sampling method: {os.path.basename(sampling_method_path)}")
                # print(f"[INFO] Processing sampling method: {os.path.basename(sampling_method_path)}")

                # 处理当前采样方法
                results = process_sampling_method(
                    sampling_method_path,
                    os.path.basename(sampling_method_path),
                    models_to_use,
                    device,
                    epochs,
                    batch_size,
                    temp_dir=os.path.join(output_dir, "temp"),
                    clean_temp=clean_temp
                )

                all_results.extend(results)

                # # 为每个采样方法保存一个csv文件
                # sampling_method_output_path = os.path.join(output_dir, f'DL_{data_root_name}_{os.path.basename(sampling_method_path)}_results.csv')
                # save_results(results, sampling_method_output_path)

        # 保存所有采样方法的结果到一个文件中
        sampling_method_output_path = os.path.join(output_dir, f'DL_{data_root_name}_results.csv')
        save_results(all_results, sampling_method_output_path)


if __name__ == '__main__':
    data_roots = [
        # '../data/feature_select+ClusterCentroids/Lasso170+ClusterCentroids',
        # '../data/feature_select+ClusterCentroids/mic190+ClusterCentroids',
        # '../data/feature_select+ClusterCentroids/rf190+ClusterCentroids',
        # '../data/feature_select+ClusterCentroids/RFE190+ClusterCentroids',
        # '../data/feature_select+ClusterCentroids/Shap170+ClusterCentroids'
        # '../data/feature_select/chi2/chi2',
        # '../data/feature_select/Lasso/lasso',
        # '../data/feature_select/mutual_info_classif/mic',
        # '../data/feature_select/Random_Forest/rf',
        # '../data/feature_select/RFE/rfe',
        '../data/feature_select/Shap/shap',
    ]
    output_dir = '../data_set'

    # 需要处理的采样方法文件夹列表
    target_folders = [
        'ClusterCentroids170',
        # 'reduce',
        # 'ENN',
        # 'KSU',
        # 'miner',
        # 'NearMiss1',
        # 'NearMiss2',
        # 'NearMiss3',
        # 'OSS',
        # 'Random'
    ]

    # 运行主函数，传入要使用的模型
    main(data_roots, output_dir, target_folders, clean_temp=True)
