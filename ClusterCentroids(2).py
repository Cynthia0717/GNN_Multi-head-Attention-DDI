import os
import shutil
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from imblearn.under_sampling import ClusterCentroids
from collections import Counter


def UnderClusterCentroids(request, X, label, k):
    n_init = int(request.form.get('CCn_init', 10))
    voting = request.form.get('CC_voting', 'hard')

    label_counts = Counter(label.flatten())
    print("原始类别样本数:", label_counts)

    # 计算新目标样本数（每个类别减少 500）
    label_count_dict = {l: max(n - k, 0) for l, n in label_counts.items()}
    print("调整后的样本数:", label_count_dict)

    x_resampled, y_resampled = Cluster_Centroids(X, label, label_count_dict, n_init, voting)
    return x_resampled, y_resampled


def Cluster_Centroids(X, y, sampling_strategy, n_init, voting):
    y = y.ravel()
    under = ClusterCentroids(
        sampling_strategy=sampling_strategy, random_state=1, voting=voting,
        estimator=MiniBatchKMeans(n_init=n_init, random_state=1, batch_size=2048)
    )
    x_resampled, y_resampled = under.fit_resample(X, y)
    return x_resampled, y_resampled


def load_data(file_path):
    """
    读取CSV文件并返回前4列、特征和标签
    参数:
    - file_path: 数据文件路径
    返回:
    - meta: 前4列数据
    - X: 特征数据
    - y: 标签数据
    """
    data_df = pd.read_csv(file_path)
    meta = data_df.iloc[:, :4]  # 前4列数据
    X = data_df.iloc[:, 4:-1].values  # 前200列是特征
    y = data_df['label'].values  # 最后一列是标签
    return meta, X, y


def save_resampled_data(meta, X, y, output_path, k):
    """
    将下采样后的数据保存到CSV文件
    参数:
    - meta: 原始前4列数据
    - X: 下采样后的特征数据
    - y: 下采样后的标签数据
    - output_path: 输出文件路径
    """
    # 生成动态文件名
    dir_name, base_name = os.path.split(output_path)
    base_name, ext = os.path.splitext(base_name)
    new_filename = f"{base_name}_reduce{k}{ext}"
    new_path = os.path.join(dir_name, new_filename)

    df_resampled = pd.DataFrame(X)
    df_resampled['label'] = y

    # 添加前4列数据
    for col in meta.columns[::-1]:
        df_resampled.insert(0, col, meta[col].values[:df_resampled.shape[0]])
    df_resampled.to_csv(new_path, index=False)
    print(f"下采样完成,下采样后的数据已保存到 {new_path}")


def main(input_root, output_root, k):
    os.makedirs(output_root, exist_ok=True)

    class MockRequest:
        def __init__(self):
            self.form = {
                'CCn_init': '5',  # 示例：初始化KMeans的n_init参数
                'CC_voting': 'hard',  # 投票机制，选择 'hard' 或 'soft'
            }

    # 创建一个模拟的request对象
    request = MockRequest()

    for fold in os.listdir(input_root):
        fold_path = os.path.join(input_root, fold)
        if not os.path.isdir(fold_path):
            continue  # 跳过非文件夹项

        output_fold_path = os.path.join(output_root, fold)
        os.makedirs(output_fold_path, exist_ok=True)

        for file in os.listdir(fold_path):
            input_file_path = os.path.join(fold_path, file)
            output_file_path = os.path.join(output_fold_path, file)

            if file.startswith("train") and file.endswith(".csv"):
                meta, X, y = load_data(input_file_path)
                X_resampled, y_resampled = UnderClusterCentroids(request, X, y, k)
                print("欠采样后数据形状:", X_resampled.shape)
                save_resampled_data(meta, X_resampled, y_resampled, output_file_path, k)
            else:
                shutil.copy2(input_file_path, output_file_path)  # 复制其他文件
                print(f"复制 {input_file_path} 到 {output_file_path}")


if __name__ == '__main__':
    # 加载数据
    # input_root = '../data/feature_select/Lasso/lasso/ClusterCentroids_170'
    # input_root = '../data/feature_select/mutual_info_classif/mic/ClusterCentroids_190'
    # input_root = '../data/feature_select/Random_Forest/rf/ClusterCentroids_190'
    # input_root = '../data/feature_select/RFE/rfe/ClusterCentroids_190'
    input_root = '../data_set/features_data'
    k_values = [200]  # 欲减少的样本数
    for k in k_values:
        # output_root = f'../data/feature_select+ClusterCentroids/Lasso170+ClusterCentroids/reduce{k}'
        # output_root = f'../data/feature_select+ClusterCentroids/mic190+ClusterCentroids/reduce{k}'
        # output_root = f'../data/feature_select+ClusterCentroids/rf190+ClusterCentroids/reduce{k}'
        # output_root = f'../data/feature_select+ClusterCentroids/RFE190+ClusterCentroids/reduce{k}'
        output_root = f'../data_set/features_data/CC+Shap170+CC{k}'
        print(f"\n开始处理 k={k}, 输出目录: {output_root}\n")
        main(input_root, output_root, k)

    print("\n所有 k 处理完成！")
