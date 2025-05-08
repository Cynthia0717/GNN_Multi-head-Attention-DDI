import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from imblearn.under_sampling import ClusterCentroids
from collections import Counter


def UnderClusterCentroids(request, X, label):
    n_init = int(request.form.get(f'CCn_init'))
    voting = request.form.get(f'CC_voting')

    # 统计原始标签分布
    label_counts = Counter(label.flatten())
    print("Original counts:", label_counts)

    # 找到负样本的数量
    negative_label = 0
    negative_count = label_counts[negative_label]

    # 设置下采样策略：仅对正样本下采样到负样本数量
    sampling_strategy = {k: (int(negative_count * 1) if k != negative_label else v) for k, v in label_counts.items()}
    print("Sampling strategy:", sampling_strategy)

    x_resampled, y_resampled = Cluster_Centroids(X, label, sampling_strategy, n_init, voting)
    return x_resampled, y_resampled


def Cluster_Centroids(X, y, sampling_strategy, n_init, voting):
    y = y.ravel()
    under = ClusterCentroids(
        sampling_strategy=sampling_strategy, random_state=1, voting=voting,
        estimator=MiniBatchKMeans(n_init=n_init, random_state=1, batch_size=1024)
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
    X = data_df.iloc[:, 4:604].values  # 前200列是特征
    y = data_df['label'].values  # 最后一列是标签
    return meta, X, y


def save_resampled_data(meta, X, y, output_path):
    """
    将下采样后的数据保存到CSV文件
    参数:
    - meta: 原始前4列数据
    - X: 下采样后的特征数据
    - y: 下采样后的标签数据
    - output_path: 输出文件路径
    """
    df_resampled = pd.DataFrame(X)
    df_resampled['label'] = y
    # 添加前4列数据
    for col in meta.columns[::-1]:
        df_resampled.insert(0, col, meta[col].values[:df_resampled.shape[0]])
    df_resampled.to_csv(output_path, index=False)
    print(f"下采样完成,下采样后的数据已保存到 {output_path}")


if __name__ == '__main__':
    # 加载数据
    data_file = '../data_set/features_data/original_mol2vec/ChChMiner_train.csv'
    meta, X, y = load_data(data_file)

    class MockRequest:
        def __init__(self):
            self.form = {
                'CCn_init': '10',  # 示例：初始化KMeans的n_init参数
                'CC_voting': 'hard',  # 投票机制，选择 'hard' 或 'soft'
            }

    # 创建一个模拟的request对象
    request = MockRequest()

    # 使用 UnderRandom 进行欠采样
    X_resampled, y_resampled = UnderClusterCentroids(request, X, y)
    print(X_resampled.shape)

    # 保存下采样后的数据到CSV文件
    output_file_path = '../data_set/features_data/ClusterCentroids_downsampling/ClusterCentroids_downsample.csv'
    save_resampled_data(meta, X_resampled, y_resampled, output_file_path)

