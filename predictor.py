import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter.ttk import Treeview, Scrollbar, Style, Progressbar

import joblib
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
from mol2vec import Mol2Vec
from GNN_Multihead_Attention import GNN_Multihead_Attention


class DrugInteractionPredictor:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.data_loader = None
        self.df_result = None
        self.drug1_ids = None
        self.drug2_ids = None
        self.true_labels = None

        # 特征处理相关属性
        self.selected_features_idx_drug1 = None  # 药物1的特征选择索引
        self.selected_features_idx_drug2 = None  # 药物2的特征选择索引
        self.scaler_drug1 = StandardScaler()  # 药物1的标准化器
        self.scaler_drug2 = StandardScaler()  # 药物2的标准化器

        # 自动加载特征选择文件
        self.load_feature_selectors()

        try:
            self.feature_extractor = Mol2Vec()
        except Exception as e:
            messagebox.showwarning("警告", f"无法加载Mol2Vec模型: {str(e)}")
            self.feature_extractor = None

        self.setup_ui()

    def setup_ui(self):
        self.root = tk.Tk()
        self.root.title("药物-药物相互作用预测系统")
        self.root.geometry("1000x750")

        # 设置样式
        self.style = Style()
        self.style.configure('TLabel', font=('Arial', 10))
        self.style.configure('TButton', font=('Arial', 10))
        self.style.configure('TEntry', font=('Arial', 10))

        # 主框架
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 数据选择部分
        self.setup_data_section()

        # 模型选择部分
        self.setup_model_section()

        # 控制按钮
        self.setup_control_buttons()

        # 进度条
        self.setup_progress_bar()

        # 状态栏
        self.setup_status_bar()

        # 结果表格
        self.setup_result_table()

    def setup_model_section(self):
        frame = ttk.Frame(self.main_frame)
        frame.pack(fill=tk.X, pady=5)

        ttk.Label(frame, text='模型文件:').pack(side=tk.LEFT, padx=5)
        self.model_entry = ttk.Entry(frame, width=70)
        self.model_entry.pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)
        ttk.Button(frame, text='加载模型', command=self.load_model).pack(side=tk.LEFT, padx=5)

    def setup_data_section(self):
        frame = ttk.Frame(self.main_frame)
        frame.pack(fill=tk.X, pady=5)

        ttk.Label(frame, text='数据文件:').pack(side=tk.LEFT, padx=5)
        self.data_entry = ttk.Entry(frame, width=70)
        self.data_entry.pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)

        ttk.Button(frame, text='加载测试集', command=self.load_csv).pack(side=tk.LEFT, padx=5)

    def setup_control_buttons(self):
        # 创建主容器框架
        container = ttk.Frame(self.main_frame)
        container.pack(fill=tk.X, pady=10)

        # 创建按钮框架并使其在容器中居中
        button_frame = ttk.Frame(container)
        button_frame.pack(expand=True)  # 使用expand使框架在容器中居中

        self.predict_button = ttk.Button(
            button_frame,
            text='开始预测',
            command=self.run_prediction,
            state=tk.DISABLED
        )
        self.predict_button.pack(side=tk.LEFT, padx=10)

        ttk.Button(
            button_frame,
            text='保存结果',
            command=self.save_results
        ).pack(side=tk.LEFT, padx=10)

    def setup_progress_bar(self):
        self.progress = Progressbar(
            self.main_frame,
            orient=tk.HORIZONTAL,
            length=300,
            mode='determinate'
        )
        self.progress.pack(fill=tk.X, pady=5)

    def setup_status_bar(self):
        self.status_var = tk.StringVar()
        self.status_var.set("就绪")
        ttk.Label(
            self.main_frame,
            textvariable=self.status_var,
            relief=tk.SUNKEN,
            anchor='w'
        ).pack(fill=tk.X, pady=5)

    def setup_result_table(self):
        frame = ttk.Frame(self.main_frame)
        frame.pack(fill=tk.BOTH, expand=True, pady=10)

        # 滚动条
        scrollbar = Scrollbar(frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # 结果表格
        self.result_table = Treeview(
            frame,
            height=25,
            yscrollcommand=scrollbar.set,
            show='headings'
        )

        # 配置列
        columns = ['样本编号', '药物1', '药物2', '预测概率', '预测标签', '真实标签']
        self.result_table['columns'] = columns

        # 设置列宽
        col_widths = {
            '样本编号': 80,
            '药物1': 100,
            '药物2': 100,
            '预测概率': 150,
            '预测标签': 100,
            '真实标签': 100
        }

        for col in columns:
            self.result_table.column(col, width=col_widths[col], anchor='center')
            self.result_table.heading(col, text=col, anchor='center')

        self.result_table.pack(fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.result_table.yview)

    def load_feature_selectors(self):
        """自动加载预定义的特征选择文件"""
        try:
            # 获取当前脚本所在目录
            base_dir = os.path.dirname(os.path.abspath(__file__))
            # 药物1特征索引路径
            drug1_path = os.path.join(base_dir, "Shap_feature_index_1.pkl")
            if os.path.exists(drug1_path):
                self.selected_features_idx_drug1 = joblib.load(drug1_path)
                print(f"已自动加载药物1特征索引，共{len(self.selected_features_idx_drug1)}个特征")
            else:
                raise FileNotFoundError(f"药物1特征索引文件不存在: {drug1_path}")

            # 药物2特征索引路径
            drug2_path = os.path.join(base_dir, "Shap_feature_index_2.pkl")
            if os.path.exists(drug2_path):
                self.selected_features_idx_drug2 = joblib.load(drug2_path)
                print(f"已自动加载药物2特征索引，共{len(self.selected_features_idx_drug2)}个特征")
            else:
                raise FileNotFoundError(f"药物2特征索引文件不存在: {drug2_path}")

        except Exception as e:
            error_msg = f"加载特征选择文件失败: {str(e)}"
            print(error_msg)
            messagebox.showerror("错误", error_msg)
            # 设置为None表示不应用特征选择
            self.selected_features_idx_drug1 = None
            self.selected_features_idx_drug2 = None

    def update_status(self, message):
        self.status_var.set(message)
        self.root.update_idletasks()

    def update_progress(self, value):
        self.progress['value'] = value
        self.root.update_idletasks()

    def load_model(self):
        model_path = filedialog.askopenfilename(
            filetypes=[("PyTorch Model Files", "*.pth")]
        )

        if model_path:
            try:
                self.update_status("正在加载模型...")
                self.update_progress(10)

                self.model = torch.load(model_path).to(self.device)
                self.model.eval()

                self.model_entry.delete(0, tk.END)
                self.model_entry.insert(0, model_path)
                self.update_status(f"模型加载成功: {os.path.basename(model_path)}")
                self.update_progress(100)

                if self.data_loader:
                    self.predict_button.config(state=tk.NORMAL)

            except Exception as e:
                messagebox.showerror("错误", f"加载模型失败: {str(e)}")
                self.update_status("模型加载失败")
                self.update_progress(0)

    def load_csv(self, file_path=None, force_reload=False):
        if not file_path:
            file_path = filedialog.askopenfilename(
                filetypes=[("CSV Files", "*.csv")]
            )
            if not file_path:
                return

        try:
            print("\n=== 开始加载数据 ===")
            self.update_status("正在加载数据...")
            self.update_progress(5)

            # 1. 加载原始数据
            print("步骤1: 加载CSV文件...")
            data = pd.read_csv(file_path)
            print(f"数据加载成功，形状: {data.shape}")
            self.update_status("CSV文件加载完成，开始验证数据格式...")
            self.update_progress(10)

            # 2. 验证数据格式
            print("步骤2: 验证数据格式...")
            required_cols = ['drugbank_id_1', 'drugbank_id_2', 'smiles_1', 'smiles_2']
            missing = [col for col in required_cols if col not in data.columns]
            if missing:
                raise ValueError(f"缺少必要列: {missing}")
            print("数据格式验证通过")

            # 3. 特征提取
            print("步骤3: 特征提取...")
            self.update_status("数据格式验证通过，开始特征提取...")
            self.update_progress(20)

            if self.feature_extractor:
                print("使用Mol2Vec提取特征...")
                self.update_status("使用Mol2Vec提取药物分子特征...")
                features = self.feature_extractor.extract_features(data)
                print(f"提取的特征形状: {features.shape}")
                self.update_status("药物分子特征提取完成!")
                self.update_progress(50)
            else:
                print("使用原始特征...")
                self.update_status("使用原始特征...")
                features = data.iloc[:, 4:-1].values.astype('float32')
                print(f"原始特征形状: {features.shape}")
                self.update_progress(50)

            # 4. 分离药物1和药物2的特征 (假设各300维)
            print("步骤4: 分离药物特征...")
            self.update_status("特征选择开始...")
            num_features_per_drug = 300
            features_drug1 = features[:, :num_features_per_drug]
            features_drug2 = features[:, num_features_per_drug:num_features_per_drug * 2]
            print(f"药物1特征形状: {features_drug1.shape}")
            print(f"药物2特征形状: {features_drug2.shape}")

            # 5. 应用特征选择
            print("步骤5: 应用特征选择...")
            print(f"药物1特征选择索引: {type(self.selected_features_idx_drug1)}")
            print(f"药物2特征选择索引: {type(self.selected_features_idx_drug2)}")

            if self.selected_features_idx_drug1 is not None:
                print(
                    f"应用药物1特征选择，原维度: {features_drug1.shape[1]}, 选择{len(self.selected_features_idx_drug1)}个特征")
                features_drug1 = features_drug1[:, self.selected_features_idx_drug1]
                print(f"选择后药物1特征形状: {features_drug1.shape}")
                self.update_status(f"已应用药物1特征选择，保留{len(self.selected_features_idx_drug1)}个特征")
            else:
                print("未应用药物1特征选择")
            if self.selected_features_idx_drug2 is not None:
                print(
                    f"应用药物2特征选择，原维度: {features_drug2.shape[1]}, 选择{len(self.selected_features_idx_drug2)}个特征")
                features_drug2 = features_drug2[:, self.selected_features_idx_drug2]
                print(f"选择后药物2特征形状: {features_drug2.shape}")
                self.update_status(f"已应用药物2特征选择，保留{len(self.selected_features_idx_drug2)}个特征")
            else:
                print("未应用药物2特征选择")

            # 6. 分别标准化药物1和药物2的特征
            print("步骤6: 标准化数据...")
            self.update_status("正在标准化数据...")
            features_drug1 = self.scaler_drug1.fit_transform(features_drug1)
            features_drug2 = self.scaler_drug2.fit_transform(features_drug2)

            # 7. 拼接特征
            print("步骤7: 拼接特征...")
            features_combined = np.concatenate([features_drug1, features_drug2], axis=1)
            print(f"最终特征形状: {features_combined.shape}")
            self.update_progress(80)

            # 获取ID和标签
            self.drug1_ids = data['drugbank_id_1'].values
            self.drug2_ids = data['drugbank_id_2'].values
            self.true_labels = data['label'].values if 'label' in data.columns else None

            # 创建数据集
            print("步骤8: 创建数据集...")
            self.update_status("正在创建数据集...")
            features_tensor = torch.tensor(features_combined, dtype=torch.float32)
            if self.true_labels is not None:
                labels_tensor = torch.tensor(self.true_labels, dtype=torch.float32)
                dataset = TensorDataset(features_tensor, labels_tensor)
            else:
                dataset = TensorDataset(features_tensor)

            self.data_loader = DataLoader(dataset, batch_size=64, shuffle=False)
            self.data_entry.delete(0, tk.END)
            self.data_entry.insert(0, file_path)

            print("=== 数据加载完成 ===")
            self.update_status(f"数据处理完成: {os.path.basename(file_path)}")
            self.update_progress(100)

            if self.model:
                self.predict_button.config(state=tk.NORMAL)

        except Exception as e:
            print(f"!!! 数据加载出错: {str(e)}")
            messagebox.showerror("错误", f"加载数据失败: {str(e)}")
            self.update_status("数据加载失败")
            self.update_progress(0)

    def run_prediction(self):
        if not self.model or not self.data_loader:
            messagebox.showerror("错误", "请先加载模型和数据！")
            return

        self.predict_button.config(state=tk.DISABLED)
        self.update_status("正在预测...")
        self.update_progress(0)

        try:
            predictions = self._predict_with_progress()
            self.display_results(predictions)
            self.update_status("预测完成")
            self.update_progress(100)

        except Exception as e:
            messagebox.showerror("错误", f"预测失败: {str(e)}")
            self.update_status("预测失败")
            self.update_progress(0)

        finally:
            self.predict_button.config(state=tk.NORMAL)

    def _predict_with_progress(self):
        sigmoid = nn.Sigmoid()
        predictions = []
        total_batches = len(self.data_loader)

        with torch.no_grad():
            for batch_idx, (X_batch, _) in enumerate(self.data_loader):
                X_batch = X_batch.to(self.device)
                y_pred = sigmoid(self.model(X_batch).squeeze()).cpu().numpy()
                predictions.extend(y_pred)

                progress = (batch_idx + 1) / total_batches * 100
                self.update_progress(progress)
                self.update_status(f"预测进度: {batch_idx + 1}/{total_batches}批次")

        return np.array(predictions)

    def display_results(self, predictions):
        predictions = np.round(predictions, 4)
        pred_labels = ['1' if p >= 0.5 else '0' for p in predictions]

        if self.true_labels is not None:
            true_labels = ['1' if l == 1 else '0' for l in self.true_labels]
        else:
            true_labels = ['N/A'] * len(predictions)

        # 创建结果DataFrame
        self.df_result = pd.DataFrame({
            '样本编号': range(1, len(predictions) + 1),
            '药物1': self.drug1_ids,
            '药物2': self.drug2_ids,
            '预测概率': predictions,
            '预测标签': pred_labels,
            '真实标签': true_labels
        })

        # 更新表格
        self.result_table.delete(*self.result_table.get_children())
        for _, row in self.df_result.iterrows():
            self.result_table.insert('', 'end', values=row.tolist())

        self.update_status("结果已显示在表格中")

    def save_results(self):
        if self.df_result is None:
            messagebox.showerror('错误', '没有可保存的结果！')
            return

        default_filename = time.strftime("预测结果_%Y%m%d_%H%M%S")

        # 根据数据来源文件生成更具体的默认名
        if hasattr(self, 'data_entry') and self.data_entry.get():
            data_filename = os.path.splitext(os.path.basename(self.data_entry.get()))[0]
            default_filename = f"{data_filename}_预测结果_{time.strftime('%Y%m%d')}"

        save_path = filedialog.asksaveasfilename(
            initialfile=default_filename,  # 设置默认文件名
            defaultextension='.csv',  # 默认扩展名
            filetypes=[
                ('CSV文件', '*.csv'),
                ('Excel文件', '*.xlsx'),
                ('所有文件', '*.*')
            ],
            title='保存预测结果',  # 对话框标题
            initialdir=os.path.expanduser('\~')  # 默认保存到用户目录
        )

        if save_path:
            try:
                self.update_status("正在保存结果...")
                self.update_progress(50)
                if save_path.endswith('.xlsx'):
                    self.df_result.to_excel(save_path, index=False)
                else:
                    self.df_result.to_csv(save_path, index=False, encoding='utf-8-sig')

                self.update_status(f"结果已保存到: {os.path.basename(save_path)}")
                self.update_progress(100)
                messagebox.showinfo('成功', f'结果已保存到: {save_path}')

            except Exception as e:
                messagebox.showerror('错误', f'保存失败: {str(e)}')
                self.update_status("保存失败")
                self.update_progress(0)

    def run(self):
        self.root.mainloop()


def main():
    app = DrugInteractionPredictor()
    app.run()


if __name__ == '__main__':
    main()
