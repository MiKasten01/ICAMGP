import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import json
import os
import tarfile
from pandas_plink import read_plink

class MyDataset(Dataset):
    def __init__(self):
        self.features = None
        self.labels = None
        self.stat = {}
        self.h2 = 0

    def __len__(self):
        if self.features is not None:
            return len(self.features)
        return 0

    def __getitem__(self, idx):
        if isinstance(self.features, pd.DataFrame):
            x = torch.tensor(self.features.iloc[idx].values, dtype=torch.float32)
        else:
            x = torch.tensor(self.features[idx], dtype=torch.float32)
            
        if isinstance(self.labels, pd.DataFrame):
            y = torch.tensor(self.labels.iloc[idx].values, dtype=torch.float32)
        else:
            y = torch.tensor(self.labels[idx], dtype=torch.float32)
            
        return x, y
    
    def to_numpy(self):
        X_np = self.features.values if isinstance(self.features, pd.DataFrame) else self.features
        Y_np = self.labels.values if isinstance(self.labels, pd.DataFrame) else self.labels
        return X_np, Y_np
    
    def _load_config(self, dataset_name, config_path):
        # If config_path is default relative path, try to resolve it relative to this file
        if config_path == 'dataset/dataset_config.json' and not os.path.exists(config_path):
             # Try finding it in the same directory as this file
             current_dir = os.path.dirname(os.path.abspath(__file__))
             potential_path = os.path.join(current_dir, 'dataset_config.json')
             if os.path.exists(potential_path):
                 config_path = potential_path

        if not os.path.exists(config_path):
             # Try relative path if absolute fails or vice versa
             if os.path.exists(os.path.join(os.getcwd(), config_path)):
                 config_path = os.path.join(os.getcwd(), config_path)
             else:
                 raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, 'r') as f:
            config = json.load(f)
            
        if dataset_name not in config:
            raise ValueError(f"Dataset {dataset_name} not found in config")
            
        return config[dataset_name]

    def _calculate_stats(self):
        if self.features is not None and self.labels is not None:
            X_np = self.features.values if isinstance(self.features, pd.DataFrame) else self.features
            Y_np = self.labels.values if isinstance(self.labels, pd.DataFrame) else self.labels
            
            self.stat = {
                'Num': X_np.shape[0],
                'SNPs': X_np.shape[1],
                'X_mean': np.mean(X_np, axis=0),
                'X_std': np.std(X_np, axis=0),
                'Y_mean': float(np.mean(Y_np)),
                'Y_std': float(np.std(Y_np)),
                'Y_var': float(np.var(Y_np)),
                'h2': self.h2
            }
             
def load_dataset(dataset_name, config_path='dataset/dataset_config.json', **kwargs):
    """
    统一读取数据集的入口函数。
    根据配置文件中的 dataset_type 自动选择对应的 Dataset 类。
    """
    # If config_path is default relative path, try to resolve it relative to this file
    if config_path == 'dataset/dataset_config.json' and not os.path.exists(config_path):
            # Try finding it in the same directory as this file
            current_dir = os.path.dirname(os.path.abspath(__file__))
            potential_path = os.path.join(current_dir, 'dataset_config.json')
            if os.path.exists(potential_path):
                config_path = potential_path

    if not os.path.exists(config_path):
         if os.path.exists(os.path.join(os.getcwd(), config_path)):
             config_path = os.path.join(os.getcwd(), config_path)
         else:
             raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = json.load(f)
        
    if dataset_name not in config:
        raise ValueError(f"Dataset {dataset_name} not found in config")
        
    ds_config = config[dataset_name]
    dataset_type = ds_config.get('dataset_type')
    
    if not dataset_type:
        raise ValueError(f"dataset_type not specified for {dataset_name} in config")
        
    if dataset_type == 'GeneralDataset1':
        return GeneralDataset1(dataset_name, config_path, **kwargs)
    elif dataset_type == 'GeneralDataset2':
        return GeneralDataset2(dataset_name, config_path, **kwargs)
    elif dataset_type == 'PlinkDataset':
        return PlinkDataset(dataset_name, config_path, **kwargs)
    elif dataset_type == 'WheatDataset':
        return WheatDataset(dataset_name, config_path, **kwargs)
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")

class GeneralDataset1(MyDataset):
    def __init__(self, dataset_name, config_path='dataset/dataset_config.json', delimiter='\t'):
        super().__init__()
        ds_config = self._load_config(dataset_name, config_path)
        self.h2 = ds_config.get('h2', 0)
        
        data_path = ds_config['data_path']
        X_txt_name = ds_config['X_txt_name']
        Y_txt_name = ds_config['Y_txt_name']
        target_col = ds_config.get('target_col')

        # Load X
        x_path = os.path.join(data_path, X_txt_name)
        self.features = pd.read_csv(x_path, sep=delimiter)
        
        # Load Y
        y_path = os.path.join(data_path, Y_txt_name)
        if target_col is not None:
            self.labels = pd.read_csv(y_path, sep=delimiter, usecols=["ID", target_col])
        else:
            self.labels = pd.read_csv(y_path, sep=delimiter)

        # Process X
        self.features.columns = ['column']
        df = self.features['column'].str.split(expand=True)
        df.rename(columns={0: 'ID'}, inplace=True)

        def transform_id(id_value):
            if isinstance(id_value, str) and "_" in id_value:
                parts = id_value.split("_")
                if len(parts) == 2 and parts[0] == parts[1]:
                    return parts[0]
            return id_value

        df["ID"] = df["ID"].astype(str).apply(transform_id)
        
        if df["ID"].str.isnumeric().all():
             df["ID"] = df["ID"].astype(int)

        self.features = df

        # Process Y ID check
        if np.issubdtype(self.labels['ID'].dtype, np.number) and np.array_equal(self.labels["ID"], np.arange(1, len(self.labels) + 1)):
            print(" Y['ID'] 是连续整数，重设 X['ID']")
            self.features["ID"] = np.arange(1, len(self.features) + 1)

        # Clear NaNs and align IDs
        self.nan_clear()

        # Drop ID columns
        self.common_IDs = self.features["ID"].copy()
        self.features.drop(columns=['ID'], inplace=True)
        self.labels.drop(columns=['ID'], inplace=True)

        # Convert to float32
        self.features = self.features.astype(np.float32)
        self.labels = self.labels.astype(np.float32)

        if len(self.labels.shape) == 1:
            self.labels = self.labels.to_frame()

        self._calculate_stats()

    def nan_clear(self):
        print("清理 NaN 前:")
        print(f"  X 维度: {self.features.shape}, Y 维度: {self.labels.shape}")
        original_X_IDs = set(self.features["ID"])
        original_Y_IDs = set(self.labels["ID"])
        common_IDs = original_X_IDs & original_Y_IDs
        removed_X_IDs = original_X_IDs - common_IDs
        removed_Y_IDs = original_Y_IDs - common_IDs

        self.features = self.features[self.features["ID"].isin(common_IDs)].reset_index(drop=True)
        self.labels = self.labels[self.labels["ID"].isin(common_IDs)].reset_index(drop=True)

        merged = pd.merge(self.features, self.labels, on='ID', how='inner')
        nan_rows = merged[merged.isnull().any(axis=1)]
        nan_ids = nan_rows["ID"].tolist()

        self.features = self.features[~self.features["ID"].isin(nan_ids)].reset_index(drop=True)
        self.labels = self.labels[~self.labels["ID"].isin(nan_ids)].reset_index(drop=True)

        print("清理 NaN 后:")
        print(f"  X 维度: {self.features.shape}, Y 维度: {self.labels.shape}")
        print(f"  只在 X 里不在 Y 里的 ID 数量: {len(removed_X_IDs)}")
        print(f"  只在 Y 里不在 X 里的 ID 数量: {len(removed_Y_IDs)}")
        print(f"  被删除的 NaN 行数: {len(nan_ids)}")

        return {
            "removed_X_IDs": len(removed_X_IDs),
            "removed_Y_IDs": len(removed_Y_IDs),
            "removed_nan_rows": len(nan_ids),
            "total_removed": len(removed_X_IDs) + len(removed_Y_IDs) + len(nan_ids),
        }

class GeneralDataset2(MyDataset):
    def __init__(self, dataset_name, config_path='dataset/dataset_config.json'):
        super().__init__()
        ds_config = self._load_config(dataset_name, config_path)
        self.h2 = ds_config.get('h2', 0)
        
        data_path = ds_config['data_path']
        csv_names = ds_config['csv_names']
        phenotype_column = ds_config.get('phenotype_column', 'PHENOTYPE')
        first_snp_column = ds_config.get('first_snp_column', 5)
        encoding = ds_config.get('encoding', 'default')

        all_y = []
        feature_columns = None

        for idx, file_name in enumerate(csv_names):
            file_path = os.path.join(data_path, file_name)
            print(f"读取文件: {file_path}")
            df = pd.read_csv(file_path)

            X = df.iloc[:, first_snp_column:]
            y = df[[phenotype_column]]

            if idx == 0:
                self.features = X.copy()
                feature_columns = X.columns.tolist()
            else:
                assert list(X.columns) == feature_columns, f"{file_name} 的特征列不一致！"

            all_y.append(y.reset_index(drop=True))

        self.labels = pd.concat(all_y, axis=1)

        self.clean_nan_data()

        features = self.features
        snp_names = self.features.columns

        print(f"编码方式: {encoding}")
        if encoding == 'one-hot':
            features_np = np.eye(3)[features.astype(int)].reshape(features.shape[0], -1)
            snp_names = [f"{name}_{i}" for name in snp_names for i in range(3)]
            print(len(snp_names))
            self.features = pd.DataFrame(features_np, columns=snp_names)
        elif encoding == 'binary':
            self.features = pd.DataFrame((features > 0).astype(np.float32), columns=snp_names)
        else:
            self.features = pd.DataFrame(features, columns=snp_names)

        print(f"特征列名: {self.features.columns}")
        
        # Ensure float32
        self.features = self.features.astype(np.float32)
        self.labels = self.labels.astype(np.float32)
        
        self._calculate_stats()

    def clean_nan_data(self):
        print("清理 NaN 前:", self.features.shape, self.labels.shape)
        combined = pd.concat([self.features, self.labels], axis=1)
        combined.dropna(inplace=True)
        self.features = combined.iloc[:, :len(self.features.columns)]
        self.labels = combined.iloc[:, len(self.features.columns):]
        print("清理 NaN 后:", self.features.shape, self.labels.shape)



class PlinkDataset(MyDataset):
    def __init__(self, dataset_name, config_path='dataset/dataset_config.json'):
        super().__init__()
        ds_config = self._load_config(dataset_name, config_path)
        self.h2 = ds_config.get('h2', 0)
        
        data_path = ds_config['data_path']
        tar_name = ds_config['tar_name']
        extract_dir = ds_config.get('extract_dir', 'plink_data')
        encoding = ds_config.get('encoding', 'default')
        label_column = ds_config.get('label_column', 'trait')

        tar_name = tar_name.lstrip("/")
        full_tar_path = os.path.join(data_path, tar_name)

        dataset_name_extracted = tar_name.replace(".tar.gz", "")
        extract_subdir = os.path.join(extract_dir, dataset_name_extracted)

        if not os.path.exists(extract_subdir):
            self.extract_tar(full_tar_path, extract_subdir)
        else:
            print(f"[INFO] 解压目录已存在，跳过解压：{extract_subdir}")

        prefix = self.find_prefix(extract_subdir)
        bim, fam, bed = read_plink(os.path.join(extract_subdir, prefix))

        snp_names = bim['snp'].tolist()
        features = bed.compute().T.astype(np.float32)
        self.features = pd.DataFrame(features, columns=snp_names)

        if label_column not in fam.columns:
            raise ValueError(f"label_column '{label_column}' 不在 fam 中，可选列：{fam.columns.tolist()}")
        self.labels = pd.DataFrame(fam[[label_column]].astype(np.float32)).reset_index(drop=True)

        self.clean_nan_data()

        print(f"编码方式: {encoding}")
        features = self.features
        snp_names = self.features.columns
        if encoding == 'one-hot':
            features_np = np.eye(3)[features.astype(int)].reshape(features.shape[0], -1)
            snp_names = [f"{name}_{i}" for name in snp_names for i in range(3)]
            print(len(snp_names))
            self.features = pd.DataFrame(features_np, columns=snp_names)
        elif encoding == 'binary':
            self.features = pd.DataFrame((features > 0).astype(np.float32), columns=snp_names)
        else:
            self.features = pd.DataFrame(features, columns=snp_names)
            
        # Ensure float32
        self.features = self.features.astype(np.float32)
        self.labels = self.labels.astype(np.float32)
        
        self._calculate_stats()

    def extract_tar(self, tar_path, extract_dir):
        if not os.path.exists(extract_dir):
            os.makedirs(extract_dir)
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=extract_dir)
            print(f"已解压 {tar_path} 至 {extract_dir}")

    def find_prefix(self, folder):
        bed_files = [f for f in os.listdir(folder) if f.endswith(".bed")]
        for bed_file in bed_files:
            prefix = bed_file[:-4]
            bim_file = os.path.join(folder, prefix + ".bim")
            fam_file = os.path.join(folder, prefix + ".fam")
            if os.path.exists(bim_file) and os.path.exists(fam_file):
                return prefix
        raise FileNotFoundError("未找到完整的 .bed/.bim/.fam 三件套文件")

    def clean_nan_data(self):
        print("清理 NaN 前:", self.features.shape, self.labels.shape)
        combined = pd.concat([self.features, self.labels], axis=1)
        combined.dropna(inplace=True)
        self.features = combined.iloc[:, :len(self.features.columns)]
        self.labels = combined.iloc[:, len(self.features.columns):]
        print("清理 NaN 后:", self.features.shape, self.labels.shape)

class WheatDataset(MyDataset):
    def __init__(self, dataset_name, config_path='dataset/dataset_config.json'):
        super().__init__()
        ds_config = self._load_config(dataset_name, config_path)
        self.h2 = ds_config.get('h2', 0)
        
        data_path = ds_config['data_path']
        X_csv_name = ds_config['X_csv_name']
        Y_csv_name = ds_config['Y_csv_name']
        target_col = ds_config.get('target_col')

        # Load X
        x_path = os.path.join(data_path, X_csv_name)
        self.features = pd.read_csv(x_path)
        
        # Load Y
        y_path = os.path.join(data_path, Y_csv_name)
        self.labels = pd.read_csv(y_path)
        
        if target_col:
            # Ensure target_col is treated as string if columns are strings
            # The csv header "1" might be read as int or str depending on pandas
            # Let's check columns type or try both
            if target_col in self.labels.columns:
                self.labels = self.labels[[target_col]]
            elif int(target_col) in self.labels.columns:
                 self.labels = self.labels[[int(target_col)]]
            elif str(target_col) in self.labels.columns:
                 self.labels = self.labels[[str(target_col)]]
            else:
                raise ValueError(f"Target column {target_col} not found in {Y_csv_name}. Available: {self.labels.columns.tolist()}")
        
        # Ensure float32
        self.features = self.features.astype(np.float32)
        self.labels = self.labels.astype(np.float32)
        
        self._calculate_stats()
