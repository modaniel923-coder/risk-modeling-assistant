"""
数据加载模块
支持CSV/Excel上传，自动识别字段类型，处理缺失值
"""
import pandas as pd
import numpy as np
from io import BytesIO
from typing import Tuple, Dict, List, Optional
import json


class DataLoader:
    """数据加载与预处理"""

    def __init__(self):
        self.df = None
        self.target_col = None
        self.feature_cols = []
        self.numeric_cols = []
        self.categorical_cols = []

    def load_csv(self, file_path: str, target_col: str = "class") -> Dict:
        """从CSV文件加载数据"""
        self.df = pd.read_csv(file_path)
        self.target_col = target_col
        self._infer_column_types()
        return self._get_summary()

    def load_csv_bytes(self, file_bytes: bytes, target_col: str = "class") -> Dict:
        """从字节流加载CSV"""
        self.df = pd.read_csv(BytesIO(file_bytes))
        self.target_col = target_col
        self._infer_column_types()
        return self._get_summary()

    def load_excel(self, file_path: str, target_col: str = "class") -> Dict:
        """从Excel文件加载数据"""
        self.df = pd.read_excel(file_path)
        self.target_col = target_col
        self._infer_column_types()
        return self._get_summary()

    def load_sample_german_credit(self) -> Dict:
        """加载内置德国信贷数据集"""
        self.df = pd.read_csv(
            "https://raw.githubusercontent.com/modaniel923-coder/risk-modeling-assistant/main/data/german_credit_openml.csv"
        )
        self.target_col = "class"
        self._infer_column_types()
        return self._get_summary()

    def _infer_column_types(self):
        """推断字段类型"""
        if self.target_col not in self.df.columns:
            # 如果目标列不存在，尝试自动识别（常见名称）
            possible_targets = ['class', 'target', 'label', 'y', 'default', 'bad_flag', 'is_bad', 'status']
            found = False
            for col in possible_targets:
                if col in self.df.columns:
                    self.target_col = col
                    found = True
                    break
            if not found:
                # 取最后一列作为目标列
                self.target_col = self.df.columns[-1]
        self.feature_cols = [c for c in self.df.columns if c != self.target_col]
        self.numeric_cols = []
        self.categorical_cols = []
        for col in self.feature_cols:
            if self.df[col].dtype in ("int64", "float64"):
                self.numeric_cols.append(col)
            else:
                try:
                    pd.to_numeric(self.df[col])
                    self.numeric_cols.append(col)
                except (ValueError, TypeError):
                    self.categorical_cols.append(col)

    def _get_summary(self) -> Dict:
        """获取数据概要"""
        total = len(self.df)
        target_vals = self.df[self.target_col].dropna().unique()
        
        # 自动识别好坏客户（支持多种格式：good/bad, 0/1, Good/Bad 等）
        good = 0
        bad = 0
        target_vals_lower = [str(v).lower() for v in target_vals]
        
        if 'good' in target_vals_lower and 'bad' in target_vals_lower:
            # 标准 good/bad 格式
            good = int((self.df[self.target_col].astype(str).str.lower() == 'good').sum())
            bad = total - good
        elif len(target_vals) == 2:
            # 二分类，假设数量少的是坏客户
            val_counts = self.df[self.target_col].value_counts()
            bad = int(val_counts.iloc[-1])
            good = total - bad
        else:
            # 其他情况：尝试找1作为坏客户
            bad = int((self.df[self.target_col] == 1).sum()) if 1 in target_vals else 0
            if bad == 0:
                bad = int((self.df[self.target_col].astype(str) == '1').sum())
            good = total - bad
        missing = self.df[self.feature_cols].isnull().sum().to_dict()
        missing = {k: int(v) for k, v in missing.items() if v > 0}
        return {
            "total_samples": total,
            "num_features": len(self.feature_cols),
            "good_count": good,
            "bad_count": bad,
            "good_rate": round(good / total, 4) if total > 0 else 0,
            "bad_rate": round(bad / total, 4) if total > 0 else 0,
            "numeric_cols": self.numeric_cols,
            "categorical_cols": self.categorical_cols,
            "feature_cols": self.feature_cols,
            "target_col": self.target_col,
            "missing_values": missing,
            "columns": list(self.df.columns),
            "dtypes": {c: str(self.df[c].dtype) for c in self.df.columns},
        }

    def get_head(self, n: int = 5) -> List[Dict]:
        """获取前n行数据"""
        return json.loads(self.df.head(n).to_json(orient="records"))

    def get_df(self) -> pd.DataFrame:
        """获取完整DataFrame"""
        return self.df
