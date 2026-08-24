"""
EDA 探索性数据分析模块
自动生成统计描述、分布分析、相关性矩阵、缺失值报告
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any
import json


class EDAAnalyzer:
    """探索性数据分析"""

    def __init__(self, df: pd.DataFrame, target_col: str, numeric_cols: list, categorical_cols: list):
        self.df = df
        self.target_col = target_col
        self.numeric_cols = numeric_cols
        self.categorical_cols = categorical_cols

    def run_full_eda(self) -> Dict:
        """执行完整EDA分析"""
        return {
            "data_quality": self._data_quality_report(),
            "numeric_stats": self._numeric_statistics(),
            "categorical_stats": self._categorical_statistics(),
            "target_distribution": self._target_distribution(),
            "correlation": self._correlation_matrix(),
            "missing_report": self._missing_report(),
        }

    def _data_quality_report(self) -> Dict:
        """数据质量报告"""
        total_cells = self.df.shape[0] * self.df.shape[1]
        missing_cells = int(self.df.isnull().sum().sum())
        duplicate_rows = int(self.df.duplicated().sum())
        return {
            "total_rows": int(len(self.df)),
            "total_cols": int(self.df.shape[1]),
            "total_cells": int(total_cells),
            "missing_cells": missing_cells,
            "missing_rate": round(missing_cells / total_cells, 4) if total_cells > 0 else 0,
            "duplicate_rows": duplicate_rows,
            "duplicate_rate": round(duplicate_rows / len(self.df), 4) if len(self.df) > 0 else 0,
            "memory_mb": round(self.df.memory_usage(deep=True).sum() / 1024 / 1024, 2),
        }

    def _numeric_statistics(self) -> List[Dict]:
        """数值型特征统计"""
        results = []
        for col in self.numeric_cols:
            s = pd.to_numeric(self.df[col], errors="coerce")
            results.append({
                "column": col,
                "count": int(s.count()),
                "mean": round(float(s.mean()), 2) if s.mean() == s.mean() else None,
                "std": round(float(s.std()), 2) if s.std() == s.std() else None,
                "min": round(float(s.min()), 2) if s.min() == s.min() else None,
                "q25": round(float(s.quantile(0.25)), 2) if s.quantile(0.25) == s.quantile(0.25) else None,
                "median": round(float(s.median()), 2) if s.median() == s.median() else None,
                "q75": round(float(s.quantile(0.75)), 2) if s.quantile(0.75) == s.quantile(0.75) else None,
                "max": round(float(s.max()), 2) if s.max() == s.max() else None,
                "missing": int(s.isnull().sum()),
                "skew": round(float(s.skew()), 4) if s.skew() == s.skew() else None,
                "kurtosis": round(float(s.kurtosis()), 4) if s.kurtosis() == s.kurtosis() else None,
            })
        return results

    def _categorical_statistics(self) -> List[Dict]:
        """类别型特征统计"""
        results = []
        for col in self.categorical_cols:
            vc = self.df[col].value_counts()
            results.append({
                "column": col,
                "unique_count": int(self.df[col].nunique()),
                "top_values": {str(k): int(v) for k, v in vc.head(5).items()},
                "missing": int(self.df[col].isnull().sum()),
            })
        return results

    def _target_distribution(self) -> Dict:
        """目标变量分布"""
        vc = self.df[self.target_col].value_counts()
        total = int(vc.sum())
        return {
            "target_col": self.target_col,
            "counts": {str(k): int(v) for k, v in vc.items()},
            "rates": {str(k): round(int(v) / total, 4) for k, v in vc.items()},
            "imbalance_ratio": round(vc.iloc[0] / vc.iloc[-1], 2) if len(vc) > 1 else 1.0,
        }

    def _correlation_matrix(self) -> Dict:
        """数值型特征相关性矩阵"""
        if len(self.numeric_cols) < 2:
            return {"columns": [], "matrix": []}
        num_df = self.df[self.numeric_cols].apply(pd.to_numeric, errors="coerce")
        corr = num_df.corr()
        cols = list(corr.columns)
        matrix = []
        for i in range(len(cols)):
            row = []
            for j in range(len(cols)):
                val = corr.iloc[i, j]
                row.append(round(float(val), 4) if val == val else 0)
            matrix.append(row)
        return {"columns": cols, "matrix": matrix}

    def _missing_report(self) -> List[Dict]:
        """缺失值报告"""
        results = []
        for col in self.df.columns:
            missing = int(self.df[col].isnull().sum())
            if missing > 0:
                results.append({
                    "column": col,
                    "missing_count": missing,
                    "missing_rate": round(missing / len(self.df), 4),
                })
        return results
