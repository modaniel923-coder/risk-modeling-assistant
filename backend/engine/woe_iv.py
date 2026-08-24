"""
WOE/IV 分箱模块
自动分箱、WOE编码、IV值计算、特征筛选
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import json


class WoeIVBinner:
    """WOE/IV自动分箱"""

    def __init__(self, df: pd.DataFrame, target_col: str, numeric_cols: list, categorical_cols: list):
        self.df = df.copy()
        self.target_col = target_col
        self.numeric_cols = numeric_cols
        self.categorical_cols = categorical_cols
        self.feature_cols = numeric_cols + categorical_cols
        self._prepare_target()
        self.woe_tables: Dict[str, pd.DataFrame] = {}
        self.iv_values: Dict[str, float] = {}
        self.bin_edges: Dict[str, list] = {}
        self.woe_maps: Dict[str, Dict] = {}

    def _prepare_target(self):
        """将目标变量转为0/1（1=坏客户）"""
        if self.df[self.target_col].dtype == "object":
            self.df["_bad"] = (self.df[self.target_col] == "bad").astype(int)
        else:
            self.df["_bad"] = self.df[self.target_col].astype(int)

    def fit(self, max_bins: int = 5, min_bin_pct: float = 0.05) -> Dict:
        """对所有特征执行WOE分箱"""
        results = []
        for col in self.feature_cols:
            if col in self.numeric_cols:
                result = self._bin_numeric(col, max_bins, min_bin_pct)
            else:
                result = self._bin_categorical(col)
            results.append(result)
        sorted_results = sorted(results, key=lambda x: x["iv"], reverse=True)
        return {
            "features": sorted_results,
            "total_features": len(sorted_results),
            "strong_features": sum(1 for r in sorted_results if r["iv"] >= 0.3),
            "medium_features": sum(1 for r in sorted_results if 0.1 <= r["iv"] < 0.3),
            "weak_features": sum(1 for r in sorted_results if r["iv"] < 0.1),
        }

    def _bin_numeric(self, col: str, max_bins: int, min_bin_pct: float) -> Dict:
        """数值型特征分箱（等频+决策树优化）"""
        s = pd.to_numeric(self.df[col], errors="coerce")
        good = self.df["_bad"] == 0
        bad = self.df["_bad"] == 1
        n_unique = s.nunique()
        if n_unique <= max_bins:
            bins = sorted(s.dropna().unique())
            edges = [-np.inf] + [(bins[i] + bins[i + 1]) / 2 for i in range(len(bins) - 1)] + [np.inf]
            edges = sorted(set(edges))
        else:
            edges = self._decision_tree_bins(col, s, max_bins)
        labels = self._make_labels(edges)
        self.df[f"_bin_{col}"] = pd.cut(s, bins=edges, labels=labels, include_lowest=True, duplicates="raise")
        woe_df = self._calc_woe(col, f"_bin_{col}")
        return woe_df

    def _decision_tree_bins(self, col: str, s: pd.Series, max_bins: int) -> list:
        """基于决策树的最优分箱"""
        try:
            from sklearn.tree import DecisionTreeClassifier
            valid = ~s.isna()
            X = s[valid].values.reshape(-1, 1)
            y = self.df.loc[valid, "_bad"].values
            dt = DecisionTreeClassifier(max_leaf_nodes=max_bins, min_samples_leaf=int(0.05 * len(s)))
            dt.fit(X, y)
            thresholds = sorted(set(dt.tree_.threshold[dt.tree_.children_left != -1]))
            edges = [-np.inf] + thresholds + [np.inf]
            return edges
        except Exception:
            quantiles = np.linspace(0, 1, max_bins + 1)
            edges = list(s.quantile(quantiles).values)
            edges[0] = -np.inf
            edges[-1] = np.inf
            return sorted(set(edges))

    def _bin_categorical(self, col: str) -> Dict:
        """类别型特征分箱（按WOE合并）"""
        self.df[f"_bin_{col}"] = self.df[col].astype(str).fillna("missing")
        return self._calc_woe(col, f"_bin_{col}")

    def _calc_woe(self, col: str, bin_col: str) -> Dict:
        """计算WOE和IV"""
        grouped = self.df.groupby(bin_col, observed=False)
        stats = grouped.agg(
            total=("_bad", "count"),
            bad=("_bad", "sum"),
        ).reset_index()
        stats["good"] = stats["total"] - stats["bad"]
        total_good = stats["good"].sum()
        total_bad = stats["bad"].sum()
        eps = 0.5
        stats["good_rate"] = (stats["good"] + eps) / (total_good + eps * len(stats))
        stats["bad_rate"] = (stats["bad"] + eps) / (total_bad + eps * len(stats))
        stats["woe"] = np.log(stats["good_rate"] / stats["bad_rate"])
        stats["iv"] = (stats["good_rate"] - stats["bad_rate"]) * stats["woe"]
        iv = float(stats["iv"].sum())
        stats["total_pct"] = stats["total"] / stats["total"].sum()
        stats = stats.sort_values(bin_col)
        woe_map = {str(row[bin_col]): float(row["woe"]) for _, row in stats.iterrows()}
        self.woe_maps[col] = woe_map
        self.woe_tables[col] = stats
        self.iv_values[col] = iv
        bins_info = []
        for _, row in stats.iterrows():
            bins_info.append({
                "bin": str(row[bin_col]),
                "total": int(row["total"]),
                "bad": int(row["bad"]),
                "good": int(row["good"]),
                "woe": round(float(row["woe"]), 4),
                "iv": round(float(row["iv"]), 4),
                "pct": round(float(row["total_pct"]), 4),
                "bad_rate": round(int(row["bad"]) / int(row["total"]), 4) if int(row["total"]) > 0 else 0,
            })
        strength = "strong" if iv >= 0.3 else "medium" if iv >= 0.1 else "weak"
        return {
            "feature": col,
            "iv": round(iv, 4),
            "strength": strength,
            "bins": bins_info,
            "num_bins": len(bins_info),
        }

    def _make_labels(self, edges: list) -> list:
        labels = []
        for i in range(len(edges) - 1):
            lo = edges[i]
            hi = edges[i + 1]
            if i == 0:
                labels.append(f"x <= {hi:.0f}")
            elif i == len(edges) - 2:
                labels.append(f"x > {lo:.0f}")
            else:
                labels.append(f"{lo:.0f} < x <= {hi:.0f}")
        return labels

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """将WOE编码应用到新数据"""
        result = df.copy()
        for col, woe_map in self.woe_maps.items():
            if col in result.columns:
                if col in self.numeric_cols:
                    s = pd.to_numeric(result[col], errors="coerce")
                    edges = self.bin_edges.get(col, [])
                    if edges:
                        labels = self._make_labels(edges)
                        binned = pd.cut(s, bins=edges, labels=labels, include_lowest=True)
                    else:
                        binned = s.astype(str)
                    result[f"woe_{col}"] = binned.map(woe_map).fillna(0)
                else:
                    result[f"woe_{col}"] = result[col].astype(str).map(woe_map).fillna(0)
        return result

    def get_woe_table(self, feature: str) -> List[Dict]:
        """获取指定特征的WOE表"""
        if feature not in self.woe_tables:
            return []
        df = self.woe_tables[feature]
        return json.loads(df.to_json(orient="records"))

    def get_iv_ranking(self) -> List[Dict]:
        """获取IV值排序"""
        return sorted(
            [{"feature": k, "iv": round(v, 4), "strength": "strong" if v >= 0.3 else "medium" if v >= 0.1 else "weak"}
             for k, v in self.iv_values.items()],
            key=lambda x: x["iv"],
            reverse=True,
        )
