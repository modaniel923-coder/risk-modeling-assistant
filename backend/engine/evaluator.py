"""
模型评估模块
KS、AUC、Gini、Lift、PSI计算与可视化数据生成
"""
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve
from typing import Dict, List, Any
import json


class ModelEvaluator:
    """模型评估器"""

    @staticmethod
    def evaluate(y_true: np.ndarray, y_pred: np.ndarray, prefix: str = "") -> Dict:
        """完整模型评估"""
        y_true = np.array(y_true).astype(float)
        y_pred = np.array(y_pred).astype(float)
        # 处理可能的NaN
        mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
        y_true = y_true[mask]
        y_pred = y_pred[mask]
        
        if len(np.unique(y_true)) < 2:
            # 只有一类样本，无法计算AUC
            auc = 0.5
            ks = 0.0
            gini = 0.0
            fpr = np.array([0, 1])
            tpr = np.array([0, 1])
            thresholds = np.array([1, 0])
        else:
            auc = roc_auc_score(y_true, y_pred)
            fpr, tpr, thresholds = roc_curve(y_true, y_pred)
            ks = float(max(tpr - fpr))
            gini = 2 * auc - 1
        lift_data = ModelEvaluator._calc_lift(y_true, y_pred, n_bins=10)
        ks_curve = ModelEvaluator._calc_ks_curve(y_true, y_pred)
        score_dist = ModelEvaluator._calc_score_distribution(y_pred)
        confusion = ModelEvaluator._confusion_at_threshold(y_true, y_pred, threshold=0.5)
        return {
            f"{prefix}auc": round(auc, 4),
            f"{prefix}ks": round(ks, 4),
            f"{prefix}gini": round(gini, 4),
            f"{prefix}n_samples": int(len(y_true)),
            f"{prefix}lift": lift_data,
            f"{prefix}ks_curve": ks_curve,
            f"{prefix}score_distribution": score_dist,
            f"{prefix}confusion": confusion,
        }

    @staticmethod
    def _calc_lift(y_true: np.ndarray, y_pred: np.ndarray, n_bins: int = 10) -> List[Dict]:
        """计算Lift表"""
        df = pd.DataFrame({"y": y_true, "p": y_pred})
        df = df.sort_values("p", ascending=False).reset_index(drop=True)
        # 使用按行号分箱代替qcut，避免重复值导致NaN
        n = len(df)
        actual_bins = min(n_bins, n)
        df["decile"] = (df.index * actual_bins // n).astype(int)
        total_bad = float(df["y"].sum())
        total = len(df)
        results = []
        for d in range(actual_bins):
            grp = df[df["decile"] == d]
            grp_n = len(grp)
            if grp_n == 0:
                continue
            bad = float(grp["y"].sum())
            cum_n = int(df[df["decile"] <= d].shape[0])
            cum_bad = float(df[df["decile"] <= d]["y"].sum())
            results.append({
                "decile": d + 1,
                "samples": int(grp_n),
                "bad": int(bad),
                "bad_rate": round(bad / grp_n, 4) if grp_n > 0 else 0,
                "cum_bad_rate": round(cum_bad / cum_n, 4) if cum_n > 0 else 0,
                "lift": round((bad / grp_n) / (total_bad / total), 2) if grp_n > 0 and total_bad > 0 else 0,
                "cum_lift": round((cum_bad / cum_n) / (total_bad / total), 2) if cum_n > 0 and total_bad > 0 else 0,
            })
        return results

    @staticmethod
    def _calc_ks_curve(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """计算KS曲线数据"""
        fpr, tpr, thresholds = roc_curve(y_true, y_pred)
        ks_values = tpr - fpr
        ks_max = float(max(ks_values))
        ks_idx = int(np.argmax(ks_values))
        n_points = min(50, len(thresholds))
        step = max(1, len(thresholds) // n_points)
        return {
            "ks_max": round(ks_max, 4),
            "ks_threshold": round(float(thresholds[ks_idx]), 4),
            "fpr": [round(float(fpr[i]), 4) for i in range(0, len(fpr), step)],
            "tpr": [round(float(tpr[i]), 4) for i in range(0, len(tpr), step)],
            "thresholds": [round(float(thresholds[i]), 4) for i in range(0, len(thresholds), step)] if len(thresholds) > 0 else [],
        }

    @staticmethod
    def _calc_score_distribution(y_pred: np.ndarray) -> Dict:
        """计算评分分布"""
        hist, edges = np.histogram(y_pred, bins=20)
        return {
            "bins": [round(float(e), 4) for e in edges],
            "counts": [int(h) for h in hist],
        }

    @staticmethod
    def _confusion_at_threshold(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> Dict:
        """计算混淆矩阵"""
        pred_label = (y_pred >= threshold).astype(int)
        tp = int(np.sum((pred_label == 1) & (y_true == 1)))
        fp = int(np.sum((pred_label == 1) & (y_true == 0)))
        tn = int(np.sum((pred_label == 0) & (y_true == 0)))
        fn = int(np.sum((pred_label == 0) & (y_true == 1)))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        return {
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(2 * precision * recall / (precision + recall), 4) if (precision + recall) > 0 else 0,
            "threshold": threshold,
        }

    @staticmethod
    def calc_psi(base: np.ndarray, current: np.ndarray, n_bins: int = 10) -> Dict:
        """计算PSI（群体稳定性指数）"""
        try:
            edges = np.linspace(min(base.min(), current.min()), max(base.max(), current.max()), n_bins + 1)
            edges[0] = -np.inf
            edges[-1] = np.inf
        except Exception:
            return {"psi": 0, "status": "stable"}
        base_hist, _ = np.histogram(base, bins=edges)
        curr_hist, _ = np.histogram(current, bins=edges)
        base_pct = base_hist / max(base_hist.sum(), 1)
        curr_pct = curr_hist / max(curr_hist.sum(), 1)
        eps = 1e-6
        psi_bins = (curr_pct - base_pct) * np.log((curr_pct + eps) / (base_pct + eps))
        psi = float(psi_bins.sum())
        status = "stable" if psi < 0.1 else "warning" if psi < 0.25 else "unstable"
        return {
            "psi": round(psi, 4),
            "status": status,
            "base_pct": [round(float(p), 4) for p in base_pct],
            "curr_pct": [round(float(p), 4) for p in curr_pct],
            "bin_edges": [round(float(e) if e != -np.inf else 0, 4) for e in edges],
        }
