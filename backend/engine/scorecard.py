"""
评分卡训练模块
Logistic Regression + WOE编码
生成标准评分卡（特征→分箱→加减分）
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve
from typing import Dict, List, Tuple, Optional, Any
import json
import pickle
import base64


class ScorecardTrainer:
    """评分卡训练器"""

    def __init__(self, woe_binner, selected_features: List[str]):
        self.binner = woe_binner
        self.selected_features = selected_features
        self.model = None
        self.coef_ = None
        self.intercept_ = None
        self.base_score = 600
        self.pdo = 20
        self.factor = None
        self.offset = None
        self.scorecard = None
        self.train_metrics = {}
        self.test_metrics = {}
        self.vif_values = {}

    def _prepare_woe_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """将WOE编码应用到数据"""
        woe_df = self.binner.transform(df)
        woe_cols = [f"woe_{col}" for col in self.selected_features if f"woe_{col}" in woe_df.columns]
        for col in self.selected_features:
            woe_col = f"woe_{col}"
            if woe_col not in woe_df:
                woe_df[woe_col] = 0
        X = woe_df[woe_cols].values
        if "class" in df.columns and df["class"].dtype == "object":
            y = (df["class"] == "bad").astype(int).values
        elif "class" in df.columns:
            y = df["class"].astype(int).values
        else:
            target = self.binner.target_col
            if df[target].dtype == "object":
                y = (df[target] == "bad").astype(int).values
            else:
                y = df[target].astype(int).values
        return X, y, woe_cols

    def _calc_vif(self, X: np.ndarray, col_names: list) -> Dict:
        """计算VIF（方差膨胀因子）"""
        try:
            from statsmodels.stats.outliers_influence import variance_inflation_factor
            vif = {}
            for i, col in enumerate(col_names):
                try:
                    vif[col] = round(float(variance_inflation_factor(X, i)), 2)
                except Exception:
                    vif[col] = float("inf")
            return vif
        except ImportError:
            return {}

    def train(self, df: pd.DataFrame, test_size: float = 0.3, random_state: int = 42) -> Dict:
        """训练Logistic Regression模型"""
        X, y, woe_cols = self._prepare_woe_data(df)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        self.model = LogisticRegression(
            penalty="l2",
            C=1.0,
            solver="lbfgs",
            max_iter=1000,
            random_state=random_state,
        )
        self.model.fit(X_train, y_train)
        self.coef_ = dict(zip(woe_cols, self.model.coef_[0]))
        self.intercept_ = float(self.model.intercept_[0])
        self.vif_values = self._calc_vif(X_train, woe_cols)
        self._calc_score_constants()
        train_pred = self.model.predict_proba(X_train)[:, 1]
        test_pred = self.model.predict_proba(X_test)[:, 1]
        self.train_metrics = self._calc_metrics(y_train, train_pred, "train")
        self.test_metrics = self._calc_metrics(y_test, test_pred, "test")
        self.scorecard = self._build_scorecard(woe_cols)
        return {
            "model_type": "Logistic Regression",
            "features": self.selected_features,
            "n_features": len(self.selected_features),
            "coef": {k: round(float(v), 6) for k, v in self.coef_.items()},
            "intercept": round(self.intercept_, 6),
            "vif": {k: (round(float(v), 2) if v == v and abs(v) != float("inf") else None) for k, v in self.vif_values.items()},
            "base_score": self.base_score,
            "pdo": self.pdo,
            "train_metrics": self.train_metrics,
            "test_metrics": self.test_metrics,
            "scorecard_preview": self.scorecard[:10],
            "scorecard_total_bins": len(self.scorecard),
        }

    def _calc_score_constants(self):
        """计算评分卡缩放常数"""
        self.factor = self.pdo / np.log(2)
        p0 = 1 / (1 + np.exp(-(-self.intercept_)))
        if p0 == 0:
            p0 = 0.0001
        elif p0 == 1:
            p0 = 0.9999
        self.offset = self.base_score - self.factor * np.log(p0 / (1 - p0))

    def _calc_metrics(self, y_true, y_pred, prefix: str) -> Dict:
        """计算评估指标"""
        auc = roc_auc_score(y_true, y_pred)
        fpr, tpr, thresholds = roc_curve(y_true, y_pred)
        ks = float(max(tpr - fpr))
        gini = 2 * auc - 1
        return {
            "auc": round(auc, 4),
            "ks": round(ks, 4),
            "gini": round(gini, 4),
            "n_samples": int(len(y_true)),
        }

    def _build_scorecard(self, woe_cols: list) -> List[Dict]:
        """构建评分卡表"""
        scorecard = []
        for woe_col in woe_cols:
            orig_col = woe_col.replace("woe_", "")
            coef = self.coef_.get(woe_col, 0)
            if orig_col in self.binner.woe_maps:
                for bin_label, woe_val in self.binner.woe_maps[orig_col].items():
                    score = round(-(self.factor * coef * woe_val), 2)
                    scorecard.append({
                        "feature": orig_col,
                        "bin": bin_label,
                        "woe": round(woe_val, 4),
                        "coef": round(float(coef), 6),
                        "score": score,
                    })
        return scorecard

    def predict_score(self, df: pd.DataFrame) -> List[Dict]:
        """对新数据评分"""
        X, _, _ = self._prepare_woe_data(df)
        preds = self.model.predict_proba(X)[:, 1]
        results = []
        for i, pred in enumerate(preds):
            odds = pred / (1 - pred) if pred < 1 else 999
            score = int(round(self.offset + self.factor * np.log(max(odds, 0.0001))))
            results.append({
                "index": i,
                "score": score,
                "probability": round(float(pred), 4),
                "risk_level": self._risk_level(score),
            })
        return results

    def _risk_level(self, score: int) -> str:
        """评分风险等级"""
        if score >= 660:
            return "low_risk"
        elif score >= 580:
            return "medium_risk"
        elif score >= 500:
            return "high_risk"
        else:
            return "very_high_risk"

    def get_scorecard_table(self) -> List[Dict]:
        """获取完整评分卡表"""
        return self.scorecard or []

    def get_feature_scores(self, sample_idx: int, df: pd.DataFrame) -> List[Dict]:
        """获取单样本特征级评分明细"""
        if not self.scorecard:
            return []
        woe_df = self.binner.transform(df.iloc[[sample_idx]])
        breakdown = []
        for item in self.scorecard:
            orig_col = item["feature"]
            if orig_col in woe_df.columns:
                val = woe_df.iloc[0].get(orig_col, "N/A")
                bin_label = str(val)
                if bin_label == item["bin"]:
                    breakdown.append({
                        "feature": orig_col,
                        "bin": item["bin"],
                        "value": str(df.iloc[sample_idx].get(orig_col, "N/A")),
                        "woe": item["woe"],
                        "score": item["score"],
                    })
        return breakdown

    def export_model(self) -> str:
        """导出模型为base64字符串"""
        model_data = {
            "model": pickle.dumps(self.model),
            "coef": self.coef_,
            "intercept": self.intercept_,
            "base_score": self.base_score,
            "pdo": self.pdo,
            "factor": self.factor,
            "offset": self.offset,
            "selected_features": self.selected_features,
            "scorecard": self.scorecard,
            "woe_maps": self.binner.woe_maps,
            "iv_values": self.binner.iv_values,
        }
        return base64.b64encode(pickle.dumps(model_data)).decode("utf-8")
