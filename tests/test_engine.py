"""
单元测试 - 后端引擎模块
覆盖数据加载、EDA、WOE分箱、评分卡训练、评估、可解释性
"""
import pytest
import pandas as pd
import numpy as np
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))
from engine.data_loader import DataLoader
from engine.eda import EDAAnalyzer
from engine.woe_iv import WoeIVBinner
from engine.scorecard import ScorecardTrainer
from engine.evaluator import ModelEvaluator
from engine.explainer import ModelExplainer
from engine.exporter import ScorecardExporter

SAMPLE_DATA_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "credit-risk-datasets", "german_credit_openml.csv"
)
SAMPLE_DATA_PATH = os.path.normpath(SAMPLE_DATA_PATH)
if not os.path.exists(SAMPLE_DATA_PATH):
    SAMPLE_DATA_PATH = r"C:\Users\Daniel\Desktop\credit-risk-datasets\german_credit_openml.csv"


@pytest.fixture
def sample_df():
    """加载样本数据"""
    if os.path.exists(SAMPLE_DATA_PATH):
        return pd.read_csv(SAMPLE_DATA_PATH)
    return pd.DataFrame({
        "checking_status": ["<0", "0<=X<200", "no checking"] * 10,
        "duration": [6, 48, 12] * 10,
        "credit_amount": [1169, 5951, 2096] * 10,
        "age": [67, 22, 49] * 10,
        "class": ["good", "bad", "good"] * 10,
    })


@pytest.fixture
def loader(sample_df):
    """数据加载器"""
    dl = DataLoader()
    dl.df = sample_df
    dl.target_col = "class"
    dl._infer_column_types()
    return dl


@pytest.fixture
def trained_model(loader):
    """训练好的模型"""
    binner = WoeIVBinner(
        loader.get_df(), loader.target_col, loader.numeric_cols, loader.categorical_cols
    )
    binner.fit(max_bins=5)
    selected = [f for f in binner.feature_cols if binner.iv_values.get(f, 0) >= 0.02]
    if not selected:
        selected = binner.feature_cols[:3]
    trainer = ScorecardTrainer(binner, selected)
    trainer.train(loader.get_df())
    return binner, trainer


class TestDataLoader:
    """测试数据加载器"""

    def test_load_csv(self):
        if os.path.exists(SAMPLE_DATA_PATH):
            dl = DataLoader()
            summary = dl.load_csv(SAMPLE_DATA_PATH)
            assert summary["total_samples"] == 1000
            assert summary["num_features"] == 20
            assert summary["good_count"] == 700
            assert summary["bad_count"] == 300
            assert "checking_status" in summary["categorical_cols"]
            assert "duration" in summary["numeric_cols"]

    def test_get_head(self, loader):
        head = loader.get_head(3)
        assert len(head) == 3
        assert "checking_status" in head[0]

    def test_column_types(self, loader):
        assert "checking_status" in loader.categorical_cols
        assert "duration" in loader.numeric_cols
        assert "credit_amount" in loader.numeric_cols
        assert loader.target_col == "class"


class TestEDA:
    """测试EDA分析器"""

    def test_run_full_eda(self, loader):
        eda = EDAAnalyzer(
            loader.get_df(), loader.target_col, loader.numeric_cols, loader.categorical_cols
        )
        result = eda.run_full_eda()
        assert "data_quality" in result
        assert "numeric_stats" in result
        assert "categorical_stats" in result
        assert "target_distribution" in result
        assert "correlation" in result
        assert "missing_report" in result

    def test_data_quality(self, loader):
        eda = EDAAnalyzer(
            loader.get_df(), loader.target_col, loader.numeric_cols, loader.categorical_cols
        )
        dq = eda._data_quality_report()
        assert dq["total_rows"] > 0
        assert dq["total_cols"] > 0
        assert dq["missing_rate"] >= 0

    def test_target_distribution(self, loader):
        eda = EDAAnalyzer(
            loader.get_df(), loader.target_col, loader.numeric_cols, loader.categorical_cols
        )
        dist = eda._target_distribution()
        assert "good" in dist["counts"] or "bad" in dist["counts"]
        assert dist["imbalance_ratio"] > 0

    def test_numeric_stats(self, loader):
        eda = EDAAnalyzer(
            loader.get_df(), loader.target_col, loader.numeric_cols, loader.categorical_cols
        )
        stats = eda._numeric_statistics()
        assert len(stats) > 0
        assert "mean" in stats[0]
        assert "std" in stats[0]


class TestWoeIVBinner:
    """测试WOE/IV分箱器"""

    def test_fit(self, loader):
        binner = WoeIVBinner(
            loader.get_df(), loader.target_col, loader.numeric_cols, loader.categorical_cols
        )
        result = binner.fit(max_bins=5)
        assert "features" in result
        assert result["total_features"] > 0
        for f in result["features"]:
            assert "iv" in f
            assert "bins" in f
            assert f["iv"] >= 0

    def test_iv_ranking(self, loader):
        binner = WoeIVBinner(
            loader.get_df(), loader.target_col, loader.numeric_cols, loader.categorical_cols
        )
        binner.fit()
        ranking = binner.get_iv_ranking()
        assert len(ranking) > 0
        ivs = [r["iv"] for r in ranking]
        assert ivs == sorted(ivs, reverse=True)

    def test_woe_table(self, loader):
        binner = WoeIVBinner(
            loader.get_df(), loader.target_col, loader.numeric_cols, loader.categorical_cols
        )
        binner.fit()
        for feature in binner.feature_cols[:1]:
            table = binner.get_woe_table(feature)
            assert len(table) > 0
            assert "woe" in table[0]

    def test_iv_strength(self, loader):
        binner = WoeIVBinner(
            loader.get_df(), loader.target_col, loader.numeric_cols, loader.categorical_cols
        )
        result = binner.fit()
        assert result["strong_features"] + result["medium_features"] + result["weak_features"] == result["total_features"]


class TestScorecardTrainer:
    """测试评分卡训练器"""

    def test_train(self, trained_model):
        binner, trainer = trained_model
        assert trainer.model is not None
        assert trainer.coef_ is not None
        assert trainer.intercept_ is not None
        assert trainer.factor is not None
        assert trainer.offset is not None

    def test_train_metrics(self, trained_model):
        binner, trainer = trained_model
        assert "auc" in trainer.train_metrics
        assert "ks" in trainer.train_metrics
        assert trainer.train_metrics["auc"] > 0.5
        assert trainer.train_metrics["ks"] > 0

    def test_scorecard_table(self, trained_model):
        binner, trainer = trained_model
        table = trainer.get_scorecard_table()
        assert len(table) > 0
        assert "feature" in table[0]
        assert "bin" in table[0]
        assert "woe" in table[0]
        assert "score" in table[0]

    def test_predict_score(self, trained_model, loader):
        binner, trainer = trained_model
        scores = trainer.predict_score(loader.get_df().head(5))
        assert len(scores) == 5
        for s in scores:
            assert "score" in s
            assert "risk_level" in s
            assert s["risk_level"] in ("low_risk", "medium_risk", "high_risk", "very_high_risk")

    def test_base_score_config(self, trained_model):
        binner, trainer = trained_model
        assert trainer.base_score == 600
        assert trainer.pdo == 20

    def test_export_model(self, trained_model):
        binner, trainer = trained_model
        exported = trainer.export_model()
        assert isinstance(exported, str)
        assert len(exported) > 100


class TestModelEvaluator:
    """测试模型评估器"""

    def test_evaluate(self):
        y_true = np.array([0, 0, 0, 1, 1, 1, 0, 1, 0, 1])
        y_pred = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9, 0.4, 0.6, 0.15, 0.85])
        result = ModelEvaluator.evaluate(y_true, y_pred)
        assert result["auc"] > 0.5
        assert result["ks"] > 0
        assert result["gini"] > 0

    def test_psi_stable(self):
        np.random.seed(42)
        base = np.random.normal(0.5, 0.15, 1000)
        current = np.random.normal(0.5, 0.15, 1000)
        psi = ModelEvaluator.calc_psi(base, current)
        assert psi["psi"] < 0.1
        assert psi["status"] == "stable"

    def test_psi_unstable(self):
        np.random.seed(42)
        base = np.random.normal(0.3, 0.1, 1000)
        current = np.random.normal(0.7, 0.2, 1000)
        psi = ModelEvaluator.calc_psi(base, current)
        assert psi["psi"] > 0.1
        assert psi["status"] != "stable"

    def test_lift(self):
        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        y_pred = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
        lift = ModelEvaluator._calc_lift(y_true, y_pred, n_bins=5)
        assert len(lift) > 0
        assert "lift" in lift[0]

    def test_confusion(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0.1, 0.6, 0.4, 0.9])
        cm = ModelEvaluator._confusion_at_threshold(y_true, y_pred, 0.5)
        assert cm["tp"] + cm["fp"] + cm["tn"] + cm["fn"] == 4


class TestModelExplainer:
    """测试模型可解释性"""

    def test_algorithm_justification(self):
        explainer = ModelExplainer(None, [])
        result = explainer.get_algorithm_justification()
        assert result["selected_algorithm"] == "Logistic Regression + WOE Encoding"
        assert len(result["comparison"]) == 7
        assert "woe_reasoning" in result

    def test_compliance_audit(self):
        explainer = ModelExplainer(None, [])
        audit = explainer.compliance_audit(
            {"auc": 0.78, "ks": 0.27},
            {"auc": 0.76, "ks": 0.25},
            {"feature1": 1.5, "feature2": 3.2},
            psi=0.06,
        )
        assert audit["total_checks"] > 0
        assert audit["passed"] > 0
        assert "checks" in audit

    def test_roadmap(self):
        explainer = ModelExplainer(None, [])
        roadmap = explainer.get_explainability_roadmap()
        assert len(roadmap) >= 5
        assert "feature" in roadmap[0]

    def test_explain_sample(self, trained_model, loader):
        binner, trainer = trained_model
        woe_df = binner.transform(loader.get_df())
        woe_cols = [f"woe_{col}" for col in trainer.selected_features if f"woe_{col}" in woe_df.columns]
        X = woe_df[woe_cols].values
        explainer = ModelExplainer(trainer.model, woe_cols, binner)
        explanation = explainer.explain_sample(X[0], woe_cols, score=550, threshold=580)
        assert "breakdown" in explanation
        assert "reason_text" in explanation
        assert "decision" in explanation


class TestScorecardExporter:
    """测试评分卡导出器"""

    def test_export_html(self, trained_model, loader):
        binner, trainer = trained_model
        exporter = ScorecardExporter(
            trainer.get_scorecard_table(),
            trainer.train_metrics,
            trainer.test_metrics,
            trainer.coef_,
            trainer.intercept_,
            trainer.base_score,
            trainer.pdo,
            state["data_summary"] if "data_summary" in dir() else {"total_samples": 1000, "num_features": 21, "good_count": 700, "bad_count": 300, "good_rate": 0.7, "bad_rate": 0.3},
            {},
        )
        html = exporter.export_html()
        assert "<html" in html
        assert "评分卡" in html
        assert "Logistic" in html or "logistic" in html.lower() or "评分" in html

    def test_export_python(self, trained_model, loader):
        binner, trainer = trained_model
        exporter = ScorecardExporter(
            trainer.get_scorecard_table(),
            trainer.train_metrics,
            trainer.test_metrics,
            trainer.coef_,
            trainer.intercept_,
            trainer.base_score,
            trainer.pdo,
            {"total_samples": 1000},
            {},
        )
        script = exporter.export_python()
        assert "def calculate_score" in script
        assert "BASE_SCORE" in script
        assert "SCORECARD" in script

    def test_export_sql(self, trained_model, loader):
        binner, trainer = trained_model
        exporter = ScorecardExporter(
            trainer.get_scorecard_table(),
            trainer.train_metrics,
            trainer.test_metrics,
            trainer.coef_,
            trainer.intercept_,
            trainer.base_score,
            trainer.pdo,
            {"total_samples": 1000},
            {},
        )
        sql = exporter.export_sql()
        assert "SELECT" in sql.upper()
        assert "CASE WHEN" in sql.upper()
