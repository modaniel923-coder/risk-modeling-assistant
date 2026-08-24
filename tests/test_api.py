"""
集成测试 - API接口测试
使用FastAPI TestClient测试所有API端点
"""
import pytest
import os
import sys
import io
import json
import pandas as pd
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))
from api.routes import app, state

client = TestClient(app)

SAMPLE_DATA_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "credit-risk-datasets", "german_credit_openml.csv"
)
SAMPLE_DATA_PATH = os.path.normpath(SAMPLE_DATA_PATH)
if not os.path.exists(SAMPLE_DATA_PATH):
    SAMPLE_DATA_PATH = r"C:\Users\Daniel\Desktop\credit-risk-datasets\german_credit_openml.csv"


class TestHealthAndInfo:
    """测试健康检查和模型信息接口"""

    def test_health_check(self):
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "version" in data

    def test_model_info_no_model(self):
        state["trainer"] = None
        response = client.get("/api/v1/model/info")
        assert response.status_code == 200
        assert response.json()["status"] == "no_model"


class TestDataAPI:
    """测试数据相关API"""

    def test_load_sample(self):
        response = client.post("/api/v1/data/load-sample")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "summary" in data
        assert data["summary"]["total_samples"] > 0

    def test_get_summary(self):
        client.post("/api/v1/data/load-sample")
        response = client.get("/api/v1/data/summary")
        assert response.status_code == 200
        data = response.json()
        assert "total_samples" in data
        assert "numeric_cols" in data
        assert "categorical_cols" in data

    def test_get_preview(self):
        client.post("/api/v1/data/load-sample")
        response = client.get("/api/v1/data/preview?n=5")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 5

    def test_upload_csv(self):
        if not os.path.exists(SAMPLE_DATA_PATH):
            pytest.skip("样本数据集不存在")
        with open(SAMPLE_DATA_PATH, "rb") as f:
            response = client.post(
                "/api/v1/data/upload",
                files={"file": ("german_credit.csv", f, "text/csv")},
                data={"target_col": "class"},
            )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["summary"]["total_samples"] > 0


class TestEDAAPI:
    """测试EDA分析API"""

    def test_run_eda(self):
        client.post("/api/v1/data/load-sample")
        response = client.post("/api/v1/eda/run")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "data_quality" in data["result"]
        assert "numeric_stats" in data["result"]
        assert "correlation" in data["result"]


class TestBinningAPI:
    """测试WOE/IV分箱API"""

    def test_run_binning(self):
        client.post("/api/v1/data/load-sample")
        response = client.post("/api/v1/binning/run")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "result" in data
        assert data["result"]["total_features"] > 0

    def test_get_iv_ranking(self):
        client.post("/api/v1/data/load-sample")
        client.post("/api/v1/binning/run")
        response = client.get("/api/v1/binning/iv-ranking")
        assert response.status_code == 200
        data = response.json()
        assert len(data["ranking"]) > 0

    def test_get_woe_table(self):
        client.post("/api/v1/data/load-sample")
        client.post("/api/v1/binning/run")
        response = client.get("/api/v1/binning/woe-table/checking_status")
        assert response.status_code == 200
        data = response.json()
        assert data["feature"] == "checking_status"
        assert len(data["woe_table"]) > 0


class TestTrainingAPI:
    """测试模型训练API"""

    def test_run_training(self):
        client.post("/api/v1/data/load-sample")
        client.post("/api/v1/binning/run")
        response = client.post("/api/v1/training/run")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "result" in data
        assert data["result"]["model_type"] == "Logistic Regression"
        assert data["result"]["train_metrics"]["auc"] > 0.5

    def test_get_training_result(self):
        client.post("/api/v1/data/load-sample")
        client.post("/api/v1/binning/run")
        client.post("/api/v1/training/run")
        response = client.get("/api/v1/training/result")
        assert response.status_code == 200
        assert "coef" in response.json()


class TestEvaluationAPI:
    """测试模型评估API"""

    def test_get_evaluation(self):
        client.post("/api/v1/data/load-sample")
        client.post("/api/v1/binning/run")
        client.post("/api/v1/training/run")
        response = client.get("/api/v1/evaluation/result")
        assert response.status_code == 200
        data = response.json()
        assert "auc" in data
        assert "ks" in data


class TestScorecardAPI:
    """测试评分卡API"""

    def test_get_scorecard(self):
        client.post("/api/v1/data/load-sample")
        client.post("/api/v1/binning/run")
        client.post("/api/v1/training/run")
        response = client.get("/api/v1/scorecard/table")
        assert response.status_code == 200
        data = response.json()
        assert len(data["scorecard"]) > 0
        assert "feature" in data["scorecard"][0]
        assert "score" in data["scorecard"][0]


class TestScoringAPI:
    """测试评分API"""

    def test_single_score(self):
        client.post("/api/v1/data/load-sample")
        client.post("/api/v1/binning/run")
        client.post("/api/v1/training/run")
        features = {
            "checking_status": "no checking",
            "duration": 12,
            "credit_history": "existing paid",
            "purpose": "radio/tv",
            "credit_amount": 2000,
            "savings_status": "<100",
            "employment": ">=7",
            "installment_commitment": 4,
            "personal_status": "male single",
            "other_parties": "none",
            "residence_since": 4,
            "property_magnitude": "real estate",
            "age": 35,
            "other_payment_plans": "none",
            "housing": "own",
            "existing_credits": 1,
            "job": "skilled",
            "num_dependents": 1,
            "own_telephone": "yes",
            "foreign_worker": "yes",
            "class": "good",
        }
        response = client.post("/api/v1/score", json={"features": features})
        assert response.status_code == 200
        data = response.json()
        assert "score" in data
        assert "risk_level" in data

    def test_batch_score(self):
        client.post("/api/v1/data/load-sample")
        client.post("/api/v1/binning/run")
        client.post("/api/v1/training/run")
        records = [
            {
                "checking_status": "no checking",
                "duration": 12,
                "credit_history": "existing paid",
                "purpose": "radio/tv",
                "credit_amount": 2000,
                "savings_status": "<100",
                "employment": ">=7",
                "installment_commitment": 4,
                "personal_status": "male single",
                "other_parties": "none",
                "residence_since": 4,
                "property_magnitude": "real estate",
                "age": 35,
                "other_payment_plans": "none",
                "housing": "own",
                "existing_credits": 1,
                "job": "skilled",
                "num_dependents": 1,
                "own_telephone": "yes",
                "foreign_worker": "yes",
                "class": "good",
            },
            {
                "checking_status": "<0",
                "duration": 48,
                "credit_history": "delayed previously",
                "purpose": "radio/tv",
                "credit_amount": 8000,
                "savings_status": "<100",
                "employment": "1<=X<4",
                "installment_commitment": 2,
                "personal_status": "male single",
                "other_parties": "none",
                "residence_since": 2,
                "property_magnitude": "no known property",
                "age": 25,
                "other_payment_plans": "none",
                "housing": "rent",
                "existing_credits": 2,
                "job": "unskilled resident",
                "num_dependents": 1,
                "own_telephone": "none",
                "foreign_worker": "yes",
                "class": "bad",
            },
        ]
        response = client.post("/api/v1/score/batch", json={"records": records})
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 2
        assert len(data["results"]) == 2


class TestExportAPI:
    """测试导出API"""

    def test_export_html(self):
        client.post("/api/v1/data/load-sample")
        client.post("/api/v1/binning/run")
        client.post("/api/v1/training/run")
        response = client.post("/api/v1/export/html")
        assert response.status_code == 200
        assert "html" in response.headers.get("content-type", "").lower()

    def test_export_python(self):
        client.post("/api/v1/data/load-sample")
        client.post("/api/v1/binning/run")
        client.post("/api/v1/training/run")
        response = client.post("/api/v1/export/python")
        assert response.status_code == 200
        body = response.text
        assert "def calculate_score" in body


class TestExplainAPI:
    """测试可解释性API"""

    def test_algorithm_justification(self):
        client.post("/api/v1/data/load-sample")
        client.post("/api/v1/binning/run")
        client.post("/api/v1/training/run")
        response = client.get("/api/v1/explain/algorithm")
        assert response.status_code == 200
        data = response.json()
        assert data["selected_algorithm"] == "Logistic Regression + WOE Encoding"

    def test_feature_importance(self):
        client.post("/api/v1/data/load-sample")
        client.post("/api/v1/binning/run")
        client.post("/api/v1/training/run")
        response = client.get("/api/v1/explain/importance")
        assert response.status_code == 200
        data = response.json()
        assert len(data["importance"]) > 0

    def test_explain_sample(self):
        client.post("/api/v1/data/load-sample")
        client.post("/api/v1/binning/run")
        client.post("/api/v1/training/run")
        response = client.get("/api/v1/explain/sample/0")
        assert response.status_code == 200
        data = response.json()
        assert "breakdown" in data
        assert "reason_text" in data

    def test_audit_report(self):
        client.post("/api/v1/data/load-sample")
        client.post("/api/v1/binning/run")
        client.post("/api/v1/training/run")
        response = client.get("/api/v1/explain/audit")
        assert response.status_code == 200
        data = response.json()
        assert data["total_checks"] > 0
        assert data["passed"] > 0

    def test_roadmap(self):
        response = client.get("/api/v1/explain/roadmap")
        assert response.status_code == 200
        data = response.json()
        assert len(data["roadmap"]) >= 5


class TestPipelineAPI:
    """测试完整流程API"""

    def test_full_pipeline(self):
        client.post("/api/v1/data/load-sample")
        response = client.get("/api/v1/pipeline/run-all")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "EDA" in data["steps_completed"]
        assert "WOE/IV分箱" in data["steps_completed"]
        assert "模型训练" in data["steps_completed"]


class TestMonitorAPI:
    """测试监控API"""

    def test_psi(self):
        response = client.post(
            "/api/v1/monitor/psi",
            json={
                "base_scores": [500, 520, 530, 540, 550, 560, 580, 600, 620, 650],
                "current_scores": [480, 510, 520, 530, 540, 550, 570, 590, 610, 640],
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "psi" in data
        assert "status" in data
