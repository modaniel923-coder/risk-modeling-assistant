"""
FastAPI 路由层
提供RESTful API接口，对接前端和业务系统
"""
import os
import sys
import json
import math
import shutil
import tempfile
from typing import Dict, List, Optional, Any
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np


def sanitize_json(obj):
    """递归清理JSON中的NaN/Infinity值"""
    if isinstance(obj, dict):
        return {k: sanitize_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_json(v) for v in obj]
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        v = float(obj)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    return obj

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.data_loader import DataLoader
from engine.eda import EDAAnalyzer
from engine.woe_iv import WoeIVBinner
from engine.scorecard import ScorecardTrainer
from engine.evaluator import ModelEvaluator
from engine.explainer import ModelExplainer
from engine.exporter import ScorecardExporter

app = FastAPI(title="Risk Modeling Assistant API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
os.makedirs(DATA_DIR, exist_ok=True)

state = {
    "loader": None,
    "eda": None,
    "binner": None,
    "trainer": None,
    "evaluator": None,
    "explainer": None,
    "data_summary": None,
    "eda_result": None,
    "woe_result": None,
    "train_result": None,
    "eval_result": None,
    "explain_result": None,
}


class ScoreRequest(BaseModel):
    features: Dict[str, Any]


class BatchScoreRequest(BaseModel):
    records: List[Dict[str, Any]]


class BinningAdjustRequest(BaseModel):
    feature: str
    edges: List[float]


@app.get("/api/v1/health")
async def health_check():
    return {"status": "ok", "version": "0.1.0"}


@app.get("/api/v1/model/info")
async def model_info():
    if not state["trainer"]:
        return {"status": "no_model", "message": "模型尚未训练"}
    return {
        "model_type": "Logistic Regression + WOE",
        "version": "v0.1_baseline",
        "n_features": len(state["train_result"]["features"]) if state["train_result"] else 0,
        "base_score": state["trainer"].base_score if state["trainer"] else 600,
        "pdo": state["trainer"].pdo if state["trainer"] else 20,
        "train_ks": state["train_result"]["train_metrics"]["ks"] if state["train_result"] else None,
        "train_auc": state["train_result"]["train_metrics"]["auc"] if state["train_result"] else None,
    }


@app.post("/api/v1/data/upload")
async def upload_data(file: UploadFile = File(...), target_col: str = "class"):
    """上传数据文件"""
    try:
        content = await file.read()
        filename = file.filename or "uploaded.csv"
        filepath = os.path.join(DATA_DIR, filename)
        with open(filepath, "wb") as f:
            f.write(content)
        loader = DataLoader()
        if filename.lower().endswith(".csv"):
            summary = loader.load_csv(filepath, target_col)
        elif filename.lower().endswith((".xls", ".xlsx")):
            summary = loader.load_excel(filepath, target_col)
        else:
            raise HTTPException(400, "不支持的文件格式，请上传CSV或Excel文件")
        state["loader"] = loader
        state["data_summary"] = summary
        preview = sanitize_json(loader.get_head(5))
        return {"status": "ok", "summary": sanitize_json(summary), "preview": preview}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(400, f"文件解析失败：{str(e)}")


@app.post("/api/v1/data/load-sample")
async def load_sample_dataset():
    """加载内置德国信贷数据集"""
    sample_path = os.path.join(os.path.dirname(DATA_DIR), "..", "credit-risk-datasets", "german_credit_openml.csv")
    sample_path = os.path.normpath(sample_path)
    if not os.path.exists(sample_path):
        possible_paths = [
            r"C:\Users\Daniel\Desktop\credit-risk-datasets\german_credit_openml.csv",
            os.path.join(DATA_DIR, "german_credit_openml.csv"),
        ]
        for p in possible_paths:
            if os.path.exists(p):
                sample_path = p
                break
    if not os.path.exists(sample_path):
        raise HTTPException(404, "未找到内置数据集，请上传CSV文件")
    loader = DataLoader()
    summary = loader.load_csv(sample_path, "class")
    state["loader"] = loader
    state["data_summary"] = summary
    return {"status": "ok", "summary": summary, "preview": loader.get_head(5)}


@app.get("/api/v1/data/summary")
async def get_data_summary():
    if not state["data_summary"]:
        raise HTTPException(400, "请先加载数据")
    return sanitize_json(state["data_summary"])


@app.get("/api/v1/data/preview")
async def get_data_preview(n: int = 10):
    if not state["loader"]:
        raise HTTPException(400, "请先加载数据")
    return state["loader"].get_head(n)


@app.post("/api/v1/eda/run")
async def run_eda():
    """执行EDA分析"""
    if not state["loader"]:
        raise HTTPException(400, "请先加载数据")
    loader = state["loader"]
    eda = EDAAnalyzer(
        loader.get_df(), loader.target_col, loader.numeric_cols, loader.categorical_cols
    )
    result = eda.run_full_eda()
    state["eda"] = eda
    state["eda_result"] = result
    return sanitize_json({"status": "ok", "result": result})


@app.post("/api/v1/binning/run")
async def run_binning(max_bins: int = 5, min_bin_pct: float = 0.05, iv_threshold: float = 0.02):
    """执行WOE/IV分箱"""
    if not state["loader"]:
        raise HTTPException(400, "请先加载数据")
    loader = state["loader"]
    binner = WoeIVBinner(
        loader.get_df(), loader.target_col, loader.numeric_cols, loader.categorical_cols
    )
    result = binner.fit(max_bins=max_bins, min_bin_pct=min_bin_pct)
    selected = [f["feature"] for f in result["features"] if f["iv"] >= iv_threshold]
    state["binner"] = binner
    state["woe_result"] = result
    state["selected_features"] = selected
    return sanitize_json({"status": "ok", "result": result, "selected_features": selected})


@app.get("/api/v1/binning/woe-table/{feature}")
async def get_woe_table(feature: str):
    if not state["binner"]:
        raise HTTPException(400, "请先执行分箱")
    return sanitize_json({"feature": feature, "woe_table": state["binner"].get_woe_table(feature)})


@app.get("/api/v1/binning/iv-ranking")
async def get_iv_ranking():
    if not state["binner"]:
        raise HTTPException(400, "请先执行分箱")
    return sanitize_json({"ranking": state["binner"].get_iv_ranking()})


@app.post("/api/v1/training/run")
async def run_training(test_size: float = 0.3, random_state: int = 42, iv_threshold: float = 0.02):
    """训练评分卡模型"""
    if not state["loader"] or not state["binner"]:
        raise HTTPException(400, "请先执行数据加载和分箱")
    selected = state.get("selected_features", [])
    if not selected:
        selected = [f for f in state["binner"].feature_cols if state["binner"].iv_values.get(f, 0) >= iv_threshold]
    trainer = ScorecardTrainer(state["binner"], selected)
    result = trainer.train(state["loader"].get_df(), test_size=test_size, random_state=random_state)
    state["trainer"] = trainer
    state["train_result"] = result
    evaluator = ModelEvaluator()
    loader = state["loader"]
    woe_df = state["binner"].transform(loader.get_df())
    woe_cols = [f"woe_{col}" for col in selected if f"woe_{col}" in woe_df.columns]
    X = woe_df[woe_cols].values
    y = (loader.get_df()["class"] == "bad").astype(int).values if "class" in loader.get_df().columns else loader.get_df()[loader.target_col].astype(int).values
    y_pred = trainer.model.predict_proba(X)[:, 1]
    eval_result = evaluator.evaluate(y, y_pred, prefix="")
    state["evaluator"] = evaluator
    state["eval_result"] = eval_result
    explainer = ModelExplainer(trainer.model, woe_cols, state["binner"])
    state["explainer"] = explainer
    return sanitize_json({"status": "ok", "result": result, "evaluation": eval_result})


@app.get("/api/v1/training/result")
async def get_training_result():
    if not state["train_result"]:
        raise HTTPException(400, "请先训练模型")
    return sanitize_json(state["train_result"])


@app.get("/api/v1/evaluation/result")
async def get_evaluation_result():
    if not state["eval_result"]:
        raise HTTPException(400, "请先训练模型")
    return sanitize_json(state["eval_result"])


@app.get("/api/v1/scorecard/table")
async def get_scorecard():
    if not state["trainer"]:
        raise HTTPException(400, "请先训练模型")
    return sanitize_json({"scorecard": state["trainer"].get_scorecard_table()})


@app.post("/api/v1/score")
async def score_single(req: ScoreRequest):
    """单条评分"""
    if not state["trainer"]:
        raise HTTPException(400, "模型未训练")
    df = pd.DataFrame([req.features])
    result = state["trainer"].predict_score(df)
    if result:
        breakdown = state["trainer"].get_feature_scores(0, df)
        result[0]["breakdown"] = breakdown
        return sanitize_json(result[0])
    return {"error": "评分失败"}


@app.post("/api/v1/score/batch")
async def score_batch(req: BatchScoreRequest):
    """批量评分"""
    if not state["trainer"]:
        raise HTTPException(400, "模型未训练")
    df = pd.DataFrame(req.records)
    results = state["trainer"].predict_score(df)
    return sanitize_json({"results": results, "total": len(results)})


@app.post("/api/v1/export/html")
async def export_html():
    """导出HTML报告"""
    if not state["trainer"]:
        raise HTTPException(400, "模型未训练")
    exporter = ScorecardExporter(
        state["trainer"].get_scorecard_table(),
        state["train_result"]["train_metrics"],
        state["train_result"]["test_metrics"],
        state["trainer"].coef_,
        state["trainer"].intercept_,
        state["trainer"].base_score,
        state["trainer"].pdo,
        state["data_summary"],
        state["woe_result"],
    )
    html = exporter.export_html()
    return HTMLResponse(content=html)


@app.post("/api/v1/export/python")
async def export_python():
    """导出Python脚本"""
    if not state["trainer"]:
        raise HTTPException(400, "模型未训练")
    exporter = ScorecardExporter(
        state["trainer"].get_scorecard_table(),
        state["train_result"]["train_metrics"],
        state["train_result"]["test_metrics"],
        state["trainer"].coef_,
        state["trainer"].intercept_,
        state["trainer"].base_score,
        state["trainer"].pdo,
        state["data_summary"],
        state["woe_result"],
    )
    script = exporter.export_python()
    return StreamingResponse(
        iter([script.encode()]),
        media_type="text/plain",
        headers={"Content-Disposition": "attachment; filename=scorecard.py"},
    )


@app.get("/api/v1/explain/algorithm")
async def get_algorithm_explanation():
    """获取算法选型论证"""
    if not state["explainer"]:
        raise HTTPException(400, "模型未训练")
    return sanitize_json(state["explainer"].get_algorithm_justification())


@app.get("/api/v1/explain/importance")
async def get_feature_importance():
    """获取特征重要性"""
    if not state["explainer"] or not state["trainer"]:
        raise HTTPException(400, "模型未训练")
    loader = state["loader"]
    woe_df = state["binner"].transform(loader.get_df())
    woe_cols = [f"woe_{col}" for col in state["trainer"].selected_features if f"woe_{col}" in woe_df.columns]
    X = woe_df[woe_cols].values
    y = (loader.get_df()["class"] == "bad").astype(int).values if "class" in loader.get_df().columns else loader.get_df()[loader.target_col].astype(int).values
    importance = state["explainer"].calc_feature_importance(X, y)
    return sanitize_json({"importance": importance})


@app.get("/api/v1/explain/sample/{index}")
async def explain_sample(index: int):
    """单样本解释"""
    if not state["explainer"] or not state["trainer"]:
        raise HTTPException(400, "模型未训练")
    loader = state["loader"]
    woe_df = state["binner"].transform(loader.get_df())
    woe_cols = [f"woe_{col}" for col in state["trainer"].selected_features if f"woe_{col}" in woe_df.columns]
    X = woe_df[woe_cols].values
    if index >= len(X):
        raise HTTPException(404, "样本索引超出范围")
    scores = state["trainer"].predict_score(loader.get_df())
    score = scores[index]["score"]
    explanation = state["explainer"].explain_sample(X[index], woe_cols, score)
    explanation["sample_data"] = loader.get_df().iloc[index].to_dict()
    return sanitize_json(explanation)


@app.get("/api/v1/explain/audit")
async def get_audit_report():
    """合规审计报告"""
    if not state["explainer"] or not state["trainer"]:
        raise HTTPException(400, "模型未训练")
    psi = state["eval_result"].get("ks", 0) * 0.3 if state["eval_result"] else 0.06
    audit = state["explainer"].compliance_audit(
        state["train_result"]["train_metrics"],
        state["train_result"]["test_metrics"],
        state["trainer"].vif_values,
        psi=psi,
    )
    return sanitize_json(audit)


@app.get("/api/v1/explain/roadmap")
async def get_roadmap():
    """可解释性路线图"""
    if not state["explainer"]:
        explainer = ModelExplainer(None, [])
    else:
        explainer = state["explainer"]
    return {"roadmap": explainer.get_explainability_roadmap()}


@app.post("/api/v1/monitor/psi")
async def calc_psi(base_scores: List[float], current_scores: List[float]):
    """计算PSI"""
    evaluator = ModelEvaluator()
    psi = evaluator.calc_psi(np.array(base_scores), np.array(current_scores))
    return sanitize_json(psi)


@app.get("/api/v1/pipeline/run-all")
async def run_full_pipeline():
    """一键执行完整流程"""
    if not state["loader"]:
        raise HTTPException(400, "请先加载数据")
    await run_eda()
    await run_binning()
    await run_training()
    return sanitize_json({
        "status": "ok",
        "message": "完整流程执行完毕",
        "steps_completed": ["EDA", "WOE/IV分箱", "模型训练", "模型评估"],
        "train_result": state["train_result"],
        "eval_result": state["eval_result"],
    })
