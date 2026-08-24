# 风控建模助手 Risk Modeling Assistant

> 零代码信用评分卡建模平台：从原始数据到可解释的风控评分卡，业务人员也能独立完成。

一个面向风控 / 精算 / 策略人员的可视化建模工具。无需写代码，即可完成 **数据接入 → 自动 EDA → WOE/IV 分箱 → 评分卡训练 → 模型评估 → 可解释报告 → 多格式导出** 的完整流程。

## 功能特性（已全部实现）

| 模块 | 功能 |
|------|------|
| 数据接入 | 上传 CSV/Excel，自动推断字段类型、识别目标列、输出缺失率统计 |
| 自动 EDA | 目标分布、单变量统计、相关性矩阵、缺失值概览 |
| WOE/IV 分箱 | 决策树最优分箱、IV 计算与特征筛选（支持手动调整分箱边界） |
| 评分卡建模 | 逻辑回归训练，PDO/基础分/翻倍分参数化，输出标准评分卡表 |
| 模型评估 | KS、AUC/Gini、混淆矩阵、Lift 曲线、评分分布、PSI 稳定性 |
| 可解释输出 | 算法选型解释、特征重要性、单样本评分明细与特征贡献度 |
| 多格式导出 | 一键导出 HTML 报告 / Python 评分代码 / SQL 评分规则 |
| 部署与监控 | 部署指南、PSI 监控看板 |

## 技术栈

- **前端**：HTML + CSS + 原生 JS（纯静态 SPA，无需构建）
- **后端**：Python + FastAPI（25+ RESTful 接口）
- **算法引擎**：pandas / numpy / scikit-learn（WOE/IV + 逻辑回归评分卡）
- **数据**：内置 German Credit 示例数据集（开箱即用）

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
# 如有代理导致 SSL 错误：
# pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 2. 启动后端

```bash
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8080
```

### 3. 打开前端

直接双击 `frontend/index.html`，或在浏览器访问。前端默认调用 `http://127.0.0.1:8080`。

> 也可使用 `start.bat` 一键启动（Windows）。

### 4. 运行测试

```bash
pytest tests/
# 或使用 run_tests.bat（Windows）
```

## 项目结构

```
risk-modeling-assistant/
├── frontend/              # 纯静态前端（10 个功能页面）
│   ├── index.html         # 主页面
│   ├── styles.css         # 样式
│   └── app.js             # 交互逻辑与 API 调用
├── backend/
│   ├── main.py            # 应用入口
│   ├── api/routes.py      # FastAPI 路由（25+ 接口）
│   ├── engine/            # 核心算法引擎
│   │   ├── data_loader.py # 数据加载 + 类型推断
│   │   ├── eda.py         # EDA 分析
│   │   ├── woe_iv.py      # WOE/IV 分箱
│   │   ├── scorecard.py   # 评分卡训练
│   │   ├── evaluator.py   # 模型评估
│   │   ├── explainer.py   # 模型解释
│   │   └── exporter.py    # 多格式导出
│   └── data/              # 内置示例数据（german_credit.csv）
├── tests/                 # 54 个测试用例
├── requirements.txt
├── start.bat              # 一键启动
└── run_tests.bat          # 一键测试
```

## API 接口一览

| 方法 | 路径 | 功能 |
|------|------|------|
| GET | /api/v1/health | 健康检查 |
| POST | /api/v1/data/upload | 上传数据文件 |
| POST | /api/v1/data/load-sample | 加载内置示例数据 |
| GET | /api/v1/data/summary | 数据概要 |
| GET | /api/v1/data/preview | 数据预览 |
| POST | /api/v1/eda/, run | 执行 EDA 分析 |
| POST | /api/v1/binning/run | 执行 WOE 分箱 |
| GET | /api/v1/binning/iv-ranking | IV 排名 |
| GET | /api/v1/binning/woe/{feature} | 指定特征 WOE 表 |
| POST | /api/v1/training/run | 训练评分卡 |
| GET | /api/v1/scorecard/table | 评分卡表 |
| POST | /api/v1/evaluation/run | 模型评估 |
| POST | /api/v1/explain/sample | 单样本解释 |
| POST | /api/v1/export/html | 导出 HTML |
| POST | /api/v1/export/python | 导出 Python |
| POST | /api/v1/export/sql | 导出 SQL |

完整接口见仓库内 `backend/api/routes.py`。

## 作者

modaniel923 · 前保险精算师，现海外信贷风控从业者，AI Builder 转型中

---

📄 English documentation: [README_EN.md](README_EN.md)
