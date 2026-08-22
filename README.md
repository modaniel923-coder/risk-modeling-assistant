# 风控建模助手 Risk Modeling Assistant

> 零代码搭建信用风险评分卡模型：从原始数据到可解释的风控模型，业务人员也能独立完成。

## 解决什么问题

在现金贷、消费信贷等场景中，风控评分卡是授信决策的核心，但建模流程专业门槛高：

- 业务人员有数据和经验，但不会写代码，无法独立完成模型开发；
- 外包给数据科学团队，沟通成本高，业务逻辑在传递中失真；
- 模型建完了缺乏可解释性，策略同事不敢直接上线使用。

**风控建模助手**把评分卡建模的全流程（数据接入 → 自动 EDA → 特征工程 → 模型训练 → 评估报告 → 策略导出）封装成零代码工作流，让风控策略、精算、运营等业务角色也能快速产出可解释、可落地的评分卡模型。

## 核心能力

| 模块 | 功能 |
|------|------|
| 数据接入 | 支持 CSV / Excel 上传，自动识别字段类型与缺失情况 |
| 自动 EDA | 一键生成目标变量分布、特征统计、相关性分析 |
| 特征工程 | WOE / IV 自动分箱，支持手动调整分箱边界 |
| 评分卡建模 | 基于 Logistic Regression 的标准评分卡训练 |
| 模型评估 | KS、AUC、混淆矩阵、Lift 曲线、PSI 稳定性 |
| 可解释输出 | 单样本评分明细、特征贡献度、规则化策略导出 |
| AI Copilot | 自然语言解读模型结果、生成风控策略建议 |

## 产品 Roadmap

- [ ] v0.1 数据上传 + 自动 EDA 报告
- [ ] v0.2 WOE / IV 自动分箱与特征筛选
- [ ] v0.3 评分卡模型训练与评估
- [ ] v0.4 模型可解释报告 + 单样本评分明细
- [ ] v0.5 AI Copilot：模型解读与策略建议
- [ ] v1.0 Web 可视化界面 + 模型版本管理

## 技术架构

- **后端**：Python + FastAPI
- **建模核心**：pandas, scikit-learn, scorecardpy / optbinning
- **LLM / AI Copilot**：DeepSeek / GPT-4o + RAG（挂载建模方法论与业务知识库）
- **前端**：Web Demo（React / Gradio，待确定）
- **部署**：Docker + GitHub Actions CI/CD

## 快速开始

```bash
git clone https://github.com/modaniel923-coder/risk-modeling-assistant.git
cd risk-modeling-assistant
# 安装依赖（待项目初始化后补充）
```

## 作者

modaniel923 · 前保险精算师，现海外信贷风控从业者，AI Builder 转型中
