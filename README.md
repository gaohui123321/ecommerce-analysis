# E-Commerce User Behavior & Sales Analysis

> 电商用户行为与销售数据分析 | 面试作品集项目

基于真实电商数据集（10,000 用户 × 120,000 交易 × 80,000 会话），从 0 到 1 完成全链路数据分析，产出可落地的运营优化建议。

## 数据集

- **来源**: [Synthetic E-Commerce Customer Behavior Dataset](https://www.kaggle.com/datasets/lorenzoscaturchio/ecommerce-behavior) (Kaggle)
- **时间范围**: 2023-01 ~ 2025-12
- **5 张关系表**: customers (10K) / products (1K) / transactions (120K) / sessions (80K) / reviews (25K)

## 项目结构

```
├── README.md
├── requirements.txt
├── data/                          # 数据集 (需从 Kaggle 下载)
├── notebooks/
│   ├── 01_data_preprocessing.ipynb    # 数据清洗与特征工程
│   ├── 02_sales_overview.ipynb        # 整体销售分析
│   ├── 03_user_analysis.ipynb         # 用户行为分析
│   ├── 04_product_analysis.ipynb      # 商品结构分析
│   ├── 05_funnel_analysis.ipynb       # 转化漏斗分析
│   ├── 06_rfm_segmentation.ipynb      # RFM 用户分层
│   └── 07_predictive_model.ipynb      # 购买预测模型
└── output/
    └── tableau_export.csv             # Tableau 可视化宽表
```

## 分析模块

| 模块 | 核心业务问题 | 方法/模型 |
|------|-------------|----------|
| 01 数据清洗 | 数据质量评估、异常值处理、特征构造 | pandas, numpy, scipy |
| 02 销售总览 | GMV 趋势、客单价、大促效果量化 | 时间序列, 同比环比, matplotlib, seaborn |
| 03 用户分析 | 新老客贡献、留存率、复购周期、同期群 | 同期群分析, plotly, seaborn |
| 04 商品分析 | ABC 分层、品类矩阵、关联购买 | 帕累托分析, mlxtend 关联规则 |
| 05 漏斗分析 | 浏览→加购→下单→支付, 流失价值量化 | 漏斗可视化, plotly |
| 06 RFM 分层 | 客户价值分群 + 精细化运营策略 | RFM 模型, numpy, seaborn |
| 07 预测模型 | 复购倾向预测, 特征重要性业务解读 | scikit-learn, statsmodels |

## 技术栈

`pandas` `numpy` `matplotlib` `seaborn` `plotly` `scipy` `scikit-learn` `statsmodels` `mlxtend`

## 快速开始

```bash
# 1. 下载数据集
kaggle datasets download lorenzoscaturchio/ecommerce-behavior -p data/ --unzip

# 2. 安装依赖
pip install -r requirements.txt

# 3. 按序号运行 notebooks/
```

## 可视化

Tableau 
仪表盘1：
https://public.tableau.com/views/_17791786520900/1?:language=zh-CN&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link
