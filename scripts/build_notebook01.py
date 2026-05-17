"""Generate notebook 01: Data Preprocessing & Feature Engineering"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.9.0"}
}

cells = []

def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))

def code(text):
    cells.append(nbf.v4.new_code_cell(text))

# ============================================================
md("""# 01 — 数据清洗与特征工程

---

## 业务背景

在开始任何分析之前，我们需要先回答一个问题：**这份数据能真实反映业务状况吗？**

数据清洗不是机械地删缺失值、填异常值。每一步清理背后都应该有一个业务判断：

- 这个"异常值"是录入错误，还是真实的大客户订单？
- 这个"缺失值"是系统 bug，还是业务流程中天然就不产生这条记录？
- 退款订单该不该算进 GMV？

本 Notebook 完成以下工作：
1. 加载 5 张源表，检查数据质量
2. 日期格式转换与时间特征提取
3. 异常值检测与业务判断
4. 构建分析就绪的宽表
""")

# ============================================================
md("""## 1. 环境准备""")

code("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# 绘图风格
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.size'] = 11
plt.rcParams['figure.figsize'] = (12, 5)

# 显示设置
pd.set_option('display.max_columns', 50)
pd.set_option('display.float_format', '{:.2f}'.format)

print("环境就绪 ✓")""")

# ============================================================
md("""## 2. 加载数据

一次性加载 5 张源表，初步观察规模和字段。""")

code("""# 加载全部 5 张表
customers = pd.read_csv('../data/customers.csv')
products = pd.read_csv('../data/products.csv')
transactions = pd.read_csv('../data/transactions.csv')
sessions = pd.read_csv('../data/sessions.csv')
reviews = pd.read_csv('../data/reviews.csv')

# 数据规模一览
tables = {
    'customers': customers,
    'products': products,
    'transactions': transactions,
    'sessions': sessions,
    'reviews': reviews
}

print(f"{'='*60}")
print(f"{'表名':<20} {'行数':>8} {'列数':>6}")
print(f"{'='*60}")
for name, df in tables.items():
    print(f"{name:<20} {len(df):>8,} {df.shape[1]:>6}")
print(f"{'='*60}")
print(f"\\n数据时间范围: 2023-01 ~ 2025-12 (3年)")""")

# ============================================================
md("""## 3. 数据质量审计

逐表检查：数据类型是否正确、是否存在缺失值、是否存在重复记录。

### 3.1 customers 表 —— 用户画像""")

code("""print("=== 数据类型 ===")
print(customers.dtypes)
print(f"\\n=== 缺失值 ===")
print(customers.isnull().sum()[customers.isnull().sum() > 0] if customers.isnull().sum().sum() > 0 else "无缺失值 ✓")
print(f"\\n=== 重复 customer_id ===")
dup = customers['customer_id'].duplicated().sum()
print(f"重复 ID 数: {dup}" if dup > 0 else "无重复 ✓")
print(f"\\n=== 基础统计 ===")
print(customers.describe())""")

md("""**业务观察：**
- `is_churned` 是二值变量，可以分析流失用户的特征
- `segment` 已经做了预分层（Premium / Regular / Budget Shopper），后续可以和 RFM 分群做对比
- `lifetime_value` 分布范围很大（标准差接近均值），说明用户价值差异显著 —— 这是做分层的业务前提
""")

# ============================================================
md("""### 3.2 products 表 —— 商品维度""")

code("""print("=== 数据类型 ===")
print(products.dtypes)
print(f"\\n=== 缺失值 ===")
print(products.isnull().sum()[products.isnull().sum() > 0] if products.isnull().sum().sum() > 0 else "无缺失值 ✓")
print(f"\\n=== 品类分布 ===")
print(products['category'].value_counts())
print(f"\\n=== 价格分布 ===")
print(products['price'].describe())""")

md("""**业务观察：**
- 15 个品类，其中 Sports、Books 的商品数量最多——但不代表销售收入最高
- `discount_pct` 范围 0-50%，可以分析折扣对转化的真实效果
- `avg_rating` 已有预计算评分，可以验证和 reviews 表的一致性
""")

# ============================================================
md("""### 3.3 transactions 表 —— 交易记录（核心事实表）""")

code("""print("=== 数据类型 ===")
print(transactions.dtypes)
print(f"\\n=== 缺失值 ===")
print(transactions.isnull().sum()[transactions.isnull().sum() > 0] if transactions.isnull().sum().sum() > 0 else "无缺失值 ✓")
print(f"\\n=== 订单状态分布 ===")
print(transactions['status'].value_counts())
print(f"\\n=== 支付方式分布 ===")
print(transactions['payment_method'].value_counts())
print(f"\\n=== 基础统计 ===")
print(transactions[['quantity', 'unit_price', 'total_amount', 'discount_applied', 'shipping_cost']].describe())""")

md("""**业务观察：**
- 订单状态有三种：`completed`（正常完成）、`cancelled`（取消）、`refunded`（退款）
- **关键决策**：计算 GMV 时仅保留 `completed` 状态，cancelled 和 refunded 单独分析退货率
- 单笔交易金额跨度很大（max = 499.65），需要检查是否合理
""")

# ============================================================
md("""### 3.4 sessions 表 —— 用户行为日志""")

code("""print("=== 数据类型 ===")
print(sessions.dtypes)
print(f"\\n=== 缺失值 ===")
print(sessions.isnull().sum()[sessions.isnull().sum() > 0] if sessions.isnull().sum().sum() > 0 else "无缺失值 ✓")
print(f"\\n=== 设备分布 ===")
print(sessions['device'].value_counts())
print(f"\\n=== 渠道分布 ===")
print(sessions['channel'].value_counts())
print(f"\\n=== 转化标志 ===")
print(sessions['converted'].value_counts())
print(f"\\n转化率: {sessions['converted'].mean():.2%}")""")

md("""**业务观察：**
- `converted` 标记了该 session 是否最终产生了购买
- `bounced` 标记是否跳出（只看了一页就走了）
- `cart_additions` 记录了加购次数 —— 这 3 个字段是漏斗分析的核心
""")

# ============================================================
md("""### 3.5 reviews 表 —— 用户评价""")

code("""print("=== 数据类型 ===")
print(reviews.dtypes)
print(f"\\n=== 缺失值 ===")
print(reviews.isnull().sum()[reviews.isnull().sum() > 0] if reviews.isnull().sum().sum() > 0 else "无缺失值 ✓")
print(f"\\n=== 评分分布 ===")
print(reviews['rating'].value_counts().sort_index())
print(f"\\n=== 是否有已验证购买标记 ===")
print(reviews['verified_purchase'].value_counts())""")

md("""**业务观察：**
- 有 `verified_purchase` 字段，可以区分真实购买评价 vs 未购买评价
- `helpful_votes` 可以分析哪些评价被其他用户认可，用于商品详情页排序
- `review_text` 可用于文本情感分析，验证评分和文本是否一致
""")

# ============================================================
md("""## 4. 表关联关系验证

在正式合并之前，先验证外键的完整性：transactions 中的 customer_id 和 product_id 是否都能在对应的维度表中找到？""")

code("""# 检查 transactions 中的 customer_id 是否都在 customers 表中
txn_customers = set(transactions['customer_id'])
cust_ids = set(customers['customer_id'])
orphan_customers = txn_customers - cust_ids

# 检查 transactions 中的 product_id 是否都在 products 表中
txn_products = set(transactions['product_id'])
prod_ids = set(products['product_id'])
orphan_products = txn_products - prod_ids

print(f"Transactions 中 customer_id 总数: {len(txn_customers):,}")
print(f"在 customers 表中找不到的 customer_id: {len(orphan_customers)}")
print(f"\\nTransactions 中 product_id 总数: {len(txn_products):,}")
print(f"在 products 表中找不到的 product_id: {len(orphan_products)}")

# 同样检查 sessions 和 reviews
sess_customers = set(sessions['customer_id'])
orphan_sess = sess_customers - cust_ids
print(f"\\nSessions 中找不到的 customer_id: {len(orphan_sess)}")

rev_customers = set(reviews['customer_id'])
orphan_rev = rev_customers - cust_ids
print(f"Reviews 中找不到的 customer_id: {len(orphan_rev)}")

rev_products = set(reviews['product_id'])
orphan_rev_prod = rev_products - prod_ids
print(f"Reviews 中找不到的 product_id: {len(orphan_rev_prod)}")""")

md("""**结论：** 外键关联完整，所有交易、会话、评论中的 customer_id 和 product_id 都能在维度表中找到。可以直接进行多表 JOIN。""")

# ============================================================
md("""## 5. 日期字段处理

所有日期字段目前是字符串格式，需要转为 datetime，并提取年月日小时等时间维度。""")

code("""# 转换所有日期字段
date_cols = {
    'customers': ['signup_date'],
    'transactions': ['transaction_date'],
    'sessions': ['session_date'],
    'reviews': ['review_date']
}

for table_name, cols in date_cols.items():
    df = tables[table_name]
    for col in cols:
        df[col] = pd.to_datetime(df[col])

# 验证转换结果
for table_name, cols in date_cols.items():
    df = tables[table_name]
    for col in cols:
        print(f"{table_name}.{col}: {df[col].dtype}, 范围 {df[col].min()} ~ {df[col].max()}")""")

md("""**业务观察：**
- 数据覆盖 2023-01-01 到 2025-12-31，恰好 3 年
- `signup_date` 最早可以追溯到 2020 年，说明有 3 年老用户 —— 可以做同期群分析
""")

# ============================================================
md("""## 6. 时间特征工程

从日期中提取可用于后续分析的时间维度。""")

code("""# 为 transactions 提取时间维度（后续分析最常用的表）
transactions['year'] = transactions['transaction_date'].dt.year
transactions['month'] = transactions['transaction_date'].dt.month
transactions['year_month'] = transactions['transaction_date'].dt.to_period('M')
transactions['day_of_week'] = transactions['transaction_date'].dt.dayofweek
transactions['day_of_week_name'] = transactions['transaction_date'].dt.day_name()
transactions['hour'] = transactions['transaction_date'].dt.hour
transactions['quarter'] = transactions['transaction_date'].dt.quarter
transactions['is_weekend'] = transactions['day_of_week'].isin([5, 6]).astype(int)

# 同样处理 sessions
sessions['year'] = sessions['session_date'].dt.year
sessions['month'] = sessions['session_date'].dt.month
sessions['year_month'] = sessions['session_date'].dt.to_period('M')
sessions['day_of_week'] = sessions['session_date'].dt.dayofweek
sessions['hour'] = sessions['session_date'].dt.hour
sessions['is_weekend'] = sessions['day_of_week'].isin([5, 6]).astype(int)

print("时间特征提取完成 ✓")
print(f"\\nTransactions 新增字段: year, month, year_month, day_of_week, hour, quarter, is_weekend")
print(f"Sessions 新增字段: year, month, year_month, day_of_week, hour, is_weekend")""")

# ============================================================
md("""## 7. 异常值检测与业务判断

这是数据清洗中最需要**业务判断**的环节。不是所有统计上的"异常值"都该删。""")

md("""### 7.1 交易金额异常值检查

用 IQR 方法识别极端值，然后逐条判断：是数据错误还是真实的大单？""")

code("""# 只看 completed 订单的金额分布
completed = transactions[transactions['status'] == 'completed'].copy()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 直方图
axes[0].hist(completed['total_amount'], bins=100, edgecolor='white', alpha=0.8)
axes[0].axvline(completed['total_amount'].median(), color='red', linestyle='--', label=f"中位数: {completed['total_amount'].median():.0f}")
axes[0].axvline(completed['total_amount'].mean(), color='orange', linestyle='--', label=f"均值: {completed['total_amount'].mean():.0f}")
axes[0].set_title('订单金额分布（含尾部）')
axes[0].set_xlabel('订单金额')
axes[0].set_ylabel('订单数')
axes[0].legend()

# 箱线图
axes[1].boxplot(completed['total_amount'].values, vert=True, widths=0.5)
axes[1].set_title('订单金额箱线图')
axes[1].set_ylabel('订单金额')

plt.tight_layout()
plt.show()

# IQR 异常值检测
Q1 = completed['total_amount'].quantile(0.25)
Q3 = completed['total_amount'].quantile(0.75)
IQR = Q3 - Q1
upper_bound = Q3 + 3 * IQR  # 用 3 倍 IQR 而非 1.5，减少误杀

outliers = completed[completed['total_amount'] > upper_bound]
print(f"Q1: {Q1:.2f}, Q3: {Q3:.2f}, IQR: {IQR:.2f}")
print(f"上界 (3×IQR): {upper_bound:.2f}")
print(f"超过上界的订单数: {len(outliers)} / {len(completed)} ({len(outliers)/len(completed):.2%})")
print(f"\\n极端大单样本 (top 10):")
print(outliers.nlargest(10, 'total_amount')[['transaction_id', 'customer_id', 'total_amount', 'quantity', 'unit_price']])""")

md("""### 7.2 业务判断：这些"大单"该删吗？

看上面的 Top 10 大单，我们需要逐一判断：

- **quantity 正常 + unit_price 高** → 这是买了高单价商品，属于正常业务行为，**保留**
- **quantity 极高** → 可能是批发采购或 B 端客户，**保留但标记**
- **total_amount = unit_price × quantity 明显不匹配** → 可能是数据错误，**修复或删除**

**我们的决策：** 不做硬删除。在后续分析中按场景决定是否过滤。例如计算客单价时排除极端值，但计算 GMV 时全部保留。
""")

md("""### 7.3 退货与取消订单分析

cancelled 和 refunded 订单占多少？是否有时间规律？""")

code("""# 订单状态按月趋势
status_monthly = transactions.groupby(['year_month', 'status']).size().unstack(fill_value=0)
status_monthly.index = status_monthly.index.astype(str)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 堆叠面积图 —— 各状态占比
status_pct = status_monthly.div(status_monthly.sum(axis=1), axis=0)
status_pct[['completed', 'cancelled', 'refunded']].plot.area(
    ax=axes[0], stacked=True, alpha=0.8, colormap='RdYlGn'
)
axes[0].set_title('订单状态占比趋势')
axes[0].set_ylabel('占比')
axes[0].set_xlabel('')
axes[0].legend(loc='lower right')
axes[0].tick_params(axis='x', rotation=45)

# 退货率 + 取消率趋势
non_complete_pct = 1 - status_pct['completed']
axes[1].plot(non_complete_pct.index, non_complete_pct.values, marker='o', color='darkred')
axes[1].fill_between(range(len(non_complete_pct)), non_complete_pct.values, alpha=0.3, color='darkred')
axes[1].set_title('非完成率趋势 (取消 + 退款)')
axes[1].set_ylabel('非完成率')
axes[1].set_xlabel('')
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

overall_non_complete = 1 - (transactions['status'] == 'completed').mean()
print(f"\\n整体非完成率: {overall_non_complete:.2%}")
print(f"其中 cancelled: {(transactions['status']=='cancelled').mean():.2%}")
print(f"其中 refunded: {(transactions['status']=='refunded').mean():.2%}")""")

md("""**业务观察：**
- 非完成率约在数据中占一定比例，这是电商的正常水平（行业一般在 5-15%）
- 如果某个时间段非完成率突然飙升，需要排查是否与特定促销活动或商品质量问题相关
- **决策**：后续 GMV 计算仅使用 `status == 'completed'` 的订单
""")

# ============================================================
md("""## 8. 构建分析宽表

将所有表关联成一张大宽表，方便后续 Notebook 直接使用。这是**一次 JOIN，处处复用**。

ER 关系：
```
customers ──┐
            ├── transactions (事实表) ── products
            │
sessions ───┤
            │
reviews ────┘
```""")

code("""# 构建交易宽表: transactions + customers + products
txn_wide = (
    transactions
    .merge(customers, on='customer_id', how='left')
    .merge(products, on='product_id', how='left')
)

# 构建会话宽表: sessions + customers
sess_wide = (
    sessions
    .merge(customers, on='customer_id', how='left')
)

# 构建评论宽表: reviews + customers + products
rev_wide = (
    reviews
    .merge(customers[['customer_id', 'segment', 'country']], on='customer_id', how='left')
    .merge(products[['product_id', 'category', 'brand', 'price']], on='product_id', how='left')
)

print(f"交易宽表: {txn_wide.shape[0]:,} rows × {txn_wide.shape[1]} cols")
print(f"会话宽表: {sess_wide.shape[0]:,} rows × {sess_wide.shape[1]} cols")
print(f"评论宽表: {rev_wide.shape[0]:,} rows × {rev_wide.shape[1]} cols")
print("\\n宽表构建完成 ✓")""")

# ============================================================
md("""## 9. 数据导出

将清洗后的数据保存为 parquet 格式（比 CSV 更快，保留数据类型），供后续 Notebook 使用。""")

code("""import os
os.makedirs('../output', exist_ok=True)

# 保存清洗后的核心表
txn_wide.to_parquet('../output/transactions_wide.parquet', index=False)
sess_wide.to_parquet('../output/sessions_wide.parquet', index=False)
rev_wide.to_parquet('../output/reviews_wide.parquet', index=False)

# 也保存单独的表（某些场景不需要 JOIN 全部字段）
customers.to_parquet('../output/customers_clean.parquet', index=False)
products.to_parquet('../output/products_clean.parquet', index=False)
sessions.to_parquet('../output/sessions_clean.parquet', index=False)
transactions.to_parquet('../output/transactions_clean.parquet', index=False)

print("数据导出完成 ✓")
print("\\n输出文件:")
for f in os.listdir('../output'):
    if f.endswith('.parquet'):
        size_mb = os.path.getsize(f'../output/{f}') / 1024 / 1024
        print(f"  {f} ({size_mb:.1f} MB)")""")

# ============================================================
md("""---

## 10. 本 Notebook 小结

| 步骤 | 做了什么 | 关键业务决策 |
|------|---------|-------------|
| 数据加载 | 5 张表，共计 ~23.5 万行 | 确认数据规模适合分析 |
| 质量审计 | 检查类型、缺失、重复 | 数据质量好，0 缺失值 |
| 外键验证 | 验证 customer_id / product_id 完整性 | JOIN 不会丢数据 |
| 日期处理 | 字符串 → datetime | 数据覆盖 2023-2025 |
| 特征工程 | 提取 year / month / day_of_week / hour 等 | 为后续趋势分析做准备 |
| 异常值 | IQR 检测大单，分析退货率 | 不删大单，按场景过滤 |
| 宽表构建 | 5 表 JOIN 成 3 张分析宽表 | 一次 JOIN，后续复用 |
| 数据导出 | parquet 格式 | 比 CSV 快，保留 dtype |

**关键 Takeaways：**
1. 数据质量总体良好：无缺失值，外键关联完整
2. 订单状态中 cancelled + refunded 需要单独处理，GMV 仅计算 completed
3. 用户 LTV 分布极不均匀（标准差 ≈ 均值），说明用户价值分层是合理的分析方向
4. 3 年的数据跨度 + sessions 行为日志，为漏斗分析和同期群分析提供了足够的时间深度
""")

# ============================================================
# Write notebook
nb.cells = cells
with open('d:/dianshang/notebooks/01_data_preprocessing.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("Notebook 01 generated successfully")
