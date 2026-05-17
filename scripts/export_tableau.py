"""导出 Tableau 可视化宽表（中文命名）
运行完所有 Notebook 后执行（需要 output/*.parquet 文件）
"""
import pandas as pd
import os

print("加载数据...")

txn = pd.read_parquet('output/transactions_wide.parquet')
sessions = pd.read_parquet('output/sessions_wide.parquet')
customers = pd.read_parquet('output/customers_clean.parquet')

# ============================================================
# Sheet 1: 月度KPI
# ============================================================
print("生成: 月度KPI...")
completed = txn[txn['status'] == 'completed']

# 已完成订单的月度指标
monthly_kpi = completed.groupby(completed['transaction_date'].dt.to_period('M')).agg(
    销售额=('total_amount', 'sum'),
    订单量=('transaction_id', 'nunique'),
    客户数=('customer_id', 'nunique'),
    客单价=('total_amount', 'mean'),
    平均购买件数=('quantity', 'mean')
).reset_index()

# 用全量订单计算退货/取消率（不能用 completed 子集算）
monthly_status = txn.groupby(txn['transaction_date'].dt.to_period('M')).agg(
    总订单=('transaction_id', 'nunique'),
    已完成=('status', lambda x: (x == 'completed').sum()),
    已取消=('status', lambda x: (x == 'cancelled').sum()),
    已退款=('status', lambda x: (x == 'refunded').sum()),
    挂起中=('status', lambda x: (x == 'pending').sum())
).reset_index()
monthly_status['退货退款率'] = (
    (monthly_status['已取消'] + monthly_status['已退款']) / monthly_status['总订单'] * 100
).round(2)
monthly_status['完成率'] = (monthly_status['已完成'] / monthly_status['总订单'] * 100).round(2)

# 合并
monthly_kpi = monthly_kpi.merge(
    monthly_status[['transaction_date', '总订单', '已取消', '已退款', '挂起中', '退货退款率', '完成率']],
    on='transaction_date', how='left'
)

monthly_kpi['月份'] = monthly_kpi['transaction_date'].astype(str)
monthly_kpi.drop(columns=['transaction_date'], inplace=True)
monthly_kpi = monthly_kpi[['月份', '销售额', '订单量', '客户数', '客单价', '平均购买件数',
                            '总订单', '完成率', '退货退款率', '已取消', '已退款', '挂起中']]
monthly_kpi['销售额'] = monthly_kpi['销售额'].round(2)
monthly_kpi['客单价'] = monthly_kpi['客单价'].round(2)

# ============================================================
# Sheet 2: 品类表现
# ============================================================
print("生成: 品类表现...")
category_kpi = completed.groupby('category').agg(
    销售额=('total_amount', 'sum'),
    订单量=('transaction_id', 'nunique'),
    客单价=('total_amount', 'mean'),
    售出商品数=('product_id', 'nunique')
).reset_index()
category_kpi['销售占比'] = (category_kpi['销售额'] / category_kpi['销售额'].sum() * 100).round(2)
category_kpi['累计占比'] = category_kpi['销售占比'].cumsum().round(2)
category_kpi.columns = ['品类', '销售额', '订单量', '客单价', '售出商品数', '销售占比', '累计占比']
category_kpi['销售额'] = category_kpi['销售额'].round(2)
category_kpi['客单价'] = category_kpi['客单价'].round(2)

# ============================================================
# Sheet 3: 客户RFM
# ============================================================
print("生成: 客户RFM...")
NOW = completed['transaction_date'].max() + pd.Timedelta(days=1)
rfm = completed.groupby('customer_id').agg(
    最近购买距今天数=('transaction_date', lambda x: (NOW - x.max()).days),
    购买次数=('transaction_id', 'nunique'),
    消费总金额=('total_amount', 'sum')
).reset_index()
rfm.rename(columns={'customer_id': '客户ID'}, inplace=True)
rfm = rfm.merge(
    customers[['customer_id', 'segment', 'country', 'is_churned', 'lifetime_value']].rename(columns={
        'customer_id': '客户ID',
        'segment': '客户分层',
        'country': '国家',
        'is_churned': '是否流失',
        'lifetime_value': '生命周期价值'
    }),
    on='客户ID', how='left'
)
rfm['消费总金额'] = rfm['消费总金额'].round(2)
rfm['生命周期价值'] = rfm['生命周期价值'].round(2)

# ============================================================
# Sheet 4: 每日销售
# ============================================================
print("生成: 每日销售...")
daily_sales = completed.groupby(completed['transaction_date'].dt.date).agg(
    销售额=('total_amount', 'sum'),
    订单量=('transaction_id', 'nunique'),
    客户数=('customer_id', 'nunique')
).reset_index()
daily_sales.columns = ['日期', '销售额', '订单量', '客户数']
daily_sales['销售额'] = daily_sales['销售额'].round(2)

# ============================================================
# Sheet 5: 支付方式 & 设备漏斗
# ============================================================
print("生成: 支付方式 & 设备漏斗...")
payment_stats = completed.groupby('payment_method').agg(
    订单量=('transaction_id', 'nunique'),
    销售额=('total_amount', 'sum'),
    客单价=('total_amount', 'mean')
).reset_index()
payment_stats.columns = ['支付方式', '订单量', '销售额', '客单价']
payment_stats['销售额'] = payment_stats['销售额'].round(2)
payment_stats['客单价'] = payment_stats['客单价'].round(2)

device_funnel = sessions.groupby('device').agg(
    会话数=('session_id', 'nunique'),
    转化数=('converted', 'sum'),
    跳出数=('bounced', 'sum'),
    加购数=('cart_additions', 'sum')
).reset_index()
device_funnel['转化率'] = (device_funnel['转化数'] / device_funnel['会话数'] * 100).round(2)
device_funnel.columns = ['设备', '会话数', '转化数', '跳出数', '加购数', '转化率']

# ============================================================
# 导出 Excel (一个文件, 多个 Sheet)
# ============================================================
output_path = 'output/tableau_data.xlsx'
print(f"\n导出到 {output_path}...")
with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
    monthly_kpi.to_excel(writer, sheet_name='月度KPI', index=False)
    category_kpi.to_excel(writer, sheet_name='品类表现', index=False)
    rfm.to_excel(writer, sheet_name='客户RFM', index=False)
    daily_sales.to_excel(writer, sheet_name='每日销售', index=False)
    payment_stats.to_excel(writer, sheet_name='支付方式', index=False)
    device_funnel.to_excel(writer, sheet_name='设备漏斗', index=False)

# ============================================================
# 同时导出 CSV（Tableau Public 可能只支持 CSV）
# ============================================================
exports = [
    ('tableau_01_月度KPI', monthly_kpi),
    ('tableau_02_品类表现', category_kpi),
    ('tableau_03_客户RFM', rfm),
    ('tableau_04_每日销售', daily_sales),
]
for name, df in exports:
    csv_path = f'output/{name}.csv'
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"  -> {csv_path}")

print(f"\n导出完成! output/ 下共 {len(os.listdir('output'))} 个文件")
print("将 output/*.csv 或 output/tableau_data.xlsx 导入 Tableau 即可")
