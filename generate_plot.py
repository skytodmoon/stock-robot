import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # macOS系统可用的中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 读取预测数据
predictions_df = pd.read_csv('stock_data/603777_predictions.csv')

# 将日期列转换为datetime格式
predictions_df['date'] = pd.to_datetime(predictions_df['date'])

# 创建图表
plt.figure(figsize=(15, 8))

# 绘制预测价格和实际价格的线图
plt.plot(predictions_df['date'], predictions_df['predicted_price'], label='预测价格', color='blue')
plt.plot(predictions_df['date'], predictions_df['actual_price'], label='实际价格', color='red')

# 标注交易信号
for idx, row in predictions_df.iterrows():
    if row['signal'] == '买入':
        plt.scatter(row['date'], row['predicted_price'], color='green', marker='^', s=100, label='买入' if idx == 0 else '')
    elif row['signal'] == '卖出':
        plt.scatter(row['date'], row['predicted_price'], color='red', marker='v', s=100, label='卖出' if idx == 0 else '')

# 设置图表格式
plt.title('603777股票价格预测与实际对比', fontsize=14)
plt.xlabel('日期', fontsize=12)
plt.ylabel('价格变化率', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

# 设置x轴日期格式
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gcf().autofmt_xdate()

# 保存图表
plt.savefig('603777_price_predictions.png', dpi=300, bbox_inches='tight')
plt.close()