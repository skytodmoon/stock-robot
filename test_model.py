import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from tensorflow.keras.models import load_model
from datetime import datetime
import os
from main import DataHandler, AutoTrader

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # macOS系统可用的中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

def test_model_2023():
    # 设置测试参数
    stock_code = "603777"
    start_date = "20230101"
    end_date = "20231231"
    
    # 获取2023年的实际数据
    data_handler = DataHandler(stock_code)
    df = data_handler.get_data(start_date=start_date, end_date=end_date)
    df = data_handler.add_technical_indicators(df)
    
    # 加载最新保存的模型
    model_dir = 'saved_models'
    model_files = [f for f in os.listdir(model_dir) if f.startswith(stock_code) and f.endswith('.keras')]
    if not model_files:
        raise ValueError("没有找到保存的模型文件")
    latest_model = sorted(model_files)[-1]
    model_path = os.path.join(model_dir, latest_model)
    model = load_model(model_path)
    
    # 准备预测数据
    auto_trader = AutoTrader(stock_code)
    df = auto_trader.data_handler.add_technical_indicators(df)
    
    # 添加KDJ指标
    df['K'], df['D'], df['J'] = auto_trader.calculate_kdj(df)
    df['rsi_5'] = auto_trader.calculate_rsi(df['close'], period=5)
    df['rsi_9'] = auto_trader.calculate_rsi(df['close'], period=9)
    
    # 选择特征列
    feature_columns = ['close', 'volume', 'ma5', 'ma20', 'rsi', 'rsi_5', 'rsi_9',
                      'macd', 'signal', 'hist', 'upper', 'middle', 'lower',
                      'cci', 'price_change_rate', 'K', 'D', 'J']
    
    # 数据标准化
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    df[feature_columns] = scaler.fit_transform(df[feature_columns])
    
    # 创建时间窗口特征
    window_size = 5
    X = []
    dates = []
    
    for i in range(len(df) - window_size):
        X.append(df[feature_columns].values[i:(i + window_size)])
        dates.append(df['date'].iloc[i + window_size])
    
    X = np.array(X)
    
    # 生成预测结果
    predictions = model.predict(X)
    
    # 创建结果DataFrame
    results_df = pd.DataFrame({
        'date': dates,
        'actual_price': df['close'].values[window_size:],
        'predicted_price': predictions.flatten()
    })
    
    # 删除包含NaN的行
    results_df = results_df.dropna()
    
    # 绘制对比图
    plt.figure(figsize=(15, 8))
    plt.plot(results_df['date'], results_df['actual_price'], label='实际价格', color='blue')
    plt.plot(results_df['date'], results_df['predicted_price'], label='预测价格', color='red')
    
    plt.title(f'{stock_code} 2023年股票价格预测与实际对比', fontsize=14)
    plt.xlabel('日期', fontsize=12)
    plt.ylabel('价格', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # 设置x轴日期格式
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gcf().autofmt_xdate()
    
    # 保存图表
    plt.savefig(f'{stock_code}_2023_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 计算预测准确性指标
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    mse = mean_squared_error(results_df['actual_price'], results_df['predicted_price'])
    mae = mean_absolute_error(results_df['actual_price'], results_df['predicted_price'])
    r2 = r2_score(results_df['actual_price'], results_df['predicted_price'])
    
    print(f"\n模型评估指标:")
    print(f"均方误差 (MSE): {mse:.4f}")
    print(f"平均绝对误差 (MAE): {mae:.4f}")
    print(f"R2分数: {r2:.4f}")

if __name__ == "__main__":
    test_model_2023()