import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from tensorflow.keras.models import load_model
from main import DataHandler, AutoTrader
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # macOS系统可用的中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

def analyze_feature_importance():
    # 设置参数
    stock_code = "603777"
    start_date = "20230101"
    end_date = "20231231"
    
    # 获取数据
    data_handler = DataHandler(stock_code)
    df = data_handler.get_data(start_date=start_date, end_date=end_date)
    df = data_handler.add_technical_indicators(df)
    
    # 准备数据
    auto_trader = AutoTrader(stock_code)
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
    y = []
    
    for i in range(len(df) - window_size):
        X.append(df[feature_columns].values[i:(i + window_size)])
        y.append(df['close'].values[i + window_size])
    
    X = np.array(X)
    y = np.array(y)
    
    # 加载最新保存的模型
    model_dir = 'saved_models'
    model_files = [f for f in os.listdir(model_dir) if f.startswith(stock_code) and f.endswith('.keras')]
    if not model_files:
        raise ValueError("没有找到保存的模型文件")
    latest_model = sorted(model_files)[-1]
    model_path = os.path.join(model_dir, latest_model)
    model = load_model(model_path)
    
    # 重塑输入数据为2D格式
    n_samples = X.shape[0]
    n_features = X.shape[1] * X.shape[2]
    X_reshaped = X.reshape(n_samples, n_features)
    
    # 定义评分函数（使用负MSE，因为sklearn假设更高的分数更好）
    def custom_score(estimator, X, y):
        X_3d = X.reshape(-1, window_size, len(feature_columns))
        y_pred = estimator.predict(X_3d)
        return -np.mean((y - y_pred) ** 2)
    
    # 计算特征重要性
    result = permutation_importance(
        model, X_reshaped, y,
        scoring=custom_score,
        n_repeats=10,
        random_state=42
    )
    
    # 创建特征重要性DataFrame，合并时间窗口的特征
    all_features = [f"{col}_t{t}" for col in feature_columns for t in range(window_size)]
    importance_df = pd.DataFrame({
        'feature': all_features,
        'importance': result.importances_mean
    })
    
    # 按重要性降序排序
    importance_df = importance_df.sort_values('importance', ascending=False)
    
    # 绘制特征重要性条形图
    plt.figure(figsize=(12, 8))
    plt.bar(importance_df['feature'], importance_df['importance'])
    plt.xticks(rotation=45, ha='right')
    plt.title('特征重要性分析', fontsize=14)
    plt.xlabel('特征', fontsize=12)
    plt.ylabel('重要性得分', fontsize=12)
    plt.tight_layout()
    
    # 保存图表
    plt.savefig(f'{stock_code}_feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 打印特征重要性排名
    print("\n特征重要性排名:")
    for idx, row in importance_df.iterrows():
        print(f"{row['feature']}: {row['importance']:.4f}")

if __name__ == "__main__":
    analyze_feature_importance()