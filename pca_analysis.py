import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from factor_analyzer import FactorAnalyzer
from main import DataHandler, AutoTrader
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # macOS系统可用的中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

def analyze_features():
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
    
    # 删除包含NaN的行
    df = df.dropna()
    
    # 数据标准化和预处理
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(df[feature_columns])
    
    # 检查特征相关性
    correlation_matrix = pd.DataFrame(features_scaled, columns=feature_columns).corr()
    
    # 处理高度相关的特征
    high_correlation = 0.95
    high_corr_features = set()
    for i in range(len(correlation_matrix.columns)):
        for j in range(i):
            if abs(correlation_matrix.iloc[i, j]) > high_correlation:
                colname = correlation_matrix.columns[i]
                high_corr_features.add(colname)
    
    # 移除高度相关的特征
    if high_corr_features:
        print("\n移除以下高度相关的特征:")
        print(high_corr_features)
        feature_columns = [f for f in feature_columns if f not in high_corr_features]
        features_scaled = scaler.fit_transform(df[feature_columns])
    
    # 执行PCA分析
    pca = PCA()
    pca_result = pca.fit_transform(features_scaled)
    
    # 计算解释方差比
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
    
    # 绘制碎石图
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, 'bo-')
    plt.plot(range(1, len(cumulative_variance_ratio) + 1), cumulative_variance_ratio, 'ro-')
    plt.xlabel('主成分数量')
    plt.ylabel('解释方差比率')
    plt.title('PCA碎石图')
    plt.grid(True)
    plt.savefig(f'{stock_code}_pca_scree.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 计算特征贡献度
    components = pd.DataFrame(
        pca.components_,
        columns=feature_columns,
        index=[f'PC{i+1}' for i in range(len(feature_columns))]
    )
    
    # 计算每个特征的总体贡献度
    feature_importance = np.abs(components.values).mean(axis=0)
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': feature_importance
    })
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    
    # 绘制特征重要性条形图
    plt.figure(figsize=(12, 6))
    plt.bar(feature_importance['feature'], feature_importance['importance'])
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('特征')
    plt.ylabel('重要性得分')
    plt.title('特征重要性分析 (基于PCA)')
    plt.tight_layout()
    plt.savefig(f'{stock_code}_feature_importance_pca.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 执行因子分析
    # 首先计算相关矩阵的特征值，以确定合适的因子数量
    fa_initial = FactorAnalyzer(rotation=None, n_factors=len(feature_columns))
    fa_initial.fit(features_scaled)
    ev, v = fa_initial.get_eigenvalues()
    n_factors = sum(ev > 1)  # Kaiser准则：特征值大于1的因子数
    
    # 使用确定的因子数重新进行因子分析
    fa = FactorAnalyzer(rotation='varimax', n_factors=n_factors)
    fa.fit(features_scaled)
    
    # 获取因子载荷
    factor_loading = pd.DataFrame(
        fa.loadings_,
        columns=[f'Factor{i+1}' for i in range(n_factors)],
        index=feature_columns
    )
    
    # 计算共同度
    communalities = fa.get_communalities()
    
    # 输出分析结果
    print("\nPCA分析结果:")
    print("\n前5个主成分的解释方差比率:")
    for i, ratio in enumerate(explained_variance_ratio[:5], 1):
        print(f"PC{i}: {ratio:.4f}")
    
    print("\n特征重要性排名 (基于PCA):")
    for idx, row in feature_importance.iterrows():
        print(f"{row['feature']}: {row['importance']:.4f}")
    
    print("\n因子分析结果:")
    print("\n特征共同度 (共同方差解释比例):")
    for feature, communality in zip(feature_columns, communalities):
        print(f"{feature}: {communality:.4f}")

if __name__ == "__main__":
    analyze_features()