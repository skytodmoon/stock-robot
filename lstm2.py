import numpy as np
from keras.src.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# 数据加载部分（假设从CSV文件加载）
from lstm import future_predictions, scaler, model


def load_data(csv_path):
    data = pd.read_csv(csv_path)
    # 假设数据包含 'close' 和 'volume' 列
    return data[['close', 'volume']].values


# 数据预处理部分
def preprocess_data(data, time_steps=5):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)

    X = []
    y = []

    for i in range(len(scaled_data) - time_steps):
        X.append(scaled_data[i:i + time_steps])
        y.append(scaled_data[i + time_steps, 1])  # 使用 'close' 作为目标特征

    X = np.array(X)
    y = np.array(y)

    return X, y


def train_model(X_train, y_train, epochs=50, batch_size=32):
    # 打印数据形状，调试用
    print("Training input shape (X_train):", X_train.shape)
    print("Training target shape (y_train):", y_train.shape)

    # 构建 LSTM 模型
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(1)  # 输出层
    ])

    # 编译模型
    model.compile(optimizer='adam', loss='mean_squared_error')

    # 定义早停法
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1)

    # 训练模型
    history = model.fit(X_train, y_train, epochs=epochs,
                        batch_size=batch_size,
                        validation_split=0.2,
                        callbacks=[early_stopping],
                        verbose=1)

    return model


# 测试集预测部分
def predict_model(model, X_test):
    y_test_pred = model.predict(X_test)
    return y_test_pred


# 预测未来值部分
def predict_future(model, test_set, time_steps, future_days=30):
    current_input = test_set[-time_steps:]  # 最后一个时间窗口

    for _ in range(future_days):
        next_prediction = model.predict(current_input.reshape(1, -1))  # 预测下一个时间步的目标值
        next_input = current_input.copy()  # 复制当前输入窗口

        # 更新目标特征 'close' 值
        next_input[0, -1] = next_prediction

        # 移动窗口，去掉第一个样本，加入新的预测值
        current_input = np.vstack([current_input[1:], next_input])

    return current_input  # 包含未来预测的输入窗口


def plot_results(y_test_original, y_test_pred, future_predictions):
    plt.figure(figsize=(12, 6))

    # 绘制实际值
    plt.plot(y_test_original, label='Actual Price')

    # 绘制预测值（最后 30 天）
    plt.plot(y_test_pred[-30:], 'r', label='Predicted Price (Last 30 Days)')

    # 绘制未来预测值（提取最后一列数据）
    if len(future_predictions.shape) > 1 and future_predictions.shape[1] == 1:
        # 如果是二维且只有一列，直接使用
        plt.plot(future_predictions, 'g', label='Future Predictions')
    elif len(future_predictions.shape) > 1:
        # 如果是二维，提取最后一列
        plt.plot(future_predictions[:, -1], 'g', label='Future Predictions')
    else:
        # 如果是一维，直接使用
        plt.plot(future_predictions, 'g', label='Future Predictions')

    # 添加图例和标题
    plt.legend()
    plt.title('Stock Price Prediction')
    plt.xlabel('Time Steps')
    plt.ylabel('Price')

    # 保存图像
    plt.savefig('lstm2_future.png')


# 主函数
def main():
    # 加载数据
    data = load_data('603777_historical_prices.csv')

    # 预处理数据
    X, y = preprocess_data(data, time_steps=5)

    # 划分训练集和测试集
    train_size = int(0.7 * len(X))
    X_train, y_train = X[:train_size], y[:train_size]
    X_test, y_test = X[train_size:], y[train_size:]

    # 训练模型
    model = train_model(X_train, y_train)

    # 测试集预测
    y_test_pred = model.predict(X_test)  # shape: (n_samples, n_features)
    print("Shape of y_test_pred:", y_test_pred.shape)


    # 使用正确的 reshape 操作进行逆变换，保持特征维度不变
    # 假设 y_train 的形状为 (n_samples,)
    scaler = MinMaxScaler(feature_range=(0, 1))

    # 对目标变量 y_train 进行缩放
    y_train_scaled = scaler.fit_transform(y_train.reshape(-1, 1))

    # 在测试阶段，利用训练好的模型进行预测
    y_test_pred = model.predict(X_test)

    # 将预测结果调整为适合 inverse_transform 的形状
    y_test_pred_reshaped = y_test_pred.reshape(-1, 1)

    # 对预测结果进行逆变换
    y_test_pred_inverse_transformed = scaler.inverse_transform(y_test_pred_reshaped)

    # 提取一维数组
    y_test_pred_1d = y_test_pred_inverse_transformed[:, 0]

    # 打印预测结果和 future_predictions 的形状
    print("Shape of y_test_pred_inverse_transformed:", y_test_pred_inverse_transformed.shape)

    # 如果 future_predictions 是二维数组：
    plot_results(y_test, y_test_pred_inverse_transformed, y_test_pred_inverse_transformed)

    # 如果 future_predictions 是一维数组：
    future_predictions_2d = y_test_pred_inverse_transformed.reshape(-1, 1)
    plot_results(y_test, y_test_pred_inverse_transformed, future_predictions_2d)

    # 可视化
    #plot_results(y_test, y_test_pred_inverse_transformed, y_test_pred_inverse_transformed[:, :, 0])


if __name__ == "__main__":
    main()
