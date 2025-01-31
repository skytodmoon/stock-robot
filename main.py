import pandas as pd
import akshare as ak
from datetime import datetime, timedelta
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import os


class DataHandler:
    def __init__(self, stock_code, data_dir='stock_data'):
        self.stock_code = stock_code
        self.data_dir = data_dir
        self.data_file = os.path.join(self.data_dir, f'{stock_code}.csv')  # 本地数据文件路径

    def get_data(self, start_date=None, end_date=None):
        # 检查是否存在本地文件
        if os.path.exists(self.data_file):
            print(f"Found local data for {self.stock_code}, loading from file...")
            df = pd.read_csv(self.data_file)
            df['date'] = pd.to_datetime(df['date'])
        else:
            # 从互联网获取数据
            print(f"Downloading data for {self.stock_code} from the internet...")
            df = ak.stock_zh_a_hist(
                symbol=self.stock_code,  # 股票代码
                period="daily",  # 数据周期（日线数据）
                start_date="20170301",  # 开始日期
                end_date="20250127",  # 结束日期
                adjust=""  # 不进行复权处理
            )
            df.columns = ['date', 'stock_code', 'open', 'close', 'high', 'low', 'volume', 'amount',
                          'amplitude', 'price_change_rate', 'price_change', 'turnover_rate']
            df['date'] = pd.to_datetime(df['date'])
            # 保存到本地
            if not os.path.exists(self.data_dir):
                os.makedirs(self.data_dir)  # 如果目录不存在，则创建目录
            df.to_csv(self.data_file, index=False)

        # 按照给定的日期范围进行筛选
        if start_date is not None:
            df = df[df['date'] >= pd.to_datetime(start_date)]
        if end_date is not None:
            df = df[df['date'] <= pd.to_datetime(end_date)]

        return df

    def add_technical_indicators(self, df):
        # 添加技术指标（例如移动平均）
        df['ma5'] = df['close'].rolling(window=5).mean()  # 5日移动平均
        df['ma20'] = df['close'].rolling(window=20).mean()  # 20日移动平均
        return df


class ModelTrainer:
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.model = None  # 初始化时模型为 None，防止引用未定义的模型

    def build_model(self):
        model = Sequential([
            LSTM(50, input_shape=self.input_shape, return_sequences=True),
            LSTM(25),
            Dense(self.output_shape)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model

    def train_model(self, X_train, y_train, epochs=100, batch_size=32):
        # 如果 model 已经存在，先删除原模型
        if self.model is not None:
            del self.model

        self.model = self.build_model()  # 构建并保存模型
        history = self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
        return history

    def predict(self, X_test):
        if self.model is None:
            raise ValueError("Model has not been trained yet. Please train the model first.")
        return self.model.predict(X_test)


class AutoTrader:
    def __init__(self, stock_code, balance=100000):
        self.stock_code = stock_code
        self.balance = balance

        # 初始化数据
        self.data_handler = DataHandler(stock_code)
        df = self.data_handler.get_data()
        self.df = df

        # 准备数据
        self.prepare_data()

    def prepare_data(self):
        # 添加技术指标
        self.df = self.data_handler.add_technical_indicators(self.df)

        # 使用MinMaxScaler对数据进行归一化处理
        scaler = MinMaxScaler(feature_range=(0, 1))
        self.df[['close', 'volume', 'ma5', 'ma20']] = scaler.fit_transform(self.df[['close', 'volume', 'ma5', 'ma20']])

        # 提取特征和标签
        X = self.df[['close', 'volume', 'ma5', 'ma20']].values
        y = self.df['close'].shift(-1).dropna().values  # 预测下一个收盘价
        X = X[:-1]  # 对应调整X的长度

        # 将数据转换为LSTM需要的格式：样本数量，时间步长，特征数量
        X = X.reshape(X.shape[0], 1, X.shape[1])  # 1时间步

        return X, y

    def generate_signals(self, model, X_test):
        # 预测并生成信号
        try:
            y_pred = model.predict(X_test)
        except Exception as e:
            print(f"Error during prediction: {e}")
            print(f"X_test types: {X_test.dtypes}")
            print(f"X_test shape: {X_test.shape}")

        signals = []

        for i in range(1, len(y_pred)):
            if y_pred[i] > y_pred[i - 1]:
                signals.append('买入')
            elif y_pred[i] < y_pred[i - 1]:
                signals.append('卖出')
            else:
                signals.append('持有')

        return signals

    def execute_trade(self, signals):
        current_price = self.df['close'].iloc[-1]

        for signal in signals:
            if signal == '买入':
                shares_bought = self.balance // current_price
                print(f'当前资金：{self.balance}元，以{current_price}元买入{shares_bought}股')
                self.balance -= shares_bought * current_price
            elif signal == '卖出':
                shares_sold = int(input('请输入卖出的股数：'))
                if shares_sold > 0:
                    print(f'当前资金：{self.balance + shares_sold * current_price}元，以{current_price}元卖出{shares_sold}股')
                    self.balance += shares_sold * current_price
            else:
                print("持有，无交易")

    def download_data(self, start_date=None, end_date=None):
        self.df = self.data_handler.get_data(start_date=start_date, end_date=end_date)
        if self.df.empty:
            raise ValueError("No data downloaded")
        self.prepare_data()


if __name__ == "__main__":
    # 示例使用
    stock_code = "603777"  # 替换为实际股票代码
    balance = 100000

    auto_trader = AutoTrader(stock_code, balance)
    auto_trader.download_data()  # 设置下载时间范围，确保有足够的数据用于训练和测试

    # 训练模型
    X_train, y_train = auto_trader.prepare_data()
    model_trainer = ModelTrainer(input_shape=(X_train.shape[1], X_train.shape[2]), output_shape=1)
    history = model_trainer.train_model(X_train, y_train)

    # 使用模型生成信号
    future_dates = pd.date_range(start=auto_trader.df['date'].max() + pd.Timedelta(days=1),
                                 end=auto_trader.df['date'].max() + pd.Timedelta(days=30))
    future_df = pd.DataFrame(index=future_dates, columns=['date', 'close', 'volume', 'ma5', 'ma20'])
    future_df['date'] = future_df.index
    future_df['date'] = pd.to_datetime(future_df['date'])

    # 获取未来数据并确保数据类型正确
    future_df[['close', 'volume']] = np.nan  # 填充为 NaN，表示未来的价格和交易量不可知
    X_future = future_df[['close', 'volume', 'ma5', 'ma20']].values

    # 强制转换数据类型为 float32（或者 float64）
    X_future = X_future.astype('float32')

    # 填充缺失的 NaN 值（如果有的话）
    X_future = np.nan_to_num(X_future, nan=0.0)

    # 调整形状为三维输入 (样本数, 时间步数, 特征数)
    X_future = X_future.reshape(X_future.shape[0], 1, X_future.shape[1])

    # 使用训练好的模型生成信号
    signals = auto_trader.generate_signals(model_trainer.model, X_future)

    # 执行交易
    auto_trader.execute_trade(signals)