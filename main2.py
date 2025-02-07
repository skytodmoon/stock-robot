import pandas as pd
import akshare as ak
from datetime import datetime, timedelta
import numpy as np
import os
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TechnicalIndicators:
    """技术指标计算工具类"""
    @staticmethod
    def calculate_rsi(prices, period=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def calculate_macd(prices, slow=26, fast=12, signal=9):
        exp1 = prices.ewm(span=fast, adjust=False).mean()
        exp2 = prices.ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        histogram = macd - signal_line
        return macd, signal_line, histogram

    @staticmethod
    def calculate_bollinger_bands(prices, window=20, num_std=2):
        middle = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()
        upper = middle + (std * num_std)
        lower = middle - (std * num_std)
        return upper, middle, lower

    @staticmethod
    def calculate_cci(high, low, close, period=20):
        tp = (high + low + close) / 3
        sma = tp.rolling(window=period).mean()
        mad = tp.rolling(window=period).apply(lambda x: abs(x - x.mean()).mean())
        cci = (tp - sma) / (0.015 * mad)
        return cci

    @staticmethod
    def calculate_kdj(df, n=9, m1=3, m2=3):
        low_list = df['low'].rolling(window=n, min_periods=1).min()
        high_list = df['high'].rolling(window=n, min_periods=1).max()
        rsv = (df['close'] - low_list) / (high_list - low_list) * 100
        K = rsv.ewm(com=m1-1, adjust=False).mean()
        D = K.ewm(com=m2-1, adjust=False).mean()
        J = 3 * K - 2 * D
        return K, D, J


class DataHandler:
    """数据处理类"""
    def __init__(self, stock_code, data_dir='stock_data'):
        self.stock_code = stock_code
        self.data_dir = data_dir
        self.data_file = os.path.join(self.data_dir, f'{stock_code}.csv')
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

    def get_data(self, start_date=None, end_date=None):
        """获取股票数据"""
        if os.path.exists(self.data_file):
            logging.info(f"Loading local data for {self.stock_code}...")
            df = pd.read_csv(self.data_file)
            df['date'] = pd.to_datetime(df['date'])
        else:
            logging.info(f"Downloading data for {self.stock_code}...")
            df = ak.stock_zh_a_hist(
                symbol=self.stock_code, period="daily",
                start_date="20170301", end_date="20250127", adjust=""
            )
            df.columns = ['date', 'stock_code', 'open', 'close', 'high', 'low', 'volume', 'amount',
                          'amplitude', 'price_change_rate', 'price_change', 'turnover_rate']
            df['date'] = pd.to_datetime(df['date'])
            df.to_csv(self.data_file, index=False)

        # 按日期筛选
        if start_date:
            df = df[df['date'] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df['date'] <= pd.to_datetime(end_date)]
        return df

    def add_technical_indicators(self, df):
        """添加技术指标"""
        df['ma5'] = df['close'].rolling(window=5).mean()
        df['ma20'] = df['close'].rolling(window=20).mean()
        df['rsi'] = TechnicalIndicators.calculate_rsi(df['close'])
        df['macd'], df['signal'], df['hist'] = TechnicalIndicators.calculate_macd(df['close'])
        df['upper'], df['middle'], df['lower'] = TechnicalIndicators.calculate_bollinger_bands(df['close'])
        df['cci'] = TechnicalIndicators.calculate_cci(df['high'], df['low'], df['close'])
        df['K'], df['D'], df['J'] = TechnicalIndicators.calculate_kdj(df)
        return df


class ModelTrainer:
    """模型训练类"""
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.model = None

    def build_model(self):
        """构建LSTM模型"""
        model = Sequential([
            LSTM(100, input_shape=self.input_shape, return_sequences=True,
                 kernel_regularizer=l2(0.01), recurrent_regularizer=l2(0.01)),
            Dropout(0.2),
            LSTM(50, kernel_regularizer=l2(0.01), recurrent_regularizer=l2(0.01)),
            Dropout(0.2),
            Dense(25, activation='relu', kernel_regularizer=l2(0.01)),
            Dropout(0.1),
            Dense(self.output_shape)
        ])
        optimizer = Adam(learning_rate=0.001, clipnorm=1.0)
        model.compile(optimizer=optimizer, loss='mean_squared_error')
        return model



    def train_model(self, X_train, y_train, X_val, y_val, sample_weights=None, epochs=100, batch_size=32, callbacks=None):
        """训练模型"""
        self.model = self.build_model()
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            sample_weight=sample_weights,  # 添加 sample_weight 参数
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks if callbacks else [],  # 添加 callbacks 参数
            verbose=1
        )
        return history


    def save_model(self, stock_code):
        """保存模型并返回模型路径"""
        if not self.model:
            raise ValueError("Model has not been trained yet.")
        model_dir = 'saved_models'
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_path = os.path.join(model_dir, f'{stock_code}_model_{timestamp}.keras')
        self.model.save(model_path)
        logging.info(f"Model saved to {model_path}")
        return model_path  # 返回模型路径


    def predict(self, X_test):
        """模型预测"""
        if not self.model:
            raise ValueError("Model has not been trained yet.")
        return self.model.predict(X_test)


class AutoTrader:
    """自动交易类"""
    def __init__(self, stock_code, balance=100000):
        self.stock_code = stock_code
        self.balance = balance
        self.data_handler = DataHandler(stock_code)
        self.df = None
        self.scaler = StandardScaler()

    def prepare_data(self):
        """准备训练数据"""
        end_date = datetime.now().strftime('%Y%m%d')
        start_date = (datetime.now() - timedelta(days=90)).strftime('%Y%m%d')
        self.df = self.data_handler.get_data(start_date=start_date, end_date=end_date)
        self.df = self.data_handler.add_technical_indicators(self.df)

        # 添加短期RSI
        self.df['rsi_5'] = TechnicalIndicators.calculate_rsi(self.df['close'], period=5)
        self.df['rsi_9'] = TechnicalIndicators.calculate_rsi(self.df['close'], period=9)

        # 选择特征列
        feature_columns = ['close', 'volume', 'ma5', 'ma20', 'rsi', 'rsi_5', 'rsi_9',
                          'macd', 'signal', 'hist', 'upper', 'middle', 'lower',
                          'cci', 'price_change_rate', 'K', 'D', 'J']

        # 删除缺失值
        self.df = self.df.dropna()

        # 标准化特征
        self.df[feature_columns] = self.scaler.fit_transform(self.df[feature_columns])

        # 创建时间窗口特征
        window_size = 5
        X, y, weights = [], [], []
        for i in range(len(self.df) - window_size):
            X.append(self.df[feature_columns].values[i:(i + window_size)])
            y.append(self.df['close'].values[i + window_size])
            weights.append(np.exp(-0.1 * (len(self.df) - window_size - i)))

        X = np.array(X)
        y = np.array(y)
        weights = np.array(weights)

        # 划分训练集和验证集
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        return X_train, X_val, y_train, y_val, weights[:len(X_train)]

    def generate_signals(self, model, X_test, future_dates):
        """生成交易信号"""
        try:
            y_pred = model.predict(X_test)
            confidence_threshold = 0.02
            signals = []
            prev_signal = '持有'

            results_df = pd.DataFrame({
                'date': future_dates,
                'predicted_price': y_pred.flatten(),
                'actual_price': [self.df['close'].iloc[-1]] * len(y_pred)
            })

            for i in range(1, len(y_pred)):
                price_change = (y_pred[i] - y_pred[i - 1]) / y_pred[i - 1]
                if price_change > confidence_threshold and prev_signal != '买入':
                    signals.append('买入')
                    prev_signal = '买入'
                elif price_change < -confidence_threshold and prev_signal != '卖出':
                    signals.append('卖出')
                    prev_signal = '卖出'
                else:
                    signals.append('持有')
                    prev_signal = '持有'

            results_df['signal'] = ['持有'] + signals
            results_file = os.path.join(self.data_handler.data_dir, f'{self.stock_code}_predictions.csv')
            results_df.to_csv(results_file, index=False)
            logging.info(f"Predictions saved to {results_file}")
            return signals
        except Exception as e:
            logging.error(f"Error during prediction: {e}")
            return []

    def execute_trade(self, signals):
        """执行交易"""
        current_price = self.df['close'].iloc[-1]
        for signal in signals:
            if signal == '买入':
                shares_bought = self.balance // current_price
                self.balance -= shares_bought * current_price
                logging.info(f"Bought {shares_bought} shares at {current_price}. Remaining balance: {self.balance}")
            elif signal == '卖出':
                shares_sold = int(input('请输入卖出的股数：'))
                if shares_sold > 0:
                    self.balance += shares_sold * current_price
                    logging.info(f"Sold {shares_sold} shares at {current_price}. Remaining balance: {self.balance}")
            else:
                logging.info("Holding position, no trade executed.")

import matplotlib.pyplot as plt

class ModelTester:
    """模型测试类"""
    def __init__(self, stock_code, model_path, output_dir='test_results'):
        self.stock_code = stock_code
        self.model_path = model_path
        self.output_dir = output_dir
        self.data_handler = DataHandler(stock_code)
        self.scaler = StandardScaler()
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def load_model(self):
        """加载训练好的模型"""
        from tensorflow.keras.models import load_model
        return load_model(self.model_path)

    def prepare_test_data(self):
        """准备测试数据"""
        end_date = datetime.now().strftime('%Y%m%d')
        start_date = (datetime.now() - timedelta(days=180)).strftime('%Y%m%d')  # 使用6个月的数据
        df = self.data_handler.get_data(start_date=start_date, end_date=end_date)
        df = self.data_handler.add_technical_indicators(df)

        # 添加短期RSI
        df['rsi_5'] = TechnicalIndicators.calculate_rsi(df['close'], period=5)
        df['rsi_9'] = TechnicalIndicators.calculate_rsi(df['close'], period=9)

        # 选择特征列
        feature_columns = ['close', 'volume', 'ma5', 'ma20', 'rsi', 'rsi_5', 'rsi_9',
                          'macd', 'signal', 'hist', 'upper', 'middle', 'lower',
                          'cci', 'price_change_rate', 'K', 'D', 'J']

        # 删除缺失值
        df = df.dropna()

        # 标准化特征
        df[feature_columns] = self.scaler.fit_transform(df[feature_columns])

        # 创建时间窗口特征
        window_size = 5
        X, y = [], []
        for i in range(len(df) - window_size):
            X.append(df[feature_columns].values[i:(i + window_size)])
            y.append(df['close'].values[i + window_size])

        X = np.array(X)
        y = np.array(y)
        return X, y, df['date'].iloc[window_size:]

    def evaluate_model(self):
        """评估模型表现"""
        model = self.load_model()
        X_test, y_test, dates = self.prepare_test_data()

        # 预测价格
        y_pred = model.predict(X_test).flatten()

        # 计算误差
        mse = np.mean((y_test - y_pred) ** 2)
        mae = np.mean(np.abs(y_test - y_pred))
        logging.info(f"Test MSE: {mse:.4f}, Test MAE: {mae:.4f}")

        # 绘制实际价格与预测价格
        plt.figure(figsize=(12, 6))
        plt.plot(dates, y_test, label='Actual Price', color='blue')
        plt.plot(dates, y_pred, label='Predicted Price', color='red', linestyle='--')
        plt.title(f'{self.stock_code} - Actual vs Predicted Price')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(self.output_dir, f'{self.stock_code}_actual_vs_predicted.png'))  # 保存图片
        plt.show()

        # 评估交易策略
        self.evaluate_trading_strategy(y_test, y_pred, dates)

    def evaluate_trading_strategy(self, y_test, y_pred, dates):
        """评估交易策略表现"""
        signals = []
        prev_signal = '持有'
        confidence_threshold = 0.02

        for i in range(1, len(y_pred)):
            price_change = (y_pred[i] - y_pred[i - 1]) / y_pred[i - 1]
            if price_change > confidence_threshold and prev_signal != '买入':
                signals.append('买入')
                prev_signal = '买入'
            elif price_change < -confidence_threshold and prev_signal != '卖出':
                signals.append('卖出')
                prev_signal = '卖出'
            else:
                signals.append('持有')
                prev_signal = '持有'

        # 计算策略收益
        balance = 100000
        shares = 0
        portfolio_values = []

        for i, signal in enumerate(signals):
            if signal == '买入' and balance >= y_test[i]:
                shares = balance // y_test[i]
                balance -= shares * y_test[i]
            elif signal == '卖出' and shares > 0:
                balance += shares * y_test[i]
                shares = 0
            portfolio_values.append(balance + shares * y_test[i])

        # 绘制策略收益
        plt.figure(figsize=(12, 6))
        plt.plot(dates[1:], portfolio_values, label='Portfolio Value', color='green')
        plt.title(f'{self.stock_code} - Trading Strategy Performance')
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(self.output_dir, f'{self.stock_code}_trading_strategy.png'))  # 保存图片
        plt.show()



if __name__ == "__main__":
    # 初始化参数
    stock_code = "603777"
    balance = 100000

    # 初始化自动交易器
    auto_trader = AutoTrader(stock_code, balance)
    X_train, X_val, y_train, y_val, sample_weights = auto_trader.prepare_data()

    # 初始化模型训练器
    model_trainer = ModelTrainer(input_shape=(X_train.shape[1], X_train.shape[2]), output_shape=1)

    # 训练模型
    early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
    history = model_trainer.train_model(
        X_train, y_train, X_val, y_val,
        sample_weights=sample_weights,
        epochs=300,
        callbacks=[early_stopping]
    )

    # 保存模型并获取模型路径
    model_path = model_trainer.save_model(stock_code)

    # 测试模型
    model_tester = ModelTester(stock_code, model_path=model_path)  # 使用动态获取的模型路径
    model_tester.evaluate_model()


