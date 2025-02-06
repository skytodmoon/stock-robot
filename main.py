import pandas as pd
import akshare as ak
from datetime import datetime, timedelta
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
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
        # 添加技术指标
        df['ma5'] = df['close'].rolling(window=5).mean()  # 5日移动平均
        df['ma20'] = df['close'].rolling(window=20).mean()  # 20日移动平均
        df['rsi'] = self.calculate_rsi(df['close'])  # RSI指标
        df['macd'], df['signal'], df['hist'] = self.calculate_macd(df['close'])  # MACD指标
        df['upper'], df['middle'], df['lower'] = self.calculate_bollinger_bands(df['close'])  # 布林带
        df['cci'] = self.calculate_cci(df['high'], df['low'], df['close'])  # CCI指标
        return df

    def calculate_rsi(self, prices, period=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def calculate_macd(self, prices, slow=26, fast=12, signal=9):
        exp1 = prices.ewm(span=fast, adjust=False).mean()
        exp2 = prices.ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        histogram = macd - signal_line
        return macd, signal_line, histogram

    def calculate_bollinger_bands(self, prices, window=20, num_std=2):
        middle = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()
        upper = middle + (std * num_std)
        lower = middle - (std * num_std)
        return upper, middle, lower

    def calculate_cci(self, high, low, close, period=20):
        tp = (high + low + close) / 3
        sma = tp.rolling(window=period).mean()
        # 使用更新的方法计算平均绝对偏差
        mad = tp.rolling(window=period).apply(lambda x: abs(x - x.mean()).mean())
        cci = (tp - sma) / (0.015 * mad)
        return cci


class ModelTrainer:
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.model = None

    def build_model(self):
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.regularizers import l2
        
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

    def save_model(self, stock_code):
        if self.model is None:
            raise ValueError("No model to save. Please train the model first.")
        
        # 创建模型保存目录
        model_dir = 'saved_models'
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        # 生成包含股票代码和时间戳的文件名
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_path = os.path.join(model_dir, f'{stock_code}_model_{timestamp}.keras')
        
        # 保存模型
        self.model.save(model_path)
        print(f'模型已保存到：{model_path}')

    def train_model(self, X_train, y_train, sample_weights=None, epochs=100, batch_size=32, callbacks=None):
        if self.model is not None:
            del self.model

        self.model = self.build_model()
        history = self.model.fit(X_train, y_train, 
                               sample_weight=sample_weights,
                               epochs=epochs, 
                               batch_size=batch_size, 
                               callbacks=callbacks,
                               verbose=1)
        return history

    def predict(self, X_test):
        if self.model is None:
            raise ValueError("Model has not been trained yet. Please train the model first.")
        return self.model.predict(X_test)


class AutoTrader:
    def __init__(self, stock_code, balance=100000):
        self.stock_code = stock_code
        self.balance = balance
        self.data_handler = DataHandler(stock_code)
        self.df = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def prepare_data(self):
        if self.df is None:
            # 获取最近3个月的数据
            end_date = datetime.now().strftime('%Y%m%d')
            start_date = (datetime.now() - timedelta(days=90)).strftime('%Y%m%d')
            self.df = self.data_handler.get_data(start_date=start_date, end_date=end_date)
            self.df = self.data_handler.add_technical_indicators(self.df)

        # 添加KDJ指标
        self.df['K'], self.df['D'], self.df['J'] = self.calculate_kdj(self.df)
        
        # 添加短周期RSI
        self.df['rsi_5'] = self.calculate_rsi(self.df['close'], period=5)
        self.df['rsi_9'] = self.calculate_rsi(self.df['close'], period=9)

        # 选择特征列，包含新增的短期指标
        feature_columns = ['close', 'volume', 'ma5', 'ma20', 'rsi', 'rsi_5', 'rsi_9',
                          'macd', 'signal', 'hist', 'upper', 'middle', 'lower',
                          'cci', 'price_change_rate', 'K', 'D', 'J']

        # 删除包含NaN的行
        self.df = self.df.dropna()

        # 使用StandardScaler对数据进行标准化处理
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        self.df[feature_columns] = scaler.fit_transform(self.df[feature_columns])

        # 创建时间窗口特征，使用较短的时间窗口
        window_size = 5  # 减小窗口大小以更好地捕捉短期特征
        X = []
        y = []
        weights = []

        for i in range(len(self.df) - window_size):
            X.append(self.df[feature_columns].values[i:(i + window_size)])
            y.append(self.df['close'].values[i + window_size])
            # 添加时间衰减权重，最近的数据权重更大
            weight = np.exp(-0.1 * (len(self.df) - window_size - i))
            weights.append(weight)

        X = np.array(X)
        y = np.array(y)
        weights = np.array(weights)

        # 划分训练集和验证集
        train_size = int(len(X) * 0.8)
        X_train = X[:train_size]
        y_train = y[:train_size]
        sample_weights = weights[:train_size]

        return X_train, y_train, sample_weights

    def calculate_kdj(self, df, n=9, m1=3, m2=3):
        low_list = df['low'].rolling(window=n, min_periods=1).min()
        high_list = df['high'].rolling(window=n, min_periods=1).max()
        rsv = (df['close'] - low_list) / (high_list - low_list) * 100

        K = rsv.ewm(com=m1-1, adjust=False).mean()
        D = K.ewm(com=m2-1, adjust=False).mean()
        J = 3 * K - 2 * D

        return K, D, J

    def calculate_rsi(self, prices, period):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def generate_signals(self, model, X_test, future_dates):
        try:
            y_pred = model.predict(X_test)
            # 添加预测值的置信度阈值
            confidence_threshold = 0.02  # 2%的变化阈值
            
            signals = []
            prev_signal = '持有'  # 记录前一个信号
            
            # 创建预测结果DataFrame
            results_df = pd.DataFrame({
                'date': future_dates,
                'predicted_price': y_pred.flatten(),
                'actual_price': [self.df['close'].iloc[-1]] * len(y_pred)  # 使用最新的实际价格
            })
            
            for i in range(1, len(y_pred)):
                price_change = (y_pred[i] - y_pred[i - 1]) / y_pred[i - 1]
                
                # 根据价格变化幅度和前一个信号来决定交易信号
                if price_change > confidence_threshold and prev_signal != '买入':
                    signals.append('买入')
                    prev_signal = '买入'
                elif price_change < -confidence_threshold and prev_signal != '卖出':
                    signals.append('卖出')
                    prev_signal = '卖出'
                else:
                    signals.append('持有')
                    prev_signal = '持有'
            
            # 添加交易信号到结果DataFrame
            results_df['signal'] = ['持有'] + signals  # 第一个信号设为'持有'
            
            # 保存预测结果到CSV文件
            results_file = os.path.join(self.data_handler.data_dir, f'{self.stock_code}_predictions.csv')
            results_df.to_csv(results_file, index=False)
            print(f'预测结果已保存到：{results_file}')
                    
            return signals
        except Exception as e:
            print(f"Error during prediction: {e}")
            return []

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


if __name__ == "__main__":
    stock_code = "603777"
    balance = 100000

    auto_trader = AutoTrader(stock_code, balance)
    X_train, y_train, sample_weights = auto_trader.prepare_data()

    model_trainer = ModelTrainer(input_shape=(X_train.shape[1], X_train.shape[2]), output_shape=1)
    # 添加early stopping回调
    from tensorflow.keras.callbacks import EarlyStopping
    early_stopping = EarlyStopping(
        monitor='loss',
        patience=20,  # 如果20个epoch内loss没有改善，则停止训练
        min_delta=0.0001,  # loss改善的最小阈值
        restore_best_weights=True  # 恢复最佳权重
    )
    
    # 增加训练轮次并添加early stopping
    history = model_trainer.train_model(
        X_train, 
        y_train, 
        sample_weights=sample_weights,
        epochs=300,  # 增加最大训练轮次
        callbacks=[early_stopping]
    )
    
    # 保存训练好的模型
    model_trainer.save_model(stock_code)

    # 准备未来数据
    future_dates = pd.date_range(
        start=auto_trader.df['date'].max() + pd.Timedelta(days=1),
        end=auto_trader.df['date'].max() + pd.Timedelta(days=30)
    )

    # 创建未来数据的特征矩阵
    feature_columns = ['close', 'volume', 'ma5', 'ma20', 'rsi', 'rsi_5', 'rsi_9',
                      'macd', 'signal', 'hist', 'upper', 'middle', 'lower',
                      'cci', 'price_change_rate', 'K', 'D', 'J']
    future_df = pd.DataFrame(index=future_dates, columns=feature_columns)
    future_df = future_df.fillna(0)  # 填充为0

    # 转换为模型所需的格式
    X_future = future_df.values.reshape(future_df.shape[0], 1, len(feature_columns))
    X_future = X_future.astype('float32')

    # 生成交易信号并执行交易
    signals = auto_trader.generate_signals(model_trainer.model, X_future, future_dates)
    auto_trader.execute_trade(signals)