import os
import numpy as np
import pandas as pd
import pandas_datareader.data as web
import datetime
from collections import deque
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model  # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input  # type: ignore
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping  # type: ignore
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import logging
from datetime import timedelta

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 忽略 TensorFlow 的一些警告信息
logging.getLogger('tensorflow').setLevel(logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=all, 1=info, 2=warning, 3=error


# 行业名称到英文的映射
def englishname(industry: str) -> str:
    if industry == "科技":
        englishname = "tech"
    elif industry == "医药":
        englishname = "pharma"
    elif industry == "金融":
        englishname = "finance"
    elif industry == "消费":
        englishname = "consumer"
    elif industry == "能源":
        englishname = "energy"
    else:
        englishname = "unknown"
    return englishname


# --- 特征工程函数 (与CNN.py中保持一致) ---
def compute_technical_indicators(df):
    """计算技术指标特征"""
    # 简单移动平均线
    df['SMA5'] = df['Close'].rolling(window=5).mean()
    df['SMA10'] = df['Close'].rolling(window=10).mean()
    df['SMA20'] = df['Close'].rolling(window=20).mean()

    # 指数移动平均线
    df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()

    # MACD
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['Histogram'] = df['MACD'] - df['Signal']

    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # 布林带
    df['MiddleBand'] = df['Close'].rolling(window=20).mean()
    std = df['Close'].rolling(window=20).std()
    df['UpperBand'] = df['MiddleBand'] + (std * 2)
    df['LowerBand'] = df['MiddleBand'] - (std * 2)

    # 标准差
    df['StdDev'] = df['Close'].rolling(window=10).std()

    # 成交量指标
    df['Vol_SMA10'] = df['Volume'].rolling(window=10).mean()
    df['Vol_Ratio'] = df['Volume'] / df['Vol_SMA10']

    # 价格与均线比率
    df['Price_SMA5_Ratio'] = df['Close'] / df['SMA5']
    df['Price_SMA10_Ratio'] = df['Close'] / df['SMA10']
    df['Price_SMA20_Ratio'] = df['Close'] / df['SMA20']

    # 动量指标
    df['Momentum'] = df['Close'] - df['Close'].shift(5)

    # 清理NaN值
    df.dropna(inplace=True)
    return df


# --- 数据预处理函数 (修改为多标签输出) ---
def enhanced_stock_data_preprocessing(df, mem_his_days, pre_days):
    """增强版股票数据预处理，包括特征工程和序列化"""
    # 1. 计算技术指标
    df = compute_technical_indicators(df.copy())

    # 2. 定义特征列（固定顺序）
    feature_columns = [
        'Open', 'High', 'Low', 'Close', 'Volume',
        'SMA5', 'SMA10', 'SMA20', 'EMA12', 'EMA26',
        'MACD', 'Signal', 'Histogram', 'RSI',
        'MiddleBand', 'UpperBand', 'LowerBand', 'StdDev',
        'Vol_SMA10', 'Vol_Ratio',
        'Price_SMA5_Ratio', 'Price_SMA10_Ratio', 'Price_SMA20_Ratio',
        'Momentum'
    ]

    # 3. 选择特征并标准化
    features = df[feature_columns]
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    # 4. 创建多标签（未来1~pre_days天的收盘价）
    for i in range(1, pre_days + 1):
        df[f'label_{i}'] = df['Close'].shift(-i)

    # 删除包含NaN的行
    df.dropna(inplace=True)

    # 5. 序列化数据（为LSTM模型准备）
    X, y = [], []
    deq = deque(maxlen=mem_his_days)

    for i in range(len(scaled_features)):
        deq.append(scaled_features[i])
        if len(deq) == mem_his_days:
            # 检查是否有对应的标签
            if i < len(df) and not np.isnan(df.iloc[i]['label_1']):
                X.append(np.array(deq))
                # 收集所有标签
                labels = [df.iloc[i][f'label_{j}'] for j in range(1, pre_days + 1)]
                y.append(labels)

    X = np.array(X)
    y = np.array(y)

    # 6. 提取最近的数据用于实时预测
    X_lately = np.array([scaled_features[-mem_his_days:]])

    return X, y, X_lately, scaler, feature_columns


# --- 创建专门的LSTM模型构建函数 (修改为多输出) ---
def build_lstm_model(input_shape, lstm_layers=2, dense_layers=2, units=64, dropout_rate=0.2):
    """
    构建LSTM模型用于时间序列预测

    Args:
        input_shape: 输入形状，如(timesteps, features)
        lstm_layers: LSTM层数量
        dense_layers: 全连接层数量
        units: LSTM单元数量
        dropout_rate: Dropout比率

    Returns:
        构建好的Keras模型
    """
    model = Sequential(name=f"LSTM_Model_l{lstm_layers}_d{dense_layers}_u{units}")
    model.add(Input(shape=input_shape))

    # 添加LSTM层
    for i in range(lstm_layers):
        return_sequences = i < lstm_layers - 1
        model.add(LSTM(units,
                       activation='tanh',
                       recurrent_activation='sigmoid',
                       return_sequences=return_sequences))
        model.add(Dropout(dropout_rate))

    # 添加全连接层
    for i in range(dense_layers):
        model.add(Dense(units, activation='relu'))
        model.add(Dropout(dropout_rate))

    # 输出层 - 预测未来pre_days天的价格
    model.add(Dense(5, name='output'))

    return model


# --- LSTM模型训练与评估函数 ---
def train_lstm_model(X_train, y_train, X_val, y_val, lstm_params, industry, ticker, epochs=50, batch_size=32):
    """
    训练LSTM模型并评估

    Args:
        X_train, y_train: 训练数据
        X_val, y_val: 验证数据
        lstm_params: LSTM模型参数字典
        industry: 行业名称（用于保存模型）
        ticker: 股票代码（用于保存模型）
        epochs: 训练轮数
        batch_size: 批大小

    Returns:
        训练好的模型、训练历史和模型路径
    """
    # 解析参数
    input_shape = X_train.shape[1:]
    lstm_layers = lstm_params.get('lstm_layers', 2)
    dense_layers = lstm_params.get('dense_layers', 2)
    units = lstm_params.get('units', 64)
    dropout_rate = lstm_params.get('dropout_rate', 0.2)

    # 构建模型
    model = build_lstm_model(
        input_shape=input_shape,
        lstm_layers=lstm_layers,
        dense_layers=dense_layers,
        units=units,
        dropout_rate=dropout_rate
    )

    # 编译模型
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(
        optimizer=optimizer,
        loss='mse',
        metrics=['mape', 'mae']
    )

    # 打印模型概要
    model.summary()

    # 设置回调
    checkpoint_dir = 'model_checkpoints_lstm'
    os.makedirs(checkpoint_dir, exist_ok=True)

    model_id = f"lstm_{lstm_layers}_dense_{dense_layers}_units_{units}"
    filepath = os.path.join(checkpoint_dir, f'lstm_model-{model_id}-{{epoch:02d}}-{{val_mape:.2f}}.keras')

    checkpoint = ModelCheckpoint(
        filepath=filepath,
        save_weights_only=False,
        monitor='val_mape',
        mode='min',
        save_best_only=True,
        verbose=1
    )

    early_stopping = EarlyStopping(
        monitor='val_mape',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )

    # 训练模型
    print(f"\n--- 开始训练LSTM模型 [{model_id}] ---")
    history = model.fit(
        X_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_val, y_val),
        callbacks=[checkpoint, early_stopping],
        verbose=2
    )

    # 保存最终模型
    model_dir = 'models'
    os.makedirs(model_dir, exist_ok=True)
    final_model_path = os.path.join(model_dir, f'final_lstm_{englishname(industry)}.keras')
    model.save(final_model_path)
    print(f"模型已保存至: {final_model_path}")

    # 评估模型
    val_loss, val_mape, val_mae = model.evaluate(X_val, y_val, verbose=0)
    print(f"\n验证集评估结果:")
    print(f"Loss (MSE): {val_loss:.4f}")
    print(f"MAPE: {val_mape:.2f}%")
    print(f"MAE: {val_mae:.4f}")

    return model, history, final_model_path


# --- LSTM预测函数 (适配多输出) ---
def predict_with_lstm(model, X_test, scaler=None, actual_values=None):
    """
    使用训练好的LSTM模型进行预测

    Args:
        model: 训练好的模型
        X_test: 测试数据
        scaler: 用于反标准化的scaler对象
        actual_values: 实际值(可选)

    Returns:
        预测结果和评估指标(如果提供了actual_values)
    """
    # 预测
    predictions = model.predict(X_test)

    # 如果提供了实际值，计算评估指标
    if actual_values is not None:
        # 计算每个预测天数的独立指标
        metrics = {}
        for day in range(predictions.shape[1]):
            pred_day = predictions[:, day]
            actual_day = actual_values[:, day]

            # 计算指标
            mae = mean_absolute_error(actual_day, pred_day)
            mse = mean_squared_error(actual_day, pred_day)
            rmse = np.sqrt(mse)
            r2 = r2_score(actual_day, pred_day)
            mape = np.mean(np.abs((actual_day - pred_day) / actual_day)) * 100

            metrics[f'day_{day + 1}'] = {
                'mae': mae,
                'mse': mse,
                'rmse': rmse,
                'r2': r2,
                'mape': mape
            }

        # 打印每日指标
        print("\n分日预测评估指标:")
        for day, vals in metrics.items():
            print(f"{day}: MAE={vals['mae']:.4f}, MAPE={vals['mape']:.2f}%, R²={vals['r2']:.4f}")

        return predictions, metrics

    return predictions


# --- LSTM可视化函数 (适配多输出) ---
def visualize_lstm_predictions(actual_values, predictions, ticker, title=None):
    """
    可视化LSTM预测结果

    Args:
        actual_values: 实际值 (n_samples, n_days)
        predictions: 预测值 (n_samples, n_days)
        ticker: 股票代码
        title: 图表标题
    """
    plt.figure(figsize=(12, 8))

    # 计算整体MAPE
    all_actual = actual_values.flatten()
    all_preds = predictions.flatten()
    mape = np.mean(np.abs((all_actual - all_preds) / all_actual)) * 100

    # 绘制每日预测效果
    for day in range(actual_values.shape[1]):
        plt.subplot(2, 3, day + 1)  # 2行3列布局
        x = range(len(actual_values))
        plt.plot(x, actual_values[:, day], 'b-', label='实际价格', linewidth=1.5)
        plt.plot(x, predictions[:, day], 'r--', label='预测价格', linewidth=1.5)
        plt.title(f'第 {day + 1} 天预测', fontsize=12)
        plt.grid(True, alpha=0.3)

        # 添加每日MAPE
        day_mape = np.mean(np.abs((actual_values[:, day] - predictions[:, day]) / actual_values[:, day])) * 100
        plt.figtext(0.15 + 0.3 * (day % 3), 0.45 - 0.45 * (day // 3), f'MAPE: {day_mape:.2f}%',
                    fontsize=10, bbox={'facecolor': 'lightgray', 'alpha': 0.3, 'pad': 3})

    # 添加主标题
    if title:
        plt.suptitle(f"{title}\n整体MAPE: {mape:.2f}%", fontsize=16)
    else:
        plt.suptitle(f'{ticker} 股票价格 LSTM 预测\n整体MAPE: {mape:.2f}%', fontsize=16)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # 创建picture文件夹并保存
    output_dir = "picture"
    os.makedirs(output_dir, exist_ok=True)  # 创建文件夹（如果不存在）
    output_path = os.path.join(output_dir, f'lstm_prediction_{ticker}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


# --- LSTM训练历史可视化函数 ---
def plot_lstm_training_history(history):
    """
    绘制LSTM训练历史曲线

    Args:
        history: 模型训练历史对象
    """
    plt.figure(figsize=(15, 6))

    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='训练损失')
    plt.plot(history.history['val_loss'], label='验证损失')
    plt.title('模型损失曲线', fontsize=15)
    plt.xlabel('训练轮次', fontsize=12)
    plt.ylabel('损失值(MSE)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)

    # 绘制MAPE曲线
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mape'], label='训练MAPE')
    plt.plot(history.history['val_mape'], label='验证MAPE')
    plt.title('模型MAPE曲线', fontsize=15)
    plt.xlabel('训练轮次', fontsize=12)
    plt.ylabel('MAPE(%)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    # 创建picture文件夹并保存
    output_dir = "picture"
    os.makedirs(output_dir, exist_ok=True)  # 创建文件夹（如果不存在）
    output_path = os.path.join(output_dir, f'lstm_training_history.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


# --- 可视化未来价格预测 (添加置信区间) ---
def visualize_future_predictions(history_df, future_prices, ticker, industry):
    """可视化历史价格和未来预测"""
    plt.figure(figsize=(14, 7))

    # 绘制历史价格
    plt.plot(history_df.index, history_df['Close'], 'b-', label='历史价格', linewidth=2)

    # 计算未来日期
    last_date = history_df.index[-1]
    future_dates = [last_date + timedelta(days=i + 1) for i in range(len(future_prices))]

    # 绘制预测价格
    plt.plot(future_dates, future_prices, 'ro-', label='预测价格', linewidth=2, markersize=8)

    # 添加置信区间（基于历史波动率）
    volatility = history_df['Close'].pct_change().std() * 100  # 百分比波动率
    upper_bound = [p * (1 + volatility / 100) for p in future_prices]
    lower_bound = [p * (1 - volatility / 100) for p in future_prices]

    plt.fill_between(future_dates, lower_bound, upper_bound,
                     color='orange', alpha=0.2, label='置信区间')

    # 添加标记
    for i, (date, price) in enumerate(zip(future_dates, future_prices)):
        plt.annotate(f'{price:.2f}',
                     (date, price),
                     textcoords="offset points",
                     xytext=(0, 10),
                     ha='center',
                     fontsize=10)

    plt.title(f'{ticker} ({industry}行业) 未来价格预测', fontsize=16)
    plt.xlabel('日期', fontsize=14)
    plt.ylabel('价格', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)

    # 设置x轴范围以更好显示未来预测
    plt.xlim([history_df.index[-30], future_dates[-1] + timedelta(days=2)])

    # 保存图像
    plt.tight_layout()
    plot_filename = f'future_prediction_{ticker}_{englishname(industry)}.png'
    output_dir = "picture"
    os.makedirs(output_dir, exist_ok=True)  # 创建文件夹（如果不存在）
    output_path = os.path.join(output_dir, plot_filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"   未来价格预测图已保存至: {plot_filename}")


# --- 实施LSTM模型训练主函数 (修改未来预测逻辑) ---
def train_stock_lstm_model(ticker, industry, lstm_params=None, test_size=0.2, start_date=None, end_date=None):
    """
    完整的LSTM模型训练流程

    Args:
        ticker: 股票代码
        industry: 行业名称
        lstm_params: LSTM模型参数
        test_size: 测试集比例
        start_date: 开始日期
        end_date: 结束日期

    Returns:
        训练好的模型、scaler对象和特征列表
    """
    print(f"\n{'=' * 50}")
    print(f"开始LSTM模型训练流程: {ticker} ({industry}行业)")
    print(f"{'=' * 50}")

    # 默认LSTM参数
    if lstm_params is None:
        lstm_params = {
            'lstm_layers': 2,
            'dense_layers': 2,
            'units': 64,
            'dropout_rate': 0.2,
            'mem_days': 20,
            'pred_days': 5
        }

    # 设置日期范围
    if start_date is None:
        start_date = datetime.datetime(2015, 1, 1)
    if end_date is None:
        end_date = datetime.datetime.now()

    # 1. 获取股票数据
    print(f"\n1. 获取{ticker}股票数据 ({start_date.date()} - {end_date.date()})...")
    try:
        df_original_data = web.DataReader(ticker, 'stooq', start_date, end_date)
        # 确保日期升序
        df_original_data.sort_index(ascending=True, inplace=True)
        print(f"   数据获取成功! 共 {len(df_original_data)} 个交易日")
    except Exception as e:
        print(f"   数据获取失败: {e}")
        return None, None, None

    # 2. 数据预处理
    print(f"\n2. 数据预处理...")
    X, y, X_lately, scaler, feature_list = enhanced_stock_data_preprocessing(
        df_original_data,
        lstm_params['mem_days'],
        lstm_params['pred_days']
    )

    if X is None or len(X) == 0:
        print("   数据预处理失败!")
        return None, None, None

    print(f"   预处理完成! 特征形状: {X.shape}, 标签形状: {y.shape}")
    print(f"   特征列表: {feature_list}")

    # 3. 划分训练集和测试集
    print(f"\n3. 划分数据集 (测试集比例: {test_size})...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
    print(f"   训练集: {X_train.shape}, {y_train.shape}")
    print(f"   测试集: {X_test.shape}, {y_test.shape}")

    # 4. 训练模型
    print(f"\n4. 训练LSTM模型...")
    model, history, model_path = train_lstm_model(
        X_train, y_train,
        X_test, y_test,
        lstm_params,
        industry,
        ticker,
        epochs=50,
        batch_size=32
    )

    # 5. 可视化训练历史
    print(f"\n5. 可视化训练历史...")
    plot_lstm_training_history(history)

    # 6. 在测试集上预测
    print(f"\n6. 在测试集上进行预测...")
    predictions, metrics = predict_with_lstm(model, X_test, scaler, y_test)

    # 7. 可视化预测结果
    print(f"\n7. 可视化预测结果...")
    visualize_lstm_predictions(y_test, predictions, ticker, f"{ticker} ({industry}行业) LSTM预测结果")

    # 8. 保存scaler和特征列表
    model_dir = 'models'
    os.makedirs(model_dir, exist_ok=True)

    scaler_path = os.path.join(model_dir, f'lstm_scaler_{englishname(industry)}.pkl')
    joblib.dump(scaler, scaler_path)
    print(f"Scaler已保存至: {scaler_path}")

    feature_list_path = os.path.join(model_dir, f'lstm_features_{englishname(industry)}.pkl')
    joblib.dump(feature_list, feature_list_path)
    print(f"特征列表已保存至: {feature_list_path}")

    # 9. 预测未来价格
    print(f"\n9. 预测未来{lstm_params['pred_days']}天价格...")
    if X_lately is not None and len(X_lately) > 0:
        # 直接预测未来5天价格
        future_predictions = model.predict(X_lately)[0]

        # 打印预测结果
        for i, price in enumerate(future_predictions):
            print(f"   预测未来第{i + 1}天收盘价: {price:.2f}")
    else:
        print("   没有最新数据用于预测")
        future_predictions = np.zeros(lstm_params['pred_days'])

    # 10. 可视化未来价格预测
    print(f"\n10. 可视化未来价格预测...")
    visualize_future_predictions(df_original_data, future_predictions, ticker, industry)

    print(f"\n{'=' * 50}")
    print(f"LSTM模型训练完成: {ticker} ({industry}行业)")
    print(f"{'=' * 50}")

    return model, scaler, feature_list


# --- 主执行函数 (使用更优参数) ---
def main():
    # 设置日期范围
    start = datetime.datetime(2015, 1, 1)
    end = datetime.datetime.now()

    # 定义行业和股票
    INDUSTRY_TICKERS = {
        '科技': ['AAPL', 'MSFT', 'GOOGL', 'META', 'NVDA'],
        '医药': ['JNJ', 'PFE', 'MRK', 'ABBV', 'LLY'],
        '金融': ['JPM', 'BAC', 'GS', 'MS', 'WFC'],
        '能源': ['XOM', 'CVX', 'COP', 'BP', 'SLB'],
        '消费': ['PG', 'KO', 'PEP', 'WMT', 'COST']
    }

    # 为演示选择股票
    industry = '能源'
    ticker = 'XOM'

    # 设置优化的LSTM参数
    lstm_params = {
        'lstm_layers': 2,
        'dense_layers': 2,
        'units': 64,
        'dropout_rate': 0.2,
        'mem_days': 20,
        'pred_days': 5
    }

    # 训练模型
    model, scaler, feature_list = train_stock_lstm_model(
        ticker,
        industry,
        lstm_params,
        start_date=start,
        end_date=end
    )

    if model and scaler and feature_list:
        print("\n训练完成! 模型、scaler和特征列表已保存")
    else:
        print("\n训练失败!")


if __name__ == "__main__":
    main()