import os
import joblib
import numpy as np
import pandas as pd
import pandas_datareader.data as web

# 设置 matplotlib 后端为 Agg（非交互式）
import matplotlib
matplotlib.use('Agg')  # 必须在导入 pyplot 之前设置
import matplotlib.pyplot as plt

from io import BytesIO
import base64
from datetime import datetime, timedelta
from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model  # type: ignore
import traceback
import CNN
import LSTM_Stock
from flask import Flask, request, jsonify, render_template

# 全局缓存
LOADED_MODELS_CACHE = {}
LOADED_SCALERS_CACHE = {}
LOADED_FEATURE_LISTS_CACHE = {}

# 行业最优模型配置 (更新CNN配置)
INDUSTRY_OPTIMAL_MODELS_INFO = {
    "科技": {
        "model_type": "cnn",
        "model_file": "models/final_cnn_tech.keras",
        "scaler_file": "models/cnn_scaler_tech.pkl",
        "feature_list_file": "models/cnn_features_tech.pkl",
        "mem_days": 30,  # 延长历史天数
        "pred_days": 5
    },
    "医药": {
        "model_type": "transformer",
        "model_file": "models/final_transformer_pharma.keras",
        "scaler_file": "models/transformer_scaler_pharma.pkl",
        "feature_list_file": "models/transformer_features_pharma.pkl",
        "mem_days": 20,
        "pred_days": 5
    },
    "金融": {
        "model_type": "cnn",
        "model_file": "models/final_cnn_金融.keras",
        "scaler_file": "models/cnn_scaler_金融.pkl",
        "feature_list_file": "models/cnn_features_金融.pkl",
        "mem_days": 30,  # 延长历史天数
        "pred_days": 5
    },
    "能源": {
        "model_type": "lstm",
        "model_file": "models/final_lstm_energy.keras",
        "scaler_file": "models/lstm_scaler_energy.pkl",
        "feature_list_file": "models/lstm_features_energy.pkl",
        "mem_days": 20,
        "pred_days": 5
    },
    "消费": {
        "model_type": "lstm",
        "model_file": "models/final_lstm_consumer.keras",
        "scaler_file": "models/lstm_scaler_consumer.pkl",
        "feature_list_file": "models/lstm_features_consumer.pkl",
        "mem_days": 20,
        "pred_days": 5
    }
}


# 从CSV文件加载行业最佳模型类型
def load_industry_model_types():
    """从CSV文件加载行业最佳模型类型"""
    csv_path = 'industry_best_models.csv'
    model_types = {}

    try:
        if os.path.exists(csv_path):
            # 读取CSV文件
            df = pd.read_csv(csv_path)

            # 确保列名正确
            if 'Industry' in df.columns and 'Best_Model' in df.columns:
                # 转换为字典：行业->最佳模型
                model_types = pd.Series(df['Best_Model'].values, index=df['Industry']).to_dict()
                print(f"成功从 {csv_path} 加载行业模型类型:")
                for industry, model in model_types.items():
                    print(f"  {industry}: {model}")
            else:
                print(f"CSV文件缺少必要的列: 'Industry' 或 'Best_Model'")
        else:
            print(f"警告: 最佳行业模型CSV文件不存在: {csv_path}")
    except Exception as e:
        print(f"加载行业模型类型时出错: {str(e)}")

    # 提供默认值以防文件加载失败
    default_types = {
        "科技": "cnn",
        "金融": "cnn",
        "能源": "lstm",
        "消费": "lstm",
        "医药": "transformer"
    }

    # 合并加载的和默认的值
    for industry, model in default_types.items():
        if industry not in model_types:
            model_types[industry] = model

    return model_types

# 从CSV文件加载行业最佳模型类型
INDUSTRY_MODEL_TYPE = load_industry_model_types()

# INDUSTRY_MODEL_TYPE = {
#     "科技": "cnn",
#     "金融": "cnn",
#     "能源": "lstm",
#     "消费": "lstm",
#     "医药": "transformer"  # transformer暂不支持训练
# }

app = Flask(__name__)


# 行业名称到英文的映射
def industry_to_english(industry):
    """将行业中文名称转换为英文简写"""
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



# 从文件读取图片并转换为Base64
def image_to_base64(image_path):
    """将图片文件转换为Base64编码"""
    if not os.path.exists(image_path):
        return None

    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def fetch_stock_data(ticker, days=100):
    """获取股票历史数据"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    try:
        # 获取数据
        df = web.DataReader(ticker, 'stooq', start_date, end_date)

        # 确保数据按日期升序排列（从旧到新）
        df.sort_index(ascending=True, inplace=True)

        # 重置索引并转换为datetime
        df = df.reset_index()
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)

        return df
    except Exception as e:
        print(f"获取{ticker}数据失败: {str(e)}")
        return None


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """股票预测API端点 - 直接返回训练时生成的预测图片"""
    try:
        # 解析请求数据
        data = request.get_json()
        if not data or 'industry' not in data or 'ticker' not in data:
            return jsonify({'error': '请求体必须包含"industry"和"ticker"字段'}), 400

        industry = data['industry']
        ticker = data['ticker'].upper()

        # 检查行业是否支持
        if industry not in INDUSTRY_OPTIMAL_MODELS_INFO:
            return jsonify({'error': f'不支持的行业: {industry}'}), 400

        # 获取模型配置
        model_info = INDUSTRY_OPTIMAL_MODELS_INFO[industry]
        model_type = model_info["model_type"]

        # 将行业名称转换为英文简写
        english_name = industry_to_english(industry)

        # 构建图片文件名
        image_filename = f'future_prediction_{ticker}_{english_name}.png'
        image_path = os.path.join('picture', image_filename)

        # if not os.path.exists(image_path):
        #     train()

        # 检查图片是否存在
        if not os.path.exists(image_path):
            return jsonify({
                'error': f'预测图片不存在，请先训练模型',
                'message': f'图片路径: {image_path}'
            }), 404

        # 将图片转换为Base64
        img_base64 = image_to_base64(image_path)
        if not img_base64:
            return jsonify({'error': '图片读取失败'}), 500

        # 构建响应
        response = {
            'industry': industry,
            'ticker': ticker,
            'model_type': model_type,
            'currency': 'USD',
            'prediction_days': 5,
            'plot_image': img_base64
        }

        return jsonify(response)

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': f'内部服务器错误: {str(e)}'}), 500


@app.route('/train', methods=['POST'])
def train():
    """训练模型并生成预测图片"""
    data = request.get_json()
    if not data or 'industry' not in data or 'ticker' not in data:
        return jsonify({'error': '请求体必须包含 industry 和 ticker'}), 400

    industry = data['industry']
    ticker = data['ticker'].upper()

    if industry not in INDUSTRY_MODEL_TYPE:
        return jsonify({'error': f'不支持的行业: {industry}'}), 400

    model_type = INDUSTRY_MODEL_TYPE[industry]

    try:
        # 根据模型类型调用相应的训练函数
        if model_type == 'cnn':
            CNN.train_stock_cnn_model(ticker, industry)
        elif model_type == 'lstm':
            LSTM_Stock.train_stock_lstm_model(ticker, industry)
        else:
            return jsonify({'error': f'{model_type} 模型训练暂不支持'}), 400

        # 检查图片是否生成
        english_name = industry_to_english(industry)
        image_filename = f'future_prediction_{ticker}_{english_name}.png'
        image_path = os.path.join('picture', image_filename)

        if os.path.exists(image_path):
            return jsonify({
                'message': f'{ticker} 的 {industry} 行业 {model_type} 模型训练完成',
                'image_path': image_path
            })
        else:
            return jsonify({
                'warning': f'模型训练完成但预测图片未生成',
                'message': f'预期图片路径: {image_path}'
            })

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': f'训练失败: {str(e)}'}), 500


if __name__ == '__main__':
    # 确保pictures目录存在
    os.makedirs('picture', exist_ok=True)
    app.run(host='0.0.0.0', port=5001, debug=True)