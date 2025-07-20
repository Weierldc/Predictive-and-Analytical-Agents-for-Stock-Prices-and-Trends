# BestModel.py
import os
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error, r2_score
import CNN
import GRU
import LSTM_Stock
import Transformer

# 定义行业及其对应的模型列表
INDUSTRY_MODELS = {
    '科技': ['cnn', 'lstm', 'transformer', 'xgboost'],
    '医药': ['transformer', 'lstm', 'cnn', 'random_forest'],
    '金融': ['cnn', 'xgboost', 'lstm', 'transformer'],
    '能源': ['lstm', 'xgboost', 'cnn', 'random_forest'],
    '消费': ['lstm', 'cnn', 'xgboost', 'transformer']
}

# 定义行业及其代表性股票
INDUSTRY_TICKERS = {
    '科技': ['AAPL', 'MSFT', 'GOOGL', 'META', 'NVDA'],
    # '医药': ['JNJ', 'PFE', 'MRK', 'ABBV', 'LLY'],
    # '金融': ['JPM', 'BAC', 'GS', 'MS', 'WFC'],
    # '能源': ['XOM', 'CVX', 'COP', 'BP', 'SLB'],
    # '消费': ['PG', 'KO', 'PEP', 'WMT', 'COST']
}


class IndustryModelSelector:
    def __init__(self, start_date, end_date):
        self.start_date = start_date
        self.end_date = end_date
        self.industry_tickers = INDUSTRY_TICKERS
        self.model_modules = {
            'lstm': LSTM_Stock,
            'gru': GRU,
            'cnn': CNN,
            'transformer': Transformer
        }
        self.industry_best_models = {}
        self.performance_results = {}

    def evaluate_all_models_on_stock(self, ticker, industry):
        """在单只股票上评估所有模型"""
        model_performance = {}

        for model_type in INDUSTRY_MODELS[industry]:
            try:
                print(f"\n评估 {ticker} ({industry}) 的 {model_type.upper()} 模型...")

                # 调用对应的模型训练函数
                if model_type == 'lstm':
                    _, metrics = LSTM_Stock.train_stock_lstm_model(
                        ticker, self.start_date, self.end_date,
                        test_size=0.2
                    )
                elif model_type == 'gru':
                    _, metrics = GRU.train_stock_gru_model(
                        ticker, self.start_date, self.end_date,
                        test_size=0.2
                    )
                # elif model_type == 'cnn':
                #     _, metrics = CNN.train_stock_cnn_model(
                #         ticker, self.start_date, self.end_date,
                #         cnn_params={
                #             'conv_layers': 2,
                #             'dense_layers': 2,
                #             'filters': 64,
                #             'kernel_size': 3,
                #             'pool_size': 2,
                #             'mem_days': 20,
                #             'pred_days': 5
                #         },
                #         test_size=0.2
                #     )
                elif model_type == 'cnn':
                    _, metrics = CNN.train_stock_cnn_model(
                        ticker, self.start_date, self.end_date,
                        test_size=0.2
                    )
                elif model_type == 'transformer':
                    _, metrics = Transformer.train_stock_transformer_model(
                        ticker, self.start_date, self.end_date,
                        test_size=0.2
                    )

                if metrics:
                    model_performance[model_type] = {
                        'mape': metrics['mape'],
                        'r2': metrics['r2'],
                        'rmse': metrics['rmse']
                    }
                    print(f"{model_type.upper()} 评估完成: MAPE={metrics['mape']:.2f}%, R²={metrics['r2']:.4f}")

            except Exception as e:
                print(f"评估 {model_type} 失败: {str(e)}")

        return model_performance

    def aggregate_industry_performance(self, industry, stock_performance):
        """聚合行业内的模型性能"""
        industry_performance = {}

        for model_type in INDUSTRY_MODELS[industry]:
            model_perf_list = []

            for ticker, perf in stock_performance.items():
                if model_type in perf:
                    model_perf_list.append(perf[model_type])

            if model_perf_list:
                # 计算平均性能指标
                avg_mape = np.mean([p['mape'] for p in model_perf_list])
                avg_r2 = np.mean([p['r2'] for p in model_perf_list])
                avg_rmse = np.mean([p['rmse'] for p in model_perf_list])

                industry_performance[model_type] = {
                    'mape': avg_mape,
                    'r2': avg_r2,
                    'rmse': avg_rmse
                }

        return industry_performance

    def select_best_model_for_industry(self, industry_performance):
        """为行业选择最佳模型"""
        best_model = None
        best_score = float('-inf')

        for model_type, metrics in industry_performance.items():
            mape = metrics['mape']
            r2 = metrics['r2']

            # 计算综合得分 (MAPE权重60%，R²权重40%)
            if mape > 0:  # 确保MAPE有效
                score = 0.6 * (1 / mape) + 0.4 * r2
            else:
                score = float('-inf')

            if score > best_score:
                best_score = score
                best_model = model_type

        return best_model

    def run_selection(self):
        """运行模型选择流程"""
        for industry, tickers in self.industry_tickers.items():
            print(f"\n{'=' * 50}")
            print(f"处理行业: {industry}")
            print(f"股票列表: {', '.join(tickers)}")
            print(f"{'=' * 50}")

            stock_performance = {}

            # 评估每只股票的所有模型
            for ticker in tickers:
                print(f"\n评估股票: {ticker}")
                stock_perf = self.evaluate_all_models_on_stock(ticker, industry)
                if stock_perf:
                    stock_performance[ticker] = stock_perf

            # 聚合行业性能
            industry_perf = self.aggregate_industry_performance(industry, stock_performance)
            self.performance_results[industry] = industry_perf

            # 选择最佳模型
            best_model = self.select_best_model_for_industry(industry_perf)
            self.industry_best_models[industry] = best_model

            print(f"\n{industry} 行业选择的最佳模型: {best_model}")
            print(f"性能指标: MAPE={industry_perf[best_model]['mape']:.2f}%, "
                  f"R²={industry_perf[best_model]['r2']:.4f}, "
                  f"RMSE={industry_perf[best_model]['rmse']:.4f}")

        return self.industry_best_models

    def visualize_results(self):
        """可视化模型选择结果"""
        industries = list(self.industry_best_models.keys())
        best_models = list(self.industry_best_models.values())
        mape_values = [self.performance_results[ind][model]['mape']
                       for ind, model in self.industry_best_models.items()]

        plt.figure(figsize=(12, 6))

        # 创建条形图
        plt.bar(industries, mape_values, color=['blue', 'green', 'red', 'purple', 'orange'])

        # 添加数值标签
        for i, v in enumerate(mape_values):
            plt.text(i, v + 0.2, f"{v:.2f}%", ha='center')

        plt.title('各行业最佳模型及其MAPE', fontsize=16)
        plt.xlabel('行业', fontsize=14)
        plt.ylabel('MAPE(%)', fontsize=14)
        plt.xticks(fontsize=12)
        plt.grid(axis='y', alpha=0.3)

        # 添加模型类型标注
        for i, model in enumerate(best_models):
            plt.text(i, mape_values[i] / 2, model.upper(),
                     ha='center', va='center', color='white', fontsize=12, fontweight='bold')

        plt.tight_layout()
        plt.savefig('industry_best_models.png', dpi=300)
        plt.show()

    def save_results(self):
        """保存选择结果"""
        # 创建结果DataFrame
        results = []
        for industry, best_model in self.industry_best_models.items():
            metrics = self.performance_results[industry][best_model]
            results.append({
                'Industry': industry,
                'Best_Model': best_model,
                'MAPE': metrics['mape'],
                'R2': metrics['r2'],
                'RMSE': metrics['rmse']
            })

        results_df = pd.DataFrame(results)

        # 保存到CSV
        results_df.to_csv('industry_best_models.csv', index=False)
        print("结果已保存到 industry_best_models.csv")

        # 保存到pickle
        joblib.dump(self.industry_best_models, 'industry_best_models.pkl')
        print("模型选择结果已保存到 industry_best_models.pkl")


if __name__ == "__main__":
    # 设置日期范围
    start = datetime(2015, 1, 1)
    end = datetime(2024, 12, 31)

    # 创建模型选择器
    selector = IndustryModelSelector(start, end)

    # 运行模型选择
    best_models = selector.run_selection()

    # 可视化和保存结果
    selector.visualize_results()
    selector.save_results()

    print("\n行业最佳模型选择完成:")
    for industry, model in best_models.items():
        print(f"{industry}: {model}")