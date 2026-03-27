import pandas as pd
import numpy as np
import warnings

FEATURES_COLS = [
    'feat_price_ma5_ratio', 'feat_price_ma20_ratio', 'feat_ma_spread', 'feat_bb_position',
    'feat_K', 'feat_D', 'feat_volume_ratio', 'feat_KD_diff',
    'ret_1', 'ret_3', 'ret_5', 'ma5_slope', 'ma20_slope', 'K_change', 'D_change'
]

def clean_stock_data(df):
    # 1. 轉換日期格式
    df['date'] = pd.to_datetime(df['date'], errors='coerce')

    # 2. 刪除日期有缺的
    df = df.dropna(subset=['date'])

    # 3. 只留週一到週五 (交易日)
    df = df[df['date'].dt.weekday < 5]

    # 4. 依照日期排序
    df = df.sort_values(by='date').reset_index(drop=True)

    return df


def calculate_technical_indicators(df, periods=[5, 20]):
    """
    計算移動平均線、布林通道及「台股標準版」KD 指標
    - 解決 KD 提早交叉或與市面軟體不一致的問題
    - 確保多支股票數據完全隔離計算
    """
    if df is None or df.empty:
        return df

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # --- 1. 基礎數據清洗 ---
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            # 確保價格與成交量為數值，移除可能存在的逗號
            for col in ['price', 'open', 'high', 'low', 'volume']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce')

            # 排序
            df = df.sort_values(by=['name', 'date']).reset_index(drop=True)
            grouped_df = df.groupby('name')

            # --- 2. 計算 MA, VMA 與布林通道 ---
            for p in periods:
                # 移動平均線
                df[f'MA{p}'] = grouped_df['price'].transform(lambda x: x.rolling(window=p).mean())

                # 成交量均線
                if 'volume' in df.columns:
                    df[f'VMA{p}'] = grouped_df['volume'].transform(lambda x: x.rolling(window=p).mean())

                # 布林通道 (以 20 日線為基準)
                if p == 20:
                    std = grouped_df['price'].transform(lambda x: x.rolling(window=p).std())
                    df['Upper_Band'] = df['MA20'] + (std * 2)
                    df['Lower_Band'] = df['MA20'] - (std * 2)

            df['Regime'] = 'Sideways'
            df.loc[df['price'] > df['MA20'], 'Regime'] = 'Bull'
            df.loc[df['price'] < df['MA20'], 'Regime'] = 'Bear'

            # --- 3. 手動計算「台股標準版」KD (9, 3, 3) ---
            def compute_group_kd(group):
                stock_name = group.name
                group = group.copy()
                group['name'] = stock_name

                # 至少要有 9 天資料才能開始算 RSV
                if len(group) < 9:
                    group['K'] = np.nan
                    group['D'] = np.nan
                    return group

                # A. 計算 RSV
                low_9 = group['low'].rolling(window=9).min()
                high_9 = group['high'].rolling(window=9).max()

                # 處理最高等於最低的情況，避免除以 0
                denom = high_9 - low_9
                rsv = 100 * (group['price'] - low_9) / denom

                # B. 遞迴計算 K 與 D (台股標準 1/3 平滑法)
                k_list = []
                d_list = []
                last_k = 50.0  # 初始值設定為 50
                last_d = 50.0

                for val in rsv:
                    if pd.isna(val):
                        k_list.append(np.nan)
                        d_list.append(np.nan)
                    else:
                        if last_k is None:
                            # 第一筆有效 RSV 直接作為初始 K, D
                            current_k = val
                            current_d = val
                        else:
                            # 今日 K = 前日 K * (2/3) + 今日 RSV * (1/3)
                            current_k = (2 / 3) * last_k + (1 / 3) * val
                            # 今日 D = 前日 D * (2/3) + 今日 K * (1/3)
                            current_d = (2 / 3) * last_d + (1 / 3) * current_k

                        k_list.append(current_k)
                        d_list.append(current_d)
                        last_k = current_k
                        last_d = current_d

                group['K'] = k_list
                group['D'] = d_list
                return group

            # 套用分組計算 KD
            # 當 K 值在高檔（K 值 > 80）連續三天，即為高檔鈍化，表示多方非常強勢；
            # 反之當 K 值在低檔（K 值 < 20）連續三天，即為低檔鈍化，表示空方非常強勢。
            df = (
                df.sort_values(['name','date'])
                .groupby('name', group_keys=False)
                .apply(compute_group_kd)
                .reset_index(drop=True)
            )

            if 'name' not in df.columns:
                df = df.reset_index()

                if 'index' in df.columns:
                    df = df.drop(columns=['index'])

            print("✅ [DEBUG] 技術指標校正完成 (已對齊台股標準)。")

    except Exception as e:
        print(f"❌ calculate_technical_indicators 發生錯誤: {e}")

    return df


def detect_crossover_MA_signals(df):
    """偵測 MA5 和 MA20 的黃金交叉 (金叉) 與死亡交叉 (死叉) 訊號"""
    # 1. 確保按名稱和日期排序
    df = df.sort_values(['name', 'date']).reset_index(drop=True)

    # 2. 使用 groupby 分組計算
    df['MA_Above'] = df['MA5'] > df['MA20']

    # 針對每一支股票獨立計算 diff
    df['Signal_Change'] = df.groupby('name')['MA_Above'].transform(lambda x: x.astype(int).diff())

    # 3. 標記訊號
    df['Signal'] = 'Hold'
    df.loc[df['Signal_Change'] == 1.0, 'Signal'] = 'Buy'
    df.loc[df['Signal_Change'] == -1.0, 'Signal'] = 'Sell'

    return df


def detect_crossover_KD_signals(df):
    """偵測 KD 交叉，放寬為全區顯示，並區分『強力區』與『一般區』"""
    # 0. 設定安檢門標準
    OVERSOLD_LINE = 20
    OVERBOUGHT_LINE = 80

    # 1. 基本排序與計算交叉 (維持原本邏輯)
    df = df.sort_values(['name', 'date']).reset_index(drop=True)
    df['KD_Above'] = df['K'] > df['D']
    df['KD_Above'] = df['KD_Above'].fillna(False)

    # 2. 找出交叉的一瞬間 (1.0 = 金叉, -1.0 = 死叉)
    df['KD_Change'] = df.groupby('name')['KD_Above'].transform(lambda x: x.astype(int).diff())

    # 3. 初始化所有訊號為 Hold
    df['KD_Signal'] = 'Hold'
    df['KD_Category'] = 'None'

    # 4. 放寬後的「黃金交叉」判斷：只要是金叉都給 'Buy'
    buy_condition = (df['KD_Change'] == 1.0)
    df.loc[buy_condition, 'KD_Signal'] = 'Buy'

    # 標註金叉類別：區分「強力」與「一般」
    strong_buy = buy_condition & (df['K'] <= OVERSOLD_LINE)
    df.loc[strong_buy, 'KD_Category'] = '🔥 KD 強力金叉'
    df.loc[buy_condition & (df['K'] > OVERSOLD_LINE), 'KD_Category'] = '🚀 KD 一般金叉'

    # 5. 放寬後的「死亡交叉」判斷：只要是死叉都給 'Sell'
    sell_condition = (df['KD_Change'] == -1.0)
    df.loc[sell_condition, 'KD_Signal'] = 'Sell'

    # 標註死叉類別：區分「強力」與「一般」
    strong_sell = sell_condition & (df['K'] >= OVERBOUGHT_LINE)
    df.loc[strong_sell, 'KD_Category'] = '💀 KD 強力死叉'
    df.loc[sell_condition & (df['K'] < OVERBOUGHT_LINE), 'KD_Category'] = '📉 KD 一般死叉'

    return df


def label_data_for_ai(df):
    """
    二分類標籤（Phase 1）：
    1 : 看多（第 3 天收盤價 > 今日收盤價）
    0 : 非看多（第 3 天收盤價 <= 今日收盤價）
    """
    df = df.sort_values(['name', 'date']).copy()

    # 第 3 天之後的收盤價
    df['future_price_3d'] = df.groupby('name')['price'].shift(-3)

    # 二分類標籤
    df['target'] = (df['future_price_3d'] > df['price']).astype(int)

    # 移除無法計算未來第 3 天收盤價的資料
    return df.dropna(subset=['future_price_3d'])


def build_features(df):
    # 確保資料是有序的
    df = df.sort_values(['name', 'date']).copy()

    # 1. 計算特徵
    df['feat_price_ma5_ratio'] = (df['price'] - df['MA5']) / df['MA5']  # 5日乖離率
    df['feat_price_ma20_ratio'] = (df['price'] - df['MA20']) / df['MA20']  # 20日乖離率
    df['feat_ma_spread'] = df['MA5'] / df['MA20']  # 趨勢強度
    df['feat_bb_position'] = (df['price'] - df['Lower_Band']) / (df['Upper_Band'] - df['Lower_Band']).replace(0, np.nan)
    df['feat_K'] = df['K'] / 100
    df['feat_D'] = df['D'] / 100

    vol_ratio = df['volume'] / df['VMA5'].replace(0, np.nan)
    df['feat_volume_ratio'] = np.log1p(vol_ratio)

    df['feat_KD_diff'] = (df['K'] - df['D']) / 100

    # 新增 momentum 特徵
    df['ret_1'] = df.groupby('name')['price'].pct_change(1)
    df['ret_3'] = df.groupby('name')['price'].pct_change(3)
    df['ret_5'] = df.groupby('name')['price'].pct_change(5)

    # 新增趨勢變化特徵
    df['ma5_slope'] = df.groupby('name')['MA5'].pct_change(3, fill_method=None)
    df['ma20_slope'] = df.groupby('name')['MA20'].pct_change(5, fill_method=None)

    # 新增 KD 變化特徵
    df['K_change'] = df.groupby('name')['K'].diff(1)
    df['D_change'] = df.groupby('name')['D'].diff(1)

    return df


def prepare_features_for_ai(df):
    """
    將原始數據轉換為 AI 專用的強化特徵
    """
    df = build_features(df)

    final_df = df[FEATURES_COLS + ['target', 'Regime']].dropna().copy()

    X = final_df[FEATURES_COLS]
    y = final_df['target']
    regime = final_df['Regime']

    # 清洗掉因為計算指標產生的 NaN
    # final_df = df.dropna(subset=feature_cols + ['target'])

    return X, y, regime


def get_ai_prediction_for_stock(model, full_df, stock_name):
    """
    回傳：(預測類別, 信心度)
    類別:
        1 = 看多（未來短期上漲機率較高）
        0 = 非看多
        信心度 = 模型對該預測類別的機率
    """
    stock_df = full_df[full_df['name'] == stock_name].copy()

    if stock_df.empty:
        # 必須回傳兩個值
        return None, 0

    # 2. 計算特徵 (確保跟訓練時一模一樣)
    try:
        stock_df = build_features(stock_df)
        stock_data = stock_df.tail(1)

        input_data = stock_data[FEATURES_COLS]

        # 處理可能的缺失值或無限值 (預防萬一)
        input_data = input_data.replace([np.inf, -np.inf], np.nan).fillna(0)

        # 預測類別 (0, 1, 2)
        prediction = model.predict(input_data)[0]

        # 取得機率分布
        probabilities = model.predict_proba(input_data)[0]
        confidence = probabilities[prediction]  # 模型對「自己選的那個類別」的信心

        return prediction, confidence

    except Exception as e:
        print(f"預測出錯: {e}")
        # 發生錯誤時也要回傳兩個值
        return None, 0


def get_feature_importance(model, feature_names, top_n=10):
    """
    取得模型（RF 或 XGB）的 Feature Importance
    """
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": model.feature_importances_
    })

    # 將權重轉換為百分比格式，方便觀察
    importance_df["importance_pct"] = (importance_df["importance"] * 100).round(2)
    importance_df = importance_df.sort_values(by="importance", ascending=False)

    return importance_df.head(top_n)