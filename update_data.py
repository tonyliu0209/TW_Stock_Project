import os
import json
import time
import yfinance as yf
import pandas as pd
import joblib
from datetime import date, timedelta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from features import (
    clean_stock_data,
    calculate_technical_indicators,
    detect_crossover_MA_signals,
    detect_crossover_KD_signals,
    label_data_for_ai,
    prepare_features_for_ai
)


def fetch_all_history_to_csv(json_path="data/stock_urls.json", output_csv="data/stock_history_yfinance.csv"):
    # 1. 讀取股票清單
    with open(json_path, 'r', encoding='utf-8') as f:
        stock_list = json.load(f)

    all_data = []
    print(f"🌟 開始執行「全歷史資料」同步計畫，預計處理 {len(stock_list)} 支股票...")

    # 算出明天的日期，用來逼迫 yfinance 抓到最新
    tomorrow_str = (date.today() + timedelta(days=1)).strftime('%Y-%m-%d')

    for stock in stock_list:
        stock_name = stock["name"]
        # 取得純數字代號
        stock_id = stock["url"].split("/")[-1].split(".")[0]

        df = pd.DataFrame()
        # 嘗試抓取歷史
        for suffix in [".TW", ".TWO"]:
            symbol = f"{stock_id}{suffix}"
            print(f"🔍 正在抓取: {stock_name} ({symbol})...", end="\r")

            ticker = yf.Ticker(symbol)
            # 捨棄 period="max"，改用強制指定日期
            # 從 2000 年開始抓到「明天」
            temp_df = ticker.history(start="2000-01-01", end=tomorrow_str)

            if not temp_df.empty:
                df = temp_df
                break

        if not df.empty:
            # 格式化資料
            df = df.reset_index()
            df = df.rename(columns={
                "Date": "date", "Open": "open", "High": "high",
                "Low": "low", "Close": "price", "Volume": "volume"
            })

            # 統一日期與數值格式
            # 1. 先確保它是 datetime 格式
            # 2. 用 .dt.tz_localize(None) 把時區資訊丟掉，不進行任何時間加減
            # 3. 再格式化成字串
            df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None).dt.strftime('%Y-%m-%d')
            df["stock_id"] = str(stock_id)
            df["name"] = stock_name

            # 數值處理
            for col in ["open", "high", "low", "price"]:
                df[col] = df[col].astype(float).round(2)
            df["volume"] = (df["volume"] // 1000).astype(int)  # 換算為張數

            df = df[["date", "stock_id", "name", "price", "volume", "open", "high", "low"]]
            all_data.append(df)
            print(f"✅ {stock_name} 抓取完畢，共 {len(df)} 筆資料。")
        else:
            print(f"❌ 無法取得 {stock_name} ({stock_id}) 的任何資料。")

        # 稍微停頓，避免被 Yahoo 封鎖 IP
        time.sleep(0.5)

    # 2. 合併所有股票資料
    if all_data:
        final_df = pd.concat(all_data, ignore_index=True)
        # 根據日期排序
        final_df = final_df.sort_values(by=["stock_id", "date"])

        # 加入 drop_duplicates（避免重複）
        final_df = final_df.drop_duplicates(
            subset=["date", "stock_id"],
            keep="last"
        )

        # 避免 NaN
        final_df = final_df.dropna(subset=["price", "open", "high", "low"])

        # 3. 存檔
        final_df.to_csv(output_csv, index=False, encoding="utf-8-sig")
        print(f"\n✨ 任務完成！全新的資料庫已建立，共 {len(final_df)} 筆數據。")
    else:
        print("\n💥 發生意外，沒有抓到任何資料。")


def load_and_preprocess_data(path):
    df = pd.read_csv(path)
    df = clean_stock_data(df)
    df = calculate_technical_indicators(df)
    df = detect_crossover_MA_signals(df)
    df = detect_crossover_KD_signals(df)
    return df


def train_all_stocks(df):
    # 建立一個資料夾來存放所有模型
    os.makedirs("models", exist_ok=True)

    # --- 2. 針對「每一支股票」獨立訓練 ---
    # 利用 groupby 把每一支股票的資料分開
    for stock_id, stock_data in df.groupby("stock_id"):
        print(f"\n{'=' * 15} 開始訓練 {stock_id} {'=' * 15}")

        # 該支股票的標註與特徵工程
        df_labeled = label_data_for_ai(stock_data.copy())

        # 確保資料量足夠 (例如大於 500 筆才訓練，避免新上市股票資料太少)
        if len(df_labeled) < 500:
            print(f"⚠️ {stock_id} 資料筆數不足 ({len(df_labeled)} 筆)，跳過訓練。")
            continue

        X, y, regime = prepare_features_for_ai(df_labeled)

        # 時間序列絕對不能 shuffle
        X_train, X_test, y_train, y_test, r_train, r_test = train_test_split(
            X, y, regime, shuffle=False, test_size=0.2
        )

        # --- 1. Random Forest 測試 ---
        rf_model = RandomForestClassifier(
            n_estimators=100, max_depth=4, min_samples_leaf=7,
            class_weight='balanced', random_state=42
        )
        rf_model.fit(X_train, y_train)  # 明確訓練
        rf_acc = rf_model.score(X_test, y_test)

        # --- 2. XGBoost 測試 ---
        num_neg = (y_train == 0).sum()
        num_pos = (y_train == 1).sum()
        ratio = num_neg / num_pos if num_pos > 0 else 1

        xgb_model = XGBClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, scale_pos_weight=ratio,
            random_state=42, eval_metric='logloss'
        )
        xgb_model.fit(X_train, y_train)  # 明確訓練
        xgb_acc = xgb_model.score(X_test, y_test)

        # --- 3. 決選最佳模型並存檔 ---
        best_model = rf_model if rf_acc > xgb_acc else xgb_model

        # 儲存最佳模型給 Streamlit 使用
        model_filename = f"models/best_model_{stock_id}.pkl"
        joblib.dump(best_model, model_filename)


def full_training_pipeline():
    # 1. 抓資料
    fetch_all_history_to_csv()

    # 2. 載入 + 特徵工程
    df = load_and_preprocess_data("data/stock_history_yfinance.csv")

    # 3. 訓練
    train_all_stocks(df)

    print("🎉 全部完成")


if __name__ == "__main__":
    print("🚀 啟動每日資料更新與模型訓練流程...")
    full_training_pipeline()

