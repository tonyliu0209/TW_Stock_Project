import os
import pandas as pd
import joblib
import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

plt.rcParams['font.family'] = 'Microsoft JhengHei'  # 微軟正黑體
plt.rcParams['axes.unicode_minus'] = False  # 避免負號變方塊
plt.style.use('ggplot')

from features import (
    FEATURES_COLS,
    clean_stock_data,
    calculate_technical_indicators,
    detect_crossover_MA_signals,
    detect_crossover_KD_signals,
    get_ai_prediction_for_stock,
    get_feature_importance
)

# --- 1. 新增：動態載入個股模型的快取函數 ---
@st.cache_resource
def load_specific_model(stock_id):
    """根據 stock_id 去抓取對應的最佳模型"""
    model_path = f"models/best_model_{stock_id}.pkl"
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

# 使用快取，避免重複讀取與計算
@st.cache_data
def load_and_preprocess_data():
    df = pd.read_csv("data/stock_history_yfinance.csv")
    full_df = clean_stock_data(df)
    full_df = calculate_technical_indicators(full_df)
    full_df = detect_crossover_MA_signals(full_df)
    full_df = detect_crossover_KD_signals(full_df)
    return full_df


def plot_stock_chart_interactive(df, stock_name, plot_pred, plot_conf, plot_threshold):

    stock_df = df[df['name'] == stock_name].tail(60).copy()
    stock_df = stock_df.sort_values('date')

    # 保證型態正確
    stock_df['date'] = pd.to_datetime(stock_df['date'])

    # =======================
    # 建立 subplot
    # =======================
    interactive_fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.7, 0.3]
    )

    # =======================
    # K線
    # =======================
    interactive_fig.add_trace(go.Candlestick(
        x=stock_df['date'],
        open=stock_df['open'],
        high=stock_df['high'],
        low=stock_df['low'],
        close=stock_df['price'],
        name='K線',
        increasing_line_color='#d62728',
        decreasing_line_color='#2ca02c'
    ), row=1, col=1)

    # MA
    interactive_fig.add_trace(go.Scatter(
        x=stock_df['date'],
        y=stock_df['MA5'],
        line=dict(color='orange', width=1.5),
        name='MA5'
    ), row=1, col=1)

    interactive_fig.add_trace(go.Scatter(
        x=stock_df['date'],
        y=stock_df['MA20'],
        line=dict(color='#2E5A88', width=1.5),
        name='MA20'
    ), row=1, col=1)

    # =======================
    # Volume
    # =======================
    colors = ['#d62728' if c >= o else '#2ca02c'
              for o, c in zip(stock_df['open'], stock_df['price'])]

    interactive_fig.add_trace(go.Bar(
        x=stock_df['date'],
        y=stock_df['volume'],
        marker_color=colors,
        name='成交量',
        showlegend=False
    ), row=2, col=1)

    # =======================
    # 非交易日處理
    # =======================
    all_days = pd.date_range(
        start=stock_df['date'].min(),
        end=stock_df['date'].max(),
        freq='D'
    )

    missing_days = all_days.difference(stock_df['date']).to_list()

    # =======================
    # Layout
    # =======================
    interactive_fig.update_layout(
        title=f"{stock_name} 技術分析（K線）",
        yaxis_title="股價",
        template="plotly_white",
        height=550,
        hovermode='x unified',
        legend=dict(
            orientation="h",
            y=1.02,
            x=1,
            xanchor="right"
        )
    )

    # 預測點
    interactive_fig = add_ai_prediction_marker(
        interactive_fig,
        stock_df,
        plot_pred,
        plot_conf,
        plot_threshold
    )

    # X 軸統一設定
    interactive_fig.update_xaxes(
        rangebreaks=[
            dict(bounds=["sat", "mon"]),
            dict(values=missing_days)
        ],
        type="date",
        showgrid=True,
        gridcolor='#f0f0f0'
    )

    # 上圖不顯示日期
    interactive_fig.update_xaxes(showticklabels=False, rangeslider_visible=False, row=1, col=1)

    # 下圖顯示日期
    interactive_fig.update_xaxes(title_text="日期", row=2, col=1)

    # Y 軸
    interactive_fig.update_yaxes(gridcolor='#f0f0f0', row=1, col=1)
    interactive_fig.update_yaxes(title_text="成交量", showgrid=False, row=2, col=1)

    return interactive_fig


def add_ai_prediction_marker(pred_fig, stock_df, prediction, confidence, threshold):
    """
    在 K 線圖上加入 AI 預測點（含觀望邏輯）

    Parameters:
    - fig: plotly figure
    - stock_df: 當前股票資料（已排序）
    - prediction: 1 (漲) / 0 (跌)
    - confidence: 模型信心 (0~1)
    - threshold: 信心門檻
    """

    # =======================
    # 🎯 Decision Layer
    # =======================
    if confidence < threshold:
        label = "觀望"
        color = "gray"
    else:
        if prediction == 1:
            label = "看多"
            color = "#d62728"  # 台股紅
        else:
            label = "看空"
            color = "#2ca02c"  # 台股綠

    # =======================
    # 預測位置（X軸）
    # =======================
    last_date = stock_df['date'].iloc[-1]

    # 往後推一天（簡化版）
    pred_date = last_date + pd.Timedelta(days=1)

    # =======================
    # 預測位置（Y軸）
    # =======================
    last_price = stock_df['price'].iloc[-1]

    y_offset = last_price * 0.01
    pred_price = last_price + y_offset

    # =======================
    # 畫「預測點」
    # =======================
    pred_fig.add_trace(go.Scatter(
        x=[pred_date],
        y=[pred_price],
        mode='markers+text',
        marker=dict(
            color=color,
            size=12,
            line=dict(color='black', width=1)
        ),
        text=[label],
        textposition='top center',
        name='AI預測'
    ), row=1, col=1)

    # =======================
    # Annotation
    # =======================
    pred_fig.add_annotation(
        x=pred_date,
        y=pred_price,
        text=f"{label}<br>Conf: {confidence:.0%}",
        showarrow=True,
        arrowhead=2,
        ax=0,
        ay=-40,
        # bgcolor="rgba(255,255,255,0.8)",
        bordercolor=color,
        borderwidth=1
    )

    return pred_fig


def plot_feature_importance(importance_df):
    feat_fig, ax = plt.subplots(figsize=(8, 5))

    # 取前10個
    df = importance_df.head(10).sort_values(by='importance')

    # 橫條圖
    ax.barh(df['feature'], df['importance'])

    # 旋轉 x 軸
    plt.xticks(rotation=45, ha='right')

    ax.set_title("Feature Importance")
    ax.set_xlabel("重要性")

    # 移除框線
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    return feat_fig


full_df = load_and_preprocess_data()
data_date = full_df['date'].max().strftime("%Y-%m-%d")

st.title("📈 AI 個股決策支援系統")
st.caption(f"📊 資料更新至：{data_date}")

# --- 側邊欄設定 (Sidebar) ---
with st.sidebar:
    st.header("📊 系統控制盤")

    # 將選單移至左側
    available_stocks = sorted(full_df['name'].unique())
    selected_stock = st.selectbox(
        "選擇預測目標：",
        options=["-- 請選擇 --"] + available_stocks
    )

    st.divider()

    # 讓使用者自己調整「信心門檻」
    # 預設 0.65，如果 AI 信心沒超過這個值，就保守看待
    ai_threshold = st.slider("🤖 AI 信心門檻設定",
                                 0.50,
                                 1.00,
                                 0.60,
                                 help="調整系統過濾訊號的強度"
    )
    st.info("💡 提高門檻可以過濾雜訊，但可能會錯過波段。")

# 只有當使用者選了真正的股票時才執行預測
if selected_stock != "-- 請選擇 --":
    with st.spinner(f"正在分析 {selected_stock} 的數據..."):
        # 找出選中股票的 stock_id
        # 從 full_df 中過濾出名稱對應的 ID
        target_id = full_df[full_df['name'] == selected_stock]['stock_id'].iloc[0]

        # 載入該股票專屬的 pkl 模型
        current_model = load_specific_model(target_id)

        if current_model is not None:
            # 使用該專屬模型進行預測
            pred, conf = get_ai_prediction_for_stock(current_model, full_df, selected_stock)

            # --- Subtitle 1: 預測決策層 ---
            st.header(f"🎯 AI 決策建議：{selected_stock} (代號: {target_id})")

            res_col1, res_col2 = st.columns(2)
            with res_col1:
                # 根據信心門檻決定最終決策
                if conf >= ai_threshold:
                    if pred == 1:
                        if "雙重共鳴" in full_df['KD_Category'] or "強力金叉" in full_df['KD_Category']:
                            st.success("### 🔥 強烈看多 (技術面共振)")
                        else:
                            st.success("### 🚀 決策建議：看多")
                        decision_color = 'red'
                    else:
                        if "雙重死叉" in full_df['KD_Category'] or "強力死叉" in full_df['KD_Category']:
                            st.error("### 💀 強烈看空 (技術面共振)")
                        else:
                            st.error("### 📉 決策建議：看空")
                        decision_color = 'green'
                else:
                    st.warning("### 👀 決策建議：觀望")
                    st.write(f"原因：信心度 ({conf:.1%}) 未達設定門檻 ({ai_threshold:.1%})")

            with res_col2:
                diff = conf - ai_threshold  # 計算信心度與門檻的差距
                st.metric(
                    label = "模型計算信心度",
                    value = f"{conf * 100:.1f}%",
                    delta = f"{diff:.1%}",
                    delta_color = "normal"
                )

            # 分隔線
            st.divider()

            # --- Subtitle 2: 技術圖 ---
            st.header("📈 技術趨勢分析")
            fig = plot_stock_chart_interactive(full_df, selected_stock, pred, conf, ai_threshold)
            st.plotly_chart(fig, use_container_width=True)

            # 分隔線
            st.divider()

            # --- Subtitle 3: 模型解釋層 (Feature Importance) ---
            st.header("🔍 模型決策關鍵因素")
            feat_importance_df = get_feature_importance(current_model, FEATURES_COLS)

            feat_importance_df['feature'] = feat_importance_df['feature'].replace({
                'ma5_slope': '5日趨勢變化',
                # 'ret_1': '短期動能',
                'feat_K': 'K指標',
                'ma20_slope': '20日趨勢變化',
                'feat_price_ma5_ratio': '5日乖離率',
                'feat_price_ma20_ratio': '20日乖離率',
                'K_change': 'K指標變化'
            })

            fig2 = plot_feature_importance(feat_importance_df)
            st.pyplot(fig2)
        else:
            st.error(f"找不到 {selected_stock} ({target_id}) 的專屬模型，請確認 models/ 資料夾中有該檔案。")
