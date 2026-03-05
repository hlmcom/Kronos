# -*- coding: utf-8 -*-
"""
predict_usdjpy_mt5.py

Description:
    使用 Kronos 模型对 USD/JPY 1H K线进行零样本推理预测。
    通过 MetaTrader5 获取历史数据，运行模型推理，并可视化结果。

Usage:
    python predict_usdjpy_mt5.py
    python predict_usdjpy_mt5.py --symbol USDJPY --pred_len 48
    python predict_usdjpy_mt5.py --model NeoQuasar/Kronos-base

Requirements:
    - MetaTrader5 终端已安装并正在运行
    - pip install MetaTrader5
"""

import os
import argparse
import datetime
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import MetaTrader5 as mt5

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from model import Kronos, KronosTokenizer, KronosPredictor

# ========================================================================
# 默认配置
# ========================================================================
DEFAULT_SYMBOL = "USDJPY.cl"
DEFAULT_TIMEFRAME = mt5.TIMEFRAME_H1
DEFAULT_LOOKBACK = 400
DEFAULT_PRED_LEN = 24
DEFAULT_MODEL = "NeoQuasar/Kronos-small"
DEFAULT_TOKENIZER = "NeoQuasar/Kronos-Tokenizer-base"
MAX_CONTEXT = 512
DEVICE = "cpu"

# 模型大小配置
MODEL_SIZE_MAP = {
    "mini":  {"model": "NeoQuasar/Kronos-mini",  "tokenizer": "NeoQuasar/Kronos-Tokenizer-2k",   "max_context": 2048},
    "small": {"model": "NeoQuasar/Kronos-small", "tokenizer": "NeoQuasar/Kronos-Tokenizer-base", "max_context": 512},
    "base":  {"model": "NeoQuasar/Kronos-base",  "tokenizer": "NeoQuasar/Kronos-Tokenizer-base", "max_context": 512},
}

# 时区设置
TIMEZONE = "Asia/Tokyo"  # 日本时间 (JST, UTC+9)

# 采样参数
TEMPERATURE = 0.8
TOP_P = 0.9
SAMPLE_COUNT = 5

# 输出目录基础路径
OUTPUTS_ROOT = os.path.join(os.path.dirname(__file__), "outputs")
today_jst = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).strftime("%Y-%m-%d")

def get_save_dir(timeframe_str: str) -> str:
    """获取输出目录: outputs/{timeframe}/{date}/"""
    return os.path.join(OUTPUTS_ROOT, timeframe_str, today_jst)

# MT5 时间周期映射
TIMEFRAME_MAP = {
    "M1": mt5.TIMEFRAME_M1,
    "M5": mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "M30": mt5.TIMEFRAME_M30,
    "H1": mt5.TIMEFRAME_H1,
    "H4": mt5.TIMEFRAME_H4,
    "D1": mt5.TIMEFRAME_D1,
    "W1": mt5.TIMEFRAME_W1,
}

# 时间周期对应的 pandas 频率
TIMEFRAME_FREQ = {
    "M1": "1min",
    "M5": "5min",
    "M15": "15min",
    "M30": "30min",
    "H1": "1h",
    "H4": "4h",
    "D1": "1D",
    "W1": "1W",
}


def init_mt5():
    """初始化 MT5 连接"""
    if not mt5.initialize():
        print(f"❌ MT5 初始化失败: {mt5.last_error()}")
        print("   请确保 MetaTrader5 终端已安装并正在运行。")
        sys.exit(1)

    terminal_info = mt5.terminal_info()
    print(f"✅ MT5 已连接: {terminal_info.name}")
    print(f"   路径: {terminal_info.path}")
    print(f"   公司: {terminal_info.company}")


def fetch_mt5_data(symbol: str, timeframe, n_bars: int) -> pd.DataFrame:
    """从 MT5 获取 K 线数据"""
    print(f"📥 正在获取 {symbol} 数据（最近 {n_bars} 根 K线）...")

    # 检查品种是否存在
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        print(f"❌ 品种 {symbol} 未找到。")
        available = mt5.symbols_get()
        fx_symbols = [s.name for s in available if "JPY" in s.name][:10]
        if fx_symbols:
            print(f"   相关品种: {fx_symbols}")
        mt5.shutdown()
        sys.exit(1)

    if not symbol_info.visible:
        if not mt5.symbol_select(symbol, True):
            print(f"❌ 无法选择品种 {symbol}")
            mt5.shutdown()
            sys.exit(1)

    # 获取数据
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, n_bars)
    if rates is None or len(rates) == 0:
        print(f"❌ 获取 {symbol} 数据失败: {mt5.last_error()}")
        mt5.shutdown()
        sys.exit(1)

    df = pd.DataFrame(rates)

    # MT5 copy_rates_from_pos 返回的 time 字段是 broker 服务器时间的伪 Unix 时间戳
    # （OANDA 服务器为 UTC+2/+3），不是标准 UTC。
    # 需要：服务器时间 → UTC → JST
    # 用最新 bar 的时间戳自动检测偏移（tick 可能缓存过期，bar 数据始终实时）
    import math
    latest_bar_ts = rates[-1][0]  # 最新 bar 的 raw 时间戳
    bar_server_time = datetime.datetime.fromtimestamp(latest_bar_ts, tz=datetime.timezone.utc)
    utc_now = datetime.datetime.now(datetime.timezone.utc)
    # 最新 bar 是当前正在走的 bar，其服务器时间 ≥ 真实 UTC
    # 用 ceil 确保即使在整点附近也能正确检测
    diff_seconds = (bar_server_time - utc_now).total_seconds()
    offset_hours = math.ceil(diff_seconds / 3600)
    # 合理性校验
    if not (0 <= offset_hours <= 14):
        print(f"   ⚠️ 检测到异常偏移 {offset_hours}h，回退为 UTC+2（OANDA 默认）")
        offset_hours = 2
    print(f"   券商服务器时区: UTC+{offset_hours}")

    df["timestamps"] = (
        pd.to_datetime(df["time"], unit="s")
        - pd.Timedelta(hours=offset_hours)  # 服务器时间 → UTC
        + pd.Timedelta(hours=9)             # UTC → JST (UTC+9)
    )
    print(f"   时间转换: 服务器(UTC+{offset_hours}) → JST(UTC+9)")
    df = df.rename(columns={
        "tick_volume": "volume",
    })

    # 外汇 tick_volume 作为 volume，amount = volume * close
    df["amount"] = df["volume"] * df["close"]

    # 只保留需要的列
    df = df[["timestamps", "open", "high", "low", "close", "volume", "amount"]]
    df = df.sort_values("timestamps").reset_index(drop=True)

    print(f"✅ 数据获取成功: {len(df)} 行")
    print(f"   时间范围 (JST): {df['timestamps'].iloc[0]} ~ {df['timestamps'].iloc[-1]}")
    print(f"   最新价格: {df['close'].iloc[-1]:.3f}")
    print(f"   数据预览:")
    print(df.tail(5).to_string(index=False))

    return df


def generate_future_timestamps(last_timestamp: pd.Timestamp, pred_len: int,
                                timeframe_str: str) -> pd.Series:
    """生成未来的时间戳（跳过外汇市场周末休市时段）

    外汇市场（JST 基准）：
    - 收盘：周六 ~07:00 JST（UTC 周五 22:00）
    - 开盘：周一 ~07:00 JST（UTC 周日 22:00）
    - 保守设定：周六 08:00 ~ 周一 07:00 为休市时段
    """
    freq = TIMEFRAME_FREQ.get(timeframe_str, "1h")
    delta = pd.Timedelta(freq)

    timestamps = []
    current = last_timestamp + delta

    while len(timestamps) < pred_len:
        weekday = current.weekday()  # 0=Mon, 5=Sat, 6=Sun
        hour = current.hour

        # 判断是否在周末休市时段 (JST)
        # 周六 08:00 之后 ~ 周一 07:00 之前
        is_weekend = False
        if weekday == 5 and hour >= 8:  # 周六 08:00 以后
            is_weekend = True
        elif weekday == 6:  # 整个周日
            is_weekend = True
        elif weekday == 0 and hour < 7:  # 周一 07:00 之前
            is_weekend = True

        if is_weekend:
            # 跳到下周一 07:00 JST
            days_to_monday = (7 - weekday) % 7
            if days_to_monday == 0:
                days_to_monday = 7 if weekday == 0 and hour >= 7 else 0
            if weekday == 0 and hour < 7:
                # 已经是周一，只需跳到 07:00
                current = current.replace(hour=7, minute=0, second=0)
            else:
                # 跳到下周一 07:00
                next_monday = current + pd.Timedelta(days=days_to_monday)
                current = next_monday.replace(hour=7, minute=0, second=0)
            continue

        timestamps.append(current)
        current = current + delta

    return pd.Series(timestamps)


def run_prediction(df: pd.DataFrame, lookback: int, pred_len: int,
                   model_name: str, tokenizer_name: str,
                   timeframe_str: str, sample_count: int = SAMPLE_COUNT):
    """执行模型推理"""
    # 加载模型
    print(f"\n🚀 加载模型...")
    print(f"   Tokenizer: {tokenizer_name}")
    print(f"   Model: {model_name}")
    print(f"   Device: {DEVICE}")

    tokenizer = KronosTokenizer.from_pretrained(tokenizer_name)
    model = Kronos.from_pretrained(model_name)
    predictor = KronosPredictor(model, tokenizer, device=DEVICE, max_context=MAX_CONTEXT)

    # 准备输入
    actual_lookback = min(lookback, len(df))
    x_df = df.iloc[-actual_lookback:][["open", "high", "low", "close", "volume", "amount"]].reset_index(drop=True)
    x_timestamp = df.iloc[-actual_lookback:]["timestamps"].reset_index(drop=True)
    y_timestamp = generate_future_timestamps(
        df["timestamps"].iloc[-1], pred_len, timeframe_str
    )

    print(f"\n🔮 开始推理...")
    print(f"   回看窗口: {actual_lookback} 根 K线")
    print(f"   预测长度: {pred_len} 根 K线")
    print(f"   采样温度: {TEMPERATURE}")
    print(f"   采样次数: {sample_count}")

    import torch
    with torch.no_grad():
        pred_df = predictor.predict(
            df=x_df,
            x_timestamp=x_timestamp,
            y_timestamp=y_timestamp,
            pred_len=pred_len,
            T=TEMPERATURE,
            top_p=TOP_P,
            sample_count=sample_count,
            verbose=True,
        )

    pred_df["timestamps"] = y_timestamp.values

    print(f"\n✅ 推理完成！")
    print(f"   预测结果预览:")
    print(pred_df[["timestamps", "open", "high", "low", "close"]].head(10).to_string(index=False))

    return pred_df



def plot_results(df: pd.DataFrame, pred_df: pd.DataFrame, symbol: str,
                 lookback: int, timeframe_str: str, save_dir: str):
    """绘制K线蜡烛图（去除周末非交易时段空白）"""
    os.makedirs(save_dir, exist_ok=True)

    # 取最近的历史数据用于展示
    display_len = min(lookback, 120)
    hist_df = df.iloc[-display_len:].copy().reset_index(drop=True)
    pred_df_plot = pred_df.copy().reset_index(drop=True)

    # 合并历史和预测数据，使用顺序整数索引（消除周末间隔）
    n_hist = len(hist_df)
    n_pred = len(pred_df_plot)
    n_total = n_hist + n_pred

    all_open = list(hist_df["open"]) + list(pred_df_plot["open"])
    all_high = list(hist_df["high"]) + list(pred_df_plot["high"])
    all_low = list(hist_df["low"]) + list(pred_df_plot["low"])
    all_close = list(hist_df["close"]) + list(pred_df_plot["close"])
    all_timestamps = list(hist_df["timestamps"]) + list(pred_df_plot["timestamps"])

    fig, axes = plt.subplots(2, 1, figsize=(16, 9), sharex=True,
                             gridspec_kw={"height_ratios": [4, 1]})

    # ---- 子图1: K线蜡烛图 ----
    ax1 = axes[0]
    candle_width = 0.6
    wick_width = 0.15

    for i in range(n_total):
        o, h, l, c = all_open[i], all_high[i], all_low[i], all_close[i]
        is_pred = i >= n_hist

        if is_pred:
            # 预测K线：标准红绿色
            color_up = "#26A69A"    # 绿色（阳线）
            color_down = "#EF5350"  # 红色（阴线）
        else:
            # 历史K线：蓝灰色系
            color_up = "#64B5F6"    # 浅蓝（阳线）
            color_down = "#90A4AE"  # 灰蓝（阴线）

        if c >= o:
            body_color = color_up
            body_bottom = o
            body_height = c - o
        else:
            body_color = color_down
            body_bottom = c
            body_height = o - c

        # 绘制影线（wick）
        ax1.plot([i, i], [l, h], color=body_color, linewidth=wick_width * 2)
        # 绘制实体（body）
        ax1.bar(i, body_height, bottom=body_bottom, width=candle_width,
                color=body_color, edgecolor=body_color, linewidth=0.5)

    # 分界线
    divider_x = n_hist - 0.5
    ax1.axvline(x=divider_x, color="#FF9800", linestyle="--",
                alpha=0.8, linewidth=1.5, label="← Historical | Predicted →")

    ax1.set_title(f"Kronos Prediction: {symbol} ({timeframe_str}) - JST",
                  fontsize=14, fontweight="bold")
    ax1.set_ylabel("Price", fontsize=12)
    ax1.legend(loc="upper left", fontsize=10)
    ax1.grid(True, alpha=0.2, axis="y")

    # ---- 子图2: 预测涨跌柱状图 ----
    ax2 = axes[1]
    for i in range(n_pred):
        idx = n_hist + i
        change = all_close[idx] - all_open[idx]
        color = "#26A69A" if change >= 0 else "#EF5350"
        ax2.bar(idx, change, color=color, alpha=0.8, width=candle_width)

    ax2.axhline(y=0, color="gray", linewidth=0.5)
    ax2.set_ylabel("Pred Δ Price", fontsize=10)
    ax2.set_xlabel("Time (JST)", fontsize=12)
    ax2.grid(True, alpha=0.2, axis="y")

    # ---- 横轴标签：用时间标注，间隔显示避免拥挤 ----
    tick_interval = max(1, n_total // 15)
    tick_positions = list(range(0, n_total, tick_interval))
    tick_labels = []
    for pos in tick_positions:
        ts = all_timestamps[pos]
        if isinstance(ts, pd.Timestamp):
            tick_labels.append(ts.strftime("%m-%d\n%H:%M"))
        else:
            tick_labels.append(str(ts))

    ax2.set_xticks(tick_positions)
    ax2.set_xticklabels(tick_labels, fontsize=8)
    ax2.set_xlim(-1, n_total)

    plt.tight_layout()

    chart_path = os.path.join(save_dir, f"pred_{symbol}_{timeframe_str}_chart.png")
    plt.savefig(chart_path, dpi=150)
    plt.close()
    print(f"📊 图表已保存: {chart_path}")

    return chart_path


def save_results(df: pd.DataFrame, pred_df: pd.DataFrame, symbol: str,
                 timeframe_str: str, save_dir: str):
    """保存预测结果到 CSV"""
    os.makedirs(save_dir, exist_ok=True)

    # 保存预测数据
    pred_path = os.path.join(save_dir, f"pred_{symbol}_{timeframe_str}_data.csv")
    pred_df.to_csv(pred_path, index=False)
    print(f"💾 预测数据已保存: {pred_path}")

    # 保存完整数据（历史 + 预测）
    hist_cols = ["timestamps", "open", "high", "low", "close", "volume", "amount"]
    pred_export = pred_df[hist_cols].copy()
    full_df = pd.concat([df[hist_cols], pred_export]).reset_index(drop=True)
    full_path = os.path.join(save_dir, f"pred_{symbol}_{timeframe_str}_full.csv")
    full_df.to_csv(full_path, index=False)
    print(f"💾 完整数据已保存: {full_path}")


def load_latest_prediction(symbol: str, timeframe_str: str):
    """加载最近一次的预测数据（排除今天）"""
    outputs_root = os.path.join(OUTPUTS_ROOT, timeframe_str)
    filename = f"pred_{symbol}_{timeframe_str}_data.csv"

    if not os.path.isdir(outputs_root):
        print(f"⚠️ 输出目录不存在: {outputs_root}")
        return None, None

    # 扫描所有日期子文件夹（格式 YYYY-MM-DD），排除今天
    date_dirs = []
    for name in os.listdir(outputs_root):
        dirpath = os.path.join(outputs_root, name)
        if os.path.isdir(dirpath) and name != today_jst:
            # 验证是否为有效日期格式
            try:
                datetime.datetime.strptime(name, "%Y-%m-%d")
                pred_file = os.path.join(dirpath, filename)
                if os.path.exists(pred_file):
                    date_dirs.append((name, pred_file))
            except ValueError:
                continue

    # 也检查 outputs 根目录（兼容旧版）
    root_pred = os.path.join(outputs_root, filename)
    if os.path.exists(root_pred) and not date_dirs:
        # 旧版文件，用其修改时间作为日期
        mtime = os.path.getmtime(root_pred)
        old_date = datetime.datetime.fromtimestamp(
            mtime, tz=datetime.timezone(datetime.timedelta(hours=9))
        ).strftime("%Y-%m-%d")
        if old_date != today_jst:
            date_dirs.append((old_date, root_pred))

    if not date_dirs:
        print(f"⚠️ 没有找到之前的预测文件")
        return None, None

    # 按日期排序，取最新的
    date_dirs.sort(key=lambda x: x[0], reverse=True)
    latest_date, latest_path = date_dirs[0]

    pred_df = pd.read_csv(latest_path, parse_dates=["timestamps"])
    print(f"📂 加载最近预测 ({latest_date}): {latest_path}")
    print(f"   共 {len(pred_df)} 行")
    return pred_df, latest_date


def plot_evaluation(actual_df: pd.DataFrame, pred_df: pd.DataFrame,
                    symbol: str, timeframe_str: str, eval_date: str,
                    save_dir: str):
    """绘制昨天预测 vs 今天实际走势的对比K线图"""
    os.makedirs(save_dir, exist_ok=True)

    # 找到预测时间范围对应的实际数据
    pred_start = pred_df["timestamps"].iloc[0]
    pred_end = pred_df["timestamps"].iloc[-1]

    # 取预测范围内的实际数据
    mask = (actual_df["timestamps"] >= pred_start) & (actual_df["timestamps"] <= pred_end)
    actual_in_range = actual_df[mask].copy().reset_index(drop=True)

    if len(actual_in_range) == 0:
        print(f"⚠️ 预测时间范围内没有实际数据（可能市场还未开盘）")
        return None

    # 也取预测范围前的一些历史数据作为背景
    before_mask = actual_df["timestamps"] < pred_start
    hist_before = actual_df[before_mask].tail(24).copy().reset_index(drop=True)

    n_hist = len(hist_before)
    n_actual = len(actual_in_range)
    n_pred = len(pred_df)

    print(f"📊 评估对比: 预测 {n_pred} 根 vs 实际 {n_actual} 根")

    fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1]})

    # ---- 子图1: 对比K线图 ----
    ax1 = axes[0]
    candle_width = 0.35

    # 绘制历史背景K线（灰色）
    for i in range(n_hist):
        o, h, l, c = hist_before.iloc[i][["open", "high", "low", "close"]]
        color = "#B0BEC5" if c >= o else "#78909C"
        body_bottom = min(o, c)
        body_height = abs(c - o)
        ax1.plot([i, i], [l, h], color=color, linewidth=0.3)
        ax1.bar(i, body_height, bottom=body_bottom, width=candle_width * 2,
                color=color, edgecolor=color, linewidth=0.5)

    # 分界线
    divider_x = n_hist - 0.5
    ax1.axvline(x=divider_x, color="#FF9800", linestyle="--",
                alpha=0.8, linewidth=1.5)

    # 绘制实际K线（左偏移，蓝色系）
    for i in range(n_actual):
        idx = n_hist + i
        o, h, l, c = actual_in_range.iloc[i][["open", "high", "low", "close"]]
        color = "#1565C0" if c >= o else "#42A5F5"
        body_bottom = min(o, c)
        body_height = abs(c - o)
        ax1.plot([idx - candle_width / 2, idx - candle_width / 2], [l, h],
                 color=color, linewidth=0.3)
        ax1.bar(idx - candle_width / 2, body_height, bottom=body_bottom,
                width=candle_width, color=color, edgecolor=color, linewidth=0.5,
                label="Actual" if i == 0 else "")

    # 绘制预测K线（右偏移，红绿色）
    for i in range(n_pred):
        idx = n_hist + i
        o, h, l, c = pred_df.iloc[i][["open", "high", "low", "close"]]
        color_up = "#26A69A"
        color_down = "#EF5350"
        color = color_up if c >= o else color_down
        body_bottom = min(o, c)
        body_height = abs(c - o)
        ax1.plot([idx + candle_width / 2, idx + candle_width / 2], [l, h],
                 color=color, linewidth=0.3)
        ax1.bar(idx + candle_width / 2, body_height, bottom=body_bottom,
                width=candle_width, color=color, edgecolor=color, linewidth=0.5,
                alpha=0.6, label="Predicted" if i == 0 else "")

    ax1.set_title(f"Evaluation: {symbol} ({timeframe_str}) - {eval_date} Prediction vs Actual - JST",
                  fontsize=13, fontweight="bold")
    ax1.set_ylabel("Price", fontsize=12)
    ax1.legend(loc="upper left", fontsize=10)
    ax1.grid(True, alpha=0.2, axis="y")

    # ---- 子图2: 误差图 (预测close - 实际close) ----
    ax2 = axes[1]
    n_compare = min(n_actual, n_pred)
    errors = []
    for i in range(n_compare):
        pred_close = pred_df.iloc[i]["close"]
        actual_close = actual_in_range.iloc[i]["close"]
        error = pred_close - actual_close
        errors.append(error)
        idx = n_hist + i
        color = "#26A69A" if error >= 0 else "#EF5350"
        ax2.bar(idx, error, color=color, alpha=0.8, width=candle_width * 2)

    ax2.axhline(y=0, color="gray", linewidth=0.5)
    ax2.set_ylabel("Error\n(Pred - Actual)", fontsize=10)
    ax2.set_xlabel("Time (JST)", fontsize=12)
    ax2.grid(True, alpha=0.2, axis="y")

    # 计算误差指标
    if errors:
        mae = np.mean(np.abs(errors))
        rmse = np.sqrt(np.mean(np.array(errors) ** 2))
        direction_correct = sum(
            1 for i in range(n_compare)
            if (pred_df.iloc[i]["close"] - pred_df.iloc[i]["open"]) *
               (actual_in_range.iloc[i]["close"] - actual_in_range.iloc[i]["open"]) > 0
        )
        direction_acc = direction_correct / n_compare * 100 if n_compare > 0 else 0

        stats_text = f"MAE={mae:.3f}  RMSE={rmse:.3f}  Direction Acc={direction_acc:.0f}% ({direction_correct}/{n_compare})"
        ax2.set_title(stats_text, fontsize=10, color="#666")

    # 横轴标签
    all_timestamps = list(hist_before["timestamps"]) + list(pred_df["timestamps"])
    n_total = n_hist + n_pred
    tick_interval = max(1, n_total // 12)
    tick_positions = list(range(0, n_total, tick_interval))
    tick_labels = [all_timestamps[p].strftime("%m-%d\n%H:%M") if isinstance(all_timestamps[p], pd.Timestamp) else str(all_timestamps[p]) for p in tick_positions]
    ax2.set_xticks(tick_positions)
    ax2.set_xticklabels(tick_labels, fontsize=8)
    ax2.set_xlim(-1, n_total)

    plt.tight_layout()

    chart_path = os.path.join(save_dir, f"eval_{symbol}_{timeframe_str}_{eval_date}.png")
    plt.savefig(chart_path, dpi=150)
    plt.close()
    print(f"📊 评估图表已保存: {chart_path}")

    return chart_path


def main():
    parser = argparse.ArgumentParser(
        description="Kronos USD/JPY 预测脚本（MT5 数据源）"
    )
    parser.add_argument("--symbol", type=str, default=DEFAULT_SYMBOL,
                        help=f"MT5 品种名 (默认: {DEFAULT_SYMBOL})")
    parser.add_argument("--timeframe", type=str, default="H1",
                        choices=list(TIMEFRAME_MAP.keys()),
                        help="K线周期 (默认: H1)")
    parser.add_argument("--lookback", type=int, default=DEFAULT_LOOKBACK,
                        help=f"回看窗口长度 (默认: {DEFAULT_LOOKBACK})")
    parser.add_argument("--pred_len", type=int, default=DEFAULT_PRED_LEN,
                        help=f"预测长度 (默认: {DEFAULT_PRED_LEN})")
    parser.add_argument("--model_size", type=str, default="small",
                        choices=list(MODEL_SIZE_MAP.keys()),
                        help="模型大小: mini(4.1M), small(24.7M), base(102.3M) (默认: small)")
    parser.add_argument("--sample_count", type=int, default=SAMPLE_COUNT,
                        help=f"采样次数 (默认: {SAMPLE_COUNT})")
    parser.add_argument("--skip_eval", action="store_true",
                        help="跳过昨天预测的评估")
    parser.add_argument("--start_time", type=str, default=None,
                        help="预测起始时间 JST (格式: YYYYMMDDHHmm, 如 202602210300)")
    args = parser.parse_args()

    # 解析模型配置
    model_cfg = MODEL_SIZE_MAP[args.model_size]
    model_name = model_cfg["model"]
    tokenizer_name = model_cfg["tokenizer"]
    max_context = model_cfg["max_context"]

    print("=" * 60)
    print("  Kronos 外汇预测 - USD/JPY")
    print(f"  模型: {model_name} ({args.model_size})")
    print("=" * 60)

    # 1. 初始化 MT5
    init_mt5()

    try:
        # 2. 获取数据（多取一些以应对数据缺失）
        mt5_timeframe = TIMEFRAME_MAP[args.timeframe]
        n_bars = args.lookback + 100  # 多取 100 根作为缓冲
        df = fetch_mt5_data(args.symbol, mt5_timeframe, n_bars)

        # 如果指定了 start_time，截断数据到该时间点
        if args.start_time:
            try:
                start_dt = pd.to_datetime(args.start_time, format="%Y%m%d%H%M")
            except ValueError:
                print(f"❌ start_time 格式错误: {args.start_time}，应为 YYYYMMDDHHmm")
                sys.exit(1)
            df = df[df["timestamps"] <= start_dt].copy().reset_index(drop=True)
            if len(df) == 0:
                print(f"❌ 截断后无数据，请检查 start_time: {start_dt}")
                sys.exit(1)
            print(f"⏰ 数据截断至: {start_dt} (JST)")
            print(f"   截断后剩余 {len(df)} 行，最后时间: {df['timestamps'].iloc[-1]}")

        # ============================================
        # 图1: 评估最近一次预测
        # ============================================
        if not args.skip_eval and not args.start_time:
            print(f"\n{'=' * 60}")
            print("  📈 Step 1: 评估最近一次预测")
            print(f"{'=' * 60}")

            latest_pred, eval_date = load_latest_prediction(
                args.symbol, args.timeframe
            )
            if latest_pred is not None:
                save_dir = get_save_dir(args.timeframe)
                plot_evaluation(df, latest_pred, args.symbol,
                              args.timeframe, eval_date, save_dir)
            else:
                print("   跳过评估（无历史预测数据）")

        # ============================================
        # 图2: 生成预测
        # ============================================
        print(f"\n{'=' * 60}")
        print("  🔮 Step 2: 生成预测")
        print(f"{'=' * 60}")

        pred_df = run_prediction(
            df=df,
            lookback=args.lookback,
            pred_len=args.pred_len,
            model_name=model_name,
            tokenizer_name=tokenizer_name,
            timeframe_str=args.timeframe,
            sample_count=args.sample_count,
        )

        # 4. 保存结果
        save_dir = get_save_dir(args.timeframe)
        save_results(df, pred_df, args.symbol, args.timeframe, save_dir)

        # 5. 绘图
        plot_results(df, pred_df, args.symbol, args.lookback, args.timeframe, save_dir)

        print(f"\n🎉 预测完成！结果保存在 {save_dir}/")

    finally:
        mt5.shutdown()
        print("🔌 MT5 连接已关闭")


if __name__ == "__main__":
    main()
