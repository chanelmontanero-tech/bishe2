"""
Web dashboard for ML-based Network Traffic Anomaly Detection.
Provides an interactive browser UI to view results and predict traffic.

Usage:
    pip install streamlit
    streamlit run app.py
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="网络流量异常检测",
    page_icon="🛡️",
    layout="wide",
)

MODELS_DIR = "models"
OUTPUT_DIR = os.path.join("output", "figures")
PROCESSED_DIR = os.path.join("data", "processed")

MODEL_NAMES = ["RandomForest", "DecisionTree", "XGBoost", "KNN", "SVM"]

# ── Sidebar ─────────────────────────────────────────────────────────────────────
st.sidebar.title("🛡️ 网络流量异常检测")
st.sidebar.markdown("基于机器学习的网络入侵检测系统")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "选择页面",
    ["📊 模型结果总览", "🔍 单条流量检测", "📈 详细图表"]
)

# ── Helper: load results CSV ────────────────────────────────────────────────────
@st.cache_data
def load_results():
    path = os.path.join(OUTPUT_DIR, "results.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

@st.cache_resource
def load_model(name):
    path = os.path.join(MODELS_DIR, f"{name}.pkl")
    if os.path.exists(path):
        return joblib.load(path)
    return None

@st.cache_resource
def load_scaler():
    path = os.path.join(MODELS_DIR, "scaler.pkl")
    if os.path.exists(path):
        return joblib.load(path)
    return None

@st.cache_resource
def load_selector():
    path = os.path.join(MODELS_DIR, "selector.pkl")
    if os.path.exists(path):
        return joblib.load(path)
    return None

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1: Results overview
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊 模型结果总览":
    st.title("📊 模型性能对比")
    st.markdown("以下是在 NSL-KDD 测试集上，5 种机器学习模型的检测效果对比。")

    results = load_results()
    if results is None:
        st.error("未找到结果文件。请先运行 `python main.py` 完成训练。")
    else:
        # Metrics cards
        best = results.loc[results["f1"].idxmax()]
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("最佳模型", best["model"])
        col2.metric("最高 Accuracy", f"{best['accuracy']:.4f}")
        col3.metric("最高 F1-Score", f"{best['f1']:.4f}")
        col4.metric("最高 AUC", f"{best['auc']:.4f}" if pd.notna(best.get('auc')) else "N/A")

        st.markdown("---")

        # Results table
        st.subheader("各模型指标对比表")
        display_df = results.copy()
        for col in ["accuracy", "precision", "recall", "f1", "auc"]:
            if col in display_df.columns:
                display_df[col] = display_df[col].map(
                    lambda x: f"{x:.4f}" if pd.notna(x) else "N/A"
                )
        display_df.columns = ["模型", "准确率", "精确率", "召回率", "F1分数", "AUC"]
        st.dataframe(display_df, use_container_width=True, hide_index=True)

        st.markdown("---")

        # Bar chart
        st.subheader("性能指标柱状图")
        metric_choice = st.selectbox(
            "选择指标",
            ["accuracy", "precision", "recall", "f1"],
            format_func=lambda x: {"accuracy":"准确率","precision":"精确率",
                                    "recall":"召回率","f1":"F1分数"}[x]
        )
        fig, ax = plt.subplots(figsize=(8, 4))
        colors = plt.cm.Set2(np.linspace(0, 1, len(results)))
        bars = ax.bar(results["model"], results[metric_choice], color=colors)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel(metric_choice.upper())
        ax.set_title(f"各模型 {metric_choice.upper()} 对比")
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.005,
                    f"{h:.4f}", ha="center", va="bottom", fontsize=9)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2: Single traffic prediction
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 单条流量检测":
    st.title("🔍 单条网络流量检测")
    st.markdown("输入网络流量特征，选择模型，点击检测按钮，系统将判断该流量是否为攻击。")

    scaler = load_scaler()
    selector = load_selector()
    selected_features = joblib.load(os.path.join(MODELS_DIR, "selected_features.pkl")) \
        if os.path.exists(os.path.join(MODELS_DIR, "selected_features.pkl")) else None

    if scaler is None or selector is None:
        st.error("未找到模型文件。请先运行 `python main.py` 完成训练。")
    else:
        st.markdown("### 选择检测模型")
        model_choice = st.selectbox("模型", MODEL_NAMES)
        model = load_model(model_choice)

        st.markdown("### 输入流量特征（常用字段）")
        col1, col2, col3 = st.columns(3)

        with col1:
            duration = st.number_input("持续时长 duration", min_value=0, value=0)
            src_bytes = st.number_input("源字节数 src_bytes", min_value=0, value=491)
            dst_bytes = st.number_input("目标字节数 dst_bytes", min_value=0, value=0)
            land = st.selectbox("是否同源目 land", [0, 1])
            wrong_fragment = st.number_input("错误分片数 wrong_fragment", min_value=0, value=0)

        with col2:
            urgent = st.number_input("紧急包数 urgent", min_value=0, value=0)
            hot = st.number_input("热点指示 hot", min_value=0, value=0)
            logged_in = st.selectbox("是否已登录 logged_in", [0, 1])
            count = st.number_input("连接数 count", min_value=0, value=2)
            srv_count = st.number_input("服务连接数 srv_count", min_value=0, value=2)

        with col3:
            protocol_type = st.selectbox("协议类型 protocol_type", [0, 1, 2],
                format_func=lambda x: ["icmp","tcp","udp"][x])
            service = st.number_input("服务编码 service", min_value=0, max_value=65, value=21)
            flag = st.number_input("标志编码 flag", min_value=0, max_value=10, value=9)
            same_srv_rate = st.slider("相同服务率 same_srv_rate", 0.0, 1.0, 1.0)
            diff_srv_rate = st.slider("不同服务率 diff_srv_rate", 0.0, 1.0, 0.0)

        if st.button("🚀 开始检测", type="primary"):
            # Build full 41-feature vector with defaults
            full_features = [
                duration, protocol_type, service, flag, src_bytes,
                dst_bytes, land, wrong_fragment, urgent, hot,
                0, logged_in, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                count, srv_count, 0.0, 0.0, 0.0, 0.0,
                same_srv_rate, diff_srv_rate, 0.0,
                0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            ]

            X = np.array(full_features).reshape(1, -1)
            X_scaled = scaler.transform(X)
            X_selected = selector.transform(X_scaled)

            prediction = model.predict(X_selected)[0]
            proba = model.predict_proba(X_selected)[0] \
                if hasattr(model, "predict_proba") else None

            st.markdown("---")
            if prediction == 0:
                st.success("✅ 检测结果：**正常流量 (Normal)**")
            else:
                st.error("🚨 检测结果：**异常流量 / 攻击 (Attack)**")

            if proba is not None:
                col_a, col_b = st.columns(2)
                col_a.metric("正常概率", f"{proba[0]*100:.1f}%")
                col_b.metric("攻击概率", f"{proba[1]*100:.1f}%")

                fig, ax = plt.subplots(figsize=(4, 2.5))
                ax.bar(["Normal", "Attack"], proba, color=["#2ecc71", "#e74c3c"])
                ax.set_ylim(0, 1)
                ax.set_ylabel("Probability")
                ax.set_title(f"Prediction — {model_choice}")
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3: Detailed charts
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 详细图表":
    st.title("📈 详细分析图表")

    figures = {
        "模型对比图": "model_comparison.png",
        "ROC 曲线": "roc_curves_all.png",
        "特征重要性": "feature_importance.png",
        "混淆矩阵 — RandomForest": "confusion_matrix_RandomForest.png",
        "混淆矩阵 — DecisionTree": "confusion_matrix_DecisionTree.png",
        "混淆矩阵 — XGBoost": "confusion_matrix_XGBoost.png",
        "混淆矩阵 — KNN": "confusion_matrix_KNN.png",
        "混淆矩阵 — SVM": "confusion_matrix_SVM.png",
    }

    choice = st.selectbox("选择图表", list(figures.keys()))
    img_path = os.path.join(OUTPUT_DIR, figures[choice])

    if os.path.exists(img_path):
        st.image(img_path, use_column_width=True)
    else:
        st.warning(f"图表文件不存在：{img_path}\n请先运行 `python main.py`")
