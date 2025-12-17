import numpy as np
import pandas as pd
from pathlib import Path
import joblib

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from tensorflow.keras.models import load_model

BASE_DIR = Path(__file__).parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed" / "processed_testbed"


# =========================
# 공통 유틸
# =========================
def print_result(name, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    cm = confusion_matrix(y_true, y_pred)

    print(f"\n[{name}]")
    print(f"Accuracy : {acc:.6f}")
    print(f"Macro F1 : {f1:.6f}")
    print("Confusion Matrix:")
    print(cm)
    return acc, f1, cm


def safe_metrics(name, y_true, y_pred):
    """
    표/그림용: 전환 이벤트가 0개면 NaN 반환
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    if len(y_true) == 0:
        print(f"\n[{name}] transition events = 0 -> metrics = NaN")
        return np.nan, np.nan, np.zeros((2, 2), dtype=int)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    cm = confusion_matrix(y_true, y_pred)
    return acc, f1, cm


# =========================
# Markov baseline 관련 함수
# =========================
def load_all_sessions(processed_dir=PROCESSED_DIR):
    dfs = []
    for csv_path in processed_dir.glob("*.csv"):
        df = pd.read_csv(csv_path)
        df = df.sort_values("time_sec")
        df["session_id"] = csv_path.stem
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def fit_markov_from_df(df: pd.DataFrame):
    d = df.sort_values(["session_id", "time_sec"]).reset_index(drop=True)
    r = d["rrc_state"].to_numpy().astype(int)
    sid = d["session_id"].to_numpy()

    counts = np.zeros((2, 2), dtype=np.float64)

    for i in range(1, len(r)):
        # 세션 경계 skip
        if sid[i] != sid[i - 1]:
            continue
        s = r[i - 1]
        t = r[i]
        if s in (0, 1) and t in (0, 1):
            counts[s, t] += 1

    counts += 1.0  # Laplace smoothing
    row_sums = counts.sum(axis=1, keepdims=True)
    trans_mat = counts / row_sums
    return trans_mat


def eval_markov_nextstate(df: pd.DataFrame, trans_mat, name="Markov (RRC-only)"):
    """
    전체 시점 평가(다음 상태 예측): y_true=r[t], y_pred=argmax(P(r[t-1]->next))
    세션 경계는 제외
    """
    d = df.sort_values(["session_id", "time_sec"]).reset_index(drop=True)
    r = d["rrc_state"].to_numpy().astype(int)
    sid = d["session_id"].to_numpy()

    y_true, y_pred = [], []
    for i in range(1, len(r)):
        if sid[i] != sid[i - 1]:
            continue
        prev_s = r[i - 1]
        cur_t = r[i]
        if prev_s not in (0, 1) or cur_t not in (0, 1):
            continue
        pred = int(np.argmax(trans_mat[prev_s]))
        y_true.append(cur_t)
        y_pred.append(pred)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return print_result(name, y_true, y_pred)


def markov_transition_only_from_val_df(val_df, trans_mat):
    """
    Markov transition-only:
    - 세션 경계 제거
    - 실제 전환 이벤트(r[t] != r[t-1])인 시점만 평가
    - y_pred는 prev state 기반 argmax(P(prev->next))
    """
    d = val_df.sort_values(["session_id", "time_sec"]).reset_index(drop=True)
    r = d["rrc_state"].to_numpy().astype(int)
    sid = d["session_id"].to_numpy()

    y_true, y_pred = [], []
    for i in range(1, len(r)):
        if sid[i] != sid[i - 1]:
            continue

        prev_s = r[i - 1]
        cur_t = r[i]

        if prev_s not in (0, 1) or cur_t not in (0, 1):
            continue

        # 전환 이벤트만
        if cur_t == prev_s:
            continue

        pred = int(np.argmax(trans_mat[prev_s]))
        y_true.append(cur_t)
        y_pred.append(pred)

    return np.array(y_true), np.array(y_pred)


# =========================
# Transition-only 평가/시각화 유틸
# =========================
def filter_transition_events(y_true, y_pred):
    """
    (일반 모델) transition-only:
    y_true[t] != y_true[t-1] 인 t만 남김 (첫 원소 제외)
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    if len(y_true) < 2:
        return y_true[:0], y_pred[:0]

    idx = np.where(y_true[1:] != y_true[:-1])[0] + 1
    return y_true[idx], y_pred[idx]


def plot_cm_heatmap(cm, title, save_path):
    """
    Confusion Matrix를 '정규화 + count 표기' heatmap으로 저장
    """
    cm = np.asarray(cm).astype(float)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums != 0)

    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    im = ax.imshow(cm_norm, interpolation="nearest")

    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Idle(0)", "Conn(1)"])
    ax.set_yticklabels(["Idle(0)", "Conn(1)"])

    for i in range(2):
        for j in range(2):
            txt = f"{cm_norm[i, j]:.3f}\n({int(cm[i, j])})"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


def eval_transition_only_general(name, y_true, y_pred, results_dir):
    """
    일반 모델(LSTM/CNN/RF) transition-only 평가 + heatmap 저장
    """
    yt, yp = filter_transition_events(y_true, y_pred)
    n_tr = len(yt)

    acc, f1, cm = safe_metrics(f"{name} [transition-only]", yt, yp)

    # 출력은 전환이 있을 때만 print_result 스타일로 보여주고 싶으면 아래처럼:
    if n_tr > 0:
        print_result(f"{name} [transition-only]", yt, yp)

    fname = (
        name.lower()
        .replace(" ", "_")
        .replace("+", "plus")
        .replace("/", "_")
        .replace("(", "")
        .replace(")", "")
    )
    save_path = results_dir / f"cm_transition_{fname}.png"
    if n_tr > 0:
        plot_cm_heatmap(cm, f"{name} (transition-only)", save_path)

    return n_tr, acc, f1, cm


def eval_transition_only_markov(name, val_df, trans_mat, results_dir):
    """
    Markov transition-only 평가 + heatmap 저장
    """
    yt, yp = markov_transition_only_from_val_df(val_df, trans_mat)
    n_tr = len(yt)

    acc, f1, cm = safe_metrics(f"{name} [transition-only]", yt, yp)
    if n_tr > 0:
        print_result(f"{name} [transition-only]", yt, yp)

    save_path = results_dir / "cm_transition_markov_rrc_only.png"
    if n_tr > 0:
        plot_cm_heatmap(cm, f"{name} (transition-only)", save_path)

    return n_tr, acc, f1, cm


def plot_transition_summary(trans_df, results_dir):
    """
    transition-only 성능 비교: 가로 막대 그래프 + 표를 한 장에 저장
    """
    df = trans_df.copy()

    model_order = ["Markov", "LSTM", "1D-CNN", "RandomForest"]
    input_order = ["RRC-only", "traffic", "RRC+traffic"]

    df["Model"] = pd.Categorical(df["Model"], categories=model_order, ordered=True)
    df["Input"] = pd.Categorical(df["Input"], categories=input_order, ordered=True)
    df = df.sort_values(["Input", "Model"])

    labels = df["Model"].astype(str) + " (" + df["Input"].astype(str) + ")"
    y_pos = np.arange(len(labels))

    # 색상 매핑: Input 타입별
    color_map = {"RRC-only": "gray", "traffic": "steelblue", "RRC+traffic": "crimson"}
    colors = [color_map[inp] for inp in df["Input"]]

    fig, (ax_bar, ax_table) = plt.subplots(
        nrows=2, ncols=1,
        figsize=(10, 16),
        gridspec_kw={"height_ratios": [2.5, 1]}
    )

    # Accuracy 가로 막대
    bar_width = 0.4
    ax_bar.barh(y_pos - bar_width/2, df["Accuracy"], bar_width, 
                label="Accuracy", color=colors, alpha=0.9, edgecolor='black', linewidth=0.8)
    ax_bar.barh(y_pos + bar_width/2, df["MacroF1"], bar_width, 
                label="Macro F1-score", color=colors, alpha=0.6, edgecolor='black', linewidth=0.8, hatch='///')

    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(labels, fontsize=16)
    ax_bar.set_xlabel("Score", fontsize=18)
    ax_bar.set_xlim(0.0, 1.05)
    ax_bar.set_title("Transition-only Model Comparison", fontsize=19, fontweight="bold")
    ax_bar.legend(loc="lower right", fontsize=15)
    ax_bar.grid(axis="x", alpha=0.3)
    ax_bar.tick_params(axis='x', labelsize=14)
    ax_bar.invert_yaxis()

    ax_table.axis("off")

    show_df = df.copy()
    # 표에는 보기 좋게 반올림
    show_df["Accuracy"] = show_df["Accuracy"].round(4)
    show_df["MacroF1"] = show_df["MacroF1"].round(4)

    table = ax_table.table(
        cellText=show_df.values,
        colLabels=show_df.columns,
        bbox=[0, 0, 1, 1]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(14)

    plt.subplots_adjust(left=0.34, right=0.98, top=0.96, bottom=0.06, hspace=0.25)
    # 표 축을 그림 전체 폭에 가깝게 확장해 왼쪽 공백 제거
    pos_tbl = ax_table.get_position()
    ax_table.set_position([0.02, pos_tbl.y0, 0.96, pos_tbl.height])
    fig.savefig(results_dir / "models_comparison_transition_only.png", dpi=300)
    plt.close(fig)


# =========================
# 메인
# =========================
if __name__ == "__main__":
    results_dir = BASE_DIR / "artifacts" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # 1) Traffic-only 데이터셋 로드
    traffic_npz = np.load(BASE_DIR / "data" / "processed" / "traffic_only_seq_dataset.npz")
    Xv_t = traffic_npz["X_val"]
    yv_t = traffic_npz["y_val"]
    print("Traffic-only X_val:", Xv_t.shape)

    # 2) RRC+Traffic 데이터셋 로드
    rrc_npz = np.load(BASE_DIR / "data" / "processed" / "seq_dataset.npz")
    Xv_r = rrc_npz["X_val"]
    yv_r = rrc_npz["y_val"]
    print("RRC+Traffic X_val:", Xv_r.shape)

    # 3) Markov baseline (60/30 temporal split, same as other models)
    print("\n=== Markov baseline (60min train / 30min val) ===")
    full_df = load_all_sessions()

    # Apply 60/30 temporal split per session (assumes each session has time_sec 0~5400)
    train_df = full_df[full_df["time_sec"] < 3600].copy()
    val_df = full_df[full_df["time_sec"] >= 3600].copy()

    trans_mat = fit_markov_from_df(train_df)
    print("Transition matrix:\n", trans_mat)

    markov_acc, markov_f1, markov_cm = eval_markov_nextstate(val_df, trans_mat, name="Markov (RRC-only)")

    # 4) Traffic-only 모델들 평가
    print("\n=== Traffic-only models ===")

    lstm_t = load_model(BASE_DIR / "artifacts" / "models" / "lstm_traffic_best.keras")
    yv_t_pred_lstm = (lstm_t.predict(Xv_t, verbose=0) > 0.5).astype("int32").ravel()
    lstm_t_acc, lstm_t_f1, lstm_t_cm = print_result("LSTM (traffic)", yv_t, yv_t_pred_lstm)

    cnn_t = load_model(BASE_DIR / "artifacts" / "models" / "cnn1d_traffic_best.keras")
    yv_t_pred_cnn = (cnn_t.predict(Xv_t, verbose=0) > 0.5).astype("int32").ravel()
    cnn_t_acc, cnn_t_f1, cnn_t_cm = print_result("1D-CNN (traffic)", yv_t, yv_t_pred_cnn)

    rf_t = joblib.load(BASE_DIR / "artifacts" / "models" / "rf_traffic_model.joblib")
    Xv_t_flat = Xv_t.reshape(Xv_t.shape[0], -1)
    yv_t_pred_rf = rf_t.predict(Xv_t_flat)
    rf_t_acc, rf_t_f1, rf_t_cm = print_result("RandomForest (traffic)", yv_t, yv_t_pred_rf)

    # 5) RRC+Traffic 모델들 평가
    print("\n=== RRC + Traffic models ===")

    lstm_r = load_model(BASE_DIR / "artifacts" / "models" / "lstm_best.keras")
    yv_r_pred_lstm = (lstm_r.predict(Xv_r, verbose=0) > 0.5).astype("int32").ravel()
    lstm_r_acc, lstm_r_f1, lstm_r_cm = print_result("LSTM (RRC+traffic)", yv_r, yv_r_pred_lstm)

    cnn_r = load_model(BASE_DIR / "artifacts" / "models" / "cnn1d_best.keras")
    yv_r_pred_cnn = (cnn_r.predict(Xv_r, verbose=0) > 0.5).astype("int32").ravel()
    cnn_r_acc, cnn_r_f1, cnn_r_cm = print_result("1D-CNN (RRC+traffic)", yv_r, yv_r_pred_cnn)

    rf_r = joblib.load(BASE_DIR / "artifacts" / "models" / "rf_model.joblib")
    Xv_r_flat = Xv_r[:, :60, :].reshape(Xv_r.shape[0], -1)
    yv_r_pred_rf = rf_r.predict(Xv_r_flat)
    rf_r_acc, rf_r_f1, rf_r_cm = print_result("RandomForest (RRC+traffic)", yv_r, yv_r_pred_rf)

    # 6) 전체 성능 요약 표 + 그림(이미 네가 쓰던 방식 유지)
    summary = [
        ("Markov",       "RRC-only",     markov_acc, markov_f1),
        ("LSTM",         "traffic",      lstm_t_acc, lstm_t_f1),
        ("1D-CNN",       "traffic",      cnn_t_acc,  cnn_t_f1),
        ("RandomForest", "traffic",      rf_t_acc,   rf_t_f1),
        ("LSTM",         "RRC+traffic",  lstm_r_acc, lstm_r_f1),
        ("1D-CNN",       "RRC+traffic",  cnn_r_acc,  cnn_r_f1),
        ("RandomForest", "RRC+traffic",  rf_r_acc,   rf_r_f1),
    ]
    summary_df = pd.DataFrame(summary, columns=["Model", "Input", "Accuracy", "MacroF1"])

    print("\n=== Summary Table ===")
    print(summary_df.to_string(index=False))

    # 전체 비교 그림 저장
    model_order = ["Markov", "LSTM", "1D-CNN", "RandomForest"]
    input_order = ["RRC-only", "traffic", "RRC+traffic"]
    s = summary_df.copy()
    s["Model"] = pd.Categorical(s["Model"], categories=model_order, ordered=True)
    s["Input"] = pd.Categorical(s["Input"], categories=input_order, ordered=True)
    s = s.sort_values(["Input", "Model"])

    labels = s["Model"].astype(str) + " (" + s["Input"].astype(str) + ")"
    y_pos = np.arange(len(labels))

    # 색상 매핑: Input 타입별
    color_map = {"RRC-only": "gray", "traffic": "steelblue", "RRC+traffic": "crimson"}
    colors = [color_map[inp] for inp in s["Input"]]

    fig, (ax_bar, ax_table) = plt.subplots(
        nrows=2, ncols=1,
        figsize=(10, 16),
        gridspec_kw={"height_ratios": [2.5, 1]}
    )

    # Accuracy & MacroF1 가로 막대
    bar_width = 0.4
    ax_bar.barh(y_pos - bar_width/2, s["Accuracy"], bar_width, 
                label="Accuracy", color=colors, alpha=0.9, edgecolor='black', linewidth=0.8)
    ax_bar.barh(y_pos + bar_width/2, s["MacroF1"], bar_width, 
                label="Macro F1-score", color=colors, alpha=0.6, edgecolor='black', linewidth=0.8, hatch='///')

    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(labels, fontsize=16)
    ax_bar.set_xlabel("Score", fontsize=18)
    ax_bar.set_xlim(0.0, 1.05)
    ax_bar.set_title("Model Comparison", fontsize=19, fontweight="bold")
    ax_bar.legend(loc="lower right", fontsize=15)
    ax_bar.grid(axis="x", alpha=0.3)
    ax_bar.tick_params(axis='x', labelsize=14)
    ax_bar.invert_yaxis()

    ax_table.axis("off")
    table = ax_table.table(
        cellText=s.round(4).values,
        colLabels=s.columns,
        bbox=[0, 0, 1, 1]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(14)

    plt.subplots_adjust(left=0.34, right=0.98, top=0.96, bottom=0.06, hspace=0.25)
    # 표 축을 그림 전체 폭에 가깝게 확장해 왼쪽 공백 제거
    pos_tbl = ax_table.get_position()
    ax_table.set_position([0.02, pos_tbl.y0, 0.96, pos_tbl.height])
    fig.savefig(results_dir / "models_comparison.png", dpi=300)
    plt.close(fig)

    # CSV 저장
    summary_df.to_csv(results_dir / "models_comparison_summary.csv", index=False, encoding="utf-8-sig")

    print("\n그래프 파일 저장 완료:")
    print(" - artifacts/results/models_comparison.png")
    print(" - artifacts/results/models_comparison_summary.csv")

    # =========================
    # 7) Transition-only 평가 + 요약표(N_transitions 포함) + CM heatmap + 그림 저장
    # =========================
    print("\n=== Transition-only Evaluation (전환 이벤트만) ===")

    trans_rows = []

    # Markov transition-only (세션 경계 처리 + 전환 이벤트만)
    n_tr, acc, f1, cm = eval_transition_only_markov("Markov (RRC-only)", val_df, trans_mat, results_dir)
    trans_rows.append(("Markov", "RRC-only", n_tr, acc, f1))

    # Traffic-only
    n_tr, acc, f1, cm = eval_transition_only_general("LSTM (traffic)", yv_t, yv_t_pred_lstm, results_dir)
    trans_rows.append(("LSTM", "traffic", n_tr, acc, f1))

    n_tr, acc, f1, cm = eval_transition_only_general("1D-CNN (traffic)", yv_t, yv_t_pred_cnn, results_dir)
    trans_rows.append(("1D-CNN", "traffic", n_tr, acc, f1))

    n_tr, acc, f1, cm = eval_transition_only_general("RandomForest (traffic)", yv_t, yv_t_pred_rf, results_dir)
    trans_rows.append(("RandomForest", "traffic", n_tr, acc, f1))

    # RRC+Traffic
    n_tr, acc, f1, cm = eval_transition_only_general("LSTM (RRC+traffic)", yv_r, yv_r_pred_lstm, results_dir)
    trans_rows.append(("LSTM", "RRC+traffic", n_tr, acc, f1))

    n_tr, acc, f1, cm = eval_transition_only_general("1D-CNN (RRC+traffic)", yv_r, yv_r_pred_cnn, results_dir)
    trans_rows.append(("1D-CNN", "RRC+traffic", n_tr, acc, f1))

    n_tr, acc, f1, cm = eval_transition_only_general("RandomForest (RRC+traffic)", yv_r, yv_r_pred_rf, results_dir)
    trans_rows.append(("RandomForest", "RRC+traffic", n_tr, acc, f1))

    trans_df = pd.DataFrame(trans_rows, columns=["Model", "Input", "N_transitions", "Accuracy", "MacroF1"])

    print("\n=== Transition-only Summary Table ===")
    print(trans_df.to_string(index=False))

    trans_df.to_csv(results_dir / "eval_transition_only_summary.csv", index=False, encoding="utf-8-sig")
    print("\nTransition-only 결과 저장 완료:")
    print(" - artifacts/results/eval_transition_only_summary.csv")
    print(" - artifacts/results/cm_transition_*.png")

    # Transition-only 비교 그림 저장
    plot_transition_summary(trans_df, results_dir)
    print(" - artifacts/results/models_comparison_transition_only.png")
