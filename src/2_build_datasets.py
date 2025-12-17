# build_datasets.py

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import sys

# =========================
# 설정
# =========================
WINDOW_SIZE = 300   # 300초짜리 윈도우
STEP = 5          # 5초씩 밀면서 샘플 생성
# 기본적으로 이 스크립트 파일과 같은 폴더 안의 `processed_testbed`를 사용합니다.
PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed" / "processed_testbed"  # 네가 만든 CSV 폴더 경로


# =========================
# 1) 모든 세션 로딩
# =========================
def load_all_sessions(processed_dir=PROCESSED_DIR) -> pd.DataFrame:
    """
    processed_testbed 폴더 안의 모든 CSV를 읽어서
    하나의 DataFrame으로 합친다.
    각 행에 session_id 컬럼(파일 이름 기반) 추가.
    """
    all_dfs = []
    for csv_path in processed_dir.glob("*.csv"):
        df = pd.read_csv(csv_path)
        df = df.sort_values("time_sec")
        session_id = csv_path.stem  # 예: S22_Clean_WiFiOff
        df["session_id"] = session_id
        all_dfs.append(df)

    if len(all_dfs) == 0:
        sys.exit(f"No CSV files found in {processed_dir.resolve()}. Put CSV files into this folder or set PROCESSED_DIR correctly.")

    full_df = pd.concat(all_dfs, ignore_index=True)
    return full_df


# =========================
# 2) 시퀀스 데이터셋 생성 (트래픽만 feature)
# =========================
def make_sequence_dataset(
    df: pd.DataFrame,
    feature_cols,
    target_col="rrc_state",
    window_size=WINDOW_SIZE,
    step=STEP,
):
    """
    여러 session_id가 섞인 df에서
    세션별로 슬라이딩 윈도우를 생성해
    X: (N, window_size, num_features)
    y: (N,)  을 만든다.

    입력:  t ~ t+window_size-1 의 feature 시퀀스
    라벨: t+window_size 시점의 target_col 값
    """
    X_list = []
    y_list = []
    meta_list = []

    for session_id, session_df in df.groupby("session_id"):
        s = session_df.reset_index(drop=True)

        # feature / target / 메타 준비
        s_feat = s[feature_cols].to_numpy()
        s_target = s[target_col].to_numpy()
        s_device = s["device"].iloc[0]
        s_env = s["env"].iloc[0]
        s_wifi = s["wifi"].iloc[0]
        s_network = s["network"].iloc[0]

        num_steps = len(s)
        # 마지막 시작 인덱스: y가 t+window_size 이므로 -1
        last_start = num_steps - window_size - 1

        if last_start <= 0:
            continue

        for start in range(0, last_start + 1, step):
            end = start + window_size
            x_window = s_feat[start:end]       # (window_size, num_features)
            y_label = s_target[end]            # t+window_size 시점의 rrc_state

            X_list.append(x_window)
            y_list.append(y_label)
            meta_list.append({
                "session_id": session_id,
                "device": s_device,
                "env": s_env,
                "wifi": s_wifi,
                "network": s_network,
            })

    X = np.stack(X_list)  # (N, T, F)
    y = np.array(y_list)
    meta = pd.DataFrame(meta_list)
    return X, y, meta


# =========================
# 3) Markov Chain baseline
# =========================
def fit_markov_from_df(df: pd.DataFrame):
    """
    전체 df에서 rrc_state 시퀀스를 이용해 2x2 마르코프 전이 행렬을 추정.
    """
    d = df.sort_values(["session_id", "time_sec"]).reset_index(drop=True)
    r = d["rrc_state"].to_numpy().astype(int)

    counts = np.zeros((2, 2), dtype=np.float64)

    for i in range(len(r) - 1):
        s = r[i]
        t = r[i + 1]
        if s in [0, 1] and t in [0, 1]:
            counts[s, t] += 1

    # 라플라스 스무딩
    counts += 1.0
    row_sums = counts.sum(axis=1, keepdims=True)
    trans_mat = counts / row_sums
    return trans_mat  # shape (2,2)


def evaluate_markov(df: pd.DataFrame, trans_mat):
    """
    df 전체에 대해 마르코프 체인으로 한 스텝 앞 RRC 상태를 예측하고,
    Accuracy / F1 / Confusion Matrix를 계산.
    """
    d = df.sort_values(["session_id", "time_sec"]).reset_index(drop=True)
    r = d["rrc_state"].to_numpy().astype(int)

    y_true = []
    y_pred = []

    for i in range(len(r) - 1):
        s = r[i]
        t = r[i + 1]
        if s not in [0, 1] or t not in [0, 1]:
            continue

        probs = trans_mat[s]
        pred = int(np.argmax(probs))

        y_true.append(t)
        y_pred.append(pred)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    cm = confusion_matrix(y_true, y_pred)
    return acc, f1, cm


# =========================
# 메인 실행
# =========================
if __name__ == "__main__":
    # 1) 세션 전체 로딩
    full_df = load_all_sessions(PROCESSED_DIR)
    print("전체 로우 수:", len(full_df))

    # 2) 트래픽 feature 스케일링 (0~1 사이로 정규화)
    traffic_cols = ["ul_bytes", "dl_bytes", "ul_pkts", "dl_pkts"]
    scaler = MinMaxScaler()
    full_df[traffic_cols] = scaler.fit_transform(full_df[traffic_cols])

    # 3) 시퀀스 데이터셋 생성 (트래픽만 feature)
    FEATURE_COLS = ["rrc_state", "ul_bytes", "dl_bytes", "ul_pkts", "dl_pkts"]
    X_seq, y_seq, meta_seq = make_sequence_dataset(full_df, FEATURE_COLS)
    print("시퀀스 데이터셋 X_seq shape:", X_seq.shape)   # (N, 60, 4)
    print("시퀀스 라벨 y_seq shape:", y_seq.shape)      # (N,)

    # Train / Val 분할 (여기서는 단순 랜덤 split, 나중에 세션 기반 split도 가능)
    X_train, X_val, y_train, y_val = train_test_split(
        X_seq, y_seq,
        test_size=0.2,
        random_state=42,
        stratify=y_seq
    )
    print("Train:", X_train.shape, y_train.shape)
    print("Val  :", X_val.shape, y_val.shape)

    # 4) Markov baseline 계산
    trans_mat = fit_markov_from_df(full_df)
    acc_mk, f1_mk, cm_mk = evaluate_markov(full_df, trans_mat)
    print("\n[Markov Chain baseline]")
    print("Transition Matrix:\n", trans_mat)
    print("Accuracy:", acc_mk)
    print("Macro F1:", f1_mk)
    print("Confusion Matrix:\n", cm_mk)

    # 5) 원하면 npz로 저장해서 나중에 모델 학습 때 바로 로드
    out_path = Path(__file__).parent.parent / "data" / "processed" / "seq_dataset.npz"
    np.savez(
        str(out_path),
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
    )
    print(f"\n{out_path.resolve()} 저장 완료")
