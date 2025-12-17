# lstm_traffic.py

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.utils import class_weight

# 1) 데이터 로드
from pathlib import Path
BASE_DIR = Path(__file__).parent.parent
data = np.load(BASE_DIR / "data" / "processed" / "traffic_only_seq_dataset.npz")
X_train = data["X_train"]   # (N, 60, 4)
y_train = data["y_train"]   # (N,)
X_val   = data["X_val"]
y_val   = data["y_val"]

print(X_train.shape, y_train.shape)
print(X_val.shape, y_val.shape)

# 2) 클래스 불균형 보정 (0이 더 많아서)
cw = class_weight.compute_class_weight(
    class_weight="balanced",
    classes=np.array([0, 1]),
    y=y_train
)
class_weights = {0: cw[0], 1: cw[1]}
print("class_weights:", class_weights)

# 3) LSTM 모델 정의
model = Sequential([
    LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=False),
    Dropout(0.3),
    Dense(32, activation="relu"),
    Dropout(0.3),
    Dense(1, activation="sigmoid")  # binary classification
])

model.compile(
    loss="binary_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)

model.summary()

# 4) 학습
es = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

# 체크포인트: 검증 손실이 가장 좋을 때의 모델만 저장
ckpt_path = BASE_DIR / "artifacts" / "models" / "lstm_traffic_best.keras"
ckpt_cb = ModelCheckpoint(
    filepath=str(ckpt_path),
    monitor="val_loss",
    save_best_only=True,
    mode="min",
    verbose=1
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=256,
    class_weight=class_weights,
    callbacks=[es, ckpt_cb],
    verbose=1
)

# 5) 평가
y_val_pred_prob = model.predict(X_val)
y_val_pred = (y_val_pred_prob > 0.5).astype("int32").ravel()

acc = accuracy_score(y_val, y_val_pred)
f1  = f1_score(y_val, y_val_pred, average="macro")
cm  = confusion_matrix(y_val, y_val_pred)

print("\n[LSTM traffic-only]")
print("Val Accuracy:", acc)
print("Val Macro F1:", f1)
print("Confusion Matrix:\n", cm)
print("\nClassification report:\n", classification_report(y_val, y_val_pred))
