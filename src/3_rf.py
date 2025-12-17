# rf_traffic.py

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from pathlib import Path
import joblib

BASE_DIR = Path(__file__).parent.parent
data = np.load(BASE_DIR / "data" / "processed" / "seq_dataset.npz")
X_train = data["X_train"]   # (N, 60, 5)
y_train = data["y_train"]
X_val   = data["X_val"]
y_val   = data["y_val"]

# 1) flatten: (N, 60*5)
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_val_flat   = X_val.reshape(X_val.shape[0], -1)

print("X_train_flat shape:", X_train_flat.shape)

# 2) RF 모델
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    n_jobs=-1,
    random_state=42
)
rf.fit(X_train_flat, y_train)

# 모델 저장 
out_path = BASE_DIR / "artifacts" / "models" / "rf_model.joblib"
joblib.dump(rf, str(out_path))
print(f"\nRandom Forest 모델 저장 완료: {out_path.resolve()}")

# 3) 평가
y_val_pred = rf.predict(X_val_flat)

acc = accuracy_score(y_val, y_val_pred)
f1  = f1_score(y_val, y_val_pred, average="macro")
cm  = confusion_matrix(y_val, y_val_pred)

print("\n[Random Forest RRC+traffic-only (flatten)]")
print("Val Accuracy:", acc)
print("Val Macro F1:", f1)
print("Confusion Matrix:\n", cm)
print("\nClassification report:\n", classification_report(y_val, y_val_pred))
