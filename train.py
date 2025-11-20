import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import shap
import matplotlib.pyplot as plt

from lstm_model import build_lstm_model
from transformer_model import build_transformer_model

# -----------------------------
# 1. LOAD DATA
# -----------------------------
df = pd.read_csv("data.csv")

# Convert date to index (optional)
df['date'] = pd.to_datetime(df['date'])
df = df.set_index('date')

# Features and target
X = df[['feature1', 'feature2']].values
y = df['target'].values.reshape(-1, 1)

# Scale data
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X = scaler_X.fit_transform(X)
y = scaler_y.fit_transform(y)

# Windowing function
def create_dataset(X, y, window=3):
    Xs, ys = [], []
    for i in range(len(X) - window):
        Xs.append(X[i:i+window])
        ys.append(y[i+window])
    return np.array(Xs), np.array(ys)

window_size = 3
X, y = create_dataset(X, y, window_size)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# -----------------------------
# 2. CHOOSE MODEL (LSTM OR TRANSFORMER)
# -----------------------------
use_model = "lstm"     # change to "transformer" if needed

if use_model == "lstm":
    model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
else:
    model = build_transformer_model((X_train.shape[1], X_train.shape[2]))

# -----------------------------
# 3. TRAIN MODEL
# -----------------------------
model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=30,
    batch_size=8,
    verbose=1
)

# -----------------------------
# 4. PREDICT
# -----------------------------
pred = model.predict(X_test)
pred = scaler_y.inverse_transform(pred)
actual = scaler_y.inverse_transform(y_test)

print("Prediction Complete!")

# -----------------------------
# 5. EXPLAINABILITY (SHAP)
# -----------------------------
explainer = shap.DeepExplainer(model, X_train[:50])
shap_values = explainer.shap_values(X_test[:10])

print("SHAP Explainability Complete")

# Save summary plot
shap.summary_plot(shap_values, X_test[:10], show=False)
plt.savefig("shap_summary.png")

print("SHAP summary plot saved as shap_summary.png")
