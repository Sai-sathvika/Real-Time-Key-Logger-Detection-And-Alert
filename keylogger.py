#importing all the libraries
import psutil
import numpy as np
import pandas as pd
import random
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import time
import hashlib
import json
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
 accuracy_score, classification_report,
 confusion_matrix, roc_curve, auc,
 precision_score, recall_score, f1_score
)
# ---------------- Step 1: Generate System Behavior Data ----------------
def generate_data(samples=500):
 data = []
 for _ in range(samples):
 cpu = psutil.cpu_percent(interval=0.5)
 memory = psutil.virtual_memory().percent
 keystrokes = random.randint(0, 50)
 network = random.uniform(0, 1)
 label = 1 if keystrokes > 30 and network > 0.6 else 0
 data.append([cpu, memory, keystrokes, network, label])
 return pd.DataFrame(data, columns=['CPU_Usage', 'Memory_Usage', 'Keystrokes',
'Network_Activity', 'Keylogger']) 
df = generate_data()
df.to_csv("keylogger_data.csv", index=False)
print(" Data Generated!\n", df.head())
# ---------------- Step 2: Data Preprocessing ----------------
df = pd.read_csv("keylogger_data.csv")
scaler = MinMaxScaler()
X = scaler.fit_transform(df[['CPU_Usage', 'Memory_Usage', 'Keystrokes',
'Network_Activity']])
y = df['Keylogger'].values
# ---------------- Step 2.5: Machine Learning Models ----------------
X_ml_train, X_ml_test, y_ml_train, y_ml_test = train_test_split(X, y, test_size=0.2,
random_state=42)
# Random Forest
start_time = time.time()
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_ml_train, y_ml_train)
rf_train_time = time.time() - start_time
rf_pred = rf_model.predict(X_ml_test)
rf_accuracy = accuracy_score(y_ml_test, rf_pred)
print(f"\n Random Forest Accuracy: {rf_accuracy * 100:.2f}%")
print(f" Random Forest Training Time: {rf_train_time:.2f} seconds")
print(classification_report(y_ml_test, rf_pred))
# Logistic Regression
start_time = time.time()
lr_model = LogisticRegression()
lr_model.fit(X_ml_train, y_ml_train)
lr_train_time = time.time() - start_time
lr_pred = lr_model.predict(X_ml_test)
lr_accuracy = accuracy_score(y_ml_test, lr_pred)
print(f"\n Logistic Regression Accuracy: {lr_accuracy * 100:.2f}%")
print(f"       Logistic Regression Training Time: {lr_train_time:.2f} seconds")
print(classification_report(y_ml_test, lr_pred)) 
# ---------------- Step 3: Prepare Data for LSTM ----------------
X_dl = X.reshape(X.shape[0], 1, X.shape[1])
X_train, X_test, y_train, y_test = train_test_split(X_dl, y, test_size=0.2, random_state=42)
# ---------------- Step 4: Build & Train LSTM ----------------
model = Sequential([
 LSTM(64, return_sequences=True, input_shape=(1, 4)),
 Dropout(0.2),
 LSTM(32),
 Dropout(0.2),
 Dense(16, activation='relu'),
 Dense(1, activation='sigmoid')
])
29
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
start_time = time.time()
history = model.fit(X_train, y_train, epochs=20, batch_size=16, validation_data=(X_test,
y_test), verbose=0)
lstm_train_time = time.time() - start_time
# ---------------- Step 5: Evaluate LSTM ----------------
y_pred_dl = (model.predict(X_test) > 0.5).astype("int32")
dl_accuracy = accuracy_score(y_test, y_pred_dl)
print(f"\n LSTM Accuracy: {dl_accuracy * 100:.2f}%")
print(f"       LSTM Training Time: {lstm_train_time:.2f} seconds")
print(classification_report(y_test, y_pred_dl)) 
# ---------------- Step 6: Plot Accuracy Graphs ----------------
plt.figure(figsize=(8, 5))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('LSTM Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()
models = ['LSTM', 'Random Forest', 'Logistic Regression']
accuracies = [dl_accuracy * 100, rf_accuracy * 100, lr_accuracy * 100]
plt.figure(figsize=(8, 5))
plt.bar(models, accuracies, color=['blue', 'green', 'orange'])
plt.title('Model Accuracy Comparison')
plt.ylabel('Accuracy (%)')
plt.ylim(0, 100)
for i, acc in enumerate(accuracies):
 plt.text(i, acc + 1, f"{acc:.2f}%", ha='center')
plt.show()
# Confusion Matrices
def plot_conf_matrix(y_true, y_pred, title):
 cm = confusion_matrix(y_true, y_pred)
 plt.figure(figsize=(4, 3))
 sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
 plt.xlabel('Predicted')
 plt.ylabel('Actual')
 plt.title(f'{title} Confusion Matrix')
 plt.show() 
plot_conf_matrix(y_test, y_pred_dl, "LSTM")
plot_conf_matrix(y_ml_test, rf_pred, "Random Forest")
plot_conf_matrix(y_ml_test, lr_pred, "Logistic Regression")
# ---------------- Step 7: ROC Curve Comparison ----------------
30
def plot_roc_curve(y_true, y_probs, model_name):
 fpr, tpr, _ = roc_curve(y_true, y_probs)
 roc_auc = auc(fpr, tpr)
 plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.2f})')
plt.figure(figsize=(8, 6))
plot_roc_curve(y_ml_test, rf_model.predict_proba(X_ml_test)[:, 1], "Random Forest")
plot_roc_curve(y_ml_test, lr_model.predict_proba(X_ml_test)[:, 1], "Logistic Regression")
plot_roc_curve(y_test, model.predict(X_test).ravel(), "LSTM")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.grid(True)
plt.show()
# ---------------- Step 8: Feature Importance ----------------
importances = rf_model.feature_importances_
features = ['CPU_Usage', 'Memory_Usage', 'Keystrokes', 'Network_Activity']
plt.figure(figsize=(6, 4))
sns.barplot(x=importances, y=features, palette='viridis')
plt.title("Random Forest Feature Importance")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.show()
# ---------------- Step 9: LSTM Model Summary ----------------
print("\nLSTM Model Summary:")
model.summary() 
# ---------------- Step 10: LSTM Threshold Sensitivity ----------------
thresholds = np.arange(0.1, 0.91, 0.1)
print("\n LSTM Threshold Sensitivity Analysis:")
for thresh in thresholds:
 preds = (model.predict(X_test) > thresh).astype("int32")
 acc = accuracy_score(y_test, preds)
 print(f"Threshold {thresh:.1f} -> Accuracy: {acc*100:.2f}%")
# ---------------- Step 11: Real-Time Detection (LSTM) ----------------
def detect_keylogger_lstm():
 cpu = psutil.cpu_percent(interval=0.5)
 memory = psutil.virtual_memory().percent
 keystrokes = random.randint(0, 50)
 network = random.uniform(0, 1)
 input_data = scaler.transform([[cpu, memory, keystrokes, network]])
 input_data = input_data.reshape(1, 1, 4)
 prediction = model.predict(input_data)[0][0]
 print(f"\n Real-time Input - CPU: {cpu}%, Mem: {memory}%, Keys: {keystrokes}, Net:
31
{network:.2f}")
 if prediction > 0.5:
 print("ALERT: Suspicious keylogger activity detected!")
 else:
 print(" System behavior is normal.") 
# Run real-time detection once
detect_keylogger_lstm()
# ---------------- LSTM vs Logistic Regression Metric Comparison ----------------
metrics_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
# Metrics
lstm_metrics = [
 accuracy_score(y_test, y_pred_dl),
 precision_score(y_test, y_pred_dl),
 recall_score(y_test, y_pred_dl),
 f1_score(y_test, y_pred_dl)
]
lr_metrics = [
 accuracy_score(y_ml_test, lr_pred),
 precision_score(y_ml_test, lr_pred),
 recall_score(y_ml_test, lr_pred),
 f1_score(y_ml_test, lr_pred)
]
x = np.arange(len(metrics_labels))
width = 0.35
plt.figure(figsize=(9, 6))
plt.bar(x - width/2, lstm_metrics, width, label='LSTM', color='skyblue')
plt.bar(x + width/2, lr_metrics, width, label='Logistic Regression', color='salmon')
plt.xticks(x, metrics_labels)
plt.ylim(0, 1)
plt.ylabel("Score")
plt.title(" LSTM vs Logistic Regression - Performance Metrics")
plt.legend()
plt.grid(True)
plt.show() 
# ---------------- Scatter Plot: LSTM vs Logistic Regression ----------------
plt.figure(figsize=(8, 6))
for i, metric in enumerate(metrics_labels):
 plt.scatter(i, lstm_metrics[i], color='blue', label='LSTM' if i == 0 else "")
 plt.scatter(i, lr_metrics[i], color='red', label='Logistic Regression' if i == 0 else "")
plt.xticks(x, metrics_labels)
plt.ylim(0, 1.1)
plt.ylabel("Score")
32
plt.title(" LSTM vs Logistic Regression - Scatter Plot of Performance Metrics")
plt.legend()
plt.grid(True)
plt.show()
plt.figure(figsize=(8, 5))
plt.plot(metrics_labels, lstm_metrics, marker='o', linestyle='-', color='blue', label='LSTM')
plt.plot(metrics_labels, lr_metrics, marker='o', linestyle='--', color='red', label='Logistic
Regression')
plt.ylabel("Score")
plt.title(" Line Chart - LSTM vs Logistic Regression Metrics")
plt.ylim(0, 1.1)
plt.grid(True)
plt.legend()
plt.show() 
# Step 1: Generate Synthetic System Behavior Data
np.random.seed(42)
data_size = 500
data = {
 'CPU_Usage': np.random.uniform(30, 50, data_size),
 'Memory_Usage': np.random.uniform(60, 80, data_size),
 'Keystrokes': np.random.randint(5, 50, data_size),
 'Network_Activity': np.random.uniform(0.1, 0.8, data_size),
 'Keylogger': np.random.choice([0, 1], data_size, p=[0.5, 0.5])
}
df = pd.DataFrame(data)
df.to_csv('keylogger_data.csv', index=False)
# Step 2: Data Preprocessing
X = df.drop('Keylogger', axis=1).values
y = df['Keylogger'].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# Prepare data for LSTM (reshape to [samples, timesteps, features])
X_lstm = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
train_size = int(0.8 * len(X_lstm))
X_train, X_test = X_lstm[:train_size], X_lstm[train_size:]
y_train, y_test = y[:train_size], y[train_size:]
# Step 3: Define Blockchain Class
class Blockchain:
 def __init__(self):
 self.chain = []
 self.create_genesis_block()
 def create_genesis_block(self):
 genesis_block = {
 'index': 0,
 'timestamp': str(datetime.now()),
 'data': {'model_updates': [], 'trust_score': 1.0},
 'previous_hash': '0',
'hash': self.calculate_hash(0, '0', str(datetime.now()), {'model_updates': [], 'trust_score':
1.0})
 }
 self.chain.append(genesis_block)
 def calculate_hash(self, index, previous_hash, timestamp, data):
 value = str(index) + previous_hash + timestamp + json.dumps(data, sort_keys=True)
 return hashlib.sha256(value.encode('utf-8')).hexdigest()
 def add_block(self, data):
 previous_block = self.chain[-1]
 index = previous_block['index'] + 1
 timestamp = str(datetime.now())
 block_hash = self.calculate_hash(index, previous_block['hash'], timestamp, data)
 new_block = {
 'index': index,
 'timestamp': timestamp,
 'data': data,
 'previous_hash': previous_block['hash'],
 'hash': block_hash
 }
 self.chain.append(new_block)
 return new_block
 def is_chain_valid(self):
 for i in range(1, len(self.chain)):
 current_block = self.chain[i]
 previous_block = self.chain[i-1]
 if current_block['hash'] != self.calculate_hash(
 current_block['index'], previous_block['hash'], current_block['timestamp'],
current_block['data']):
 return False
 if current_block['previous_hash'] != previous_block['hash']:
 return False
 return True 
# Step 4: Define LSTM Model
def create_lstm_model():
 model = Sequential([
 LSTM(64, input_shape=(1, 4), return_sequences=True),
 Dropout(0.2),
 LSTM(32),
 Dropout(0.2),
 Dense(16, activation='relu'),
 Dense(1, activation='sigmoid')
 ])
 model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
 return model
# Step 5: Train Local Models on 12 Devices
num_devices = 12
blockchain = Blockchain()
34
local_models = []
trust_scores = [] 
for device_id in range(num_devices):
 print(f"\nTraining on Device_{device_id}...")
 model = create_lstm_model()
 # Simulate different data subsets for each device
 indices = np.random.choice(range(len(X_train)), size=int(len(X_train) / num_devices),
replace=False)
 X_device = X_train[indices]
 y_device = y_train[indices]
 # Train model
 model.fit(X_device, y_device, epochs=5, batch_size=16, verbose=0, validation_split=0.2)
 # Evaluate validation accuracy
 val_loss, val_accuracy = model.evaluate(X_device, y_device, verbose=0)
 # Calculate trust score (initial trust * validation accuracy)
 initial_trust = 0.8
 trust_score = initial_trust * val_accuracy
 trust_scores.append(trust_score)
 local_models.append(model.get_weights())
 # Add to blockchain
 block_data = {
 'model_updates': [float(w) for w in model.get_weights()[0].flatten()[:5]], # Simplified for
display
 'trust_score': trust_score,
 'device_id': f'Device_{device_id}'
 }
 blockchain.add_block(block_data)
 print(f"Device_{device_id} - Validation Accuracy: {val_accuracy:.4f}, Trust Score:
{trust_score:.4f}")
# Step 6: Aggregate Model Updates Using Trust Scores
total_trust = sum(trust_scores)
global_weights = [np.zeros_like(w) for w in local_models[0]]
for i, weights in enumerate(local_models):
 weight_contribution = trust_scores[i] / total_trust
 for layer in range(len(weights)):
 global_weights[layer] += weights[layer] * weight_contribution
# Create global model and apply aggregated weights
global_model = create_lstm_model()
global_model.set_weights(global_weights) 
# Step 7: Evaluate Global Model
print("\nEvaluating Global Model...")
global_model.evaluate(X_test, y_test, verbose=0)
y_pred = (global_model.predict(X_test) > 0.5).astype(int)
print("Global Model Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
35
# Step 8: Real-Time Detection
real_time_input = np.array([[45.1, 68.2, 40, 0.72]]) # Example input
real_time_input_scaled = scaler.transform(real_time_input)
real_time_input_lstm = real_time_input_scaled.reshape((1, 1, 4))
prediction = global_model.predict(real_time_input_lstm)[0][0]
threshold = 0.5
is_keylogger = prediction > threshold
# Log detection event to blockchain
detection_event = {
 'detection_event': {
 'cpu': 45.1,
 'memory': 68.2,
 'keystrokes': 40,
 'network': 0.72,
 'prediction': float(prediction),
 'timestamp': str(datetime.now())
 },
 'trust_score': 1.0,
 'device_id': 'RealTimeDetector'
}
blockchain.add_block(detection_event)
# Step 9: Blockchain Summary and Result
print("\nBlockchain Summary:")
print(f"Total Blocks: {len(blockchain.chain)}")
print(f"Chain Validity: {blockchain.is_chain_valid()}")
print("\nReal-Time Detection Result:")
if is_keylogger:
 print(f" ALERT: Keylogger detected! (Prediction Probability: {prediction:.4f})")
else:
 print(f"No keylogger detected. (Prediction Probability: {prediction:.4f})") 






























