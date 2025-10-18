import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import seaborn as sns

# Load your dataset
# Replace with your actual data path
df = pd.read_csv("source path")

# Sample: Let's say we have the following columns
# ['transaction_id', 'customer_id', 'amount', 'transaction_type' ]

# Data preprocessing
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['hour'] = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.dayofweek

# Selecting features for anomaly detection
features = ['amount', 'hour', 'day_of_week']
X = df[features]

# Handling missing values
X.fillna(0, inplace=True)

# Isolation Forest
model = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
df['anomaly_score'] = model.fit_predict(X)

# Anomalies are marked as -1
df['is_anomaly'] = df['anomaly_score'].apply(lambda x: 1 if x == -1 else 0)

# Show anomalies
anomalies = df[df['is_anomaly'] == 1]

print("Detected potential suspicious transactions:")
print(anomalies[['transaction_id', 'customer_id', 'amount', 'hour', 'day_of_week']])

# Optional: Visualize anomalies
plt.figure(figsize=(12, 6))
sns.scatterplot(x='hour', y='amount', hue='is_anomaly', data=df, palette={0: 'blue', 1: 'red'})
plt.title("Transaction Anomalies Detected (Red = Potential Fraud)")
plt.show()

