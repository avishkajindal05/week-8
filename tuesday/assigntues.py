# =============================
# Hospital Readmission Prediction
# =============================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score
from sklearn.linear_model import LogisticRegression

# =============================
# SUBSTEP 1: LOAD + AUDIT
# =============================

df = pd.read_csv("C:\\Users\\avish\\Desktop\\iitgn AINPT\\week-8\\hospital_records.csv")

print("Shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nMissing Values (%):\n", (df.isnull().mean()*100))

# =============================
# SUBSTEP 2: CLEANING
# =============================

# Example cleaning (robust for messy hospital dataset)
def clean_data(df):
    df = df.copy()
    
    # Strip spaces
    df.columns = df.columns.str.strip()

    # AGE cleaning
    if 'Age' in df.columns:
        df['Age'] = df['Age'].astype(str).str.extract('(\\d+)')
        df['Age'] = pd.to_numeric(df['Age'], errors='coerce')

    # BMI cleaning
    if 'BMI' in df.columns:
        df['BMI'] = pd.to_numeric(df['BMI'], errors='coerce')

    # Billing cleaning
    if 'Billing Amount' in df.columns:
        df['Billing Amount'] = df['Billing Amount'].astype(str)
        df['Billing Amount'] = df['Billing Amount'].str.replace('[^0-9.]', '', regex=True)
        df['Billing Amount'] = pd.to_numeric(df['Billing Amount'], errors='coerce')

    # Fill missing
    for col in df.select_dtypes(include=np.number).columns:
        df[col] = df[col].fillna(df[col].median())

    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].fillna(df[col].mode()[0])

    # Encode categoricals
    df = pd.get_dummies(df, drop_first=True)

    return df


df_clean = clean_data(df)

print("\nClean Shape:", df_clean.shape)
print("Remaining nulls:", df_clean.isnull().sum().sum())

# =============================
# TARGET SEPARATION
# =============================

# Detect target column
possible_targets = [col for col in df_clean.columns if 'readmit' in col.lower()]
target_col = possible_targets[0]

X = df_clean.drop(columns=[target_col]).values
y = df_clean[target_col].values

# =============================
# SCALING
# =============================

scaler = StandardScaler()
X = scaler.fit_transform(X)

# =============================
# TRAIN TEST SPLIT
# =============================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =============================
# SUBSTEP 3: NEURAL NETWORK
# =============================

class NeuralNetwork:
    def __init__(self, input_size):
        self.W1 = np.random.randn(input_size, 16) * 0.01
        self.b1 = np.zeros((1, 16))
        self.W2 = np.random.randn(16, 8) * 0.01
        self.b2 = np.zeros((1, 8))
        self.W3 = np.random.randn(8, 1) * 0.01
        self.b3 = np.zeros((1, 1))

    def relu(self, Z):
        return np.maximum(0, Z)

    def relu_deriv(self, Z):
        return Z > 0

    def sigmoid(self, Z):
        return 1 / (1 + np.exp(-Z))

    def forward(self, X):
        self.Z1 = X @ self.W1 + self.b1
        self.A1 = self.relu(self.Z1)

        self.Z2 = self.A1 @ self.W2 + self.b2
        self.A2 = self.relu(self.Z2)

        self.Z3 = self.A2 @ self.W3 + self.b3
        self.A3 = self.sigmoid(self.Z3)

        return self.A3

    def compute_loss(self, y, y_hat):
        m = y.shape[0]
        loss = -np.mean(y * np.log(y_hat + 1e-8) + (1 - y) * np.log(1 - y_hat + 1e-8))
        return loss

    def backward(self, X, y, lr=0.01):
        m = X.shape[0]

        dZ3 = self.A3 - y.reshape(-1,1)
        dW3 = self.A2.T @ dZ3 / m
        db3 = np.sum(dZ3, axis=0) / m

        dA2 = dZ3 @ self.W3.T
        dZ2 = dA2 * self.relu_deriv(self.Z2)
        dW2 = self.A1.T @ dZ2 / m
        db2 = np.sum(dZ2, axis=0) / m

        dA1 = dZ2 @ self.W2.T
        dZ1 = dA1 * self.relu_deriv(self.Z1)
        dW1 = X.T @ dZ1 / m
        db1 = np.sum(dZ1, axis=0) / m

        self.W3 -= lr * dW3
        self.b3 -= lr * db3
        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        self.W1 -= lr * dW1
        self.b1 -= lr * db1

    def train(self, X, y, epochs=200, lr=0.01):
        losses = []
        for i in range(epochs):
            y_hat = self.forward(X)
            loss = self.compute_loss(y, y_hat)
            self.backward(X, y, lr)
            losses.append(loss)

            if i % 50 == 0:
                print(f"Epoch {i} Loss: {loss:.4f}")
        return losses


# =============================
# TRAIN MODEL
# =============================

nn = NeuralNetwork(X_train.shape[1])
losses = nn.train(X_train, y_train, epochs=200, lr=0.01)

# Plot loss
plt.plot(losses)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

# =============================
# EVALUATION
# =============================

probs = nn.forward(X_test)
preds = (probs > 0.5).astype(int)

print("\nNeural Network Results:")
print(classification_report(y_test, preds))
print("ROC AUC:", roc_auc_score(y_test, probs))

# =============================
# BASELINE MODEL
# =============================

lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)

lr_preds = lr_model.predict(X_test)

print("\nLogistic Regression:")
print(classification_report(y_test, lr_preds))

# =============================
# SUBSTEP 5: COST OPTIMIZATION
# =============================

thresholds = np.linspace(0.1, 0.9, 50)
best_cost = float('inf')
best_t = 0.5

FN_cost = 50000
FP_cost = 5000

for t in thresholds:
    p = (probs > t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, p).ravel()
    cost = fn * FN_cost + fp * FP_cost

    if cost < best_cost:
        best_cost = cost
        best_t = t

print(f"\nBest Threshold: {best_t:.2f}")

# =============================
# CONFUSION MATRIX
# =============================

final_preds = (probs > best_t).astype(int)
cm = confusion_matrix(y_test, final_preds)

print("\nConfusion Matrix:")
print(cm)

# =============================
# SUBSTEP 6: FAKE ACCURACY
# =============================

all_zero_preds = np.zeros_like(y_test)

print("\nFake High Accuracy:")
print(classification_report(y_test, all_zero_preds))
print(confusion_matrix(y_test, all_zero_preds))

# =============================
# SUBSTEP 7: EMBEDDINGS
# =============================

# Extract embeddings (A2 layer)
_ = nn.forward(X_train)
train_embed = nn.A2

_ = nn.forward(X_test)
test_embed = nn.A2

clf = LogisticRegression()
clf.fit(train_embed, y_train)

embed_preds = clf.predict(test_embed)

print("\nEmbedding Model:")
print(classification_report(y_test, embed_preds))
