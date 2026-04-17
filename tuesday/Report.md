## 🔹 1. Data Quality Audit

### Dataset Overview

* Shape: **(2000, 16)**
* Target: `readmitted_30d`
* Class distribution:

  * Not readmitted (0): **~94%*
  * Readmitted (1): **~6%** ⚠️ *Highly imbalanced*

---

### Issues Identified

#### 1. Missing Values

* Columns affected (~2% missing):

  * `systolic_bp`
  * `diastolic_bp`
  * `glucose_mg_dl`
  * `creatinine_mg_dl`
  * `insurance_type`

**Problem:**

* Missing medical values can bias model
* Cannot directly feed NaNs into NN

---

#### 2. Categorical Variables

* `gender`, `department`, `insurance_type`, `icu_stay`

**Problem:**

* Not numeric → NN cannot process directly

---

#### 3. Date Column

* `admission_date`

**Problem:**

* Raw date not useful without feature engineering
* High cardinality → leads to explosion after encoding

---

#### 4. ID Column

* `patient_id`

**Problem:**

* No predictive value
* Causes **data leakage risk**

---

#### 5. Feature Explosion ⚠️ (CRITICAL)

After one-hot encoding:

* Shape becomes: **(2000, 3719)**

**Problem:**

* Too many features vs small dataset
* Leads to:

  * Overfitting
  * Poor generalization
  * Model collapse (observed)

---

#### 6. Class Imbalance ⚠️ (CRITICAL)

* Only ~25 positive samples in test set

**Impact:**

* Model learns to predict only class 0
* Leads to fake high accuracy

---

---

## 🔹 2. Data Cleaning Decisions & Rationale

### ✔️ Missing Value Handling

* Numeric → filled with **median**

  * Robust to outliers
* Categorical → filled with **mode**

---

### ✔️ Encoding

* Used **one-hot encoding**

**Rationale:**

* Avoids ordinal assumptions

**Issue Created:**

* Feature explosion → needs control

---

### ✔️ Scaling

* Applied **StandardScaler**

**Why:**

* Neural networks require normalized inputs
* Prevents gradient instability

---

### ✔️ Dropping Problematic Columns (RECOMMENDED IMPROVEMENT)

(Not implemented but should be mentioned)

* Drop:

  * `patient_id`
  * `admission_date`

**Reason:**

* Reduce dimensionality
* Avoid noise

---

---

## 🔹 3. Neural Network Architecture Decisions

### Architecture

* Input: **3719 features**
* Hidden Layer 1: **16 neurons (ReLU)**
* Hidden Layer 2: **8 neurons (ReLU)**
* Output: **1 neuron (Sigmoid)**

---

### Activation Functions

#### ReLU (Hidden Layers)

* Prevents vanishing gradients
* Faster convergence

#### Sigmoid (Output)

* Suitable for binary classification
* Produces probability output

---

### Loss Function

* Binary Cross Entropy

---

### Learning Rate = 0.01

**Justification:**

* Stable convergence (loss decreases smoothly)
* Observed from training:

```
Epoch 0   → 0.693
Epoch 150 → 0.489
```

---

### Key Observation ⚠️

Even though loss decreases:

* Model predicts **ONLY class 0**
* Indicates:

  * Severe class imbalance
  * Poor signal learning

---

---

## 🔹 4. Model Performance Analysis

### Neural Network Results

* Accuracy: **94%**
* Recall (class 1): **0.00**
* ROC-AUC: **0.50**

---

### Interpretation ⚠️

* Model is **not learning anything useful**
* ROC ≈ 0.5 → random guessing

---

---

## 🔹 5. Cost-Sensitive Optimization

### Cost Assumptions

* False Negative (missed patient): **₹50,000**
* False Positive (unnecessary care): **₹5,000**

---

### Optimal Threshold Found

* **0.34**

---

### Confusion Matrix

```
[[375   0]
 [ 25   0]]
```

---

### Insight ⚠️

Even after threshold tuning:

* Model still predicts all zeros
* Cannot reduce clinical risk

---

---

## 🔹 6. Why 94% Accuracy is Misleading

### Reproduced High Accuracy Model

Confusion Matrix:

```
[[375   0]
 [ 25   0]]
```

---

### Explanation

* Model predicts:
  → ALL patients = "Not Readmitted"

* Since majority class = 94%:
  → Accuracy = 94%

---

### Problem ⚠️

* Misses **100% of high-risk patients**
* Clinically unacceptable

---

### Correct Metrics

Use:

* Recall (critical)
* F1-score
* ROC-AUC

---

### Before vs After

| Metric   | Before (Accuracy Only) | After (Proper Evaluation) |
| -------- | ---------------------- | ------------------------- |
| Accuracy | 94%                    | 94%                       |
| Recall   | ❌ ignored              | **0.00 (bad)**            |
| ROC-AUC  | ❌ ignored              | **0.50 (random)**         |

---

---

## 🔹 7. Embedding Approach (Sub-step 7)

### Method

* Extracted hidden layer (A2) features
* Trained Logistic Regression

---

### Results

* Same behavior:

  * Predicts only class 0
  * No improvement

---

### Interpretation

Hidden layers learned:

* Majority class patterns only
* No separation for minority class

---

### Conclusion

* Embeddings did **not improve separability**
* Root issue = data imbalance + high dimensionality

---

---

## 🔹 8. Final Recommendation to Dr. Anand

### Plain Language Version

> The current model is not reliable for clinical use.
> It predicts almost all patients as low-risk and fails to identify high-risk patients entirely.

---

### Key Risks

* Missing high-risk patients → high cost
* Model gives **false sense of confidence**

---

### Recommended Actions

1. **Fix Data Issues**

   * Reduce features (remove ID, date)
   * Control one-hot explosion

2. **Handle Class Imbalance**

   * Use:

     * Class weights
     * Oversampling (SMOTE)

3. **Change Evaluation Metric**

   * Focus on:

     * Recall
     * F1-score

