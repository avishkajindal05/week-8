1. Hate Speech Classifier
Model:
TF-IDF + Logistic Regression
Class balancing enabled
Results:
Precision (hate): 0.72
Recall (hate):    0.81
F1-score:         0.76
Accuracy:         0.89
Interpretation:
High recall → model detects most harmful content
Some false positives → acceptable trade-off
2. Semantic Search (Sentence-BERT)
Model:
all-MiniLM-L6-v2
Cosine similarity
Observation:
Captures semantic similarity, not just keywords

Example:

“hate this leader”
“terrible politician”
→ Retrieved correctly
3. CNN on MNIST
Architecture:
2 Conv layers
MaxPooling + Dense
Performance:
Accuracy: ~98–99%
Learned Filters:
Edge detectors
Stroke patterns
Digit shapes