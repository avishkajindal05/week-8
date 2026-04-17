1. Dataset Overview

The dataset used is Reddit comments containing:

clean_comment: text data
category: sentiment label (-1, 0, 1)

Total samples ≈ 37,000

2. Class Distribution
Sentiment:
Positive (1): 15,830
Neutral (0): 13,142
Negative (-1): 8,277
Converted Hate Speech:
Hate (1): 8,277
Non-hate (0): 28,972

➡️ Imbalance present (~1:3 ratio)

3. Data Quality Issues
Missing text values: 100 rows
Text is already cleaned but:
lacks punctuation
contains informal language
4. Impact on Modeling
Imbalance → Accuracy unreliable
Chosen metric → Recall (important for safety systems)