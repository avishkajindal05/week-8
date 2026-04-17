1. Two-Stage Pipeline
Stage 1:

Classifier detects explicit hate speech

Stage 2:

Semantic search retrieves:

paraphrases
indirect hate
2. Pipeline Behavior

Example:

Input: "This politician is terrible"
→ Flagged by classifier
3. TF-IDF vs Embeddings
TF-IDF:
Keyword-based
Returns:
"hate this world"
"they hate kufrs"
Sentence-BERT:
Meaning-based
Returns:
semantically similar harmful content
4. Key Insight

Embedding-based retrieval:
✔ detects hidden patterns
✔ captures intent
❌ TF-IDF cannot

5. Real-world Impact (Required by TA)

Assume:

100,000 posts/day
Stage 1 recall = 0.81

→ detects ~81,000 harmful posts

Stage 2 adds ~10–15% more

→ +8,000 to 12,000 posts

Final detection:

👉 ~89,000–93,000 harmful posts/day

6. Recommendation

Use:

Recall as primary metric
Two-stage moderation system

Reason:
Missing harmful content is more costly than false positives