# MentalHealthChatbot
A terminal-based AI mental-health assistant that predicts stress levels using logistic regression, adaptive questioning, sentiment-scoring, and dynamic follow-up logic. Includes question pools, text-based sentiment heuristics, confidence-based branching, session logging, and optional CSV-based retraining.


# 🤖 MindCare – AI-Based Mental Health Chatbot (Terminal Version)

MindCare is a lightweight, terminal-based mental-health chatbot that uses a small ML model to estimate a user's stress level based on their responses.  
It combines **logistic regression**, **question-pool randomization**, **sentiment heuristics**, and **dynamic follow-up questions** to generate a final stress prediction.

This is a demo / academic project — not a medical tool.

---

## 🚀 Features

### 🔹 1. Machine Learning Model  
- Logistic Regression (sklearn)  
- Uses three numerical features:
  - `sleep_issues`
  - `feeling_anxious`
  - `loss_of_interest`
- Trained on a small demo dataset.
- Supports **custom CSV retraining** at runtime.

### 🔹 2. Adaptive Question Flow  
- Anxiety, sleep, interest, and general question pools.
- No repeated questions.
- Dynamically chooses follow-up questions when:
  - User score for a category is high  
  - Model confidence < threshold  
  - Minimum number of answers not reached

### 🔹 3. Sentiment-Enhanced Answer Scoring  
- Users answer **yes / sometimes / no**  
- Optional explanation text affects score  
- A tiny keyword-based sentiment score modifies numerical values

### 🔹 4. Result Prediction  
Outputs:
- Predicted stress level (`Low`, `Moderate`, `High`)
- Probability distribution
- Final feature values

Also gives basic non-medical general advice.

### 🔹 5. Session Logging  
Each session is saved to `chat_sessions.csv` with:
- Timestamp  
- Feature scores  
- Prediction probabilities  
- Chosen label  
- Number of Q/A interactions  

---

## 🛠️ How It Works

### 1. Install dependencies:
```bash
pip install pandas scikit-learn
