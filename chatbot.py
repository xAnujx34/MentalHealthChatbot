import random, csv, os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score

# ---------------------------
# Config
# ---------------------------
MIN_TOTAL_ANSWERS = 6         
CONFIDENCE_THRESHOLD = 0.65  
LOG_FILE = "chat_sessions.csv"

# ---------------------------
# Helpers
# ---------------------------
def convert(ans):
    a = str(ans).lower().strip()
    if a in ("yes","y"): return 1.0
    if a in ("sometimes","maybe","occasionally","sometimes/rarely"): return 0.5
    return 0.0

POS_KEYWORDS = {"relaxed","good","fine","okay","better","sleep","helpful","improved","happy"}
NEG_KEYWORDS = {"sad","depressed","hopeless","anxious","panic","angry","stressed","tired","lonely","bad","insomnia"}

def text_sentiment_score(txt):
    if not txt or not txt.strip(): return 0.5
    t = txt.lower()
    pos = sum(1 for k in POS_KEYWORDS if k in t)
    neg = sum(1 for k in NEG_KEYWORDS if k in t)
    raw = (pos - neg)
    if raw == 0:
        return 0.5
    elif raw > 0:
        return min(1.0, 0.5 + 0.1*raw)
    else:
        return max(0.0, 0.5 + 0.1*raw)

# ---------------------------
# Dataset and training
# ---------------------------
def get_default_df():
    data = {
        "sleep_issues":      [1,0,1,0,0.5,1,0.5,0,0.5,1,0,0.5,1,0,0.5,0],
        "feeling_anxious":   [1,0,1,0,0.5,0,1,0.5,0.5,1,0,0.5,1,0,0.5,0],
        "loss_of_interest":  [1,0,0.5,0,1,0,0.5,0,0.5,1,0,0.5,1,0,0.5,0],
        "stress_level": [
            "High","Low","High","Low","Moderate","Low",
            "High","Low","Moderate","High","Low","Moderate","High","Low","Moderate","Low"
        ]
    }
    return pd.DataFrame(data)

def train_model(df):
    X = df[["sleep_issues","feeling_anxious","loss_of_interest"]]
    y = df["stress_level"]
    model = LogisticRegression(max_iter=2000)
    model.fit(X,y)
    try:
        acc = round(100*float(cross_val_score(model, X, y, cv=4).mean()),2)
    except Exception:
        acc = round(100*accuracy_score(y, model.predict(X)),2)
    return model, acc

# ---------------------------
# Question pools
# ---------------------------
ANX_Q = [
 "Do you worry more than others about everyday things?",
 "Do you feel restless or on edge?",
 "Do you get sudden panicky sensations?",
 "Do small tasks make you anxious?"
]
SLEEP_Q = [
 "Do you find it hard to fall asleep?",
 "Do you wake up early and can't fall back asleep?",
 "Do you feel unrefreshed after sleep?",
 "Do you nap often during the day?"
]
INTEREST_Q = [
 "Have you lost interest in hobbies?",
 "Do you feel unmotivated for tasks you used to enjoy?",
 "Do social interactions feel draining recently?",
 "Have you stopped taking pleasure in food or activities?"
]
GENERAL_Q = [
 "Do you feel more irritable than usual?",
 "Have there been big changes in appetite?",
 "Do you avoid social situations?",
 "Do you have trouble concentrating while studying/working?"
]

# ---------------------------
# Aggregation & logging
# ---------------------------
def aggregate_features_from_answers(answers_raw):
    sleep_vals = [v for k,(v,t) in answers_raw.items() if "sleep" in k]
    anx_vals   = [v for k,(v,t) in answers_raw.items() if "anx" in k]
    int_vals   = [v for k,(v,t) in answers_raw.items() if "interest" in k]
    sleep = sum(sleep_vals)/len(sleep_vals) if sleep_vals else 0.0
    anx = sum(anx_vals)/len(anx_vals) if anx_vals else 0.0
    inter = sum(int_vals)/len(int_vals) if int_vals else 0.0
    return {"sleep_issues": round(sleep,3), "feeling_anxious": round(anx,3), "loss_of_interest": round(inter,3)}

def log_session(features, proba_map, prediction, history):
    header = ["timestamp","sleep_issues","feeling_anxious","loss_of_interest","prob_low","prob_moderate","prob_high","prediction","qa_count"]
    row = [
        pd.Timestamp.now().isoformat(),
        features.get("sleep_issues",0),
        features.get("feeling_anxious",0),
        features.get("loss_of_interest",0),
        proba_map.get("Low",0),
        proba_map.get("Moderate",0),
        proba_map.get("High",0),
        prediction,
        len(history)
    ]
    exists = os.path.exists(LOG_FILE)
    with open(LOG_FILE,"a",newline="",encoding="utf-8") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(header)
        writer.writerow(row)

# ---------------------------
# Main chat flow (no repeats)
# ---------------------------
def pick_unasked(pool, asked_texts):
    """Return a question from pool that hasn't been asked yet, or None if none left."""
    candidates = [q for q in pool if q not in asked_texts]
    return random.choice(candidates) if candidates else None

def run_chat(model):
    print("\n-----------------------------------------------")
    print("🤖 MindCare - Mental Health Chatbot (Terminal)")
    print("-----------------------------------------------")
    print("Answer with: yes / sometimes / no")
    print("You may add a short explanation after each question (press Enter to skip).")
    print()

    answers_raw = {}   # key->(numeric, text)
    question_history = []

    # Build initial queue: sample without repeats
    queue = []
    # pick 2 unique from each pool if available
    def sample_two(pool, asked):
        available = [q for q in pool if q not in asked]
        if len(available) >= 2:
            return random.sample(available,2)
        elif available:
            return available[:]  # whatever left
        else:
            return []  # none left

    asked_texts = []

    queue += [("anx_"+str(i), q, "anxiety") for i,q in enumerate(sample_two(ANX_Q, asked_texts))]
    asked_texts += [q for _,q,_ in queue]
    queue += [("sleep_"+str(i), q, "sleep") for i,q in enumerate(sample_two(SLEEP_Q, asked_texts))]
    asked_texts += [q for _,q,_ in queue if q not in asked_texts]
    queue += [("interest_"+str(i), q, "interest") for i,q in enumerate(sample_two(INTEREST_Q, asked_texts))]
    # final shuffle
    random.shuffle(queue)

    while queue:
        key, qtext, cat = queue.pop(0)
        # ensure we didn't accidentally include duplicates in queue
        if qtext in [h[1] for h in question_history]:
            continue

        print("\nQ:", qtext)
        ans_choice = input("Answer (yes/sometimes/no): ").strip()
        while ans_choice.lower() not in ("yes","no","sometimes","maybe",""):
            print("Please type yes / sometimes / no")
            ans_choice = input("Answer (yes/sometimes/no): ").strip()
        text = input("Optional (short explanation, press Enter to skip): ").strip()
        numeric = convert(ans_choice) if ans_choice else 0.0
        if text:
            txt_score = text_sentiment_score(text)
            numeric = max(0.0, min(1.0, numeric*0.8 + 0.2*txt_score))
        answers_raw[key] = (numeric, text)
        question_history.append((key, qtext, cat, numeric))
        asked_texts.append(qtext)

        # dynamic follow-ups: add only if unasked candidates exist
        if cat == "anxiety" and numeric >= 0.5:
            follow = pick_unasked(SLEEP_Q, asked_texts)
            if follow:
                queue.insert(0, ("sleep_follow_"+str(random.randint(1000,9999)), follow, "sleep"))

        if cat == "sleep" and numeric >= 0.5:
            follow = pick_unasked(INTEREST_Q, asked_texts)
            if follow:
                queue.insert(0, ("interest_follow_"+str(random.randint(1000,9999)), follow, "interest"))

        # ensure minimum number of answers: add general questions if needed (avoid repeats)
        if len(answers_raw) < MIN_TOTAL_ANSWERS and not queue:
            extras = [q for q in GENERAL_Q if q not in asked_texts]
            if extras:
                sample = random.sample(extras, min(2, len(extras)))
                for i,q in enumerate(sample):
                    queue.append(("gen_"+str(i), q, "general"))
                    # don't append to asked_texts yet until asked
            # if no extras left, we simply continue (we won't force repeats)

        # confidence-based follow ups: only add if unasked exist
        if len(answers_raw) >= MIN_TOTAL_ANSWERS:
            features = aggregate_features_from_answers(answers_raw)
            proba = model.predict_proba(pd.DataFrame([features]))[0]
            maxp = max(proba)
            if maxp < CONFIDENCE_THRESHOLD:
                # pick lowest feature and add an unasked follow-up from that pool if available
                feat_order = sorted(features.items(), key=lambda x: x[1])
                lowest = feat_order[0][0]
                if lowest == "sleep_issues":
                    cand = pick_unasked(SLEEP_Q, asked_texts)
                    if cand:
                        queue.insert(0, ("sleep_conf_"+str(random.randint(1000,9999)), cand, "sleep"))
                elif lowest == "feeling_anxious":
                    cand = pick_unasked(ANX_Q, asked_texts)
                    if cand:
                        queue.insert(0, ("anx_conf_"+str(random.randint(1000,9999)), cand, "anxiety"))
                else:
                    cand = pick_unasked(INTEREST_Q, asked_texts)
                    if cand:
                        queue.insert(0, ("int_conf_"+str(random.randint(1000,9999)), cand, "interest"))

    # final prediction
    features = aggregate_features_from_answers(answers_raw)
    df_user = pd.DataFrame([features])
    pred = model.predict(df_user)[0]
    proba_map = dict(zip(model.classes_, [round(p,3) for p in model.predict_proba(df_user)[0]]))

    print("\n\n========== RESULT ==========")
    print("Predicted stress level:", pred)
    print("Model probabilities:", proba_map)
    print("Feature scores:", features)
    if pred == "High":
        print("Advice: Consider professional help, breathing exercises, sleep routine.")
    elif pred == "Moderate":
        print("Advice: Improve sleep hygiene, light exercise, connect with friends.")
    else:
        print("Advice: Maintain healthy routine and monitor changes.")

    log_session(features, proba_map, pred, question_history)
    print("\nSession saved to", LOG_FILE)
    print("Thank you for using MindCare.")


if __name__ == "__main__":
    df = get_default_df()
    model, acc = train_model(df)
    print("Model trained on demo data. Cross-validated accuracy (approx):", acc, "%")
    resp = input("Load custom CSV to retrain model? (yes/no): ").strip().lower()
    if resp == "yes":
        path = input("Enter CSV filename (in same folder): ").strip()
        try:
            df2 = pd.read_csv(path)
            model, acc = train_model(df2)
            print("Model retrained. Accuracy:", acc, "%")
        except Exception as e:
            print("Failed to load/train on provided CSV. Using default model. Error:", e)
    run_chat(model)
