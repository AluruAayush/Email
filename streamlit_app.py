import streamlit as st
import joblib
import pandas as pd 
import re  # For regex operations
from textblob import TextBlob
import textstat

# Define the EmailPredictor class here (before loading the model)
class EmailPredictor:
    def __init__(self, scaler, open_model, ctr_model):
        self.scaler = scaler
        self.open_model = open_model
        self.ctr_model = ctr_model

        self.spam_words = ['free', 'winner', 'act now', 'limited time', 'urgent', 'sale']
        self.filler_words = [
            'just', 'really', 'very', 'actually', 'perhaps', 'maybe', 'some', 'kind', 'sort',
            'little', 'totally', 'basically', 'quite', 'literally', 'stuff', 'things', 'nice', 'good'
        ]

    def predict(self, subject_line, body):
        df = pd.DataFrame([{'subject_line': subject_line, 'body': body}])
        features = self.extract_email_features(df)
        X_scaled = self.scaler.transform(features.values)

        open_rate = self.open_model.predict(X_scaled)[0]
        ctr = self.ctr_model.predict(X_scaled)[0]
        return {
            'open_rate': open_rate,
            'ctr': ctr
        }

    # ---- Feature extraction ----
    def extract_email_features(self, df):
        f = {}
        f['subj_word_count']      = df['subject_line'].apply(lambda s: len(str(s).split()))
        f['subj_char_count']      = df['subject_line'].apply(lambda s: len(str(s)))
        f['subj_uppercase_pct']   = df['subject_line'].apply(self.uppercase_pct)
        f['subj_exclaim_q']       = df['subject_line'].apply(lambda s: self.punctuation_count(s, '!?'))
        f['subj_contains_num']    = df['subject_line'].apply(self.contains_number)
        f['subj_personalization'] = df['subject_line'].apply(self.contains_personalization)
        f['subj_sentiment']       = df['subject_line'].apply(self.sentiment)
        f['subj_spam_word_cnt']   = df['subject_line'].apply(self.spam_word_count)
        f['subj_unique_ratio']    = df['subject_line'].apply(self.unique_word_ratio)
        f['subj_emoji_count']     = df['subject_line'].apply(self.emoji_count)

        f['body_word_count']      = df['body'].apply(lambda b: len(str(b).split()))
        f['body_char_count']      = df['body'].apply(lambda b: len(str(b)))
        f['body_readability']     = df['body'].apply(lambda b: textstat.flesch_reading_ease(str(b)) if str(b).strip() else 0)
        f['body_link_count']      = df['body'].apply(self.link_count)
        f['body_spam_word_cnt']   = df['body'].apply(self.spam_word_count)
        f['body_cta_count']       = df['body'].apply(self.cta_count)
        f['body_filler_pct']      = df['body'].apply(self.filler_word_pct)
        f['body_sentiment']       = df['body'].apply(self.sentiment)
        f['body_para_count']      = df['body'].apply(self.paragraph_count)
        f['body_unique_ratio']    = df['body'].apply(self.unique_word_ratio)
        f['body_attachment']      = df['body'].apply(self.contains_attachment)
        f['body_emoji_count']     = df['body'].apply(self.emoji_count)

        return pd.DataFrame(f)

    def contains_number(self, text): return int(bool(re.search(r'\d', str(text))))
    def uppercase_pct(self, text):
        letters = [c for c in str(text) if c.isalpha()]
        return sum(1 for c in letters if c.isupper()) / len(letters) if letters else 0
    def punctuation_count(self, text, chars='!?'):
        return sum(str(text).count(c) for c in chars)
    def spam_word_count(self, text):
        return sum(bool(re.search(r'\b' + re.escape(word) + r'\b', str(text).lower())) for word in self.spam_words)
    def sentiment(self, text):
        return TextBlob(str(text)).sentiment.polarity if str(text).strip() else 0
    def unique_word_ratio(self, text):
        words = re.findall(r'\w+', str(text).lower())
        return len(set(words))/len(words) if words else 0
    def link_count(self, text): return len(re.findall(r'http[s]?://', str(text)))
    def cta_count(self, text):
        ctas = ['click here', 'buy now', 'learn more', 'sign up', 'register now']
        return sum(phrase in str(text).lower() for phrase in ctas)
    def filler_word_pct(self, text):
        words = str(text).lower().split()
        return sum(w in self.filler_words for w in words) / len(words) if words else 0
    def paragraph_count(self, text): return str(text).count('\n') + 1 if text else 1
    def contains_attachment(self, text): return int("attachment" in str(text).lower() or "see attached" in str(text).lower())
    def contains_personalization(self, text):
        personal_tokens = ['you', 'your', '{{name}}']
        return int(any(tok in str(text).lower() for tok in personal_tokens))
    def emoji_count(self, text):
        emoji_pattern = re.compile("["
            u"\U0001F600-\U0001F64F"
            u"\U0001F300-\U0001F5FF"
            u"\U0001F680-\U0001F6FF"
            u"\U0001F1E0-\U0001F1FF"
            "]+", flags=re.UNICODE)
        return len(emoji_pattern.findall(str(text)))

# Load the wrapped EmailPredictor class
@st.cache_resource
def load_predictor():
    return joblib.load("email_predictor.joblib")

predictor = load_predictor()

# Streamlit UI
st.set_page_config(page_title="Email Performance Predictor", page_icon="📧")

st.title("📧 Email Performance Predictor")
st.markdown("Enter your email subject and body to get predictions for open rate and click-through rate (CTR).")

# Input fields
subject = st.text_input("✉️ Email Subject")
body = st.text_area("📝 Email Body", height=200)

# Predict button
if st.button("📊 Predict"):
    if not subject or not body:
        st.warning("Please fill in both the subject and the body.")
    else:
        try:
            result = predictor.predict(subject, body)  # Updated to match your class's return dict
            open_rate = result['open_rate']
            ctr = result['ctr']
            st.success(f"✅ **Predicted Open Rate:** {open_rate * 100:.2f}%")
            st.success(f"✅ **Predicted CTR:** {ctr * 100:.2f}%")
        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")
