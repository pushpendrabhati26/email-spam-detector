import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Email Spam Detector", page_icon="📧", layout="centered")

@st.cache_resource
def train_model():
    data = pd.read_csv("sms_spam.csv")
    data = data.dropna(subset=["message", "label"])
    data["message"] = data["message"].astype(str)
    data["label"] = data["label"].astype(str).str.lower().str.strip()
    data = data[data["label"].isin(["spam", "ham"])]

    X_train, X_test, y_train, y_test = train_test_split(
        data["message"], data["label"], test_size=0.25,
        random_state=42, stratify=data["label"]
    )
    model = Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, stop_words="english")),
        ("classifier", MultinomialNB())
    ])
    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))
    return model, accuracy, len(data)

st.title("📧 Email Spam Detector")
st.write("Enter a message below to check whether it looks like spam or a normal message.")
st.info("Educational demo: this model uses a small sample dataset, so its predictions may be wrong. Do not use it as a security guarantee.")

try:
    model, accuracy, total_messages = train_model()
except Exception as e:
    st.error("Could not load the dataset. Make sure sms_spam.csv is in the same folder as app.py.")
    st.stop()

message = st.text_area(
    "Message to check",
    placeholder="Example: Congratulations! You have won a free prize. Click here...",
    height=150
)

if st.button("Check Message", type="primary", use_container_width=True):
    if not message.strip():
        st.warning("Please enter a message first.")
    else:
        prediction = model.predict([message])[0]
        probabilities = model.predict_proba([message])[0]
        classes = list(model.named_steps["classifier"].classes_)
        spam_probability = float(probabilities[classes.index("spam")]) * 100

        if prediction == "spam":
            st.error("⚠️ Prediction: This message looks like SPAM.")
        else:
            st.success("✅ Prediction: This message looks like a NORMAL message.")
        st.caption(f"Model's estimated spam score: {spam_probability:.1f}% (not a guarantee)")

with st.expander("About this project"):
    st.write("""
    This beginner machine-learning project uses TF-IDF to convert text into numerical features
    and a Multinomial Naive Bayes classifier to classify messages as spam or normal.
    The model is trained on the included sample dataset when the app starts.
    """)
    st.write(f"Sample messages used: {total_messages}")
    st.write(f"Hold-out test accuracy on this small sample: {accuracy:.1%}")
    st.write("Accuracy is only a rough classroom demonstration because the dataset is small.")

st.divider()
st.caption("Built with Python, Streamlit, Pandas, and Scikit-learn.")
