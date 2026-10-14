# Email Spam Detector (Beginner ML Project)

A simple educational machine-learning app that predicts whether a text message looks like **spam** or **normal**.

## Tools used
- Python
- Pandas
- Scikit-learn
- TF-IDF text features
- Multinomial Naive Bayes classifier
- Streamlit web interface

## Project files
- `app.py` — app interface and model training/prediction
- `sms_spam.csv` — small sample dataset used for the demonstration
- `requirements.txt` — Python dependencies

## Run locally
1. Install Python 3.10 or newer.
2. Download or clone this repository.
3. Open a terminal in the project folder.
4. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

5. Start the app:

   ```bash
   streamlit run app.py
   ```

6. A browser tab should open. Enter a message and click **Check Message**.

## How it works
1. The sample dataset contains messages labelled `spam` or `ham` (normal).
2. TF-IDF converts text into numerical features.
3. Multinomial Naive Bayes learns patterns from the labelled messages.
4. Streamlit displays the prediction and an estimated spam score.

## Important limitation
This repository uses a small sample dataset created for learning and demonstration. It is not a production-grade spam filter. The model can make mistakes, and its displayed score is not a guarantee that a message is spam or safe.

## Possible improvements
- Train on a larger, publicly available labelled dataset.
- Compare Naive Bayes with other classifiers.
- Add more evaluation metrics and a confusion matrix.
- Improve the interface and test on messages not in the sample dataset.
