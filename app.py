from pathlib import Path
import json
import joblib
import streamlit as st

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title='SpamShield | Email & SMS Spam Detector', page_icon='🛡️', layout='centered')
st.markdown('''<style>
.stApp {background: linear-gradient(135deg,#0b1220,#17223b);color:#f8fafc}
.block-container {max-width:850px;padding-top:3rem}
h1,h2,h3,p,label {color:#f8fafc!important}
div.stButton>button {background:#38bdf8;color:#081525;border:0;border-radius:12px;font-weight:700;padding:.7rem 2rem}
div[data-testid="stAlert"] {border-radius:12px}
</style>''', unsafe_allow_html=True)
st.caption('SPAMSHIELD  /  NLP CLASSIFICATION')
st.title('🛡️ Email & SMS Spam Classifier')
st.write('Analyze a message with a TF-IDF + Naive Bayes machine-learning model. Built using the SMS Spam Collection dataset.')
model_path = ROOT/'models'/'spam_pipeline.joblib'
if not model_path.exists():
    st.error('Trained model not found. Run `python train.py` after adding `data/spam.csv`, then commit the generated `models/spam_pipeline.joblib` file.')
    st.stop()
@st.cache_resource
def load_model():
    return joblib.load(model_path)
model = load_model()
message = st.text_area('Paste an email or SMS message', height=180, placeholder='Example: Congratulations! You won a free prize. Claim now...')
if st.button('Analyze message', use_container_width=True):
    if not message.strip():
        st.warning('Please enter a message first.')
    else:
        prediction = int(model.predict([message])[0])
        probability = float(model.predict_proba([message])[0][1])
        if prediction:
            st.error('🚨 Likely spam — treat this message with caution.')
        else:
            st.success('✅ Likely legitimate (ham).')
        st.metric('Model-estimated spam probability',f'{probability:.1%}')
        st.progress(probability)
        st.caption('This is a model estimate, not a guarantee. Never trust suspicious links or attachments solely because a message was classified as legitimate.')
metrics_path = ROOT/'models'/'metrics.json'
if metrics_path.exists():
    with st.expander('Model evaluation (held-out SMS test split)'):
        metrics=json.loads(metrics_path.read_text())
        c1,c2,c3=st.columns(3)
        c1.metric('Accuracy',f"{metrics['accuracy']:.1%}")
        c2.metric('Spam precision',f"{metrics['spam_precision']:.1%}")
        c3.metric('Spam recall',f"{metrics['spam_recall']:.1%}")
st.divider()
st.caption('Trained on English SMS messages; performance on full-length emails, other languages, and novel spam campaigns is not established. No message is sent to an external API by this app.')
