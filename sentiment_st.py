import streamlit as st
import joblib
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import string
import nltk
import requests
import random
import PyPDF2
from io import BytesIO
from streamlit_lottie import st_lottie
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize

# --- 1. APP CONFIGURATION ---
st.set_page_config(
    page_title="Sentimind | Pratham Goyal",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. CYBERPUNK THEME ---
st.markdown("""
<style>
    /* DEEP SPACE BACKGROUND */
    .stApp {
        background: #0f1116;
        background: radial-gradient(circle at 50% 0%, #1f2335 0%, #0f1116 100%);
    }

    /* GLASS CARDS */
    .glass-container {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
    }
    .glass-container:hover {
        border: 1px solid rgba(79, 139, 249, 0.4);
        box-shadow: 0 0 20px rgba(79, 139, 249, 0.2);
    }

    /* TYPOGRAPHY */
    h1, h2, h3 { font-family: 'Inter', sans-serif; letter-spacing: -0.5px; color: #fff !important; }
    p, label, li, span { color: #a0a0a0 !important; }

    /* INPUTS */
    .stTextArea textarea, .stTextInput input {
        background-color: rgba(0,0,0,0.4) !important;
        border: 1px solid #333 !important;
        color: #eee !important;
        border-radius: 12px;
    }
    .stTextArea textarea:focus { border-color: #4F8BF9 !important; }

    /* NEON BUTTONS */
    .stButton>button {
        background: linear-gradient(90deg, #4F8BF9 0%, #895CF3 100%);
        border: none;
        color: white;
        padding: 12px 24px;
        text-transform: uppercase;
        font-weight: 700;
        letter-spacing: 1px;
        border-radius: 8px;
        transition: all 0.3s ease;
        width: 100%;
    }
    .stButton>button:hover {
        box-shadow: 0 0 20px rgba(137, 92, 243, 0.6);
        transform: scale(1.02);
    }

    /* HIGHLIGHTER */
    .highlight-word {
        background-color: rgba(79, 139, 249, 0.2);
        border-bottom: 2px solid #4F8BF9;
        color: white;
        padding: 0 4px;
        border-radius: 4px;
        font-weight: bold;
    }

    /* TABS */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(255,255,255,0.02);
        border-radius: 8px;
        color: #aaa;
        padding: 10px 20px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4F8BF9;
        color: white;
    }

    /* LINKEDIN BADGE STYLE */
    .linkedin-badge {
        background-color: #0077b5;
        color: white !important;
        padding: 8px 15px;
        border-radius: 5px;
        text-decoration: none;
        font-weight: bold;
        display: block;
        text-align: center;
        margin-top: 10px;
    }
    .linkedin-badge:hover { opacity: 0.9; }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# --- 3. CORE LOGIC ---
@st.cache_resource
def load_engine():
    # Robust NLTK loading
    for p in ['stopwords', 'punkt', 'punkt_tab']:
        try:
            nltk.data.find(f'tokenizers/{p}')
        except LookupError:
            nltk.download(p, quiet=True)

    try:
        return joblib.load("emotion_pipeline.pkl")
    except:
        return None

@st.cache_data
def load_lottie(url):
    try:
        return requests.get(url).json()
    except:
        return None

pipeline = load_engine()
lottie_brain = load_lottie("https://assets5.lottiefiles.com/packages/lf20_5njp3vgg.json")

# Text Utilities
def clean_text(text):
    if not isinstance(text, str): return ""
    stop = set(stopwords.words("english"))
    t = text.lower().translate(str.maketrans('', '', string.punctuation))
    t = ''.join([i for i in t if not i.isdigit()])
    return ' '.join([w for w in t.split() if w not in stop])

def analyze(text):
    clean = clean_text(text)
    if not clean: return None, None, None

    # 1. Prediction
    probs = pipeline.predict_proba([clean])[0]
    classes = pipeline.classes_
    df = pd.DataFrame({'Emotion': classes, 'Probability': probs}).sort_values('Probability', ascending=False)
    top_emotion = df.iloc[0]['Emotion']

    # 2. XAI
    vec = pipeline.named_steps['tfidf']
    clf = pipeline.named_steps['clf']
    vector = vec.transform([clean])
    idx = list(clf.classes_).index(top_emotion)
    coefs = clf.coef_[idx]
    feature_names = vec.get_feature_names_out()
    input_idx = vector.nonzero()[1]

    scores = [(feature_names[i], coefs[i]) for i in input_idx]
    top_words = sorted(scores, key=lambda x: x[1], reverse=True)[:5]

    return top_emotion, df, top_words

def generate_heatmap_html(text, top_words):
    if not top_words: return text
    triggers = {w[0]: w[1] for w in top_words}
    words = text.split()
    html = []
    for word in words:
        clean = word.lower().strip(string.punctuation)
        if clean in triggers:
            score = triggers[clean]
            html.append(f'<span class="highlight-word" title="Impact: {score:.2f}">{word}</span>')
        else:
            html.append(word)
    return " ".join(html)

def generate_smart_reply(emotion, text):
    responses = {
        "anger": ["I completely understand your frustration.", "I apologize for the inconvenience caused.", "Let's make this right immediately."],
        "joy": ["That's fantastic news!", "I'm so happy to hear that!", "Celebrations are in order!"],
        "sadness": ["I'm so sorry you're going through this.", "Sending you my deepest sympathies.", "I'm here for you if you need anything."],
        "fear": ["It's going to be okay, we can handle this.", "I understand why you're worried.", "Let's take this one step at a time."],
        "love": ["I appreciate you so much.", "That means the world to me.", "Sending love your way!"],
        "surprise": ["Wow! I didn't see that coming.", "That is truly unexpected!", "Incredible news!"]
    }
    base = random.choice(responses.get(emotion, ["I understand."]))
    return f"Suggested Reply: \"{base} Thank you for sharing this with me.\""

def read_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# --- 4. SIDEBAR ---
with st.sidebar:
    if lottie_brain:
        st_lottie(lottie_brain, height=120, key="anim")

    st.markdown("## 🧠 Sentimind AI")
    st.caption("Enterprise Emotion Analytics")

    st.markdown("---")

    # --- AUTHOR SECTION ---
    st.markdown("### 👨‍💻 Developer")
    st.markdown("**Pratham Goyal**")
    st.markdown("""
        <a href="https://www.linkedin.com/in/pratham-goyal-801b682b3/" target="_blank" class="linkedin-badge">
           Connect on LinkedIn
        </a>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # --- MODEL SPECS ---
    with st.expander("🛠️ Model Specs", expanded=False):
        st.markdown("""
        **Architecture:**
        - `LogisticRegression` (Liblinear)
        - `TfidfVectorizer` (N-grams 1-2)

        **Performance:**
        - Accuracy: ~91.5%
        - Dataset: 16k Labeled Texts

        **Features:**
        - Explainable AI (XAI)
        - PDF/Document Scanning
        - 3D Visualization
        """)

    st.success("✅ System Online")

if not pipeline:
    st.error("🚨 Model missing. Please run `train_optimized.py`.")
    st.stop()

# --- 5. MAIN UI ---
st.title("Sentimind AI")
st.markdown("### Advanced Neural Linguistics & Emotion Recognition")

# TABS
tabs = st.tabs(["🔥 Neural Heatmap", "📄 Doc Intelligence", "🌌 Word Galaxy", "📈 Journey", "⚖️ Twin Lab"])

# ================= TAB 1: NEURAL HEATMAP =================
with tabs[0]:
    st.markdown('<div class="glass-container">', unsafe_allow_html=True)
    txt = st.text_area("Input Stream", height=100, placeholder="Type a message...", key="t1")

    if st.button("RUN DIAGNOSTICS"):
        if txt:
            emotion, df, triggers = analyze(txt)

            if emotion:
                # Top Stats
                c1, c2, c3 = st.columns([1, 1, 2])
                emojis = {"joy": "⚡", "sadness": "💧", "anger": "🔥", "fear": "🕸️", "love": "❤️", "surprise": "✨"}

                with c1:
                    st.markdown(f"""
                    <div style="text-align:center; padding:10px;">
                        <h1 style="font-size:60px; margin:0;">{emojis.get(emotion, '🤖')}</h1>
                        <h3 style="color:#4F8BF9;">{emotion.upper()}</h3>
                    </div>
                    """, unsafe_allow_html=True)

                with c2:
                    conf = df.iloc[0]['Probability']
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number", value=conf*100,
                        number={'suffix': "%", 'font': {'color': "white"}},
                        gauge={'axis': {'range': [None, 100]}, 'bar': {'color': "#895CF3"}, 'bgcolor': "rgba(255,255,255,0.05)"}
                    ))
                    fig.update_layout(height=150, margin=dict(t=20, b=0, l=20, r=20), paper_bgcolor='rgba(0,0,0,0)')
                    st.plotly_chart(fig, use_container_width=True)

                with c3:
                    st.markdown("**🧠 Smart Response System:**")
                    reply = generate_smart_reply(emotion, txt)
                    st.info(reply)

                # Heatmap
                st.markdown("---")
                st.markdown("**🔍 Neural Triggers (Heatmap):**")
                html_text = generate_heatmap_html(txt, triggers)
                st.markdown(f'<div style="background:rgba(255,255,255,0.05); padding:15px; border-radius:10px; line-height:1.6;">{html_text}</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ================= TAB 2: DOC INTELLIGENCE =================
with tabs[1]:
    st.markdown('<div class="glass-container">', unsafe_allow_html=True)
    st.markdown("#### 📄 Document Scanner (PDF/TXT)")
    uploaded_file = st.file_uploader("Upload Document", type=["pdf", "txt"])

    if uploaded_file:
        if st.button("SCAN DOCUMENT"):
            with st.spinner("Extracting text vectors..."):
                if uploaded_file.type == "application/pdf":
                    doc_text = read_pdf(uploaded_file)
                else:
                    doc_text = uploaded_file.read().decode("utf-8")

                if len(doc_text) > 50:
                    emotion, df, _ = analyze(doc_text[:2000]) # Analyze first 2000 chars

                    st.success("Document Analyzed Successfully")
                    col_d1, col_d2 = st.columns([1, 2])

                    with col_d1:
                        st.metric("Detected Mood", emotion.upper())
                        st.metric("Confidence", f"{df.iloc[0]['Probability']:.1%}")
                        st.metric("Character Count", len(doc_text))

                    with col_d2:
                        st.markdown("**Executive Summary:**")
                        st.caption(doc_text[:500] + "...")

                else:
                    st.warning("Document too short or empty.")
    st.markdown('</div>', unsafe_allow_html=True)

# ================= TAB 3: 3D WORD GALAXY =================
with tabs[2]:
    st.markdown('<div class="glass-container">', unsafe_allow_html=True)
    st.markdown("#### 🌌 3D Emotional Word Galaxy")
    st.markdown("Upload data to visualize the emotional weight of your vocabulary in 3D space.")

    upl = st.file_uploader("Upload CSV for Galaxy View", type="csv")

    if upl:
        df_g = pd.read_csv(upl)
        # Try to find a text column automatically
        text_cols = [c for c in df_g.columns if df_g[c].dtype == 'object']

        if text_cols:
            txt_col = st.selectbox("Text Column", text_cols)

            if st.button("LAUNCH GALAXY"):
                words, emotions, sizes = [], [], []

                # Sample data
                for txt in df_g[txt_col].astype(str).head(50):
                    em, _, trig = analyze(txt)
                    if trig:
                        for w, s in trig:
                            words.append(w)
                            emotions.append(em)
                            # FIX: Use Absolute Value (abs) to prevent negative size error
                            sizes.append(abs(s) * 100)

                if words:
                    galaxy_df = pd.DataFrame({'Word': words, 'Emotion': emotions, 'Size': sizes})
                    galaxy_df['X'] = [random.uniform(-10, 10) for _ in range(len(words))]
                    galaxy_df['Y'] = [random.uniform(-10, 10) for _ in range(len(words))]
                    galaxy_df['Z'] = [random.uniform(-10, 10) for _ in range(len(words))]

                    fig = px.scatter_3d(galaxy_df, x='X', y='Y', z='Z', color='Emotion',
                                        text='Word', size='Size', opacity=0.8, template="plotly_dark")
                    fig.update_layout(scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False)))
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Not enough emotional data found in text.")
        else:
             st.error("No text column found in CSV.")

    st.markdown('</div>', unsafe_allow_html=True)

# ================= TAB 4: JOURNEY =================
with tabs[3]:
    st.markdown('<div class="glass-container">', unsafe_allow_html=True)
    st.markdown("#### 📉 Emotional Timeline")
    story = st.text_area("Paste Story", height=100)

    if st.button("TRACK TIMELINE"):
        if story:
            sents = sent_tokenize(story)
            if len(sents) >= 2:
                data = []
                for i, s in enumerate(sents):
                    em, d, _ = analyze(s)
                    if em:
                        data.append({"Step": i+1, "Text": s[:30]+"...", "Confidence": d.iloc[0]['Probability'], "Emotion": em})

                df_j = pd.DataFrame(data)
                fig = px.area(df_j, x="Step", y="Confidence", color="Emotion", markers=True, template="plotly_dark")
                fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Need more sentences to build a timeline.")
    st.markdown('</div>', unsafe_allow_html=True)

# ================= TAB 5: TWIN LAB =================
with tabs[4]:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="glass-container">', unsafe_allow_html=True)
        t_a = st.text_area("Variant A", height=100)
        st.markdown('</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="glass-container">', unsafe_allow_html=True)
        t_b = st.text_area("Variant B", height=100)
        st.markdown('</div>', unsafe_allow_html=True)

    if st.button("COMPARE A/B"):
        if t_a and t_b:
            _, df_a, _ = analyze(t_a)
            _, df_b, _ = analyze(t_b)
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(r=df_a['Probability'], theta=df_a['Emotion'], fill='toself', name='Variant A'))
            fig.add_trace(go.Scatterpolar(r=df_b['Probability'], theta=df_b['Emotion'], fill='toself', name='Variant B'))
            fig.update_layout(polar=dict(radialaxis=dict(visible=True)), paper_bgcolor='rgba(0,0,0,0)', font={'color':'white'})
            st.plotly_chart(fig, use_container_width=True)
