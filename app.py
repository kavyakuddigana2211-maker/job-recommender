import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Job Recommender",
    page_icon="🚀",
    layout="centered"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Sans:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0d0d0d;
    color: #f0f0f0;
}

/* Hero header */
.hero {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    border-radius: 20px;
    padding: 40px 30px;
    text-align: center;
    margin-bottom: 30px;
    border: 1px solid #e94560;
    box-shadow: 0 0 40px rgba(233,69,96,0.2);
}
.hero h1 {
    font-family: 'Syne', sans-serif;
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(90deg, #e94560, #f5a623, #e94560);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 10px 0;
}
.hero p {
    font-size: 1.05rem;
    color: #aaa;
    margin: 0;
}

/* Input section */
.input-label {
    font-family: 'Syne', sans-serif;
    font-size: 1rem;
    font-weight: 700;
    color: #f5a623;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-bottom: 8px;
}

/* Job cards */
.job-card {
    background: linear-gradient(135deg, #1a1a2e, #16213e);
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 20px;
    border-left: 5px solid #e94560;
    box-shadow: 0 4px 20px rgba(0,0,0,0.4);
    transition: transform 0.2s;
}
.job-card:hover {
    transform: translateY(-3px);
}
.job-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.4rem;
    font-weight: 800;
    color: #ffffff;
    margin-bottom: 6px;
}
.match-badge {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 20px;
    font-weight: 700;
    font-size: 0.85rem;
    margin-bottom: 14px;
}
.match-high   { background: #1a472a; color: #2ecc71; }
.match-medium { background: #7d5a00; color: #f5a623; }
.match-low    { background: #4a1528; color: #e94560; }

.info-row {
    display: flex;
    flex-wrap: wrap;
    gap: 14px;
    margin-bottom: 12px;
}
.info-chip {
    background: rgba(255,255,255,0.07);
    border-radius: 8px;
    padding: 6px 14px;
    font-size: 0.85rem;
    color: #ccc;
}
.info-chip span { color: #f5a623; font-weight: 600; }

.reason-box {
    background: rgba(233,69,96,0.08);
    border: 1px solid rgba(233,69,96,0.25);
    border-radius: 10px;
    padding: 10px 16px;
    font-size: 0.88rem;
    color: #ddd;
    margin-top: 10px;
}
.reason-box b { color: #e94560; }

/* Progress bar */
.bar-bg {
    background: rgba(255,255,255,0.1);
    border-radius: 6px;
    height: 8px;
    margin-bottom: 14px;
}
.bar-fill {
    height: 8px;
    border-radius: 6px;
    background: linear-gradient(90deg, #e94560, #f5a623);
}

/* Empty / warning */
.warn-box {
    background: #1a1200;
    border: 1px solid #f5a623;
    border-radius: 12px;
    padding: 18px 24px;
    color: #f5a623;
    font-size: 0.95rem;
}

/* Section title */
.section-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.3rem;
    font-weight: 700;
    color: #f0f0f0;
    border-bottom: 2px solid #e94560;
    padding-bottom: 8px;
    margin-bottom: 20px;
}

/* Streamlit overrides */
div[data-testid="stTextInput"] input {
    background: #1a1a2e !important;
    color: #fff !important;
    border: 2px solid #e94560 !important;
    border-radius: 10px !important;
    font-size: 1rem !important;
    padding: 12px !important;
}
div[data-testid="stButton"] button {
    background: linear-gradient(135deg, #e94560, #c0392b) !important;
    color: white !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 12px 30px !important;
    width: 100% !important;
    letter-spacing: 1px !important;
}
</style>
""", unsafe_allow_html=True)

# ── Load data ──────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    return pd.read_csv("jobs.csv")

df = load_data()

# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>🚀 Job Recommender</h1>
    <p>Enter your skills — we'll match you with the best career paths, salaries & top companies.</p>
</div>
""", unsafe_allow_html=True)

# ── Input ──────────────────────────────────────────────────────────────────────
st.markdown('<div class="input-label">🛠️ Your Skills</div>', unsafe_allow_html=True)
user_input = st.text_input(
    "",
    placeholder="e.g. Python, Machine Learning, SQL, Deep Learning",
    label_visibility="collapsed"
)

recommend_clicked = st.button("🔍 Find My Best Jobs")

# ── Logic ──────────────────────────────────────────────────────────────────────
if recommend_clicked:
    if not user_input.strip():
        st.markdown('<div class="warn-box">⚠️ Please enter at least one skill to get recommendations.</div>', unsafe_allow_html=True)
    else:
        user_skills_list = [s.strip().lower() for s in user_input.split(",")]

        skills_corpus = df["Skills"].tolist()
        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform(skills_corpus + [user_input])

        similarity = cosine_similarity(vectors[-1], vectors[:-1])
        scores = similarity.flatten() * 100

        df["Match %"] = scores
        results = df[df["Match %"] >= 10].sort_values(by="Match %", ascending=False).head(5)

        if results.empty:
            st.markdown("""
            <div class="warn-box">
                😕 No strong matches found. Try adding more skills like <b>Python, SQL, Machine Learning</b> etc.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="section-title">✅ Top {len(results)} Job Matches For You</div>', unsafe_allow_html=True)

            for i, (_, row) in enumerate(results.iterrows()):
                match = round(row["Match %"], 1)

                if match >= 60:
                    badge_cls = "match-high"
                    badge_label = f"🟢 {match}% Match"
                    verdict = "Excellent fit"
                elif match >= 30:
                    badge_cls = "match-medium"
                    badge_label = f"🟡 {match}% Match"
                    verdict = "Good fit"
                else:
                    badge_cls = "match-low"
                    badge_label = f"🔴 {match}% Match"
                    verdict = "Partial fit — upskilling recommended"

                # Find matching skills for reason
                job_skills = [s.strip().lower() for s in row["Skills"].split(",")]
                matched = [s for s in user_skills_list if any(s in js for js in job_skills)]
                missing  = [s.title() for s in job_skills if not any(u in s for u in user_skills_list)]

                matched_str = ", ".join([s.title() for s in matched]) if matched else "General alignment"
                missing_str = ", ".join(missing[:3]) if missing else "None"

                bar_width = min(int(match), 100)

                st.markdown(f"""
                <div class="job-card">
                    <div class="job-title">{'🥇' if i==0 else '🔹'} {row['Job Title']}</div>
                    <span class="match-badge {badge_cls}">{badge_label} — {verdict}</span>

                    <div class="bar-bg"><div class="bar-fill" style="width:{bar_width}%"></div></div>

                    <div class="info-row">
                        <div class="info-chip">💰 <span>Salary:</span> {row['Salary Range']}</div>
                        <div class="info-chip">🏢 <span>Top Companies:</span> {row['Companies']}</div>
                    </div>

                    <div class="info-chip" style="margin-bottom:12px; display:block;">
                        📋 <span>Role:</span> {row['Description']}
                    </div>

                    <div class="reason-box">
                        <b>💡 Why this job?</b><br>
                        Your skills in <b>{matched_str}</b> are a strong match for this role.<br>
                        {'<b>📚 Skills to add:</b> ' + missing_str if missing_str != 'None' else '✅ You already cover the key skills!'}
                    </div>
                </div>
                """, unsafe_allow_html=True)

# ── Footer ──────────────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align:center; color:#555; font-size:0.8rem;'>
    Built with ❤️ by Kavia &nbsp;|&nbsp; Powered by Machine Learning
</div>
""", unsafe_allow_html=True)
