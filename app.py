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
    padding: 28px;
    margin-bottom: 24px;
    border-left: 5px solid #e94560;
    box-shadow: 0 4px 20px rgba(0,0,0,0.4);
}
.job-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.5rem;
    font-weight: 800;
    color: #ffffff;
    margin-bottom: 8px;
}
.match-badge {
    display: inline-block;
    padding: 5px 16px;
    border-radius: 20px;
    font-weight: 700;
    font-size: 0.88rem;
    margin-bottom: 16px;
}
.match-high   { background: #1a472a; color: #2ecc71; }
.match-medium { background: #7d5a00; color: #f5a623; }
.match-low    { background: #4a1528; color: #e94560; }

/* Progress bar */
.bar-bg {
    background: rgba(255,255,255,0.1);
    border-radius: 6px;
    height: 10px;
    margin-bottom: 20px;
}
.bar-fill {
    height: 10px;
    border-radius: 6px;
    background: linear-gradient(90deg, #e94560, #f5a623);
}

/* Detail rows */
.detail-section {
    margin-bottom: 14px;
}
.detail-label {
    font-size: 0.75rem;
    font-weight: 700;
    color: #f5a623;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 4px;
}
.detail-value {
    font-size: 1rem;
    color: #f0f0f0;
    line-height: 1.5;
}
.detail-value.salary {
    font-size: 1.15rem;
    font-weight: 700;
    color: #2ecc71;
}

/* Skills pills */
.skills-wrap {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: 4px;
}
.skill-pill {
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.82rem;
    font-weight: 500;
}
.skill-matched {
    background: rgba(46,204,113,0.15);
    color: #2ecc71;
    border: 1px solid rgba(46,204,113,0.4);
}
.skill-missing {
    background: rgba(245,166,35,0.1);
    color: #f5a623;
    border: 1px solid rgba(245,166,35,0.3);
}

/* Companies row */
.companies-row {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: 4px;
}
.company-tag {
    background: rgba(255,255,255,0.07);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 8px;
    padding: 4px 12px;
    font-size: 0.82rem;
    color: #ccc;
}

/* Tip box */
.tip-box {
    background: rgba(233,69,96,0.07);
    border: 1px solid rgba(233,69,96,0.2);
    border-radius: 10px;
    padding: 12px 16px;
    font-size: 0.9rem;
    color: #ddd;
    margin-top: 16px;
    line-height: 1.6;
}
.tip-box b { color: #e94560; }

/* Divider */
.divider {
    border: none;
    border-top: 1px solid rgba(255,255,255,0.07);
    margin: 16px 0;
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
    margin-bottom: 24px;
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
    <p>Enter your skills — we'll match you with the best career paths, salaries &amp; top companies.</p>
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

# ── Recommend Logic ──────────────────────────────────────────────────────────────────
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

                # Badge and verdict
                if match >= 60:
                    badge_cls   = "match-high"
                    badge_label = f"🟢 {match}% Match"
                    verdict     = "Excellent Fit"
                elif match >= 30:
                    badge_cls   = "match-medium"
                    badge_label = f"🟡 {match}% Match"
                    verdict     = "Good Fit"
                else:
                    badge_cls   = "match-low"
                    badge_label = f"🔴 {match}% Match"
                    verdict     = "Partial Fit — Upskilling Recommended"

                # Skills comparison
                job_skills    = [s.strip().lower() for s in str(row["Skills"]).split(",")]
                matched_skills = [s.strip().title() for s in user_skills_list if any(s in js for js in job_skills)]
                missing_skills = [s.strip().title() for s in job_skills if not any(u in s for u in user_skills_list)]

                matched_pills = "".join(
                    [f'<span class="skill-pill skill-matched">✓ {s}</span>' for s in matched_skills]
                ) if matched_skills else '<span style="color:#aaa; font-size:0.88rem;">General alignment with job requirements</span>'

                missing_pills = "".join(
                    [f'<span class="skill-pill skill-missing">+ {s}</span>' for s in missing_skills[:5]]
                ) if missing_skills else '<span style="color:#2ecc71; font-size:0.88rem;">🎉 You already cover all key skills!</span>'

                # Companies as tags
                companies     = [c.strip() for c in str(row["Companies"]).split(",")]
                company_tags  = "".join([f'<span class="company-tag">{c}</span>' for c in companies])

                bar_width = min(int(match), 100)
                crown     = "🥇" if i == 0 else f"#{i+1}"

                st.markdown(f"""
                <div class="job-card">

                    <!-- Title & Badge -->
                    <div class="job-title">{crown} &nbsp;{row['Job Title']}</div>
                    <span class="match-badge {badge_cls}">{badge_label} &nbsp;—&nbsp; {verdict}</span>

                    <!-- Match Progress Bar -->
                    <div class="bar-bg">
                        <div class="bar-fill" style="width:{bar_width}%"></div>
                    </div>

                    <!-- Role Description -->
                    <div class="detail-section">
                        <div class="detail-label">📋 What You'll Do</div>
                        <div class="detail-value">{row['Description']}</div>
                    </div>

                    <hr class="divider">

                    <!-- Salary -->
                    <div class="detail-section">
                        <div class="detail-label">💰 Expected Salary</div>
                        <div class="detail-value salary">{row['Salary Range']}</div>
                    </div>

                    <hr class="divider">

                    <!-- Companies -->
                    <div class="detail-section">
                        <div class="detail-label">🏢 Top Companies Hiring</div>
                        <div class="companies-row">{company_tags}</div>
                    </div>

                    <hr class="divider">

                    <!-- Your Matching Skills -->
                    <div class="detail-section">
                        <div class="detail-label">✅ Your Matching Skills</div>
                        <div class="skills-wrap">{matched_pills}</div>
                    </div>

                    <!-- Skills to Learn -->
                    <div class="detail-section" style="margin-top:14px;">
                        <div class="detail-label">📚 Skills to Learn / Add</div>
                        <div class="skills-wrap">{missing_pills}</div>
                    </div>

                    <!-- Tip -->
                    <div class="tip-box">
                        <b>💡 Career Tip:</b>
                        {'You are a strong candidate for this role. Apply confidently and highlight your experience with ' + (', '.join(matched_skills[:3]) if matched_skills else 'relevant projects') + '.' if match >= 60 else
                         'You have a good foundation. Strengthening your skills in ' + (', '.join(missing_skills[:2]) if missing_skills else 'related areas') + ' will make you much more competitive.' if match >= 30 else
                         'This role needs some additional skills. Focus on learning ' + (', '.join(missing_skills[:3]) if missing_skills else 'the required skills') + ' through online courses or projects.'}
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
