import re
from io import BytesIO

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="AI Resume Screening & Matching",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# -----------------------------
# Styling
# -----------------------------
st.markdown(
    """
    <style>
        /* Remove default Streamlit spacing */
        .block-container {
            padding: 0rem 2rem 2rem 2rem !important;
            max-width: 100% !important;
        }

        header[data-testid="stHeader"] {
            background: transparent;
        }

        [data-testid="stAppViewContainer"] {
            background: #18181B;
        }

        [data-testid="stSidebar"] {
            background: #18181B;
        }

        .hero {
            background: linear-gradient(135deg, #27272A 0%, #18181B 100%);
            border: 1px solid #3F3F46;
            border-radius: 20px;
            padding: 28px 32px;
            margin: 18px 0 24px 0;
        }

        .hero h1 {
            margin: 0;
            color: #F4F4F5;
            font-size: 2.25rem;
            font-weight: 800;
            letter-spacing: -1px;
        }

        .hero p {
            margin: 8px 0 0 0;
            color: #A1A1AA;
            font-size: 1rem;
        }

        .card {
            background: #27272A;
            border: 1px solid #3F3F46;
            border-radius: 16px;
            padding: 20px;
            margin-bottom: 18px;
        }

        .section-title {
            color: #F4F4F5;
            font-size: 1.15rem;
            font-weight: 700;
            margin-bottom: 12px;
        }

        .metric-card {
            background: #27272A;
            border: 1px solid #3F3F46;
            border-radius: 14px;
            padding: 18px;
            min-height: 105px;
        }

        .metric-label {
            color: #A1A1AA;
            font-size: 0.85rem;
            margin-bottom: 8px;
        }

        .metric-value {
            color: #10B981;
            font-size: 1.75rem;
            font-weight: 800;
        }

        .small-note {
            color: #A1A1AA;
            font-size: 0.82rem;
        }

        div[data-testid="stFileUploader"] {
            background: #18181B;
            border: 1px dashed #52525B;
            border-radius: 12px;
            padding: 10px;
        }

        .stButton > button {
            background: #10B981;
            color: #052E16;
            border: none;
            border-radius: 10px;
            font-weight: 700;
            padding: 0.65rem 1.2rem;
        }

        .stButton > button:hover {
            background: #34D399;
            color: #052E16;
        }

        textarea, input {
            border-radius: 10px !important;
        }

        .candidate-badge {
            display: inline-block;
            background: #064E3B;
            color: #6EE7B7;
            border-radius: 999px;
            padding: 4px 10px;
            font-size: 0.75rem;
            font-weight: 700;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------
# Utility functions
# -----------------------------
def extract_pdf_text(uploaded_file) -> str:
    """Extract text from a PDF uploaded through Streamlit."""
    try:
        reader = PdfReader(BytesIO(uploaded_file.getvalue()))
        pages = [(page.extract_text() or "") for page in reader.pages]
        return "\n".join(pages).strip()
    except Exception as exc:
        return f"ERROR: {exc}"


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9+#.\s-]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_keywords(text: str) -> set[str]:
    """
    Lightweight keyword extraction. For production use, replace this
    with a domain-specific skills taxonomy or an LLM-based extractor.
    """
    stopwords = {
        "and", "the", "for", "with", "you", "your", "are", "our", "from",
        "that", "this", "will", "have", "has", "job", "role", "work",
        "years", "year", "into", "using", "their", "they", "who", "all",
        "not", "but", "can", "should", "must", "about", "required",
        "requirements", "responsibilities", "experience", "skills",
    }

    words = re.findall(r"[a-zA-Z][a-zA-Z0-9+#.-]{1,}", text.lower())
    return {
        word.strip(".-")
        for word in words
        if len(word) > 2 and word not in stopwords
    }


def keyword_match(job_text: str, resume_text: str):
    job_keywords = extract_keywords(job_text)
    resume_keywords = extract_keywords(resume_text)

    matched = job_keywords.intersection(resume_keywords)
    missing = job_keywords.difference(resume_keywords)

    match_percent = (len(matched) / len(job_keywords) * 100) if job_keywords else 0
    return round(match_percent, 1), sorted(missing)


def calculate_scores(job_description: str, resumes: list[tuple[str, str]]):
    documents = [job_description] + [resume_text for _, resume_text in resumes]

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=8000,
    )

    matrix = vectorizer.fit_transform(documents)
    similarities = cosine_similarity(matrix[0:1], matrix[1:]).flatten()

    rows = []
    for (filename, resume_text), similarity in zip(resumes, similarities):
        skill_match, missing_keywords = keyword_match(job_description, resume_text)

        rows.append(
            {
                "Candidate": filename.rsplit(".", 1)[0],
                "File": filename,
                "Similarity Score": round(float(similarity) * 100, 1),
                "Skill Match %": skill_match,
                "Missing Keywords": ", ".join(missing_keywords[:12])
                if missing_keywords
                else "None",
                "Resume Text": resume_text,
            }
        )

    result = pd.DataFrame(rows)
    if not result.empty:
        result["Overall Score"] = (
            result["Similarity Score"] * 0.65
            + result["Skill Match %"] * 0.35
        ).round(1)
        result = result.sort_values("Overall Score", ascending=False).reset_index(drop=True)
        result.insert(0, "Rank", np.arange(1, len(result) + 1))

    return result


def skill_scores(job_description: str, resume_text: str):
    skill_groups = {
        "Python": ["python", "pandas", "numpy", "scikit-learn", "tensorflow", "pytorch"],
        "SQL": ["sql", "mysql", "postgresql", "database", "queries", "joins"],
        "ML": [
            "machine learning",
            "machine-learning",
            "regression",
            "classification",
            "clustering",
            "deep learning",
            "modeling",
        ],
        "Analytics": [
            "analytics",
            "data analysis",
            "tableau",
            "power bi",
            "excel",
            "statistics",
            "visualization",
        ],
    }

    job = normalize_text(job_description)
    resume = normalize_text(resume_text)

    scores = {}
    for skill, keywords in skill_groups.items():
        required = [kw for kw in keywords if kw in job]
        present = [kw for kw in keywords if kw in resume]

        if required:
            scores[skill] = min(100, round(len(set(required) & set(present)) / len(required) * 100))
        else:
            scores[skill] = min(100, round(len(present) / max(1, len(keywords)) * 100))

    return scores


def radar_chart(selected_rows: pd.DataFrame, all_rows: pd.DataFrame):
    categories = ["Python", "SQL", "ML", "Analytics"]
    fig = go.Figure()

    for _, row in selected_rows.iterrows():
        scores = skill_scores(
            st.session_state.get("job_description", ""),
            row["Resume Text"],
        )
        values = [scores[c] for c in categories]
        values += values[:1]

        fig.add_trace(
            go.Scatterpolar(
                r=values,
                theta=categories + [categories[0]],
                fill="toself",
                name=row["Candidate"],
                opacity=0.72,
            )
        )

    fig.update_layout(
        polar=dict(
            bgcolor="#18181B",
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                gridcolor="#52525B",
                tickfont=dict(color="#A1A1AA"),
            ),
            angularaxis=dict(
                gridcolor="#52525B",
                tickfont=dict(color="#E4E4E7"),
            ),
        ),
        paper_bgcolor="#27272A",
        plot_bgcolor="#27272A",
        font=dict(color="#E4E4E7"),
        margin=dict(l=30, r=30, t=35, b=30),
        height=450,
        legend=dict(orientation="h", y=-0.12),
    )
    return fig


# -----------------------------
# Header
# -----------------------------
st.markdown(
    """
    <div class="hero">
        <h1>🧠 AI Resume Screening & Matching</h1>
        <p>Rank candidates intelligently using TF-IDF similarity, keyword matching, and skill-level comparisons.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Inputs
# -----------------------------
left, right = st.columns([1.1, 0.9], gap="large")

with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">1. Paste Job Description</div>', unsafe_allow_html=True)
    job_description = st.text_area(
        "Job Description",
        height=265,
        placeholder=(
            "Paste the complete job description here...\n\n"
            "Example: Looking for a Data Analyst with Python, SQL, "
            "machine learning, statistics, Excel, and Power BI experience."
        ),
        label_visibility="collapsed",
    )
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">2. Upload Candidate Resumes</div>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "Upload PDF resumes",
        type=["pdf"],
        accept_multiple_files=True,
        help="You can upload multiple candidate resumes at once.",
    )
    st.markdown(
        '<div class="small-note">Supported format: PDF • Multiple files allowed</div>',
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

st.session_state["job_description"] = job_description

analyze = st.button("🚀 Analyze & Rank Candidates", use_container_width=True)

if analyze:
    if not job_description.strip():
        st.warning("Please paste a job description first.")
        st.stop()

    if not uploaded_files:
        st.warning("Please upload at least one PDF resume.")
        st.stop()

    with st.spinner("Extracting resumes and calculating matching scores..."):
        resume_data = []
        extraction_errors = []

        for uploaded_file in uploaded_files:
            text = extract_pdf_text(uploaded_file)
            if text.startswith("ERROR:"):
                extraction_errors.append(f"{uploaded_file.name}: {text}")
            elif not text.strip():
                extraction_errors.append(f"{uploaded_file.name}: No readable text found.")
            else:
                resume_data.append((uploaded_file.name, text))

        if extraction_errors:
            for error in extraction_errors:
                st.warning(error)

        if not resume_data:
            st.error("No readable resume text was found.")
            st.stop()

        results = calculate_scores(job_description, resume_data)
        st.session_state["results"] = results

if "results" in st.session_state and not st.session_state["results"].empty:
    results = st.session_state["results"]

    st.markdown("## Screening Overview")
    m1, m2, m3, m4 = st.columns(4)

    metrics = [
        ("Resumes Screened", len(results)),
        ("Top Candidate Score", f"{results.iloc[0]['Overall Score']:.1f}%"),
        ("Average Match", f"{results['Overall Score'].mean():.1f}%"),
        ("Strong Matches", int((results["Overall Score"] >= 70).sum())),
    ]

    for col, (label, value) in zip([m1, m2, m3, m4], metrics):
        with col:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value">{value}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("## 🏆 Candidate Leaderboard")

    display_df = results[
        [
            "Rank",
            "Candidate",
            "Overall Score",
            "Similarity Score",
            "Skill Match %",
            "Missing Keywords",
        ]
    ].copy()

    display_df = display_df.rename(
        columns={
            "Overall Score": "Overall Match %",
            "Similarity Score": "TF-IDF Similarity %",
        }
    )

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Rank": st.column_config.NumberColumn("Rank", width="small"),
            "Overall Match %": st.column_config.ProgressColumn(
                "Overall Match %",
                min_value=0,
                max_value=100,
                format="%.1f%%",
            ),
            "TF-IDF Similarity %": st.column_config.NumberColumn(format="%.1f%%"),
            "Skill Match %": st.column_config.NumberColumn(format="%.1f%%"),
        },
    )

    st.download_button(
        "⬇️ Download Screening Results CSV",
        data=display_df.to_csv(index=False).encode("utf-8"),
        file_name="resume_screening_results.csv",
        mime="text/csv",
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("## 📊 Top Candidate Skill Comparison")

    top_n = min(3, len(results))
    selected_candidates = st.multiselect(
        "Select candidates for radar comparison",
        options=results["Candidate"].tolist(),
        default=results.head(top_n)["Candidate"].tolist(),
        max_selections=5,
    )

    if selected_candidates:
        selected_rows = results[results["Candidate"].isin(selected_candidates)]
        st.plotly_chart(
            radar_chart(selected_rows, results),
            use_container_width=True,
        )
    else:
        st.info("Select at least one candidate to display the radar chart.")

    st.markdown("## 🔍 Candidate Details")
    selected_detail = st.selectbox(
        "View missing keywords and extracted resume text",
        results["Candidate"].tolist(),
    )

    selected_row = results[results["Candidate"] == selected_detail].iloc[0]

    detail_left, detail_right = st.columns(2)
    with detail_left:
        st.markdown(
            f"""
            <div class="card">
                <div class="section-title">{selected_row['Candidate']}</div>
                <p><b>Overall Match:</b> {selected_row['Overall Score']}%</p>
                <p><b>TF-IDF Similarity:</b> {selected_row['Similarity Score']}%</p>
                <p><b>Skill Match:</b> {selected_row['Skill Match %']}%</p>
                <p><b>Missing Keywords:</b> {selected_row['Missing Keywords']}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with detail_right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Extracted Resume Text</div>', unsafe_allow_html=True)
        st.text_area(
            "Resume text",
            selected_row["Resume Text"],
            height=230,
            label_visibility="collapsed",
        )
        st.markdown("</div>", unsafe_allow_html=True)

else:
    st.markdown(
        """
        <div class="card" style="text-align:center; padding:45px;">
            <div style="font-size:2.5rem;">📄</div>
            <h3 style="color:#F4F4F5;">Ready to screen resumes</h3>
            <p style="color:#A1A1AA;">
                Add a job description and upload PDF resumes to generate an AI-style
                candidate ranking dashboard.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    '<div class="small-note" style="text-align:center; margin-top:25px;">Built with Streamlit • PyPDF2 • Scikit-Learn • Plotly</div>',
    unsafe_allow_html=True,
)
