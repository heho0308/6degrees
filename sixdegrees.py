import streamlit as st
import pandas as pd
import re
import requests
from bs4 import BeautifulSoup
from fuzzywuzzy import fuzz
from transformers import pipeline
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords

# Download required NLTK resources
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("averaged_perceptron_tagger")

# Load Hugging Face Model for Job Criteria Extraction
@st.cache_resource
def load_nlp_model():
    return pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

nlp_model = load_nlp_model()

# ---------------------------
# Helper Functions
# ---------------------------

def clean_csv_data(df):
    """Cleans LinkedIn connections CSV."""
    expected_columns = ["First Name", "Last Name", "URL", "Email Address", "Company", "Position", "Connected On"]
    for col in expected_columns:
        if col not in df.columns:
            df[col] = ""
    df = df[expected_columns]
    df.drop_duplicates(inplace=True)
    return df

def extract_linkedin_connections(csv_file):
    """Reads LinkedIn connections CSV."""
    try:
        df = pd.read_csv(csv_file, skiprows=3, encoding="utf-8")
        if all(col in df.columns for col in ["First Name", "Last Name", "Company", "Position"]):
            return clean_csv_data(df)
    except:
        pass
    return None

# ---------------------------
# AI-Enhanced Job Description Extraction
# ---------------------------
def extract_job_description(url):
    """Extract job description from a given URL."""
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        paragraphs = [p.get_text(strip=True) for p in soup.find_all("p") if len(p.get_text(strip=True)) > 50]
        return " ".join(paragraphs) if paragraphs else None
    except Exception as e:
        st.error(f"Failed to fetch job description: {e}")
        return None

# ---------------------------
# AI-Enhanced Job Criteria Extraction
# ---------------------------
def extract_job_criteria(url):
    """Extract job title, seniority, industry, and experience requirements using AI."""
    job_desc = extract_job_description(url)
    if not job_desc:
        return None
    
    labels = ["Software Engineer", "Data Scientist", "Manager", "Director", "Analyst", "Consultant", "Marketing Specialist", "Sales Representative"]
    title_prediction = nlp_model(job_desc, labels)
    job_title = title_prediction["labels"][0] if title_prediction["scores"][0] > 0.5 else "Unknown"
    
    return {
        "job_title": st.text_input("Job Title", job_title),
        "seniority": st.selectbox("Seniority", ["Entry Level", "Mid-Level", "Senior"], index=1),
        "industry": st.text_input("Industry", "Technology" if "developer" in job_desc.lower() else "General"),
        "years_experience": st.number_input("Years of Experience", min_value=0, max_value=50, value=int(re.search(r'()years?', job_desc.lower()).group(1)) if re.search(r'()years?', job_desc.lower()) else 0)
    }

# ---------------------------
# AI-Enhanced Candidate Matching
# ---------------------------
def match_candidates(connections_df, criteria):
    """Matches candidates based on AI-extracted job criteria and suggests warm introductions."""
    if connections_df is None or connections_df.empty:
        return pd.DataFrame()

    def score_candidate(row):
        if str(row.get("Company", "")).lower() == criteria.get("company_name", "").lower():
            return 0  # Exclude current employees
        
        score = fuzz.token_sort_ratio(criteria["job_title"].lower(), str(row.get("Position", "")).lower())
        return round(score, 2)

    connections_df["match_score"] = connections_df.apply(score_candidate, axis=1)
    result_df = connections_df.sort_values(by="match_score", ascending=False).head(5)
    result_df["warm_introduction"] = result_df["First Name"] + " " + result_df["Last Name"] + " is a potential match. Reach out to their mutual connection for an introduction."
    return result_df[["First Name", "Last Name", "Position", "Company", "match_score", "URL", "warm_introduction"]]

# ---------------------------
# Streamlit App
# ---------------------------
st.title("6 Degrees")
st.subheader("Use your human capital in new ways")

st.sidebar.header("📌 How to Download LinkedIn Contacts")
st.sidebar.markdown("""
1. Go to [LinkedIn Data Export](https://www.linkedin.com/psettings/member-data)
2. Select **Connections** and request an archive.
3. Download the **CSV file** once LinkedIn sends it.
4. Upload the CSV here for analysis.
""")

job_url = st.text_input("🔗 Enter Job Posting URL")
criteria = extract_job_criteria(job_url)
if criteria:
    with st.form(key='criteria_form'):
        job_title = st.text_input("Job Title", criteria["job_title"])
        seniority = st.selectbox("Seniority", ["Entry Level", "Mid-Level", "Senior"], index=["Entry Level", "Mid-Level", "Senior"].index(criteria["seniority"]))
        industry = st.text_input("Industry", criteria["industry"])
        years_experience = st.number_input("Years of Experience", min_value=0, max_value=50, value=criteria["years_experience"])
        save_button = st.form_submit_button("Save Criteria")
    
    if save_button:
        criteria["job_title"] = job_title
        criteria["seniority"] = seniority
        criteria["industry"] = industry
        criteria["years_experience"] = years_experience
        st.success("✅ Criteria updated successfully!")

uploaded_file = st.file_uploader("📤 Upload LinkedIn Connections (CSV)", type=["csv"])
if uploaded_file:
    connections_df = extract_linkedin_connections(uploaded_file)
    st.success("✅ Connections uploaded successfully!")
    st.dataframe(connections_df.head())
    if st.button("Find Best Candidates and Suggest Introductions"):
        matches = match_candidates(connections_df, criteria)
        st.write(matches.to_html(escape=False), unsafe_allow_html=True)

