import streamlit as st
import pandas as pd
import re
import requests
from bs4 import BeautifulSoup
from fuzzywuzzy import fuzz
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords

# Download required NLTK resources
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("averaged_perceptron_tagger")

# Load AI Models
@st.cache_resource
def load_nlp_model():
    return pipeline("ner", model="dslim/bert-base-NER")

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def load_text_generator():
    return pipeline("text-generation", model="t5-small")

nlp_model = load_nlp_model()
embedder = load_embedding_model()
text_generator = load_text_generator()

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
    if not url:
        return None
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
    """Extract job title, seniority, industry, and hiring company using AI."""
    job_desc = extract_job_description(url)
    if not job_desc:
        return None
    
    extracted_entities = nlp_model(job_desc)
    job_title = next((ent['word'] for ent in extracted_entities if "JOB" in ent['entity']), "Unknown")
    hiring_company = next((ent['word'] for ent in extracted_entities if "ORG" in ent['entity']), "Unknown")
    
    return {
        "job_title": st.text_input("Job Title", job_title),
        "seniority": st.selectbox("Seniority", ["Entry Level", "Mid-Level", "Senior"], index=1),
        "industry": st.text_input("Industry", "Technology" if "developer" in job_desc.lower() else "General"),
        "company_name": st.text_input("Company That Is Hiring", hiring_company)
    }

# ---------------------------
# AI-Enhanced Candidate Matching
# ---------------------------
def match_candidates(connections_df, criteria):
    """Matches candidates based on AI-extracted job criteria and suggests warm introductions."""
    if connections_df is None or connections_df.empty:
        return pd.DataFrame()

    job_embedding = embedder.encode(criteria["job_title"], convert_to_tensor=True)
    
    def score_candidate(row):
        if str(row.get("Company", "")).lower() == criteria.get("company_name", "").lower():
            return 0  # Exclude current employees
        
        position = str(row.get("Position", "")).lower()
        candidate_embedding = embedder.encode(position, convert_to_tensor=True)
        similarity = util.pytorch_cos_sim(job_embedding, candidate_embedding).item()
        return round(similarity * 100, 2)

    connections_df["match_score"] = connections_df.apply(score_candidate, axis=1)
    result_df = connections_df.sort_values(by="match_score", ascending=False).head(5)
    
    def generate_intro(row):
        prompt = f"Generate a warm introduction for {row['First Name']} {row['Last Name']} for a {criteria['job_title']} role at {criteria['company_name']}. Their background as a {row['Position']} at {row['Company']} makes them a great fit."
        response = text_generator(prompt, max_length=100)
        return response[0]["generated_text"]

    result_df["warm_introduction"] = result_df.apply(generate_intro, axis=1)
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
        company_name = st.text_input("Company That Is Hiring", criteria["company_name"])
        save_button = st.form_submit_button("Save Criteria")
    
    if save_button:
        criteria["job_title"] = job_title
        criteria["seniority"] = seniority
        criteria["industry"] = industry
        criteria["company_name"] = company_name
        st.success("✅ Criteria updated successfully!")

uploaded_file = st.file_uploader("📤 Upload LinkedIn Connections (CSV)", type=["csv"])
if uploaded_file:
    connections_df = extract_linkedin_connections(uploaded_file)
    st.success("✅ Connections uploaded successfully!")
    st.dataframe(connections_df.head())
    if st.button("Find Best Candidates and Suggest Introductions"):
        matches = match_candidates(connections_df, criteria)
        st.write(matches.to_html(escape=False), unsafe_allow_html=True)
