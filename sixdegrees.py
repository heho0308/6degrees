import streamlit as st
import pandas as pd
import re
import requests
from bs4 import BeautifulSoup
from fuzzywuzzy import fuzz
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util

# Load AI Models
@st.cache_resource
def load_nlp_model():
    return pipeline("ner", model="dslim/bert-base-NER")  # Smaller, faster model

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")  # Fast & efficient

@st.cache_resource
def load_text_generator():
    return pipeline("text-generation", model="t5-small")  # Smaller, faster model

@st.cache_resource
def load_skill_extraction_model():
    return pipeline("token-classification", model="dslim/bert-base-NER")  # Faster skill extraction model

# Load models dynamically only when needed
def get_nlp_model():
    if 'nlp_model' not in st.session_state:
        st.session_state.nlp_model = load_nlp_model()
    return st.session_state.nlp_model

def get_embedding_model():
    if 'embedder' not in st.session_state:
        st.session_state.embedder = load_embedding_model()
    return st.session_state.embedder

def get_text_generator():
    if 'text_generator' not in st.session_state:
        st.session_state.text_generator = load_text_generator()
    return st.session_state.text_generator

def get_skill_extraction_model():
    if 'skill_extraction' not in st.session_state:
        st.session_state.skill_extraction = load_skill_extraction_model()
    return st.session_state.skill_extraction

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
def extract_skills_from_text(job_desc):
    model = get_skill_extraction_model()
    extracted_entities = model(job_desc)
    skills = [ent['word'] for ent in extracted_entities if ent['entity'] == "MISC"]  # Adjust entity type if needed
    return ", ".join(skills) if skills else "Not Found"

def extract_job_criteria(url):
    """Extract job title, hiring company, industry, and skills using AI."""
    job_desc = extract_job_description(url)
    if not job_desc:
        return None
    
    nlp_model = get_nlp_model()
    extracted_entities = nlp_model(job_desc)
    job_titles = [ent['word'] for ent in extracted_entities if "JOB" in ent['entity']]
    hiring_companies = [ent['word'] for ent in extracted_entities if "ORG" in ent['entity']]
    
    skills = extract_skills_from_text(job_desc)
    
    return {
        "job_title": st.text_input("Job Title", job_titles[0] if job_titles else "Unknown"),
        "company_name": st.text_input("Company That Is Hiring", hiring_companies[0] if hiring_companies else "Unknown"),
        "industry": st.text_input("Industry", "Technology" if "developer" in job_desc.lower() else "General"),
        "skills": st.text_area("Key Skills & Experience", skills if skills else "Not Found")
    }

# ---------------------------
# AI-Enhanced Candidate Matching
# ---------------------------
def match_candidates(connections_df, criteria):
    """Matches candidates based on AI-extracted job criteria and suggests warm introductions."""
    if connections_df is None or connections_df.empty:
        return pd.DataFrame()

    embedder = get_embedding_model()
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
        text_generator = get_text_generator()
        response = text_generator(prompt, max_length=100)
        return response[0]["generated_text"]

    result_df["warm_introduction"] = result_df.apply(generate_intro, axis=1)
    return result_df[["First Name", "Last Name", "Position", "Company", "match_score", "URL", "warm_introduction"]]






