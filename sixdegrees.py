import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
from fuzzywuzzy import fuzz
from transformers import pipeline
import torch

# ---------------------------
# AI Model Loading with Caching
# ---------------------------

@st.cache_resource
def get_nlp_model():
    """Load the AI model for Named Entity Recognition (NER)."""
    if 'nlp_model' not in st.session_state:
        st.session_state.nlp_model = pipeline("ner", model="dslim/bert-base-NER")
    return st.session_state.nlp_model

@st.cache_resource
def get_text_generator():
    """Load the AI model for text generation (warm introductions)."""
    if 'text_generator' not in st.session_state:
        st.session_state.text_generator = pipeline("text-generation", model="t5-small")
    return st.session_state.text_generator

@st.cache_resource
def get_embedding_model():
    """Load the AI model for candidate-job matching."""
    if 'embedding_model' not in st.session_state:
        st.session_state.embedding_model = pipeline("feature-extraction", model="sentence-transformers/all-MiniLM-L6-v2")
    return st.session_state.embedding_model

# ---------------------------
# Job Description Extraction
# ---------------------------

def extract_job_description(url):
    """Fetch and extract the job description from the given URL."""
    if not url:
        return None
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        response = requests.get(url, headers=headers, timeout=5)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        paragraphs = [p.get_text(strip=True) for p in soup.find_all("p") if len(p.get_text(strip=True)) > 50]
        return " ".join(paragraphs) if paragraphs else None
    except requests.exceptions.Timeout:
        st.error("Request timed out. Try again later.")
        return None
    except Exception as e:
        st.error(f"Failed to fetch job description: {e}")
        return None

# ---------------------------
# Extract Job Criteria Using AI
# ---------------------------

def extract_job_criteria(url):
    """Extract job title, hiring company, and required skills from job description using AI."""
    job_desc = extract_job_description(url)
    if not job_desc:
        return None
    
    model = get_nlp_model()
    extracted_entities = model(job_desc)
    
    job_titles = [ent['word'] for ent in extracted_entities if "JOB" in ent['entity']]
    hiring_companies = [ent['word'] for ent in extracted_entities if "ORG" in ent['entity']]
    
    skills = extract_skills_from_text(job_desc)
    
    return {
        "job_title": st.text_input("Job Title", job_titles[0] if job_titles else "Unknown"),
        "company_name": st.text_input("Company That Is Hiring", hiring_companies[0] if hiring_companies else "Unknown"),
        "industry": st.text_input("Industry", "Technology" if "developer" in job_desc.lower() else "General"),
        "skills": st.text_area("Key Skills & Experience", skills)
    }

# ---------------------------
# Extract Skills Using AI
# ---------------------------

def extract_skills_from_text(job_desc):
    """Extract key skills from the job description using AI."""
    model = get_nlp_model()
    extracted_entities = model(job_desc)
    skills = [ent['word'] for ent in extracted_entities if ent['entity'] == "MISC"]
    return ", ".join(skills) if skills else "Not Found"

# ---------------------------
# Match Candidates Using AI
# ---------------------------

def match_candidates(connections_df, criteria):
    """Match candidates to job criteria using AI embeddings for sentence similarity."""
    if connections_df is None or connections_df.empty:
        return pd.DataFrame()

    model = get_embedding_model()

    job_embedding = model(criteria["job_title"] + " " + criteria["skills"])[0]

    def score_candidate(row):
        if str(row.get("Company", "")).lower() == criteria["company_name"].lower():
            return 0  # Exclude current employees

        position = str(row.get("Position", "")).lower()
        candidate_embedding = model(position)[0]

        similarity_score = torch.cosine_similarity(
            torch.tensor(job_embedding).mean(dim=0),
            torch.tensor(candidate_embedding).mean(dim=0),
            dim=0
        ).item()

        return round(similarity_score * 100, 2)

    connections_df["match_score"] = connections_df.apply(score_candidate, axis=1)
    filtered_df = connections_df[connections_df["match_score"] > 20]
    result_df = filtered_df.sort_values(by="match_score", ascending=False).head(5)
    result_df = result_df[["First Name", "Last Name", "Position", "Company", "match_score", "URL"]]
    
    return result_df

# ---------------------------
# Generate AI Warm Introduction
# ---------------------------

def generate_warm_intro(candidate, job_title, company_name):
    """Generate a short, AI-driven warm introduction for recruiters."""
    model = get_text_generator()
    prompt = f"Write a short, warm introduction for {candidate['First Name']} {candidate['Last Name']} who currently works at {candidate['Company']} in a {candidate['Position']} role. They could be a great fit for the {job_title} role at {company_name}. Keep it brief and professional."
    response = model(prompt, max_length=100)
    return response[0]["generated_text"]

# ---------------------------
# Streamlit UI
# ---------------------------

def main():
    st.set_page_config(page_title="6 Degrees", layout="wide")
    
    st.title("6 Degrees")
    st.subheader("Use your human capital in new ways")

    st.sidebar.title("User Panel")
    role = st.sidebar.selectbox("Select Role", ["Employee", "Recruiter"])
    username = st.sidebar.text_input("Enter your username", value="user1")
    st.sidebar.write(f"Logged in as: **{username}** ({role})")

    if role == "Employee":
        st.header("📤 Upload Your LinkedIn Connections")
        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
        
        if uploaded_file:
            connections_df = pd.read_csv(uploaded_file)
            st.session_state.connections_df = connections_df
            st.success("✅ Connections uploaded successfully!")

    else:  # Recruiter role
        st.header("🔍 Find Matching Candidates")
        job_url = st.text_input("🔗 Paste Job Posting URL")

        if st.button("Extract Job Criteria"):
            if job_url:
                with st.spinner("Analyzing job posting..."):
                    criteria = extract_job_criteria(job_url)
                    if criteria:
                        st.session_state.current_criteria = criteria
                        st.success("✅ Job criteria extracted successfully!")
                    else:
                        st.error("❌ Failed to extract job criteria. Please check the URL.")

        if "current_criteria" in st.session_state:
            st.subheader("✏️ Review & Edit Job Criteria")
            criteria = st.session_state.current_criteria
            
            col1, col2 = st.columns(2)
            with col1:
                criteria["job_title"] = st.text_input("Job Title", value=criteria["job_title"])
                criteria["company_name"] = st.text_input("Company That Is Hiring", value=criteria["company_name"])
            with col2:
                criteria["industry"] = st.text_input("Industry", value=criteria["industry"])
                criteria["skills"] = st.text_area("Key Skills & Experience", value=criteria["skills"])
            
            if st.button("🎯 Find Matching Candidates"):
                if "connections_df" in st.session_state:
                    with st.spinner("Finding matches..."):
                        matching_candidates = match_candidates(st.session_state.connections_df, criteria)
                        st.success(f"Found {len(matching_candidates)} potential candidates!")

                        for _, candidate in matching_candidates.iterrows():
                            intro = generate_warm_intro(candidate, criteria["job_title"], criteria["company_name"])
                            st.write(f"👤 {candidate['First Name']} {candidate['Last Name']} - {candidate['Position']} at {candidate['Company']}")
                            st.write(f"💬 Suggested Warm Introduction: {intro}")

if __name__ == "__main__":
    main()






