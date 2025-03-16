import streamlit as st
import pandas as pd
import re
import requests
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from fuzzywuzzy import fuzz
from transformers import pipeline  # Hugging Face NLP model

# Download required NLTK resources
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("averaged_perceptron_tagger")

# Load Hugging Face Named Entity Recognition (NER) model
nlp = pipeline("ner", model="dslim/bert-base-NER")

def extract_job_description(url):
    """Extract job description from a given URL."""
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        job_desc_patterns = [
            {"class_": ["jobDescriptionContent", "job-description", "description"]},
            {"id": ["jobDescriptionText", "job-details", "job-description"]},
            {"data-testid": "jobDescriptionText"},
            {"itemprop": "description"}
        ]
        for pattern in job_desc_patterns:
            element = soup.find(**pattern)
            if element:
                return element.get_text(separator=" ", strip=True)
        paragraphs = [p.get_text(strip=True) for p in soup.find_all("p") if len(p.get_text(strip=True)) > 100]
        return " ".join(paragraphs) if paragraphs else None
    except Exception as e:
        st.error(f"Failed to fetch job description: {e}")
        return None

def extract_skills_from_text(text):
    """Extract key skills from text using NLP."""
    entities = nlp(text)
    skills = list(set(entity['word'] for entity in entities if entity['entity'].startswith('B-')))
    return skills

def extract_job_criteria(url):
    """Extract job criteria including skills from a job posting."""
    job_desc = extract_job_description(url)
    if not job_desc:
        return None
    skills = extract_skills_from_text(job_desc)
    return {
        "job_description": job_desc,
        "extracted_skills": skills,
        "job_title": "",
        "seniority": "",
        "industry": "",
        "location": "",
        "years_experience": ""
    }

def score_candidate(row, job_criteria):
    """Calculate a matching score for a candidate based on job criteria."""
    score = 0
    position = str(row.get("Position", "")).lower()
    title_similarity = fuzz.token_sort_ratio(job_criteria["job_title"].lower(), position)
    score += (title_similarity * 0.3)
    
    skills_matched = sum(1 for skill in job_criteria["extracted_skills"] if skill.lower() in position)
    score += (skills_matched / len(job_criteria["extracted_skills"])) * 40 if job_criteria["extracted_skills"] else 0
    
    seniority_match = 10 if job_criteria["seniority"].lower() in position else 0
    industry_match = 10 if job_criteria["industry"].lower() in str(row.get("Company", "")).lower() else 0
    network_strength = 10  # Placeholder for potential colleague connection strength
    
    score += seniority_match + industry_match + network_strength
    return round(score, 2)

def match_candidates(connections_df, job_criteria):
    """Find and rank top 5 candidates based on job criteria."""
    if connections_df is None or connections_df.empty:
        return pd.DataFrame()
    connections_df["match_score"] = connections_df.apply(lambda row: score_candidate(row, job_criteria), axis=1)
    filtered_df = connections_df[connections_df["match_score"] > 30]
    top_candidates = filtered_df.sort_values(by="match_score", ascending=False).head(5)
    top_candidates["LinkedIn Profile"] = top_candidates["URL"].apply(lambda x: f'<a href="{x}" target="_blank">View Profile</a>' if x else "")
    return top_candidates

def main():
    st.set_page_config(page_title="Smart Candidate Matcher with AI", layout="wide")
    st.title("🎯 Smart Candidate Matcher with AI")
    
    with st.sidebar:
        st.header("📌 Steps to Download LinkedIn Connections")
        st.write("1. Go to LinkedIn Settings.")
        st.write("2. Click on 'Get a copy of your data'.")
        st.write("3. Select 'Connections' and request archive.")
        st.write("4. Download the CSV file and upload it below.")
    
    st.header("🔍 Step 1: Upload Job Description URL")
    job_url = st.text_input("🔗 Paste Job Posting URL")
    
    if st.button("Extract Job Criteria"):
        if job_url:
            with st.spinner("Analyzing job posting..."):
                job_criteria = extract_job_criteria(job_url)
                if job_criteria:
                    st.session_state.job_criteria = job_criteria
                    st.success("✅ Job criteria extracted successfully!")
    
    if "job_criteria" in st.session_state:
        job_criteria = st.session_state.job_criteria
        st.header("✏️ Step 2: Edit & Save Job Criteria")
        job_criteria["job_title"] = st.text_input("Job Title", job_criteria["job_title"])
        job_criteria["seniority"] = st.text_input("Seniority", job_criteria["seniority"])
        job_criteria["industry"] = st.text_input("Industry", job_criteria["industry"])
        job_criteria["location"] = st.text_input("Location", job_criteria["location"])
        job_criteria["years_experience"] = st.text_input("Years of Experience", job_criteria["years_experience"])
        st.session_state.job_criteria = job_criteria
    
    st.header("📤 Step 3: Upload LinkedIn Connections CSV")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file:
        try:
            connections_df = pd.read_csv(uploaded_file, encoding='utf-8', on_bad_lines='skip')
            st.session_state.connections_df = connections_df
            st.success("✅ Connections uploaded successfully!")
        except pd.errors.ParserError:
            st.error("❌ Error reading the CSV file. Please check its formatting and try again.")
    
    if "connections_df" in st.session_state and "job_criteria" in st.session_state:
        st.header("🎯 Step 4: Match Candidates")
        if st.button("Find Matches"):
            candidates = match_candidates(st.session_state.connections_df, st.session_state.job_criteria)
            if not candidates.empty:
                st.success(f"Found {len(candidates)} potential candidates!")
                st.write(candidates.to_html(escape=False), unsafe_allow_html=True)
            else:
                st.warning("No matching candidates found. Try adjusting the criteria.")

if __name__ == "__main__":
    main()
