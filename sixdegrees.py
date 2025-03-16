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
        "extracted_skills": skills
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
    filtered_df = connections_df[connections_df["match_score"] > 30]  # Threshold to filter candidates
    top_candidates = filtered_df.sort_values(by="match_score", ascending=False).head(5)
    top_candidates["LinkedIn Profile"] = top_candidates["URL"].apply(lambda x: f'<a href="{x}" target="_blank">View Profile</a>' if x else "")
    return top_candidates

def main():
    st.set_page_config(page_title="Smart Candidate Matcher with AI", layout="wide")
    st.title("🎯 Smart Candidate Matcher with AI")
    
    st.sidebar.title("User Panel")
    role = st.sidebar.selectbox("Select Role", ["Employee", "Recruiter"])
    username = st.sidebar.text_input("Enter your username", value="user1")
    st.sidebar.write(f"Logged in as: **{username}** ({role})")
    
    if role == "Recruiter":
        st.header("🔍 Extract Job Criteria & Find Candidates")
        job_url = st.text_input("🔗 Paste Job Posting URL")
        
        if st.button("Extract Job Criteria & Find Matches"):
            if job_url:
                with st.spinner("Analyzing job posting..."):
                    job_criteria = extract_job_criteria(job_url)
                    if job_criteria:
                        st.success("✅ Job criteria extracted successfully!")
                        st.write("**Extracted Skills:**", ", ".join(job_criteria["extracted_skills"]))
                        st.text_area("📜 Job Description:", job_criteria["job_description"], height=200)
                        
                        if "connections_df" in st.session_state:
                            candidates = match_candidates(st.session_state.connections_df, job_criteria)
                            if not candidates.empty:
                                st.success(f"Found {len(candidates)} potential candidates!")
                                st.write(candidates.to_html(escape=False), unsafe_allow_html=True)
                            else:
                                st.warning("No matching candidates found. Try adjusting the criteria.")
                    else:
                        st.error("❌ Failed to extract job criteria. Please check the URL.")

if __name__ == "__main__":
    main()




