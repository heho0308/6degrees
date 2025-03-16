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
import matplotlib.pyplot as plt

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

def extract_job_details(text):
    """Extract job title and company from job description using NLP."""
    job_title = ""
    company_name = ""
    entities = nlp(text)
    for entity in entities:
        if "JOB" in entity['entity']:  # Placeholder condition, update based on model output
            job_title = entity['word']
        if "ORG" in entity['entity']:  # Placeholder condition, update based on model output
            company_name = entity['word']
    return job_title, company_name

def extract_job_criteria(url):
    """Extract job criteria including job title and company name from job posting."""
    job_desc = extract_job_description(url)
    if not job_desc:
        return None
    job_title, company_name = extract_job_details(job_desc)
    return {
        "job_description": job_desc,
        "job_title": job_title,
        "company_hiring": company_name
    }

def clean_csv_data(df):
    """Cleans LinkedIn connections CSV by ensuring required columns exist."""
    required_columns = ["First Name", "Last Name", "Company", "Position", "URL"]
    for col in required_columns:
        if col not in df.columns:
            df[col] = ""
    df = df[required_columns]
    return df.drop_duplicates()

def generate_blurb(candidate, job_criteria):
    """Generate a warm introduction blurb for the recruiter."""
    return f"Hi [Colleague's Name], I’m looking to hire for a {job_criteria['job_title']} role at {job_criteria['company_hiring']}. {candidate['First Name']} {candidate['Last Name']} seems like a great fit based on their experience at {candidate['Company']}. Would you be open to making an introduction?"

def match_candidates(connections_df, job_criteria):
    """Find and rank top 5 candidates based on job criteria."""
    top_candidates = connections_df.head(5)  # Placeholder logic
    top_candidates["Blurb"] = top_candidates.apply(lambda row: generate_blurb(row, job_criteria), axis=1)
    return top_candidates

def main():
    st.set_page_config(page_title="6 Degrees", layout="wide")
    st.title("6 Degrees")
    st.subheader("Leverage Human Capital in New Ways")
    
    with st.sidebar:
        st.header("📌 Steps to Download LinkedIn Connections")
        st.write("1. Go to LinkedIn Settings.")
        st.write("2. Click on 'Get a copy of your data'.")
        st.write("3. Select 'Connections' and request archive.")
        st.write("4. Download the CSV file and upload it below.")
    
    st.header("🔍 Step 1: Upload Job Description URL")
    job_url = st.text_input("🔗 Paste Job Posting URL")
    
    if job_url:
        with st.spinner("Analyzing job posting..."):
            job_criteria = extract_job_criteria(job_url)
            if job_criteria:
                st.session_state.job_criteria = job_criteria
                st.success("✅ Job criteria extracted successfully!")
    
    st.header("✏️ Step 2: Edit & Save Job Criteria")
    job_criteria = st.session_state.get("job_criteria", {
        "job_title": "",
        "company_hiring": ""
    })
    job_criteria["job_title"] = st.text_input("Job Title", job_criteria["job_title"])
    job_criteria["company_hiring"] = st.text_input("Company That Is Hiring", job_criteria["company_hiring"])
    if st.button("Save Criteria"):
        st.session_state.job_criteria = job_criteria
        st.success("✅ Job criteria saved successfully!")
    
    st.header("📤 Step 3: Upload LinkedIn Connections CSV")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file:
        try:
            connections_df = pd.read_csv(uploaded_file, encoding='utf-8', on_bad_lines='skip')
            connections_df = clean_csv_data(connections_df)
            st.session_state.connections_df = connections_df
            st.success("✅ Connections uploaded and formatted successfully!")
            st.write(connections_df.head(5))
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

