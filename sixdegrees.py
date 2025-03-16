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

def generate_blurb(candidate, job_criteria):
    """Generate a warm introduction blurb for the recruiter."""
    return f"Hi [Colleague's Name], I’m looking to hire for a {job_criteria['job_title']} role at {job_criteria['company_hiring']}. {candidate['First Name']} {candidate['Last Name']} seems like a great fit based on their experience at {candidate['Company']}. Would you be open to making an introduction?"

def match_candidates(connections_df, job_criteria):
    """Find and rank top 5 candidates based on job criteria while excluding current employees."""
    filtered_candidates = connections_df[connections_df["Company"].str.lower() != job_criteria["company_hiring"].lower()]
    top_candidates = filtered_candidates.head(5)  # Placeholder logic
    top_candidates["Blurb"] = top_candidates.apply(lambda row: generate_blurb(row, job_criteria), axis=1)
    return top_candidates

def main():
    st.set_page_config(page_title="6 Degrees", layout="wide")
    st.title("6 Degrees")
    st.subheader("Leverage Human Capital in New Ways")
    
    st.header("🎯 Step 4: Match Candidates")
    if "connections_df" in st.session_state and "job_criteria" in st.session_state:
        if st.button("Find Matches"):
            candidates = match_candidates(st.session_state.connections_df, st.session_state.job_criteria)
            if not candidates.empty:
                st.success(f"Found {len(candidates)} potential candidates!")
                
                for index, row in candidates.iterrows():
                    st.markdown(
                        f"""
                        <div style="border: 2px solid #ddd; padding: 10px; margin: 10px 0; border-radius: 10px; background-color: #f9f9f9;">
                            <h4>{row['First Name']} {row['Last Name']}</h4>
                            <p><a href="{row['URL']}" target="_blank">View LinkedIn Profile</a></p>
                            <p>{row['Blurb']}</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
            else:
                st.warning("No matching candidates found. Try adjusting the criteria.")

if __name__ == "__main__":
    main()
