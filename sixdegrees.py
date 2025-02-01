import streamlit as st
import pandas as pd
import re
import random
import requests
from bs4 import BeautifulSoup
from textblob import TextBlob
import nltk
from nltk.corpus import stopwords

# Ensure necessary NLP resources are installed
nltk.download("stopwords")
nltk.download("punkt")

# ---------------------------
# Helper Functions
# ---------------------------

def extract_job_description(url):
    """
    Fetches the job description from a given job posting URL.
    """
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
    except requests.RequestException as e:
        st.error(f"Failed to fetch job URL: {e}")
        return None

    soup = BeautifulSoup(response.text, "html.parser")
    job_desc_tags = ["jobDescriptionContent", "job_description", "description", "job-desc"]
    
    for tag in job_desc_tags:
        job_section = soup.find(class_=tag) or soup.find(id=tag)
        if job_section:
            return job_section.get_text(separator=" ", strip=True)

    paragraphs = [p.get_text() for p in soup.find_all("p") if len(p.get_text()) > 100]
    return " ".join(paragraphs) if paragraphs else None

def extract_job_criteria(url):
    """
    Extracts job title, experience, skills, and education using TextBlob & Regex.
    """
    job_desc = extract_job_description(url)
    if not job_desc:
        return None

    # Use TextBlob for NLP processing
    blob = TextBlob(job_desc)

    # Extract job title (First noun phrase)
    job_title = blob.noun_phrases[0] if blob.noun_phrases else "Not Found"

    # Extract required years of experience (Regex)
    experience_match = re.search(r"(\d+)\+?\s*(?:years|yrs|YRS)", job_desc, re.IGNORECASE)
    required_experience = f"{experience_match.group(1)}+ years" if experience_match else "Not specified"

    # Extract key skills (Matching against common skills list)
    common_skills = ["Python", "SQL", "Java", "JavaScript", "Machine Learning", "Data Science", "Leadership"]
    skills_found = set(word for word in blob.words if word in common_skills)

    # Extract education requirements (Regex)
    education_match = re.search(r"(Bachelor's|Master's|PhD) in ([A-Za-z ]+)", job_desc, re.IGNORECASE)
    education = f"{education_match.group(1)} in {education_match.group(2)}" if education_match else "Not specified"

    # Extract industry
    industry_keywords = ["Technology", "Finance", "Healthcare", "Education", "Retail", "Marketing"]
    industry = next((word for word in industry_keywords if word.lower() in job_desc.lower()), "Not specified")

    # Extract seniority level
    seniority_levels = ["Entry-level", "Mid-level", "Senior", "Manager", "Director", "Executive"]
    seniority = next((level for level in seniority_levels if level.lower() in job_desc.lower()), "Not specified")

    return {
        "job_title": job_title,
        "required_experience": required_experience,
        "skills": list(skills_found),
        "education": education,
        "seniority": seniority,
        "industry": industry
    }

