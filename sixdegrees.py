import streamlit as st
import pandas as pd
import re
import requests
from bs4 import BeautifulSoup
import nltk
from datetime import datetime
from fuzzywuzzy import fuzz

# Ensure necessary NLP resources are installed
nltk.download("stopwords")

# ---------------------------
# Helper Functions
# ---------------------------

def clean_csv_data(df):
    """
    Cleans LinkedIn connections CSV:
      - Standardizes column names
      - Trims whitespace from fields
      - Converts dates properly
      - Drops duplicates
    """
    expected_columns = ["First Name", "Last Name", "URL", "Email Address", "Company", "Position", "Connected On"]
    df.columns = expected_columns  # Rename columns correctly
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    df["Connected On"] = pd.to_datetime(df["Connected On"], errors="coerce")
    df.drop_duplicates(inplace=True)
    return df

def extract_linkedin_connections(csv_file):
    """
    Reads and cleans the LinkedIn connections CSV file uploaded by the employee.
    """
    try:
        df = pd.read_csv(csv_file, skiprows=3, encoding="utf-8")
        df = clean_csv_data(df)
        return df
    except Exception as e:
        st.error(f"Error processing CSV file: {e}")
        return None

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

def extract_noun_phrases(text):
    """
    Extracts capitalized words and phrases (assumed to be noun phrases like job titles).
    No external NLP models required.
    """
    words = text.split()
    noun_phrases = []
    temp_phrase = []

    for word in words:
        if word.istitle():  # Capitalized words likely indicate a title or important noun
            temp_phrase.append(word)
        else:
            if temp_phrase:
                noun_phrases.append(" ".join(temp_phrase))
                temp_phrase = []
    if temp_phrase:
        noun_phrases.append(" ".join(temp_phrase))

    return noun_phrases

def extract_job_criteria(url):
    """
    Extracts job title, experience, skills, and education using simple text processing.
    """
    job_desc = extract_job_description(url)
    if not job_desc:
        return None

    # Extract noun phrases (job titles, key terms)
    noun_phrases = extract_noun_phrases(job_desc)
    job_title = noun_phrases[0] if noun_phrases else "Not Found"

    # Extract experience requirements (Regex)
    experience_match = re.search(r"(\d+)\+?\s*(?:years|yrs|YRS)", job_desc, re.IGNORECASE)
    required_experience = f"{experience_match.group(1)}+ years" if experience_match else "Not specified"

    # Extract skills (Simple keyword matching)
    common_skills = ["Python", "SQL", "Java", "JavaScript", "Machine Learning", "Data Science", "Leadership"]
    skills_found = set(word for word in job_desc.split() if word in common_skills)

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

# ---------------------------
# Streamlit Web Application
# ---------------------------
def main():
    st.title("Employee & Recruiter Networking App")

    st.sidebar.title("User Panel")
    role = st.sidebar.selectbox("Select Role", ["Employee", "Recruiter"])
    username = st.sidebar.text_input("Enter your username", value="user1")
    st.sidebar.write(f"Logged in as: **{username}** ({role})")

    if role == "Employee":
        st.header("Upload Your LinkedIn Connections")
        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
        if uploaded_file:
            connections_df = extract_linkedin_connections(uploaded_file)
            if connections_df is not None:
                st.success("Connections uploaded and cleaned successfully!")
                st.subheader("Example Connections:")
                st.dataframe(connections_df.head(5))  # Show sample data
                st.session_state.connections_df = connections_df

    else:
        st.header("Find Candidates for a Job Posting")
        job_url = st.text_input("Paste Job Posting URL")
        if st.button("Extract Job Criteria"):
            if job_url:
                criteria = extract_job_criteria(job_url)
                st.session_state.current_criteria = criteria

        if "current_criteria" in st.session_state:
            st.subheader("Review and Edit Job Criteria")
            criteria = st.session_state.current_criteria

            criteria["job_title"] = st.text_input("Job Title", value=criteria.get("job_title", ""))
            criteria["required_experience"] = st.text_input("Required Experience", value=criteria.get("required_experience", ""))
            criteria["education"] = st.text_input("Education Requirement", value=criteria.get("education", ""))
            criteria["seniority"] = st.text_input("Seniority Level", value=criteria.get("seniority", ""))
            criteria["industry"] = st.text_input("Industry", value=criteria.get("industry", ""))
            skills = st.text_area("Skills (comma separated)", value=", ".join(criteria.get("skills", [])))
            criteria["skills"] = [skill.strip() for skill in skills.split(",") if skill.strip()]
            st.session_state.current_criteria = criteria

if __name__ == "__main__":
    main()
