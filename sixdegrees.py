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
    """Cleans LinkedIn connections CSV."""
    expected_columns = ["First Name", "Last Name", "URL", "Email Address", "Company", "Position", "Connected On"]
    df.columns = expected_columns
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    df["Connected On"] = pd.to_datetime(df["Connected On"], errors="coerce")
    df.drop_duplicates(inplace=True)
    return df

def extract_linkedin_connections(csv_file):
    """Reads and cleans LinkedIn connections CSV file uploaded by an employee."""
    try:
        df = pd.read_csv(csv_file, skiprows=3, encoding="utf-8")
        df = clean_csv_data(df)
        return df
    except Exception as e:
        st.error(f"Error processing CSV file: {e}")
        return None

def extract_job_description(url):
    """Fetches the job description from a given job posting URL."""
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
    """Extracts capitalized words and phrases as job titles."""
    words = text.split()
    noun_phrases = []
    temp_phrase = []

    for word in words:
        if word.istitle():
            temp_phrase.append(word)
        else:
            if temp_phrase:
                noun_phrases.append(" ".join(temp_phrase))
                temp_phrase = []
    if temp_phrase:
        noun_phrases.append(" ".join(temp_phrase))

    return noun_phrases

def extract_job_criteria(url):
    """Extracts job title, level, experience, and education using regex and simple text processing."""
    job_desc = extract_job_description(url)
    if not job_desc:
        return None

    noun_phrases = extract_noun_phrases(job_desc)
    job_title = noun_phrases[0] if noun_phrases else "Not Found"

    # Extract seniority level
    seniority_levels = ["Intern", "Junior", "Mid-Level", "Senior", "Lead", "Manager", "Director", "VP", "Executive"]
    seniority = next((level for level in seniority_levels if level.lower() in job_desc.lower()), "Not specified")

    # Extract years of experience
    experience_match = re.search(r"(\d+)\+?\s*(?:years|yrs|YRS|experience)", job_desc, re.IGNORECASE)
    required_experience = f"{experience_match.group(1)}+ years" if experience_match else "Not specified"

    # Extract education
    education_match = re.search(r"(Bachelor's|Master's|PhD) in ([A-Za-z ]+)", job_desc, re.IGNORECASE)
    education = f"{education_match.group(1)} in {education_match.group(2)}" if education_match else "Not specified"

    return {
        "job_title": job_title,
        "seniority": seniority,
        "required_experience": required_experience,
        "education": education
    }

def match_candidates(connections_df, criteria):
    """Matches candidates based on job criteria and excludes candidates working at the job's company."""
    if connections_df is None or connections_df.empty:
        return pd.DataFrame()

    def score_candidate(row):
        score = 0
        # Title match
        if criteria["job_title"].lower() in str(row.get("Position", "")).lower():
            score += 50

        # Seniority match
        if criteria["seniority"].lower() in str(row.get("Position", "")).lower():
            score += 20

        # Experience match
        if criteria["required_experience"].lower() in str(row.get("Position", "")).lower():
            score += 15

        # Education match
        if criteria["education"].lower() in str(row.get("Company", "")).lower():
            score += 10

        return score
    
    connections_df["match_score"] = connections_df.apply(score_candidate, axis=1)
    filtered_df = connections_df[connections_df["match_score"] > 0]
    return filtered_df.sort_values(by="match_score", ascending=False).head(5)

# ---------------------------
# Streamlit Web Application
# ---------------------------
def main():
    st.title("Employee & Recruiter Networking App")

    st.sidebar.title("User Panel")
    role = st.sidebar.selectbox("Select Role", ["Employee", "Recruiter"])
    username = st.sidebar.text_input("Enter your username", value="user1")
    st.sidebar.write(f"Logged in as: **{username}** ({role})")

    if "previous_searches" not in st.session_state:
        st.session_state.previous_searches = {}

    if role == "Employee":
        st.header("Upload Your LinkedIn Connections")
        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
        if uploaded_file:
            connections_df = extract_linkedin_connections(uploaded_file)
            if connections_df is not None:
                st.success("Connections uploaded successfully!")
                st.subheader("Example Connections:")
                st.dataframe(connections_df.head(5))
                st.session_state.connections_df = connections_df

    else:
        st.header("Find Candidates for a Job Posting")
        job_url = st.text_input("Paste Job Posting URL")
        if st.button("Extract Job Criteria"):
            if job_url:
                criteria = extract_job_criteria(job_url)
                st.session_state.previous_searches[job_url] = criteria
                st.session_state.current_criteria = criteria

        st.sidebar.subheader("Previous Searches")
        selected_search = st.sidebar.selectbox("Select a past search:", list(st.session_state.previous_searches.keys()), index=len(st.session_state.previous_searches)-1 if st.session_state.previous_searches else None)

        if selected_search:
            st.session_state.current_criteria = st.session_state.previous_searches[selected_search]

        if "current_criteria" in st.session_state:
            st.subheader("Review and Edit Job Criteria")
            criteria = st.session_state.current_criteria

            for key, value in criteria.items():
                criteria[key] = st.text_input(key.replace("_", " ").title(), value=value)

            st.session_state.current_criteria = criteria

        if st.button("Find Matching Candidates") and "connections_df" in st.session_state:
            matching_candidates = match_candidates(st.session_state.connections_df, st.session_state.current_criteria)
            st.dataframe(matching_candidates)

if __name__ == "__main__":
    main()
