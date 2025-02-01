import streamlit as st
import pandas as pd
import re
import random
import requests
from bs4 import BeautifulSoup
import spacy
import nltk
from nltk.corpus import stopwords

# Download necessary NLP resources
nltk.download("stopwords")
nlp = spacy.load("en_core_web_sm")

# ---------------------------
# Helper Functions
# ---------------------------

def clean_csv_data(df):
    """
    Cleans LinkedIn connections CSV:
      - Skips introductory rows
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

def extract_job_criteria(url):
    """
    Extracts structured job criteria (title, experience, skills, education) from a job posting URL.
    """
    job_desc = extract_job_description(url)
    if not job_desc:
        return None

    doc = nlp(job_desc)

    job_title = None
    for token in doc:
        if token.pos_ == "NOUN" and token.text.istitle():
            job_title = token.text
            break

    experience_match = re.search(r"(\d+)\+?\s*(?:years|yrs|YRS)", job_desc, re.IGNORECASE)
    required_experience = f"{experience_match.group(1)}+ years" if experience_match else "Not specified"

    common_skills = ["Python", "SQL", "Java", "JavaScript", "Machine Learning", "Data Science", "Leadership"]
    skills_found = set(token.text for token in doc if token.text in common_skills)

    education_match = re.search(r"(Bachelor's|Master's|PhD) in ([A-Za-z ]+)", job_desc, re.IGNORECASE)
    education = f"{education_match.group(1)} in {education_match.group(2)}" if education_match else "Not specified"

    industry_keywords = ["Technology", "Finance", "Healthcare", "Education", "Retail", "Marketing"]
    industry = next((word for word in industry_keywords if word.lower() in job_desc.lower()), "Not specified")

    seniority_levels = ["Entry-level", "Mid-level", "Senior", "Manager", "Director", "Executive"]
    seniority = next((level for level in seniority_levels if level.lower() in job_desc.lower()), "Not specified")

    return {
        "job_title": job_title if job_title else "Not found",
        "required_experience": required_experience,
        "skills": list(skills_found),
        "education": education,
        "seniority": seniority,
        "industry": industry
    }

def match_candidates(connections_df, criteria):
    """
    Matches candidates based on job criteria.
    """
    if connections_df is None or connections_df.empty:
        return pd.DataFrame()
    
    def score_candidate(row):
        score = 0
        if criteria["job_title"].lower() in str(row.get("Position", "")).lower():
            score += 50
        if criteria["education"].lower() in str(row.get("Company", "")).lower():
            score += 15
        return score
    
    connections_df["match_score"] = connections_df.apply(score_candidate, axis=1)
    filtered_df = connections_df[connections_df["match_score"] > 0]
    filtered_df = filtered_df.sort_values(by="match_score", ascending=False).head(5)
    return filtered_df

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
        st.session_state.previous_searches = []

    st.sidebar.subheader("Previous Searches")
    if st.session_state.previous_searches:
        for search in st.session_state.previous_searches:
            st.sidebar.write(search)

    if role == "Employee":
        st.header("Upload Your LinkedIn Connections")
        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
        if uploaded_file:
            connections_df = extract_linkedin_connections(uploaded_file)
            if connections_df is not None:
                st.success("Connections uploaded and cleaned successfully!")
                st.dataframe(connections_df.head())
                st.session_state.connections_df = connections_df

    else:
        st.header("Find Candidates for a Job Posting")
        job_url = st.text_input("Paste Job Posting URL")
        if st.button("Extract Job Criteria"):
            if job_url:
                criteria = extract_job_criteria(job_url)
                if criteria:
                    st.session_state.current_criteria = criteria
                    st.success("Job criteria extracted!")
                    st.json(criteria)
                    st.session_state.previous_searches.append(job_url)
            else:
                st.error("Please enter a job URL.")

        if "current_criteria" in st.session_state:
            criteria = st.session_state.current_criteria
            if st.button("Find Matching Candidates"):
                if "connections_df" in st.session_state:
                    connections_df = st.session_state.connections_df
                    matching_candidates = match_candidates(connections_df, criteria)
                    if not matching_candidates.empty:
                        st.subheader("Top 5 Matching Candidates")
                        for idx, row in matching_candidates.iterrows():
                            st.markdown(f"### {row['First Name']} {row['Last Name']}")
                            st.write(f"**Position:** {row['Position']}")
                            st.write(f"**Company:** {row['Company']}")
                            st.write(f"**Match Score:** {row['match_score']}")
                            rating = st.slider(f"Rate candidate {row['First Name']} {row['Last Name']}", 1, 5, 3, key=f"rating_{idx}")
                            st.write(f"Your rating: {rating}")
                    else:
                        st.error("No matching candidates found.")
                else:
                    st.error("No employee connections found.")

if __name__ == "__main__":
    main()
