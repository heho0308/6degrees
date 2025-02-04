import streamlit as st
import pandas as pd
import re
import requests
from bs4 import BeautifulSoup
import nltk
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
    df["Company"] = df["Company"].astype(str).fillna("")  # Convert NaN to empty strings
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

def match_candidates(connections_df, criteria):
    """Matches candidates based on job criteria."""
    if connections_df is None or connections_df.empty:
        return pd.DataFrame()

    def score_candidate(row):
        score = 0
        company = str(row.get("Company", "")).lower()  # Convert to string to prevent errors

        # Title match
        if criteria["job_title"].lower() in str(row.get("Position", "")).lower():
            score += 50

        # Seniority match
        if criteria["seniority"].lower() in str(row.get("Position", "")).lower():
            score += 20

        # Experience match
        if criteria["required_experience"].lower() in str(row.get("Position", "")).lower():
            score += 15

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
                criteria = extract_job_description(job_url)
                if criteria:
                    search_key = f"{criteria['job_title']} ({criteria['seniority']})"
                    st.session_state.previous_searches[search_key] = criteria
                    st.session_state.current_criteria = criteria

        st.sidebar.subheader("Previous Searches")
        selected_search = st.sidebar.selectbox("Select a past search:", list(st.session_state.previous_searches.keys()), index=len(st.session_state.previous_searches)-1 if st.session_state.previous_searches else None)

        if selected_search:
            st.session_state.current_criteria = st.session_state.previous_searches.get(selected_search, {})

        if "current_criteria" in st.session_state and st.session_state.current_criteria:
            st.subheader("Review and Edit Job Criteria")
            criteria = st.session_state.current_criteria

            for key in ["job_title", "required_experience", "seniority"]:
                criteria[key] = st.text_input(key.replace("_", " ").title(), value=criteria.get(key, ""))

            st.session_state.current_criteria = criteria

        if st.button("Find Matching Candidates") and "connections_df" in st.session_state:
            matching_candidates = match_candidates(st.session_state.connections_df, st.session_state.current_criteria)

            if not matching_candidates.empty:
                st.subheader("Top 5 Matching Candidates")

                for _, row in matching_candidates.iterrows():
                    # Assign Traffic Light Color
                    score = row["match_score"]
                    if score >= 70:
                        color = "🟢 Strong Fit"
                    elif score >= 50:
                        color = "🟠 Medium Fit"
                    else:
                        color = "🔴 Weak Fit"

                    # Display Candidate
                    st.markdown(f"### {row['First Name']} {row['Last Name']} - {color}")
                    st.write(f"**Position:** {row['Position']}")
                    st.write(f"**Company:** {row['Company']}")
                    st.write(f"**Match Score:** {score}")
                    st.markdown(f"🔗 [LinkedIn Profile]({row['URL']})", unsafe_allow_html=True)
                    st.markdown("---")

            else:
                st.error("No matching candidates found.")

if __name__ == "__main__":
    main()
