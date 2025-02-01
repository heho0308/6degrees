import streamlit as st
import pandas as pd
import re
import random

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
    # Ensure correct column names
    expected_columns = [
        "First Name", "Last Name", "URL", "Email Address",
        "Company", "Position", "Connected On"
    ]
    
    df.columns = expected_columns  # Rename columns correctly
    
    # Trim whitespace
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    
    # Convert "Connected On" to datetime
    df["Connected On"] = pd.to_datetime(df["Connected On"], errors="coerce")
    
    # Drop duplicate entries
    df.drop_duplicates(inplace=True)

    return df

def extract_linkedin_connections(csv_file):
    """
    Reads and cleans the LinkedIn connections CSV file uploaded by the employee.
    """
    try:
        # Skip the first 3 non-data rows (LinkedIn notes)
        df = pd.read_csv(csv_file, skiprows=3, encoding="utf-8")
        df = clean_csv_data(df)
        return df
    except Exception as e:
        st.error(f"Error processing CSV file: {e}")
        return None

def extract_job_criteria(job_url):
    """
    Simulates extracting job criteria from a job posting URL.
    """
    criteria = {
        "job_title": "Software Engineer",
        "required_experience": "3+ years",
        "skills": ["Python", "Machine Learning", "Data Structures"],
        "education": "Bachelor's in Computer Science",
        "seniority": "Mid-level",
        "industry": "Technology"
    }
    return criteria

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

    # Sidebar Navigation
    st.sidebar.title("User Panel")
    role = st.sidebar.selectbox("Select Role", ["Employee", "Recruiter"])
    username = st.sidebar.text_input("Enter your username", value="user1")
    st.sidebar.write(f"Logged in as: **{username}** ({role})")

    # Store previous searches in session state
    if "previous_searches" not in st.session_state:
        st.session_state.previous_searches = []

    st.sidebar.subheader("Previous Searches")
    if st.session_state.previous_searches:
        for search in st.session_state.previous_searches:
            st.sidebar.write(search)
    else:
        st.sidebar.write("No previous searches.")

    # Employee Panel - Upload LinkedIn Connections
    if role == "Employee":
        st.header("Upload Your LinkedIn Connections")
        st.write("""
        **How to Export LinkedIn Connections:**
        1. Log in to LinkedIn.
        2. Go to **My Network** → **Connections**.
        3. Click **Manage Synced & Imported Contacts**.
        4. Click **Export Contacts** → **Connections**.
        5. Download the CSV and upload it here.
        """)

        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
        if uploaded_file is not None:
            connections_df = extract_linkedin_connections(uploaded_file)
            if connections_df is not None:
                st.success("Connections uploaded and cleaned successfully!")
                st.dataframe(connections_df.head())
                st.session_state.connections_df = connections_df

    # Recruiter Panel - Find Candidates
    else:
        st.header("Find Candidates for a Job Posting")
        job_url = st.text_input("Paste Job Posting URL")
        if st.button("Extract Job Criteria"):
            if job_url:
                criteria = extract_job_criteria(job_url)
                st.session_state.current_criteria = criteria
                st.success("Job criteria extracted!")
                st.json(criteria)
                st.session_state.previous_searches.append(job_url)
            else:
                st.error("Please enter a job URL.")

        if "current_criteria" in st.session_state:
            criteria = st.session_state.current_criteria
            criteria["job_title"] = st.text_input("Job Title", value=criteria.get("job_title", ""))
            criteria["required_experience"] = st.text_input("Required Experience", value=criteria.get("required_experience", ""))
            criteria["education"] = st.text_input("Education Requirement", value=criteria.get("education", ""))
            criteria["seniority"] = st.text_input("Seniority Level", value=criteria.get("seniority", ""))
            criteria["industry"] = st.text_input("Industry", value=criteria.get("industry", ""))
            skills = st.text_area("Skills (comma separated)", value=", ".join(criteria.get("skills", [])))
            criteria["skills"] = [skill.strip() for skill in skills.split(",") if skill.strip()]
            st.session_state.current_criteria = criteria

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
                    st.error("No employee connections found. Please ask an employee to upload their connections.")

if __name__ == "__main__":
    main()
