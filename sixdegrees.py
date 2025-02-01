import streamlit as st
import pandas as pd
import re
import random

# ---------------------------
# Helper Functions
# ---------------------------

def clean_csv_data(df):
    """
    Cleans the CSV data to ensure that it is in the correct format:
      - Standardize column names by stripping whitespace and lower-casing.
      - Remove duplicate rows.
      - Drop rows missing critical columns.
      - Clean 'Experience' column to be numeric when possible.
      - Trim extra whitespace in string columns.
    """
    # Standardize column names
    df.columns = [col.strip().lower() for col in df.columns]
    
    # Define expected columns and their cleaned names
    expected_columns = {
        "name": "Name",
        "title": "Title",
        "company": "Company",
        "experience": "Experience",
        "education": "Education",
        "industry": "Industry"
    }
    
    # Rename columns if found in the CSV
    df.rename(columns={col: expected_columns[col] for col in expected_columns if col in df.columns}, inplace=True)
    
    # Drop duplicates
    df.drop_duplicates(inplace=True)
    
    # Drop rows that do not have critical information (Name and Title)
    df.dropna(subset=["Name", "Title"], inplace=True)
    
    # Clean up string columns: trim whitespace
    for col in ["Name", "Title", "Company", "Education", "Industry"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    
    # Convert 'Experience' to numeric, if possible. If not, set to 0.
    if "Experience" in df.columns:
        def parse_experience(exp):
            try:
                # Remove non-numeric characters (e.g., "years", "yrs", etc.)
                num = re.findall(r"[\d\.]+", str(exp))
                return float(num[0]) if num else 0
            except Exception:
                return 0
        df["Experience"] = df["Experience"].apply(parse_experience)
    else:
        # Create a default 'Experience' column if missing
        df["Experience"] = 0
    
    return df

def extract_linkedin_connections(csv_file):
    """
    Reads the CSV file uploaded by the employee, cleans it, and returns a cleaned DataFrame.
    
    **Steps to retrieve your LinkedIn connections CSV:**
      1. Log in to your LinkedIn account.
      2. Click on 'My Network' then 'Connections'.
      3. Click on 'Manage synced and imported contacts'.
      4. Click 'Export Contacts' and select 'Connections'.
      5. Download the CSV file and then upload it here.
    
    The CSV file should ideally include columns like "Name", "Title", "Company", "Experience", "Education", and "Industry".
    """
    df = pd.read_csv(csv_file)
    df = clean_csv_data(df)
    return df

def extract_job_criteria(job_url):
    """
    In a production system, this function would scrape the provided URL and use NLP
    to extract key job criteria such as job title, required experience, skills, education, etc.
    
    For demonstration, we simulate this extraction with dummy criteria.
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
    Simulates candidate matching by scoring each connection.  
    We assume the CSV has columns: Name, Title, Company, Experience, Education, Industry.
    
    Scoring criteria:
      - +50 points if the candidate's Title contains the job title.
      - +20 points if Experience is 3+ years.
      - +15 points if the candidate's Education matches.
      - +15 points if the Industry matches.
    
    The top 5 scoring candidates are returned.
    """
    if connections_df is None or connections_df.empty:
        return pd.DataFrame()
    
    def score_candidate(row):
        score = 0
        # Check job title match
        if criteria["job_title"].lower() in str(row.get("Title", "")).lower():
            score += 50
        # Check experience: assume Experience is numeric (years)
        try:
            exp = float(row.get("Experience", 0))
            if exp >= 3:
                score += 20
        except Exception:
            pass
        # Check education match
        if criteria["education"].lower() in str(row.get("Education", "")).lower():
            score += 15
        # Check industry match
        if criteria["industry"].lower() in str(row.get("Industry", "")).lower():
            score += 15
        return score
    
    connections_df["match_score"] = connections_df.apply(score_candidate, axis=1)
    filtered_df = connections_df[connections_df["match_score"] > 0]
    filtered_df = filtered_df.sort_values(by="match_score", ascending=False).head(5)
    return filtered_df

def get_reasoning(candidate, criteria):
    """
    Provides a short explanation for why a candidate is a match.
    """
    reasons = []
    if criteria["job_title"].lower() in str(candidate.get("Title", "")).lower():
        reasons.append("Job title matches.")
    try:
        exp = float(candidate.get("Experience", 0))
        if exp >= 3:
            reasons.append("Has sufficient experience.")
    except Exception:
        pass
    if criteria["education"].lower() in str(candidate.get("Education", "")).lower():
        reasons.append("Educational background fits.")
    if criteria["industry"].lower() in str(candidate.get("Industry", "")).lower():
        reasons.append("Industry experience is relevant.")
    return " ".join(reasons) if reasons else "No specific matching criteria found."

# ---------------------------
# Streamlit Application
# ---------------------------

def main():
    st.title("Employee & Recruiter Networking App")

    # ---------------------------
    # Sidebar: User Panel & Navigation
    # ---------------------------
    st.sidebar.title("User Panel")
    role = st.sidebar.selectbox("Select Role", ["Employee", "Recruiter"])
    username = st.sidebar.text_input("Enter your username", value="user1")
    st.sidebar.write(f"Logged in as: **{username}** ({role})")

    # Store previous searches in session_state if not already stored.
    if "previous_searches" not in st.session_state:
        st.session_state.previous_searches = []

    st.sidebar.subheader("Previous Searches")
    if st.session_state.previous_searches:
        for search in st.session_state.previous_searches:
            st.sidebar.write(search)
    else:
        st.sidebar.write("No previous searches.")

    # ---------------------------
    # Employee View
    # ---------------------------
    if role == "Employee":
        st.header("Employee Dashboard: Upload Your LinkedIn Connections")
        st.write("""
        **How to Retrieve Your LinkedIn Connections CSV:**
        1. Log in to LinkedIn.
        2. Navigate to **My Network** and click on **Connections**.
        3. Click **Manage synced and imported contacts**.
        4. Click **Export Contacts** and select **Connections**.
        5. Download the CSV file and upload it below.
        """)
        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
        if uploaded_file is not None:
            try:
                connections_df = extract_linkedin_connections(uploaded_file)
                st.success("Connections uploaded and cleaned successfully!")
                st.dataframe(connections_df.head())
                # Store connections in session state for later use by recruiters
                st.session_state.connections_df = connections_df
            except Exception as e:
                st.error(f"Error processing file: {e}")

    # ---------------------------
    # Recruiter View
    # ---------------------------
    else:  # role == "Recruiter"
        st.header("Recruiter Dashboard: Map Job Criteria Against Employee Connections")
        job_url = st.text_input("Paste the URL of the Open Job Posting")
        if st.button("Extract Job Criteria"):
            if job_url:
                criteria = extract_job_criteria(job_url)
                st.session_state.current_criteria = criteria
                st.success("Job criteria extracted!")
                st.write("Extracted Criteria:")
                st.json(criteria)
                # Save job URL to previous searches.
                st.session_state.previous_searches.append(job_url)
            else:
                st.error("Please enter a job URL.")

        # Allow recruiter to review and edit the criteria.
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

            if st.button("Find Matching Candidates"):
                # Retrieve employee connections stored earlier by an employee.
                if "connections_df" in st.session_state:
                    connections_df = st.session_state.connections_df
                    matching_candidates = match_candidates(connections_df, criteria)
                    if not matching_candidates.empty:
                        st.subheader("Top 5 Matching Candidates")
                        for idx, row in matching_candidates.iterrows():
                            st.markdown(f"### {row['Name']}")
                            st.write(f"**Title:** {row['Title']}")
                            st.write(f"**Company:** {row['Company']}")
                            st.write(f"**Match Score:** {row['match_score']}")
                            reasoning = get_reasoning(row, criteria)
                            st.write(f"**Reasoning:** {reasoning}")
                            # Allow recruiter to rate the suggestion.
                            rating = st.slider(f"Rate candidate {row['Name']}", 1, 5, 3, key=f"rating_{idx}")
                            st.write(f"Your rating: {rating}")
                            # In a production system, the ratings would be stored and used for improving the matching algorithm.
                    else:
                        st.error("No matching candidates found. Please check the uploaded connections or adjust the criteria.")
                else:
                    st.error("No employee connections found. Please ask an employee to upload their connections.")

if __name__ == "__main__":
    main()
