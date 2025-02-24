import streamlit as st
import pandas as pd
import re
import requests
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from fuzzywuzzy import fuzz

# Download required NLTK resources
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("averaged_perceptron_tagger")

# ---------------------------
# Helper Functions
# ---------------------------

def clean_csv_data(df):
    """Cleans LinkedIn connections CSV with improved handling."""
    expected_columns = ["First Name", "Last Name", "URL", "Email Address", "Company", "Position", "Connected On"]
    
    for col in expected_columns:
        if col not in df.columns:
            df[col] = ""
            
    df = df[expected_columns]
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    df["Connected On"] = pd.to_datetime(df["Connected On"], errors="coerce")
    df["Company"] = df["Company"].astype(str).fillna("")
    df["Position"] = df["Position"].astype(str).fillna("")
    df.drop_duplicates(inplace=True)
    return df

def extract_linkedin_connections(csv_file):
    """Reads LinkedIn connections CSV with flexible handling for different formats."""
    try:
        # Standard format with 3 header rows
        df = pd.read_csv(csv_file, skiprows=3, encoding="utf-8")
        if all(col in df.columns for col in ["First Name", "Last Name", "Company", "Position"]):
            return clean_csv_data(df)
    except:
        pass

    try:
        # No header rows
        df = pd.read_csv(csv_file, encoding="utf-8")
        if all(col in df.columns for col in ["First Name", "Last Name", "Company", "Position"]):
            return clean_csv_data(df)
    except:
        pass
    
    try:
        df = pd.read_csv(csv_file, encoding="utf-8", on_bad_lines='skip')
        name_cols = [col for col in df.columns if 'name' in col.lower()]
        company_cols = [col for col in df.columns if 'company' in col.lower()]
        position_cols = [col for col in df.columns if 'position' in col.lower()]
        url_cols = [col for col in df.columns if 'url' in col.lower()]
        
        if name_cols and company_cols and position_cols:
            column_mapping = {
                name_cols[0]: "First Name",
                name_cols[-1]: "Last Name" if len(name_cols) > 1 else "Last Name",
                company_cols[0]: "Company",
                position_cols[0]: "Position",
            }
            if url_cols:
                column_mapping[url_cols[0]] = "URL"
                
            df = df.rename(columns=column_mapping)
            return clean_csv_data(df)
            
    except Exception as e:
        st.error(f"Error processing CSV file: {e}")
        return None
        
    st.error("Unable to process CSV file. Please ensure it contains required columns: First Name, Last Name, Company, Position")
    return None

def extract_job_description(url):
    """Enhanced job description extraction."""
    headers = {
        "User-Agent": "Mozilla/5.0"
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        
        job_desc_section = soup.find(class_="jobDescriptionContent") or soup.find("p")
        return job_desc_section.get_text(strip=True) if job_desc_section else None
    except Exception as e:
        st.error(f"Error fetching job description: {e}")
        return None

def extract_job_criteria(url):
    """Extracts job criteria."""
    job_desc = extract_job_description(url)
    if not job_desc:
        return None

    job_title = re.search(r"(?:seeking|hiring|looking for)\s+([A-Z][A-Za-z\s]+)", job_desc)
    job_title = job_title.group(1).strip() if job_title else "Not Found"

    seniority_levels = {
        "Entry Level": ["entry", "junior", "associate", "trainee"],
        "Mid-Level": ["mid", "intermediate", "experienced"],
        "Senior": ["senior", "lead", "principal"],
        "Management": ["manager", "director", "vp", "chief", "executive"]
    }
    
    seniority = "Not specified"
    for level, keywords in seniority_levels.items():
        if any(keyword in job_desc.lower() for keyword in keywords):
            seniority = level
            break
    
    company_match = re.search(r"at\s+([A-Z][A-Za-z0-9&\s]+)", job_desc)
    company_of_job_posting = company_match.group(1).strip() if company_match else "Not specified"

    industry_match = re.search(r"industry[:\s]+([A-Za-z\s]+)", job_desc, re.IGNORECASE)
    industry = industry_match.group(1).strip() if industry_match else "Not specified"

    return {
        "job_title": job_title,
        "seniority": seniority,
        "company_of_job_posting": company_of_job_posting,
        "ideally_experience_in_following_industry": industry
    }

def match_candidates(connections_df, criteria):
    """Matches candidates based on job criteria."""
    if connections_df is None or connections_df.empty:
        return pd.DataFrame()

    def score_candidate(row):
        if str(row.get("Company", "")).lower() == criteria["company_of_job_posting"].lower():
            return 0
            
        score = 0
        position = str(row.get("Position", "")).lower()
        title_similarity = fuzz.token_sort_ratio(criteria["job_title"].lower(), position)
        score += (title_similarity * 0.5)
        
        if criteria["seniority"].lower() in position:
            score += 20
            
        return round(score, 2)

    connections_df["match_score"] = connections_df.apply(score_candidate, axis=1)
    filtered_df = connections_df[connections_df["match_score"] > 20]
    
    result_df = filtered_df.sort_values(by="match_score", ascending=False).head(5)
    result_df = result_df[["First Name", "Last Name", "Position", "Company", "match_score", "URL"]]
    
    result_df = result_df.copy()
    result_df['URL'] = result_df['URL'].apply(lambda x: f'<a href="{x}" target="_blank">View Profile</a>' if x else "")
    
    return result_df

def main():
    st.set_page_config(page_title="Smart Candidate Matcher", layout="wide")
    
    st.title("🎯 Smart Candidate Matcher")
    
    st.sidebar.title("User Panel")
    role = st.sidebar.selectbox("Select Role", ["Employee", "Recruiter"])
    username = st.sidebar.text_input("Enter your username", value="user1")
    st.sidebar.write(f"Logged in as: **{username}** ({role})")

    if role == "Employee":
        st.header("📤 Upload Your LinkedIn Connections")
        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
        if uploaded_file:
            connections_df = extract_linkedin_connections(uploaded_file)
            if connections_df is not None:
                st.success("✅ Connections uploaded successfully!")
                st.session_state.connections_df = connections_df
                st.dataframe(connections_df.head(5))

    else:
        st.header("🔍 Find Matching Candidates")
        job_url = st.text_input("🔗 Paste Job Posting URL")
        
        if st.button("Extract Job Criteria"):
            if job_url:
                criteria = extract_job_criteria(job_url)
                if criteria:
                    st.session_state.current_criteria = criteria
                    st.success("✅ Job criteria extracted successfully!")
                    st.write(criteria)
                else:
                    st.error("Failed to extract job criteria.")
        
        if st.button("🎯 Find Matching Candidates") and "connections_df" in st.session_state:
            matching_candidates = match_candidates(st.session_state.connections_df, st.session_state.current_criteria)
            if not matching_candidates.empty:
                st.write(matching_candidates.to_html(escape=False), unsafe_allow_html=True)
            else:
                st.warning("No matching candidates found.")

if __name__ == "__main__":
    main()


