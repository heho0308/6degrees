import streamlit as st
import pandas as pd
import re
import requests
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from fuzzywuzzy import fuzz

# Download required NLTK resources
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("averaged_perceptron_tagger")

# ---------------------------
# Enhanced Helper Functions
# ---------------------------

def clean_csv_data(df):
    """Cleans LinkedIn connections CSV with improved handling."""
    expected_columns = ["First Name", "Last Name", "URL", "Email Address", "Company", "Position", "Connected On"]
    
    # Ensure all expected columns exist
    for col in expected_columns:
        if col not in df.columns:
            df[col] = ""
            
    df = df[expected_columns]  # Reorder columns
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    df["Connected On"] = pd.to_datetime(df["Connected On"], errors="coerce")
    df["Company"] = df["Company"].astype(str).fillna("")
    df["Position"] = df["Position"].astype(str).fillna("")
    df.drop_duplicates(inplace=True)
    return df

def extract_linkedin_connections(csv_file):
    """Reads and cleans LinkedIn connections CSV with error handling."""
    try:
        df = pd.read_csv(csv_file, skiprows=3, encoding="utf-8")
        return clean_csv_data(df)
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(csv_file, skiprows=3, encoding="latin-1")
            return clean_csv_data(df)
        except Exception as e:
            st.error(f"Error processing CSV file: {e}")
            return None
    except Exception as e:
        st.error(f"Error processing CSV file: {e}")
        return None

def extract_skills_from_text(text):
    """Extracts potential skills from text using NLP."""
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    
    # Common technical skills and keywords
    common_skills = {
        'python', 'java', 'javascript', 'sql', 'aws', 'azure', 'agile', 'scrum',
        'management', 'leadership', 'analytics', 'machine learning', 'ai',
        'project management', 'data analysis', 'marketing', 'sales'
    }
    
    # Extract consecutive capitalized words (potential skill phrases)
    skill_phrases = re.findall(r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*', text)
    
    # Extract words that might be skills
    potential_skills = set()
    for token in tokens:
        if token not in stop_words and len(token) > 2:
            if token in common_skills or token.title() in skill_phrases:
                potential_skills.add(token)
    
    return list(potential_skills) + skill_phrases

def extract_job_description(url):
    """Enhanced job description extraction with better parsing."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Common job description container patterns
        job_desc_patterns = [
            {"class_": ["jobDescriptionContent", "job-description", "description"]},
            {"id": ["jobDescriptionText", "job-details", "job-description"]},
            {"data-testid": "jobDescriptionText"},
            {"itemprop": "description"}
        ]
        
        # Try multiple approaches to find job description
        for pattern in job_desc_patterns:
            element = soup.find(**pattern)
            if element:
                return element.get_text(separator=" ", strip=True)
        
        # Fallback: Look for largest text block
        paragraphs = [p.get_text(strip=True) for p in soup.find_all("p") if len(p.get_text(strip=True)) > 100]
        if paragraphs:
            return " ".join(paragraphs)
            
        return None
    except Exception as e:
        st.error(f"Failed to fetch job description: {e}")
        return None

def extract_job_criteria(url):
    """Enhanced job criteria extraction with skills and requirements."""
    job_desc = extract_job_description(url)
    if not job_desc:
        return None

    # Extract job title using common patterns
    title_patterns = [
        r"(?i)(?:seeking|hiring|looking for)[:\s]+([A-Z][A-Za-z\s]+(?:Developer|Engineer|Manager|Designer|Analyst|Director|Consultant|Specialist))",
        r"(?i)^([A-Z][A-Za-z\s]+(?:Developer|Engineer|Manager|Designer|Analyst|Director|Consultant|Specialist))"
    ]
    
    job_title = "Not Found"
    for pattern in title_patterns:
        match = re.search(pattern, job_desc)
        if match:
            job_title = match.group(1).strip()
            break

    # Extract seniority with expanded levels
    seniority_levels = {
        "Entry Level": ["entry", "junior", "associate", "trainee"],
        "Mid-Level": ["mid", "intermediate", "experienced"],
        "Senior": ["senior", "sr", "lead", "principal"],
        "Management": ["manager", "director", "head", "vp", "chief", "executive"]
    }
    
    seniority = "Not specified"
    for level, keywords in seniority_levels.items():
        if any(keyword in job_desc.lower() for keyword in keywords):
            seniority = level
            break

    # Extract company name
    company_patterns = [
        r"at\s+([A-Z][A-Za-z0-9&\s]+)(?:\s|$)",
        r"(?:Join|About)\s+([A-Z][A-Za-z0-9&\s]+)(?:\s|$)",
        r"([A-Z][A-Za-z0-9&\s]+)\s+is\s+(?:seeking|looking|hiring)"
    ]
    
    company_name = "Not specified"
    for pattern in company_patterns:
        match = re.search(pattern, job_desc)
        if match:
            company_name = match.group(1).strip()
            break

    # Extract required skills and experience
    skills = extract_skills_from_text(job_desc)
    
    # Extract years of experience
    experience_match = re.search(r'(\d+)[\+]?\s+years?(?:\s+of)?\s+experience', job_desc.lower())
    years_experience = int(experience_match.group(1)) if experience_match else 0

    return {
        "job_title": job_title,
        "seniority": seniority,
        "company_name": company_name,
        "required_skills": skills[:10],  # Top 10 most relevant skills
        "years_experience": years_experience
    }

def match_candidates(connections_df, criteria):
    """Enhanced candidate matching with multiple criteria and weighted scoring."""
    if connections_df is None or connections_df.empty:
        return pd.DataFrame()

    def score_candidate(row):
        if str(row.get("Company", "")).lower() == criteria["company_name"].lower():
            return 0  # Exclude current employees
            
        score = 0
        position = str(row.get("Position", "")).lower()
        
        # Title match (30%)
        title_similarity = fuzz.token_sort_ratio(criteria["job_title"].lower(), position)
        score += (title_similarity * 0.3)
        
        # Seniority match (20%)
        if criteria["seniority"].lower() in position:
            score += 20
            
        # Skills match (40%)
        candidate_text = f"{position} {str(row.get('Company', ''))}"
        candidate_skills = set(extract_skills_from_text(candidate_text))
        required_skills = set(skill.lower() for skill in criteria["required_skills"])
        
        if required_skills:
            skills_match_ratio = len(candidate_skills.intersection(required_skills)) / len(required_skills)
            score += (skills_match_ratio * 40)
            
        # Experience level (10%)
        experience_terms = re.findall(r'(\d+)\+?\s*years?', position)
        if experience_terms:
            candidate_experience = int(experience_terms[0])
            if candidate_experience >= criteria["years_experience"]:
                score += 10
        
        return round(score, 2)

    connections_df["match_score"] = connections_df.apply(score_candidate, axis=1)
    filtered_df = connections_df[connections_df["match_score"] > 20]  # Minimum score threshold
    
    # Sort by score and select columns for display
    result_df = filtered_df.sort_values(by="match_score", ascending=False).head(5)
    result_df = result_df[["First Name", "Last Name", "Position", "Company", "match_score", "Email Address"]]
    
    return result_df

# ---------------------------
# Enhanced Streamlit Application
# ---------------------------
def main():
    st.set_page_config(page_title="Smart Candidate Matcher", layout="wide")
    
    st.title("🎯 Smart Candidate Matcher")
    
    st.sidebar.title("User Panel")
    role = st.sidebar.selectbox("Select Role", ["Employee", "Recruiter"])
    username = st.sidebar.text_input("Enter your username", value="user1")
    st.sidebar.write(f"Logged in as: **{username}** ({role})")

    if "previous_searches" not in st.session_state:
        st.session_state.previous_searches = {}

    if role == "Employee":
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.header("📤 Upload Your LinkedIn Connections")
            uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
            
            if uploaded_file:
                connections_df = extract_linkedin_connections(uploaded_file)
                if connections_df is not None:
                    st.success("✅ Connections uploaded successfully!")
                    st.session_state.connections_df = connections_df
                    
                    with st.expander("Preview Connections"):
                        st.dataframe(connections_df.head(5))
        
        with col2:
            st.header("📋 Instructions")
            st.markdown("""
            1. Download your LinkedIn connections:
                - Go to LinkedIn Settings
                - Click on "Get a copy of your data"
                - Select "Connections"
                - Request archive
            2. Upload the CSV file here
            3. Preview your connections
            """)

    else:  # Recruiter role
        st.header("🔍 Find Matching Candidates")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            job_url = st.text_input("🔗 Paste Job Posting URL")
            
            if st.button("Extract Job Criteria"):
                if job_url:
                    with st.spinner("Analyzing job posting..."):
                        criteria = extract_job_criteria(job_url)
                        
                        if criteria:
                            st.session_state.current_criteria = criteria
                            st.success("✅ Job criteria extracted successfully!")
                        else:
                            st.error("❌ Failed to extract job criteria. Please check the URL.")

            if "current_criteria" in st.session_state:
                with st.expander("✏️ Review and Edit Job Criteria", expanded=True):
                    criteria = st.session_state.current_criteria
                    
                    col3, col4 = st.columns(2)
                    with col3:
                        criteria["job_title"] = st.text_input("Job Title", value=criteria["job_title"])
                        criteria["seniority"] = st.selectbox("Seniority", 
                            ["Entry Level", "Mid-Level", "Senior", "Management"],
                            index=["Entry Level", "Mid-Level", "Senior", "Management"].index(criteria["seniority"]))
                    
                    with col4:
                        criteria["company_name"] = st.text_input("Company", value=criteria["company_name"])
                        criteria["years_experience"] = st.number_input("Years of Experience Required", 
                            value=criteria["years_experience"], min_value=0, max_value=20)
                    
                    criteria["required_skills"] = st.multiselect("Required Skills", 
                        options=criteria["required_skills"] + ["Python", "Java", "JavaScript", "SQL", "AWS", "Azure", "Agile"],
                        default=criteria["required_skills"])
                    
                    st.session_state.current_criteria = criteria

                if st.button("🎯 Find Matching Candidates"):
                    if "connections_df" in st.session_state:
                        with st.spinner("Finding matches..."):
                            matching_candidates = match_candidates(st.session_state.connections_df, criteria)
                            
                            if not matching_candidates.empty:
                                st.success(f"Found {len(matching_candidates)} potential candidates!")
                                st.dataframe(matching_candidates)
                            else:
                                st.warning("No matching candidates found. Try adjusting the criteria.")
                    else:
                        st.error("Please upload connections data first!")
        
        with col2:
            st.header("📊 Matching Criteria")
            st.markdown("""
            Candidates are scored based on:
            - Job Title Match (30%)
            - Seniority Level (20%)
            - Required Skills (40%)
            - Years of Experience (10%)
            
            Current employees of the hiring company are automatically excluded.
            """)

if __name__ == "__main__":
    main()
