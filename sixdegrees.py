# Import required libraries
import streamlit as st
import pandas as pd
import re
import requests
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from fuzzywuzzy import fuzz
import json
from functools import lru_cache
import time

# Must be the first Streamlit command
st.set_page_config(
    page_title="Smart Candidate Matcher",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for cleaner sidebar
st.markdown("""
    <style>
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    .sidebar .sidebar-content {
        padding: 2rem 1rem;
    }
    section[data-testid="stSidebar"] div[class*="stSidebarUserContent"] {
        padding: 1.5rem;
    }
    .user-info {
        padding: 1rem;
        margin-bottom: 2rem;
        border-bottom: 1px solid #eee;
    }
    .nav-item {
        padding: 0.5rem 0;
        margin: 0.5rem 0;
        cursor: pointer;
    }
    .stButton button {
        width: 100%;
        border-radius: 4px;
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        padding: 0.5rem 1rem;
        margin: 0.25rem 0;
    }
    .stButton button:hover {
        background-color: #e9ecef;
        border-color: #dee2e6;
    }
    div[data-testid="stDecoration"] {
        background-image: none;
    }
    </style>
""", unsafe_allow_html=True)

# Download required NLTK resources
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("averaged_perceptron_tagger")

# Initialize session state for location cache if not exists
if 'location_cache' not in st.session_state:
    st.session_state.location_cache = {}

def clean_csv_data(df):
    """Cleans LinkedIn connections CSV with improved handling."""
    expected_columns = ["First Name", "Last Name", "URL", "Email Address", "Company", "Position", "Connected On"]
    
    # Ensure all expected columns exist
    for col in expected_columns:
        if col not in df.columns:
            df[col] = ""
            
    df = df[expected_columns]  # Reorder columns
    df = df.map(lambda x: x.strip() if isinstance(x, str) else x)
    df["Connected On"] = pd.to_datetime(df["Connected On"], errors="coerce")
    df["Company"] = df["Company"].astype(str).fillna("")
    df["Position"] = df["Position"].astype(str).fillna("")
    df.drop_duplicates(inplace=True)
    return df

def extract_linkedin_connections(csv_file):
    """Reads LinkedIn connections CSV with flexible handling for different formats."""
    try:
        # First try: Standard format with 3 header rows
        df = pd.read_csv(csv_file, skiprows=3, encoding="utf-8")
        if all(col in df.columns for col in ["First Name", "Last Name", "Company", "Position"]):
            return clean_csv_data(df)
    except:
        pass

    try:
        # Second try: No header rows
        df = pd.read_csv(csv_file, encoding="utf-8")
        if all(col in df.columns for col in ["First Name", "Last Name", "Company", "Position"]):
            return clean_csv_data(df)
    except:
        pass
    
    try:
        # Third try: CSV needs cleanup
        df = pd.read_csv(csv_file, encoding="utf-8", on_bad_lines='skip')
        
        # Try to identify relevant columns
        name_cols = [col for col in df.columns if 'name' in col.lower()]
        company_cols = [col for col in df.columns if 'company' in col.lower()]
        position_cols = [col for col in df.columns if any(term in col.lower() for term in ['position', 'title', 'role'])]
        url_cols = [col for col in df.columns if any(term in col.lower() for term in ['url', 'link', 'profile'])]
        
        if name_cols and company_cols and position_cols:
            # Rename columns to expected format
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
    """Enhanced job criteria extraction with location and industry."""
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
    
    # Extract location patterns
    location_patterns = [
        r"(?i)location[:\s]+([A-Za-z\s,]+)(?:\.|$)",
        r"(?i)based in[:\s]+([A-Za-z\s,]+)(?:\.|$)",
        r"(?i)(?:remote|hybrid|onsite) in ([A-Za-z\s,]+)(?:\.|$)"
    ]
    
    # Extract industry patterns
    industry_patterns = [
        r"(?i)industry[:\s]+([A-Za-z\s&,]+)(?:\.|$)",
        r"(?i)sector[:\s]+([A-Za-z\s&,]+)(?:\.|$)"
    ]

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
            
    # Extract location
    location = "Not specified"
    for pattern in location_patterns:
        match = re.search(pattern, job_desc)
        if match:
            location = match.group(1).strip()
            break
            
    # Extract industry
    industry = "Not specified"
    for pattern in industry_patterns:
        match = re.search(pattern, job_desc)
        if match:
            industry = match.group(1).strip()
            break

    return {
        "job_title": job_title,
        "seniority": seniority,
        "company_name": company_name,
        "location": location,
        "industry": industry
    }

@lru_cache(maxsize=100)
def extract_location_from_profile(url):
    """Extract location from LinkedIn profile with caching and multiple fallback options."""
    # Check cache first
    if url in st.session_state.location_cache:
        return st.session_state.location_cache[url]
        
    if not url or 'linkedin.com/in/' not in url.lower():
        return None
        
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "DNT": "1",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1"
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Try multiple approaches to find location
        location = None
        
        # Method 1: Look for location in meta tags
        meta_location = soup.find('meta', {'name': 'profile:location'})
        if meta_location:
            location = meta_location.get('content')
            
        # Method 2: Look for location in specific HTML elements
        if not location:
            location_elements = [
                soup.find('div', {'class': 'text-body-medium break-words'}),
                soup.find(lambda tag: tag.name == 'span' and 'location' in tag.get('class', [])),
                soup.find('div', string=re.compile(r'.*(?:Greater|Area|Region).*'))
            ]
            
            for element in location_elements:
                if element and element.text.strip():
                    location = element.text.strip()
                    break
        
        # Method 3: Look for structured data
        if not location:
            schema_data = soup.find('script', {'type': 'application/ld+json'})
            if schema_data:
                try:
                    data = json.loads(schema_data.string)
                    if 'addressLocality' in str(data):
                        location = data.get('location', {}).get('addressLocality')
                except:
                    pass
        
        # Clean and validate location
        if location:
            # Remove common suffixes and clean up
            location = re.sub(r'\s*(?:Area|Region|Greater)\s*', ' ', location).strip()
            # Validate it looks like a real location (at least 2 characters, no strange symbols)
            if len(location) >= 2 and re.match(r'^[A-Za-z\s,.-]+$', location):
                # Cache the result
                st.session_state.location_cache[url] = location
                return location
                
        # Fallback: Try URL parameters
        location_match = re.search(r'location=([^&]+)', url)
        if location_match:
            location = location_match.group(1).replace('+', ' ')
            st.session_state.location_cache[url] = location
            return location
            
    except Exception as e:
        st.warning(f"Could not extract location for a candidate. Using fallback matching.", icon="⚠️")
        
    # Cache the negative result to avoid repeated attempts
    st.session_state.location_cache[url] = None
    return None

def match_candidates(connections_df, criteria):
    """Enhanced candidate matching with updated scoring weights and location extraction."""
    if connections_df is None or connections_df.empty:
        return pd.DataFrame()

    def score_candidate(row):
        if str(row.get("Company", "")).lower() == criteria["company_name"].lower():
            return 0  # Exclude current employees
            
        score = 0
        position = str(row.get("Position", "")).lower()
        
        # Title match (50%)
        title_similarity = fuzz.token_sort_ratio(criteria["job_title"].lower(), position)
        score += (title_similarity * 0.5)
        
        # Seniority match (20%)
        if criteria["seniority"].lower() in position:
            score += 20
            
        # Location match (15%)
        candidate_location = extract_location_from_profile(str(row.get("URL", "")))
        if candidate_location and criteria["location"].lower() != "not specified":
            location_similarity = fuzz.token_sort_ratio(criteria["location"].lower(), candidate_location.lower())
            score += (location_similarity * 0.15)
            
        # Industry match (15%)
        candidate_company = str(row.get("Company", "")).lower()
        if criteria["industry"].lower() != "not specified":
            industry_similarity = fuzz.token_sort_ratio(criteria["industry"].lower(), candidate_company)
            score += (industry_similarity * 0.15)
        
        return round(score, 2)
