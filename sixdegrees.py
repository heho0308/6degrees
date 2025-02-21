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

# Modern sidebar styling
st.markdown("""
    <style>
    /* Clean, modern sidebar */
    section[data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 1px solid #EAECF0;
        padding: 0;
    }
    section[data-testid="stSidebar"] > div {
        padding: 1.5rem 1rem;
    }
    
    /* Header area styling */
    .sidebar-header {
        padding-bottom: 1.5rem;
        margin-bottom: 1.5rem;
        border-bottom: 1px solid #EAECF0;
    }
    
    /* Profile section */
    .profile-section {
        display: flex;
        align-items: center;
        padding: 0.75rem 0;
        margin: 1rem 0;
        gap: 0.75rem;
    }
    .profile-info {
        font-size: 0.875rem;
        color: #344054;
    }
    
    /* Navigation styling */
    .nav-section {
        padding: 0.5rem 0;
    }
    .nav-header {
        font-size: 0.75rem;
        text-transform: uppercase;
        color: #667085;
        letter-spacing: 0.05em;
        margin-bottom: 0.75rem;
    }
    .nav-item {
        display: flex;
        align-items: center;
        padding: 0.5rem 0.75rem;
        color: #344054;
        text-decoration: none;
        border-radius: 6px;
        margin: 0.25rem 0;
        transition: all 0.2s;
    }
    .nav-item:hover {
        background-color: #F9FAFB;
    }
    .nav-item.active {
        background-color: #F5F5F5;
        font-weight: 500;
    }
    
    /* Button improvements */
    .stButton button {
        width: 100%;
        border: none !important;
        padding: 0.5rem 0.75rem;
        background: transparent !important;
        color: #344054 !important;
        font-weight: normal !important;
        text-align: left;
        display: flex;
        align-items: center;
        justify-content: flex-start;
    }
    .stButton button:hover {
        background-color: #F9FAFB !important;
    }
    .stSelectbox > div[data-baseweb="select"] {
        background-color: transparent;
        border: none;
        padding: 0;
    }
    
    /* Custom separator */
    .separator {
        height: 1px;
        background-color: #EAECF0;
        margin: 1rem 0;
    }
    
    /* Hide default decorations */
    div[data-testid="stDecoration"] {
        display: none;
    }
    
    /* Streamlit elements cleanup */
    .stTextInput input {
        background: transparent;
        border: none;
        padding: 0;
        font-size: 0.875rem;
    }
    
    /* Main content area improvements */
    .main .block-container {
        padding-top: 2rem;
    }
    div[data-testid="stToolbar"] {
        display: none;
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
        "industry": industry,
        "years_experience": 0
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
                st.session_state.location_
