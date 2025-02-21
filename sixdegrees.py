import streamlit as st
import pandas as pd
import re
import requests
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from fuzzywuzzy import fuzz

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
    </style>
""", unsafe_allow_html=True)

# Download required NLTK resources
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("averaged_perceptron_tagger")

[Previous helper functions remain exactly the same...]

def main():
    # Modern sidebar
    with st.sidebar:
        # Header with logo and title
        st.markdown('<div class="sidebar-header">', unsafe_allow_html=True)
        col1, col2 = st.columns([1, 4])
        with col1:
            st.image("https://via.placeholder.com/40", width=40)
        with col2:
            st.markdown("<h3 style='margin: 0; padding: 8px 0; color: #101828;'>Candidate Matcher</h3>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # User profile section
        st.markdown('<div class="profile-section">', unsafe_allow_html=True)
        role = st.selectbox("", ["Employee", "Recruiter"], label_visibility="collapsed")
        username = st.text_input("", value="user1", label_visibility="collapsed", placeholder="Username")
        st.markdown(
            f"""<div class="profile-info">
                <div style="font-weight: 500">{username}</div>
                <div style="color: #667085">{role}</div>
            </div>""",
            unsafe_allow_html=True
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Navigation
        st.markdown('<div class="nav-section">', unsafe_allow_html=True)
        st.markdown('<div class="nav-header">Menu</div>', unsafe_allow_html=True)
        
        if role == "Employee":
            nav_items = [
                ("📤", "Upload Network", True),
                ("👥", "My Connections", False),
                ("📋", "Recent Activity", False),
                ("⚙️", "Settings", False)
            ]
        else:
            nav_items = [
                ("🔍", "Find Candidates", True),
                ("📊", "Match History", False),
                ("📋", "Saved Searches", False),
                ("⚙️", "Settings", False)
            ]
        
        for icon, label, active in nav_items:
            st.markdown(
                f"""<div class="nav-item {'active' if active else ''}">
                    <span style="margin-right: 12px">{icon}</span> {label}
                </div>""",
                unsafe_allow_html=True
            )
        st.markdown('</div>', unsafe_allow_html=True)

    # Main content area
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
                        criteria["location"] = st.text_input("Location", value=criteria["location"])
                    
                    with col4:
                        criteria["company_name"] = st.text_input("Company", value=criteria["company_name"])
                        criteria["industry"] = st.text_input("Industry", value=criteria["industry"])
                        criteria["years_experience"] = st.number_input("Years of Experience Required", 
                            value=criteria["years_experience"], min_value=0, max_value=20)
                    
                    st.session_state.current_criteria = criteria

                if st.button("🎯 Find Matching Candidates"):
                    if "connections_df" in st.session_state:
                        with st.spinner("Finding matches..."):
                            matching_candidates = match_candidates(st.session_state.connections_df, criteria)
                            
                            if not matching_candidates.empty:
                                st.success(f"Found {len(matching_candidates)} potential candidates!")
                                st.write(matching_candidates.to_html(escape=False), unsafe_allow_html=True)
                            else:
                                st.warning("No matching candidates found. Try adjusting the criteria.")
                    else:
                        st.error("Please upload connections data first!")
        
        with col2:
            st.header("📊 Matching Criteria")
            st.markdown("""
            Candidates are scored based on:
            - Job Title Match (50%)
            - Seniority Level (20%)
            - Location Match (15%)
            - Industry Match (15%)
            
            Current employees of the hiring company are automatically excluded.
            
            Notes:
            - Location is extracted from LinkedIn profiles
            - Industry matching uses company information
            - Fuzzy matching is used for more accurate comparisons
            """)

if __name__ == "__main__":
    main()
