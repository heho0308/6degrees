import streamlit as st
import pandas as pd
import requests
from io import StringIO
import time

# -----------------------------
# MOCK DATABASES / STORAGE
# -----------------------------
# For demonstration purposes, we'll store everything in memory.
# In a real app, these would be replaced by proper databases.

# Example structure for storing user data
mock_users = {
    "alice@example.com": {
        "password": "password123",
        "role": "Employee",
        "connections": []  # Will store dictionaries describing connections
    },
    "bob@example.com": {
        "password": "securepwd",
        "role": "Recruiter",
        "connections": []  # Recruiters may also have connections, if needed
    }
}

# Store recruiters' previous searches here
recruiter_search_history = {
    "bob@example.com": []  # Example: each item in list: {'criteria': ..., 'results': ...}
}

# Store rating data. For demonstration we store them as a list of (candidate_name, rating).
candidate_ratings = []

# -----------------------------
# HELPER FUNCTIONS
# -----------------------------
def parse_job_posting_from_url(url: str):
    """
    Mock function to simulate retrieving job criteria from a URL.
    In a real scenario, you might scrape or call an API to parse the job details.
    We'll return some placeholder criteria for demonstration.
    """
    # Simulate a small delay
    time.sleep(1)
    
    # Placeholder data
    # In real life, you'd parse the HTML or use a job API to get these details.
    criteria = {
        "title": "Software Engineer",
        "industry": "Tech",
        "seniority": "Mid-level",
        "skills": ["Python", "Django", "REST APIs"]
    }
    return criteria

def find_best_candidates(connections, job_criteria, top_n=5):
    """
    Given a list of connections and some job criteria, return a
    list of the top N candidate dictionaries plus an explanation.
    
    The logic here is naive. In real applications, you'd implement
    more sophisticated matching (e.g., text similarity, ML ranking).
    """
    
    # We'll simply assign a score for each connection based on matching
    # title, seniority, industry, or any skills in job_criteria.
    scored_candidates = []
    for c in connections:
        score = 0
        explanation_parts = []
        
        # Check same or similar job title
        # We'll do a naive "substring" check here
        if 'title' in c and c['title'] and job_criteria.get('title', '').lower() in c['title'].lower():
            score += 2
            explanation_parts.append("Has a matching job title")
        
        # Check seniority
        if 'seniority' in c and c['seniority'] and (job_criteria.get('seniority', '').lower() in c['seniority'].lower()):
            score += 2
            explanation_parts.append("Matches seniority level")
        
        # Check industry
        if 'industry' in c and c['industry'] and (job_criteria.get('industry', '').lower() in c['industry'].lower()):
            score += 2
            explanation_parts.append("Has experience in the same industry")
        
        # Check skill overlap
        if 'skills' in c:
            matched_skills = set(c['skills']).intersection(set(job_criteria.get('skills', [])))
            if matched_skills:
                score += len(matched_skills)
                explanation_parts.append(f"Has matching skills: {', '.join(matched_skills)}")
        
        # If score > 0, consider them in results
        if score > 0:
            scored_candidates.append({
                "connection": c,
                "score": score,
                "explanation": "; ".join(explanation_parts) if explanation_parts else "Some relevant match"
            })
    
    # Sort by score descending
    scored_candidates.sort(key=lambda x: x['score'], reverse=True)
    
    # Return top N
    return scored_candidates[:top_n]

# -----------------------------
# STREAMLIT UI
# -----------------------------
def main():
    st.title("Employee Network Matching")
    st.write("Connect your network with open positions.")
    
    # Sidebar
    st.sidebar.title("Navigation")
    
    # Check if user is logged in
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
        st.session_state.user_email = None
    
    if not st.session_state.logged_in:
        # Show login UI
        st.header("Login")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            if email in mock_users and mock_users[email]['password'] == password:
                st.session_state.logged_in = True
                st.session_state.user_email = email
                st.success(f"Logged in as {email}")
            else:
                st.error("Invalid email or password.")
        return  # Stop here if not logged in
    
    # If logged in, show user info and role
    user_data = mock_users[st.session_state.user_email]
    user_role = user_data["role"]
    
    st.sidebar.write(f"**User**: {st.session_state.user_email}")
    st.sidebar.write(f"**Role**: {user_role}")
    
    # Show a link to let user log out
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.user_email = None
        st.experimental_rerun()
    
    # Show previous searches if user is a recruiter
    if user_role == "Recruiter":
        st.sidebar.subheader("Previous Searches")
        history = recruiter_search_history.get(st.session_state.user_email, [])
        for i, h in enumerate(history):
            st.sidebar.write(f"Search {i+1}: {h['criteria'].get('title', 'No Title')} - {h['criteria'].get('seniority', '')}")
    
    # Show main content based on role
    if user_role == "Employee":
        show_employee_ui(user_data)
    else:
        show_recruiter_ui(user_data)

def show_employee_ui(user_data):
    st.subheader("Upload Your LinkedIn Connections (CSV)")
    uploaded_file = st.file_uploader("Select CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Preview of CSV:")
        st.write(df.head())
        
        # Convert the dataframe into a structure we can store
        # Suppose the CSV has columns: First Name, Last Name, Email Address, Current Position, Industry, ...
        # Adjust to your actual LinkedIn export schema
        connections_list = []
        for index, row in df.iterrows():
            connections_list.append({
                "name": f"{row.get('First Name', '')} {row.get('Last Name', '')}",
                "email": row.get("Email Address", ""),
                "title": row.get("Position", ""),
                "industry": row.get("Industry", ""),
                "seniority": row.get("Seniority", ""),
                # Possibly parse out skills or store custom data
                "skills": row.get("Skills", "").split(';') if "Skills" in row else []
            })
        
        # Store in mock_users
        mock_users[st.session_state.user_email]["connections"] = connections_list
        st.success("Connections uploaded successfully.")
        
    # Show number of connections if any
    current_connections = mock_users[st.session_state.user_email]["connections"]
    st.write(f"Total Connections: {len(current_connections)}")
    
def show_recruiter_ui(user_data):
    st.subheader("Paste a URL for an Open Job Posting")
    job_url = st.text_input("Job Posting URL")
    if st.button("Fetch Job Criteria"):
        if job_url:
            job_criteria = parse_job_posting_from_url(job_url)
            
            # Let recruiter review & edit criteria
            st.write("**Fetched Criteria (Editable):**")
            title = st.text_input("Job Title", value=job_criteria.get('title', ''))
            industry = st.text_input("Industry", value=job_criteria.get('industry', ''))
            seniority = st.text_input("Seniority", value=job_criteria.get('seniority', ''))
            skills = st.text_input("Skills (comma-separated)", value=", ".join(job_criteria.get('skills', [])))
            
            if st.button("Confirm Criteria"):
                edited_criteria = {
                    "title": title,
                    "industry": industry,
                    "seniority": seniority,
                    "skills": [s.strip() for s in skills.split(',')]
                }
                
                # Now we do the matching
                all_employees_connections = []
                # For a real app, you'd query all "Employee" accounts.
                # For simplicity, let's just iterate all mock_users
                for email, data in mock_users.items():
                    if data["role"] == "Employee":
                        all_employees_connections.extend(data["connections"])
                
                results = find_best_candidates(all_employees_connections, edited_criteria, top_n=5)
                
                # Save to search history
                recruiter_search_history[st.session_state.user_email].append({
                    "criteria": edited_criteria,
                    "results": results
                })
                
                st.write("### Top 5 Candidates")
                if not results:
                    st.write("No matches found.")
                else:
                    for i, r in enumerate(results, start=1):
                        c = r["connection"]
                        st.markdown(f"**{i}. {c['name']}** (Score: {r['score']})")
                        st.write(f"Title: {c['title']}, Industry: {c['industry']}, Seniority: {c.get('seniority', 'N/A')}")
                        st.write(f"Explanation: {r['explanation']}")
                        
                        # Add a rating option
                        rating = st.slider(f"Rate the suitability of {c['name']}", 1, 5, 3, key=f"rating_{i}")
                        if st.button(f"Submit Rating for {c['name']}", key=f"btn_{i}"):
                            candidate_ratings.append((c['name'], rating))
                            st.success(f"Submitted rating {rating} for {c['name']}")
        else:
            st.warning("Please enter a valid URL.")
