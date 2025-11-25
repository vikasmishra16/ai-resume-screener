import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import operator # For sorting dictionary by value

# Set Streamlit Page Config
st.set_page_config(
    page_title="AI Resume & Job Matcher", # Updated title
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply Custom Styling (Keep your existing CSS)
custom_css = """
<style>
/* Background Image */
.stApp {
    background-color: #5b806b;
    background-size: cover;
}
body { color: #FFFFFF; }
h1, h2, h3, h4, h5, h6 { color: #00ffcc; font-weight: bold; text-transform: uppercase; }
textarea, .stTextInput>div>div>input { background-color: #222; color: #fff; border: 2px solid #00ffcc; font-size: 16px; border-radius: 5px;}
.stFileUploader > div > div > button { background: linear-gradient(45deg, #ff00cc, #3333ff); color: white; font-weight: bold; border-radius: 8px; padding: 10px 15px; }
.stFileUploader > div > div > div > div { background-color: rgba(30, 30, 30, 0.8); border-radius: 10px; border: 1px dashed #00ffcc; }
.stButton>button { background: linear-gradient(45deg, #ff00cc, #3333ff); color: white; font-weight: bold; border-radius: 8px; padding: 10px 20px; }
.dataframe { border: 2px solid #00ffcc; color: #fff; font-size: 18px; text-align: left; background-color: rgba(30, 30, 30, 0.8); }
.main .block-container { background-color: rgba(18, 18, 18, 0.85); padding: 2rem; border-radius: 10px; }
div[data-testid="stSubheader"] { background-color: rgba(0, 255, 204, 0.1); padding: 5px; border-radius: 5px; border-left: 5px solid #00ffcc; }

/* Custom style for tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 24px;
}
.stTabs [data-baseweb="tab"] {
    height: 50px;
    white-space: pre-wrap;
    background-color: rgba(0,0,0,0.2);
    border-radius: 4px 4px 0px 0px;
    gap: 1px;
    padding-top: 10px;
    padding-bottom: 10px;
    color: #aaa; /* Inactive tab color */
}
.stTabs [aria-selected="true"] {
    background-color: #00ffcc;
    color: #121212 !important; /* Active tab text color */
    font-weight: bold;
}

</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# --- Predefined Job Profiles ---
PREDEFINED_JOB_PROFILES = {
    "Software Engineer (Backend)": """
        We are looking for a skilled Backend Software Engineer proficient in Python, Django, and REST APIs.
        Experience with database technologies like PostgreSQL or MySQL, and cloud platforms like AWS or Azure is essential.
        Responsibilities include designing and developing server-side logic, defining and maintaining databases,
        and ensuring high performance and responsiveness to requests from the front-end.
        Familiarity with version control (Git) and agile methodologies is a plus.
        Strong problem-solving skills and ability to work in a team.
    """,
    "Data Scientist": """
        Seeking a Data Scientist with a strong background in statistical analysis, machine learning, and data visualization.
        Proficiency in Python (Pandas, NumPy, Scikit-learn, TensorFlow/PyTorch) and SQL is required.
        The ideal candidate will be able to develop predictive models, perform data mining,
        and communicate insights effectively to stakeholders. Experience with big data technologies (e.g., Spark) is a plus.
        Master's or PhD in a quantitative field preferred.
    """,
    "Frontend Developer (React)": """
        Join our team as a Frontend Developer specializing in React.js.
        You will be responsible for developing and implementing user interface components using React.js concepts and workflows such as Redux, Flux, and Webpack.
        Strong proficiency in JavaScript, HTML, CSS, and experience with RESTful APIs are crucial.
        Experience with UI/UX design principles and modern frontend build pipelines and tools.
        Ability to translate designs and wireframes into high-quality code.
    """,
    "DevOps Engineer": """
        We need a DevOps Engineer to help us build and maintain our CI/CD pipelines and cloud infrastructure.
        Skills in scripting (Bash, Python), containerization (Docker, Kubernetes), infrastructure as code (Terraform, Ansible),
        and cloud platforms (AWS, GCP, Azure) are required.
        Experience with monitoring tools (Prometheus, Grafana) and version control (Git).
        Focus on automation, scalability, and reliability.
    """,
    "UX/UI Designer": """
        Creative UX/UI Designer needed to craft intuitive and engaging user experiences for web and mobile applications.
        Proficiency in design tools like Figma, Sketch, or Adobe XD.
        Strong portfolio showcasing user-centered design solutions, wireframes, prototypes, and visual designs.
        Understanding of usability principles, interaction design, and responsive design.
        Ability to conduct user research and translate findings into design improvements.
    """,
    "Human Resources (HR) Manager": """
        Experienced HR Manager to oversee all aspects of human resources practices and processes.
        Responsibilities include developing and implementing HR strategies and initiatives aligned with the overall business strategy,
        managing recruitment and selection process, bridging management and employee relations, and managing compensation and benefits.
        Strong knowledge of labor law and HR best practices. Degree in Human Resources or related field.
    """,
    "Recruitment Specialist (Talent Acquisition)": """
        Dynamic Recruitment Specialist to lead our talent acquisition efforts.
        This role involves sourcing candidates through various channels, planning interview and selection procedures,
        hosting or participating in career events, and developing long-term recruiting strategies.
        Proven experience as a Recruitment Specialist, Technical Recruiter or similar role.
        Excellent communication and interpersonal skills. Familiarity with Applicant Tracking Systems (ATS) and resume databases.
    """,
    "Project Manager (IT)": """
        Seeking an IT Project Manager to be responsible for planning, executing, and finalizing IT projects according to strict deadlines and within budget.
        This includes acquiring resources and coordinating the efforts of team members and third-party contractors or consultants.
        Proven working experience in project management in the information technology sector.
        Solid technical background, with understanding or hands-on experience in software development and web technologies. PMP certification is a plus.
    """,
    "Marketing Manager": """
        Innovative Marketing Manager to develop and implement marketing strategies to strengthen the company’s market presence and help it find a “voice” that will make a difference.
        Responsibilities include planning and executing campaigns, tracking and analyzing performance, managing budgets, and overseeing marketing material.
        Proven experience as Marketing Manager or similar role. Demonstrable experience leading and managing SEO/SEM, marketing database, email, social media and/or display advertising campaigns.
    """,
    "Operations Manager": """
        Detail-oriented Operations Manager to direct and coordinate the internal operational activities of the organization in accordance with policies, goals, and objectives established by the CEO and the Board of Directors.
        Key responsibilities include formulating policies, managing daily operations, personnel, and material resources to achieve specific goals.
        Proven experience as Operations Manager or relevant role. Understanding of business functions such as HR, Finance, marketing etc. Demonstrable competency in strategic planning and business development.
    """
}

# --- Helper Functions ---
def extract_text_from_pdf(uploaded_file):
    # ... (same as before)
    pdf = PdfReader(uploaded_file)
    text = ""
    for page_num, page in enumerate(pdf.pages):
        try:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        except Exception as e:
            st.warning(f"Could not extract text from page {page_num + 1} of {uploaded_file.name}. Error: {e}")
    return text.strip()

def rank_resumes_against_jd(job_description, resumes_text_list):
    # ... (same as original 'rank_resumes' function)
    documents = [job_description] + resumes_text_list
    vectorizer = TfidfVectorizer(stop_words='english').fit_transform(documents)
    vectors = vectorizer.toarray()
    job_description_vector = vectors[0]
    resume_vectors = vectors[1:]
    cosine_similarities = cosine_similarity([job_description_vector], resume_vectors).flatten()
    return cosine_similarities

def match_resume_to_profiles(resume_text, job_profiles_dict):
    # ... (same as before)
    profile_names = list(job_profiles_dict.keys())
    profile_descriptions = list(job_profiles_dict.values())
    documents = [resume_text] + profile_descriptions
    try:
        vectorizer = TfidfVectorizer(stop_words='english').fit_transform(documents)
        vectors = vectorizer.toarray()
    except ValueError as e:
        st.error(f"TF-IDF Vectorization Error: {e}.")
        return {}
    resume_vector = vectors[0]
    profile_vectors = vectors[1:]
    similarities = cosine_similarity([resume_vector], profile_vectors).flatten()
    matched_scores = dict(zip(profile_names, similarities))
    sorted_matches = dict(sorted(matched_scores.items(), key=operator.itemgetter(1), reverse=True))
    return sorted_matches

# --- Streamlit App UI ---
st.markdown("<h1 style='text-align: center;'>🚀 AI Resume & Job Profile Matcher 🎯</h1>", unsafe_allow_html=True)
st.markdown("---")

# Create tabs for different functionalities
tab1_title = "📄➡️👔 Rank Resumes vs. Job Description"
tab2_title = "👔➡️📄 Find Job Profiles for a Resume"
tab1, tab2 = st.tabs([tab1_title, tab2_title])

# --- TAB 1: Rank Resumes vs. Job Description ---
with tab1:
    st.header("Rank Multiple Resumes Against a Single Job Description")
    st.markdown("Upload a job description and multiple resumes to see which candidates are the best textual fit.")

    jd_input_ranker = st.text_area("Paste the complete job description here:", height=200, key="jd_ranker")
    
    resumes_input_ranker = st.file_uploader(
        "Upload candidate resumes (PDFs, multiple allowed):",
        type=["pdf"],
        accept_multiple_files=True,
        key="resumes_ranker"
    )

    if st.button("✨ Rank Candidate Resumes", key="rank_resumes_button"):
        if resumes_input_ranker and jd_input_ranker.strip():
            with st.spinner("Ranking resumes against JD... Please wait. 🧠"):
                resumes_data_ranker = []
                for file in resumes_input_ranker:
                    text = extract_text_from_pdf(file)
                    if text.strip():
                        resumes_data_ranker.append({"name": file.name, "text": text})
                    else:
                        st.warning(f"Could not extract text from '{file.name}' or it's empty. Skipping.")
                
                if not resumes_data_ranker:
                    st.error("No processable text found in the uploaded resumes for ranking.")
                else:
                    resume_texts_for_ranking = [r["text"] for r in resumes_data_ranker]
                    resume_names_for_display = [r["name"] for r in resumes_data_ranker]

                    scores = rank_resumes_against_jd(jd_input_ranker, resume_texts_for_ranking)
                    acceptability_scores_percent = [score * 100 for score in scores]

                    results_df_ranker = pd.DataFrame({
                        "Resume File Name": resume_names_for_display,
                        "Match Score (%)": acceptability_scores_percent
                    })
                    results_df_ranker = results_df_ranker.sort_values(by="Match Score (%)", ascending=False)
                    results_df_ranker["Match Score (%)"] = results_df_ranker["Match Score (%)"].map('{:.2f}%'.format)

                    st.markdown("---")
                    st.markdown("<h3 style='color: #00ffcc; text-align:center;'>🏆 Ranked Candidate Results 🏆</h3>", unsafe_allow_html=True)
                    st.markdown(results_df_ranker.to_html(escape=False, index=False), unsafe_allow_html=True)
                    
                    st.markdown("---")
                    st.subheader("💡 Score Interpretation Guide (Example)")
                    st.info("""
                    - **> 80%:** Strong match with the job description.
                    - **60% - 80%:** Good match, worth further review.
                    - **< 60%:** Lower textual similarity.
                    *Note: TF-IDF scores reflect keyword similarity.*
                    """)
        elif not jd_input_ranker.strip():
            st.error("🚨 Please enter a job description for the Ranker.")
        elif not resumes_input_ranker:
            st.error("🚨 Please upload resumes for the Ranker.")

# --- TAB 2: Find Job Profiles for a Resume ---
with tab2:
    st.header("Find Matching Job Profiles for a Single Resume")
    st.markdown("Upload a single resume to discover which of our predefined job profiles it matches best.")

    resume_input_matcher = st.file_uploader(
        "Upload your resume (single PDF):",
        type=["pdf"],
        accept_multiple_files=False,
        key="resume_matcher"
    )

    if st.button("🔍 Find Matching Job Profiles", key="match_profiles_button"):
        if resume_input_matcher is not None:
            with st.spinner("Matching resume to profiles... Please wait. 🛠️"):
                resume_text_matcher = extract_text_from_pdf(resume_input_matcher)

                if not resume_text_matcher:
                    st.error(f"Could not extract text from '{resume_input_matcher.name}'. Please try a different PDF.")
                else:
                    matched_profiles_scores = match_resume_to_profiles(resume_text_matcher, PREDEFINED_JOB_PROFILES)

                    if not matched_profiles_scores:
                        st.warning("Could not calculate matches for profiles.")
                    else:
                        st.markdown("---")
                        st.markdown("<h3 style='color: #00ffcc; text-align:center;'>🌟 Top Matching Job Profiles 🌟</h3>", unsafe_allow_html=True)

                        profile_names = []
                        match_percentages = []

                        for profile, score in matched_profiles_scores.items():
                            profile_names.append(profile)
                            match_percentages.append(f"{score*100:.2f}%")

                        results_df_matcher = pd.DataFrame({
                            "Job Profile": profile_names,
                            "Match Score (%)": match_percentages,
                        })
                        # No need to sort again, match_resume_to_profiles already returns sorted
                        
                        st.markdown(results_df_matcher.to_html(escape=False, index=False), unsafe_allow_html=True)
                        
                        if results_df_matcher.shape[0] > 0:
                            top_match_profile_name = results_df_matcher.iloc[0]["Job Profile"]
                            st.markdown("---")
                            st.subheader(f"📄 Details for Top Match: {top_match_profile_name}")
                            with st.expander("View Full Job Description for Top Match"):
                                st.markdown(f"<pre style='color:white; background-color:rgba(30,30,30,0.7); padding:10px; border-radius:5px; white-space:pre-wrap;'>{PREDEFINED_JOB_PROFILES[top_match_profile_name]}</pre>", unsafe_allow_html=True)
        else:
            st.error("🚨 Please upload a resume for the Profile Matcher.")


st.markdown("---")
st.markdown("<p style='text-align: center; color: #aaa;'>AI Resume & Job Profile Matcher by Vikas </p>", unsafe_allow_html=True)