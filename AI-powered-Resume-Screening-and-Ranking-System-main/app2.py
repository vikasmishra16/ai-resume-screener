import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import operator # For sorting dictionary by value

# Set Streamlit Page Config
st.set_page_config(
    page_title="AI Job Profile Matcher", # Changed title
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply Custom Styling (Keep your existing CSS or adjust as needed)
custom_css = """
<style>
/* Background Image */
.stApp {
    background: url('https://images.unsplash.com/photo-1522071901873-411886a10004') no-repeat center center fixed;
    background-size: cover;
}
body { color: #FFFFFF; }
h1, h2, h3, h4, h5, h6 { color: #00ffcc; font-weight: bold; text-transform: uppercase; }
textarea, .stTextInput>div>div>input { background-color: #222; color: #fff; border: 2px solid #00ffcc; font-size: 16px; border-radius: 5px;}
.stFileUploader > div > div > button { background: linear-gradient(45deg, #ff00cc, #3333ff); color: white; font-weight: bold; border-radius: 8px; padding: 10px 15px; }
.stFileUploader > div > div > div > div { background-color: rgba(30, 30, 30, 0.8); border-radius: 10px; border: 1px dashed #00ffcc; }
.stButton>button { background: linear-gradient(45deg, #ff00cc, #3333ff); color: white; font-weight: bold; border-radius: 8px; padding: 10px 20px; }
.dataframe { border: 2px solid #00ffcc; color: #fff; font-size: 18px; text-align: left; background-color: rgba(30, 30, 30, 0.8); } /* Text align left for readability */
.main .block-container { background-color: rgba(18, 18, 18, 0.85); padding: 2rem; border-radius: 10px; }
div[data-testid="stSubheader"] { background-color: rgba(0, 255, 204, 0.1); padding: 5px; border-radius: 5px; border-left: 5px solid #00ffcc; }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# --- Predefined Job Profiles ---
# For a real application, load these from files or a database
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
    """
}

# Function to extract text from PDFs
def extract_text_from_pdf(uploaded_file):
    pdf = PdfReader(uploaded_file)
    text = ""
    for page_num, page in enumerate(pdf.pages): # Added page_num for potential debugging
        try:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        except Exception as e:
            st.warning(f"Could not extract text from page {page_num + 1} of {uploaded_file.name}. Error: {e}")
    return text.strip()

# Function to find the best matching job profiles for a single resume
def match_resume_to_profiles(resume_text, job_profiles_dict):
    profile_names = list(job_profiles_dict.keys())
    profile_descriptions = list(job_profiles_dict.values())

    # The resume text is the "query document"
    # The profile descriptions are the "corpus documents"
    documents = [resume_text] + profile_descriptions
    
    try:
        vectorizer = TfidfVectorizer(stop_words='english').fit_transform(documents)
        vectors = vectorizer.toarray()
    except ValueError as e:
        # This can happen if all documents are empty after stopword removal
        st.error(f"TF-IDF Vectorization Error: {e}. This might be due to very short or common text in resume/profiles.")
        return {} # Return empty dict on error

    resume_vector = vectors[0]
    profile_vectors = vectors[1:]

    similarities = cosine_similarity([resume_vector], profile_vectors).flatten()

    # Create a dictionary of profile names and their scores
    matched_scores = dict(zip(profile_names, similarities))

    # Sort by score in descending order
    sorted_matches = dict(sorted(matched_scores.items(), key=operator.itemgetter(1), reverse=True))
    
    return sorted_matches

# --- Streamlit App UI ---
st.markdown("<h1 style='text-align: center;'>🎯 AI Resume to Job Profile Matcher</h1>", unsafe_allow_html=True)
st.markdown("---")

st.subheader("📄 Upload Your Resume")
uploaded_resume = st.file_uploader(
    "Upload your resume in PDF format to find suitable job profiles.",
    type=["pdf"],
    accept_multiple_files=False, # Only one resume at a time for this functionality
    key="resume_matcher_uploader"
)

if st.button("🔍 Find Matching Job Profiles", key="match_button"):
    if uploaded_resume is not None:
        with st.spinner("Analyzing your resume against job profiles... 🛠️"):
            resume_text = extract_text_from_pdf(uploaded_resume)

            if not resume_text:
                st.error(f"Could not extract any text from '{uploaded_resume.name}'. Please try a different PDF.")
            else:
                # st.write("Extracted Resume Text (First 500 chars):") # For debugging
                # st.info(resume_text[:500] + "...")

                matched_profiles_scores = match_resume_to_profiles(resume_text, PREDEFINED_JOB_PROFILES)

                if not matched_profiles_scores:
                    st.warning("Could not calculate matches. Please check the console for errors if any.")
                else:
                    st.markdown("---")
                    st.markdown("<h3 style='color: #00ffcc; text-align:center;'>🌟 Top Matching Job Profiles 🌟</h3>", unsafe_allow_html=True)

                    # Prepare data for display
                    profile_names = []
                    match_percentages = []
                    descriptions_to_show = [] # To show snippets of job descriptions

                    for profile, score in matched_profiles_scores.items():
                        profile_names.append(profile)
                        match_percentages.append(f"{score*100:.2f}%")
                        # Show first 150 chars of the job description as a snippet
                        descriptions_to_show.append(PREDEFINED_JOB_PROFILES[profile][:150].replace('\n', ' ') + "...")


                    results_df = pd.DataFrame({
                        "Job Profile": profile_names,
                        "Match Score": match_percentages,
                        "Description Snippet": descriptions_to_show 
                    })
                    
                    # Display results table
                    st.markdown(results_df.to_html(escape=False, index=False), unsafe_allow_html=True)
                    
                    # Optionally, display the full description of the top match
                    if results_df.shape[0] > 0:
                        top_match_profile_name = results_df.iloc[0]["Job Profile"]
                        st.markdown("---")
                        st.subheader(f"Details for Top Match: {top_match_profile_name}")
                        with st.expander("View Full Job Description"):
                            st.markdown(f"<pre style='color:white; background-color:rgba(30,30,30,0.7); padding:10px; border-radius:5px; white-space:pre-wrap;'>{PREDEFINED_JOB_PROFILES[top_match_profile_name]}</pre>", unsafe_allow_html=True)


    else:
        st.error("🚨 Please upload a resume before matching.")

st.markdown("---")
st.markdown("<p style='text-align: center; color: #aaa;'>AI Job Profile Matcher v1.0</p>", unsafe_allow_html=True)