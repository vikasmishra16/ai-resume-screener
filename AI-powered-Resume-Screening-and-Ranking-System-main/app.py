import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Set Streamlit Page Config
st.set_page_config(
    page_title="AI Resume Screening & Ranking",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply Custom Styling
custom_css = """
<style>
/* Background Image */
.stApp {
    background: url('https://images.unsplash.com/photo-1522071901873-411886a10004') no-repeat center center fixed;
    background-size: cover;
}

/* Dark Mode Styling */
body {
    color: #FFFFFF;
    /* background-color: #121212; Removed to let background image show */
}

/* Header Styling */
h1, h2, h3, h4, h5, h6 {
    color: #00ffcc;
    font-weight: bold;
    text-transform: uppercase;
}

/* Input Fields */
textarea, .stTextInput>div>div>input {
    background-color: #222;
    color: #fff;
    border: 2px solid #00ffcc;
    font-size: 16px;
    border-radius: 5px; /* Added for smoother edges */
}

/* File Uploader */
.stFileUploader > div > div > button { /* Target the button inside uploader */
    background: linear-gradient(45deg, #ff00cc, #3333ff);
    color: white;
    font-weight: bold;
    border-radius: 8px;
    padding: 10px 15px;
}
.stFileUploader > div > div > div > div { /* Target the drop zone text area */
    background-color: rgba(30, 30, 30, 0.8); /* Semi-transparent background */
    border-radius: 10px;
    border: 1px dashed #00ffcc;
}


/* Buttons */
.stButton>button {
    background: linear-gradient(45deg, #ff00cc, #3333ff);
    color: white;
    font-weight: bold;
    border-radius: 8px;
    padding: 10px 20px;
}

/* Table Styling */
.dataframe {
    border: 2px solid #00ffcc;
    color: #fff;
    font-size: 18px;
    text-align: center;
    background-color: rgba(30, 30, 30, 0.8); /* Semi-transparent background for table */
}

/* Ensure main content area is slightly opaque to make text readable over background */
.main .block-container {
    background-color: rgba(18, 18, 18, 0.85); /* Dark, semi-transparent background */
    padding: 2rem;
    border-radius: 10px;
}

/* Specific styling for subheaders to stand out */
div[data-testid="stExpander"] > div[role="button"] > div, /* Expander headers */
div[data-testid="stSubheader"] { /* Subheaders */
    background-color: rgba(0, 255, 204, 0.1); /* Light background for subheaders */
    padding: 5px;
    border-radius: 5px;
    border-left: 5px solid #00ffcc;
}

</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# Function to extract text from PDFs
def extract_text_from_pdf(uploaded_file):
    pdf = PdfReader(uploaded_file)
    text = ""
    for page in pdf.pages:
        page_text = page.extract_text()
        if page_text:  
            text += page_text + "\n"
    return text  

# Function to rank resumes based on job description
def rank_resumes(job_description, resumes_text_list): # Renamed for clarity
    documents = [job_description] + resumes_text_list # Use the list of resume texts
    vectorizer = TfidfVectorizer(stop_words='english').fit_transform(documents) # Added stop_words
    vectors = vectorizer.toarray()

    # Calculate cosine similarity
    job_description_vector = vectors[0]  
    resume_vectors = vectors[1:]  
    cosine_similarities = cosine_similarity([job_description_vector], resume_vectors).flatten()
    
    return cosine_similarities

# --- Streamlit App UI ---
st.markdown("<h1 style='text-align: center;'>🚀 AI Resume Screening & Candidate Ranking</h1>", unsafe_allow_html=True)
st.markdown("---") # Add a horizontal rule for separation

# Job description input
st.subheader("📌 Enter the Job Description")
job_description = st.text_area("Paste the complete job description here to find the best candidates.", height=200, key="job_desc_input")

# File uploader for resumes
st.subheader("📂 Upload Resumes")
uploaded_files = st.file_uploader(
    "Upload candidate resumes in PDF format (you can select multiple files).", 
    type=["pdf"], 
    accept_multiple_files=True,
    key="resume_uploader"
)

# "Rank Resumes" Button
if st.button("✨ Rank Resumes", key="rank_button"):
    if uploaded_files and job_description.strip(): # Ensure job description is not just whitespace
        with st.spinner("Analyzing resumes... Please wait. 🧠"):
            resumes_data = [] # To store file names and extracted text
            for file in uploaded_files:
                text = extract_text_from_pdf(file)
                if text.strip(): # Ensure extracted text is not empty
                    resumes_data.append({"name": file.name, "text": text})
                else:
                    st.warning(f"Could not extract text from '{file.name}' or it's empty. Skipping.")
            
            if not resumes_data:
                st.error("No processable text found in the uploaded resumes. Please check the PDF files.")
            else:
                resume_texts_for_ranking = [r["text"] for r in resumes_data]
                resume_names_for_display = [r["name"] for r in resumes_data]

                # Rank resumes
                scores = rank_resumes(job_description, resume_texts_for_ranking)

                # Convert scores to percentage and format
                # IMPORTANT: Consider how you want to scale this.
                # A direct multiplication by 100 might be too simplistic.
                # You might want to normalize or apply a curve if scores are often low.
                # For this example, we'll do a direct conversion and add a threshold interpretation.
                
                acceptability_scores_percent = [score * 100 for score in scores]

                # Display results
                results_df = pd.DataFrame({
                    "Resume File Name": resume_names_for_display, 
                    "Acceptability Score (%)": acceptability_scores_percent
                })
                results_df = results_df.sort_values(by="Acceptability Score (%)", ascending=False)
                
                # Format the score to two decimal places
                results_df["Acceptability Score (%)"] = results_df["Acceptability Score (%)"].map('{:.2f}%'.format)


                st.markdown("---") # Separator
                st.markdown("<h3 style='color: #00ffcc; text-align:center;'>🏆 Ranked Candidate Results 🏆</h3>", unsafe_allow_html=True)
                
                # Display results table (using st.dataframe for better interactivity if needed, or st.table for static)
                # st.dataframe(results_df.style.set_properties(**{ # Old styling
                #     'background-color': '#1e1e1e',
                #     'color': 'white',
                #     'border-color': '#00ffcc'
                # }))
                # For full width and custom CSS application:
                st.markdown(results_df.to_html(escape=False, index=False), unsafe_allow_html=True)


                # Optional: Add interpretation based on scores
                st.markdown("---")
                st.subheader("💡 Score Interpretation Guide (Example)")
                st.info("""
                - **> 80%:** Strong match, high potential.
                - **60% - 80%:** Good match, worth considering.
                - **40% - 60%:** Moderate match, review details carefully.
                - **< 40%:** Low match, likely not suitable based on text similarity.
                
                *Note: This TF-IDF based score primarily reflects keyword similarity. For deeper contextual understanding, more advanced models (like BERT, mentioned in Future Scope) would be beneficial.*
                """)
    elif not job_description.strip():
        st.error("🚨 Please enter a job description before ranking.")
    elif not uploaded_files:
        st.error("🚨 Please upload at least one resume before ranking.")

# Footer or additional information
st.markdown("---")
st.markdown("<p style='text-align: center; color: #aaa;'>AI Resume Screening Tool v1.0</p>", unsafe_allow_html=True)