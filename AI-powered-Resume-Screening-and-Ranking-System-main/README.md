# AI-powered Resume Screening and Ranking System

## 1. Introduction
The **AI-powered Resume Screening and Ranking System** is designed to assist recruiters in efficiently identifying the most suitable candidates for a given job role. By leveraging **Natural Language Processing (NLP)** and **Machine Learning (ML)** techniques, this system automates resume screening, saving time and improving hiring accuracy.

---

## 1.1 Problem Statement
In modern recruitment, organizations receive a vast number of resumes for each job opening. Manually reviewing these resumes is time-consuming, prone to errors, and inefficient. This project aims to automate the resume screening process by utilizing NLP techniques to rank resumes based on their relevance to the provided job description.

---

## 2. Proposed Solution
The proposed solution utilizes a combination of **TF-IDF Vectorization** and **Cosine Similarity** to rank resumes based on relevance. A web application built using **Streamlit** provides a user-friendly interface for job description input, resume uploads, and ranked result display.

---

## 3. System Design
The system architecture is composed of five key components:

### 3.1 Architecture Diagram
![Screenshot 2025-02-24 195631](https://github.com/user-attachments/assets/a542662b-fd27-46fc-9d8f-bae722626250)
 

### 3.2 Detailed Explanation
- **User Input and Frontend:**  
  Users can enter the job description and upload multiple resumes in PDF format.  
- **Backend Processing:**  
  Resumes undergo parsing and text extraction to prepare data for analysis.  
- **Feature Engineering:**  
  Extracted text is converted into numerical vectors using **TF-IDF Vectorization**.  
- **Similarity Computation:**  
  The system calculates the similarity score using **Cosine Similarity** to assess how well each resume matches the job description.  
- **Ranking and Output Display:**  
  Results are stored, sorted, and displayed in ranked order.

---

## 3.2 Requirement Specification

### 3.2.1 Hardware Requirements
- Processor: Intel i3 or higher  
- RAM: 4GB or more  
- Storage: Minimum 1GB free space  

### 3.2.2 Software Requirements
- **Python 3.11** or later  
- **VS Code** or any preferred IDE  
- **Streamlit** for frontend development  
- **scikit-learn**, **pandas**, and **NLTK** for ML and NLP tasks  
- **PyPDF2** or **pdfplumber** for PDF text extraction  

---


## 4. Usage
1. Enter the **Job Description** in the text area.
2. Upload multiple **PDF resumes** using the file uploader.
3. Click on the **"Rank Resumes"** button to display the top-ranked resumes.

---

## 5. Future Scope
- Integrate advanced NLP models like **BERT** or **GPT** for improved contextual understanding.
- Implement a **deep learning model** for enhanced ranking accuracy.
- Add **multi-language support** to cater to global audiences.
- Develop interactive **data visualization dashboards** for insights into candidate skills and experience.

---

## 6. Contributors
- **Soumen Bhunia**  
For inquiries, contact: (https://www.linkedin.com/in/soumen-bhunia-2b8799293/)

---

