import streamlit as st
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Function to extract text from PDF
def extract_text_from_pdf(file):
    pdf = PdfReader(file)
    text = ""
    for page in pdf.pages:
        text += page.extract_text() or ""
    return text

# Function to rank resumes based on job description
def rank_resumes(job_description, resumes):
    documents = [job_description] + resumes
    vectorizer = TfidfVectorizer().fit_transform(documents)
    vectors = vectorizer.toarray()

    job_description_vector = vectors[0]
    resume_vectors = vectors[1:]
    cosine_similarities = cosine_similarity([job_description_vector], resume_vectors).flatten()
    
    return cosine_similarities

# Streamlit app UI
st.set_page_config(page_title="Resume Screening", page_icon="📄", layout="centered")
st.markdown("""
    <style>
        html, body, [class*="css"]  {
            background: linear-gradient(135deg, #E6E6FA, #FFFFFF, #D1E8FF);
            color: #333333;
            font-family: Arial, sans-serif;
        }
        .stApp {
            background: linear-gradient(135deg, #E6E6FA, #FFFFFF, #D1E8FF);
        }
        .stTextArea textarea { height: 150px; }
        .stButton>button { 
            background: linear-gradient(135deg, #4CAF50, #87CEEB); 
            color: white; 
            border-radius: 8px; 
            padding: 8px 16px; 
            font-size: 16px; 
            border: none;
            transition: background 0.3s ease;
        }
        .stButton>button:hover { 
            background: linear-gradient(135deg, #87CEEB, #FF69B4); 
            color: white; 
        }
        .rankings { 
            margin-top: 20px; 
            padding: 15px; 
            border-radius: 10px; 
            background: #f0f2f6; 
            box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1); 
        }
        .resume-card {
            margin: 10px 0;
            padding: 15px;
            border-radius: 10px;
            background: linear-gradient(135deg, #FF69B4, #87CEEB, #DDA0DD);
            color: white;
            box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1);
            transition: transform 0.2s ease;
        }
        .resume-card:hover {
            transform: scale(1.03);
        }
    </style>
""", unsafe_allow_html=True)

st.title("AI Resume Screening & Candidate Ranking System")
st.write("Upload the job description and resumes to get a ranked list based on relevance.")

# Input for job description
st.subheader("📝 Job Description")
job_description = st.text_area("Enter the Job Description:", height=200)

# Upload resumes
st.subheader("📤 Upload Resumes")
uploaded_files = st.file_uploader("Upload Resumes (PDF format):", accept_multiple_files=True, type=["pdf"], help="You can upload multiple PDF files")

# Submit button
if st.button("Rank Resumes"):
    if job_description and uploaded_files:
        resumes = []
        resume_names = []

        with st.spinner("Analyzing resumes..."):
            # Extract text from uploaded resumes
            for file in uploaded_files:
                text = extract_text_from_pdf(file)
                resumes.append(text)
                resume_names.append(file.name)

            # Rank resumes based on the job description
            similarity_scores = rank_resumes(job_description, resumes)
            ranked_resumes = sorted(zip(resume_names, similarity_scores), key=lambda x: x[1], reverse=True)

        # Display results
        st.subheader("📊 Resume Rankings")
        for i, (name, score) in enumerate(ranked_resumes, start=1):
            st.markdown(f'<div class="resume-card">{i}. <strong>{name}</strong> - {score * 100:.2f}% Match</div>', unsafe_allow_html=True)
    else:
        st.warning("⚠️ Please upload both a job description and resumes.")
