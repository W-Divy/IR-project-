import streamlit as st
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
import PyPDF2
import docx
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
st.set_page_config(page_title="Document Topic Discovery & IR", layout="wide")

def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"
    return text

def extract_text_from_docx(file):
    doc = docx.Document(file)
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"
    return text

def extract_text_from_txt(file):
    return file.getvalue().decode("utf-8", errors="ignore")

@st.cache_resource
def load_and_train_model():
    sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    categories = [
        'sci.space', 
        'sci.med', 
        'comp.graphics', 
        'rec.autos', 
        'talk.politics.mideast'
    ]
    data = fetch_20newsgroups(subset='train', categories=categories, remove=('headers', 'footers', 'quotes'))
    docs = [d for d in data.data if len(d.strip()) > 80][:350]
    
    # Strip English stop words and discard words appearing in only 1 document
    vectorizer_model = CountVectorizer(stop_words="english", min_df=2)
    
    topic_model = BERTopic(
        embedding_model=sentence_model, 
        vectorizer_model=vectorizer_model,
        verbose=False
    )
    topic_model.fit(docs)
    return sentence_model, topic_model

def main():
    st.title("Topic Modeling for Document Discovery & Information Retrieval")
    st.write("Upload a document (.pdf, .docx, .txt) to discover its hidden topic and extract key themes.")

    # 1. Initialize models
    with st.spinner("Initializing models and discovering latent topics..."):
        sentence_model, topic_model = load_and_train_model()

    # 2. Main Page Topic Overview Table
    st.subheader("All Discovered Corpus Topics")
    all_topics_df = topic_model.get_topic_info()
    st.dataframe(all_topics_df, use_container_width=True)

    # 3. Sidebar: Quick overview
    st.sidebar.header("Corpus Topic Clusters")
    st.sidebar.dataframe(all_topics_df[['Topic', 'Count', 'Name']], use_container_width=True)

    # 4. Document Upload
    uploaded_file = st.file_uploader(
        "Choose a file to analyze", 
        type=['txt', 'pdf', 'docx'],
        help="Upload a document to extract its latent topics"
    )

    if uploaded_file is not None:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            status_text.text("Extracting text from document...")
            progress_bar.progress(30)
            
            file_type = uploaded_file.name.split('.')[-1].lower()
            if file_type == 'pdf':
                text = extract_text_from_pdf(uploaded_file)
            elif file_type == 'docx':
                text = extract_text_from_docx(uploaded_file)
            else:
                text = extract_text_from_txt(uploaded_file)

            if not text.strip():
                st.warning("The uploaded document contains no readable text.")
                return

            progress_bar.progress(60)
            status_text.text("Discovering latent topics & vector similarity...")

            # Predict topic cluster
            topics, probs = topic_model.transform([text])
            topic_id = topics[0]
            topic_details = topic_model.get_topic(topic_id)
            
            progress_bar.progress(90)
            status_text.text("Rendering results...")

            st.subheader("Classification & Thematic Discovery")
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("### Document Metadata")
                st.write(f"**Filename:** `{uploaded_file.name}`")
                st.write(f"**Assigned Topic Cluster:** `Topic {topic_id}`")
                if topic_id == -1:
                    st.info("Topic -1 represents an outlier document with distinct semantic properties.")

            with col2:
                st.markdown("### Top Keywords (c-TF-IDF Score)")
                if topic_details:
                    words, weights = zip(*topic_details)
                    topic_df = pd.DataFrame({
                        'Thematic Keyword': words,
                        'Relevance (c-TF-IDF)': [round(w, 4) for w in weights]
                    })
                    st.dataframe(topic_df, use_container_width=True)
                else:
                    st.write("No distinct topic cluster extracted.")

            with st.expander("Document Text Preview"):
                st.text(text[:800] + ("..." if len(text) > 800 else ""))

            # Robust Visualizations
            st.markdown("---")
            st.subheader("Topic Space Visualizations")
            try:
                fig = topic_model.visualize_barchart(top_n_topics=5)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as viz_err:
                st.info(f"Visualizer notice: {viz_err}")

            progress_bar.progress(100)
            status_text.text("Analysis complete!")

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            progress_bar.empty()
            status_text.empty()

if __name__ == "__main__":
    main()