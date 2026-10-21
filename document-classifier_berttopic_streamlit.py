import streamlit as st
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.datasets import fetch_20newsgroups
import PyPDF2
import docx
import pandas as pd

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
    
    # Fast, self-contained corpus across multiple domains (no internet download required)
    docs = [
        # Space & Astronomy
        "NASA rover explores the surface of Mars searching for ancient water and microbial life.",
        "The Hubble and James Webb Space Telescopes capture distant galaxies and stellar nebulae.",
        "Rocket propulsion systems and orbital satellite launches by SpaceX and planetary exploration.",
        "Astronomers discover exoplanets orbiting distant stars in habitable zones of deep space.",
        "Lunar missions aim to establish a permanent human outpost on the Moon's South Pole.",
        "Cosmology studies dark matter, cosmic background radiation, and expansion of the universe.",
        
        # Medicine & Healthcare
        "Clinical trials evaluate patient response to novel immunotherapy cancer treatments.",
        "Cardiovascular diseases, heart attacks, and blood pressure monitoring in clinical healthcare.",
        "Neuroscientists study synaptic plasticity, Alzheimer's disease, and brain neuropathology.",
        "Vaccine development uses mRNA technology to stimulate immune responses against pathogens.",
        "Physicians diagnose diabetes, glucose metabolic regulation, and dietary insulin requirements.",
        "Hospital intensive care units track vital signs, arterial oxygen, and patient outcomes.",
        
        # Computer Science & AI
        "Convolutional neural networks and deep learning models classify images and visual features.",
        "Natural language processing uses transformer architectures for text summarization and search.",
        "Database indexing, SQL queries, and distributed storage systems optimize retrieval latency.",
        "Cybersecurity algorithms detect malware, encrypt network communication, and mitigate DDoS attacks.",
        "Compiler optimization and runtime execution speeds up memory allocation in software programs.",
        "Reinforcement learning agents optimize reward policies in complex simulated environments.",
        
        # Finance & Economics
        "Central banks adjust interest rates to control inflation and stimulate economic investment.",
        "Stock market volatility reflects equity earnings reports, dividend yields, and bond rates.",
        "Cryptocurrency and decentralized ledger technologies challenge traditional banking systems.",
        "Fiscal policy and government tax revenue impact national debt and employment trends.",
        "Portfolio diversification minimizes asset risk while maximizing expected annualized returns."
    ]
    
    topic_model = BERTopic(
        embedding_model=sentence_model, 
        min_topic_size=2, 
        verbose=False
    )
    topic_model.fit(docs)
    return sentence_model, topic_model

def main():
    st.title("Topic Modeling to Retrieve Information")
    st.write("Upload a document (.pdf, .docx, .txt) to discover its hidden topic and extract key themes.")

    with st.spinner("Initializing models and discovering latent topics..."):
        sentence_model, topic_model = load_and_train_model()

    # Sidebar: Discovered topics across the corpus
    st.sidebar.header("Corpus Topic Clusters")
    topic_info = topic_model.get_topic_info()
    st.sidebar.dataframe(topic_info[['Topic', 'Count', 'Name']].head(10))

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

            # Document Preview
            with st.expander("Document Text Preview"):
                st.text(text[:800] + ("..." if len(text) > 800 else ""))

            # Visualizations for presentation screenshots
            st.markdown("---")
            st.subheader("Topic Space Visualizations")
            fig = topic_model.visualize_topics()
            st.plotly_chart(fig, use_container_width=True)

            progress_bar.progress(100)
            status_text.text("Analysis complete!")

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            progress_bar.empty()
            status_text.empty()

if __name__ == "__main__":
    main()