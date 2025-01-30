import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize session state for PDF summary
if 'pdf_summary' not in st.session_state:
    st.session_state.pdf_summary = ""

def get_pdf_text(pdf_docs):
    text = ""
    metadata = []
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        # Extract metadata
        info = pdf_reader.metadata
        metadata.append({
            "filename": pdf.name,
            "pages": len(pdf_reader.pages),
            "author": info.get('/Author', 'Unknown')
        })
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text, metadata

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000,
        chunk_overlap=1000,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
    )
    chunks = text_splitter.split_text(text)
    return chunks

def generate_summary(text):
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    summary_prompt = """Please provide a concise summary of the following text, highlighting the main points and key takeaways:
    
    Text: {text}
    
    Summary:"""
    response = model.invoke(summary_prompt.format(text=text[:5000]))
    return response.content

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the following question based on the provided context. 
    If the answer cannot be found in the context, politely state that you cannot answer based on the given information.
    
    Context: {context}
    Question: {question}
    
    Answer:
    """
    
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    if os.path.exists("faiss_index/index.faiss"):
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)
        
        chain = get_conversational_chain()
        response = chain(
            {
                "input_documents": docs,
                "question": user_question
            },
            return_only_outputs=True
        )
        
        # Display the response in a clean format
        st.markdown("### Answer:")
        st.write(response["output_text"])
    else:
        st.error("Error: FAISS index file not found. Please process the PDF files first.")

def main():
    st.set_page_config(page_title="PDF Chat", layout="wide")
    st.header("Enhanced PDF Chat using Gemini")

    # Create two columns for better layout
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Document Upload")
        pdf_docs = st.file_uploader(
            "Upload your PDF Files",
            accept_multiple_files=True,
            help="Select one or multiple PDF files to process"
        )
        
        if st.button("Submit & Process", type="primary"):
            with st.spinner("Processing documents..."):
                # Process PDFs
                raw_text, metadata = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                
                # Generate and store summary
                st.session_state.pdf_summary = generate_summary(raw_text)
                
                # Display metadata
                st.subheader("Document Information")
                for doc in metadata:
                    st.write(f"📄 Filename: {doc['filename']}")
                    st.write(f"📑 Pages: {doc['pages']}")
                    st.write(f"✍️ Author: {doc['author']}")
                    st.write("---")
                
                st.success("Processing complete!")

    with col2:
        # Display document summary if available
        if st.session_state.pdf_summary:
            with st.expander("Document Summary"):
                st.write(st.session_state.pdf_summary)
        
        st.subheader("Ask Questions")
        user_question = st.text_input(
            "Enter your question",
            placeholder="Ask something about the uploaded documents...",
            help="Type your question and press Enter"
        )
        
        if user_question:
            user_input(user_question)

if __name__ == "__main__":
    main()