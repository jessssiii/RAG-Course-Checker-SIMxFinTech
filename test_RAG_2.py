import streamlit as st
import pandas as pd
from langchain.schema import Document
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_openai import OpenAIEmbeddings, OpenAI
import os
    
# Load OpenAI key securely
if "openai" in st.secrets and "api_key" in st.secrets["openai"]:
    openai_key = st.secrets["openai"]["api_key"]
else:
    st.error("❌ OpenAI API key not found. Set it under 'Secrets' in Streamlit Cloud.")
    st.stop()    

# Streamlit UI setup
st.set_page_config(page_title="SIM/FinTech RAG Course Checker")
st.title("SIM/FinTech Course Checker")
st.write("Ask any question about which FinTech courses are accepted for SIM credit.")

query = st.text_area("Your question", placeholder="e.g. Is Introduction to Cryptography and Cybersecurity accepted?")

# Load and embed course data
@st.cache_resource
def setup_vectorstore():
    sim_df = pd.read_csv("sim_courses_clean.csv", sep=";")
    fintech_df = pd.read_csv("fintech_courses_clean.csv", sep=";")

    sim_df["source"] = "SIM"
    fintech_df["source"] = "FinTech"

    sim_df["text"] = sim_df.apply(
        lambda row: f"{row['event']} ({row['course_number']}, {row['lecturer']}, {row['classification']})", axis=1
    )
    fintech_df["text"] = fintech_df.apply(
        lambda row: f"{row['Name']} ({row['Number']}, {row['Lecturer']}, {row['Semester']}, {row['ECTS']} ECTS)", axis=1
    )

    full_df = pd.concat([sim_df, fintech_df])
    docs = [Document(page_content=row["text"], metadata={"source": row["source"]}) for _, row in full_df.iterrows()]

    embeddings = OpenAIEmbeddings(openai_api_key=openai_key)
    vectorstore = Chroma.from_documents(docs, embeddings)  #changed line here
    return vectorstore

if st.button("Submit") and query:
    vectorstore = setup_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    llm = OpenAI(openai_api_key=openai_key)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    
    with st.spinner("Searching and generating answer..."):
        response = qa_chain.run(query)
        st.markdown("### Answer:")
        st.markdown(response)
elif not query:
    st.info("Please enter a question.")
