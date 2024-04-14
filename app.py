import cohere
import numpy as np
import pandas as pd
import pdfplumber
import streamlit as st
from io import StringIO
from typing import Sequence

co = cohere.Client('lxIvoNPeLLQy4d5QQq41zfro6076VpS2681QjziY')

def extractTextFromPdf(pdf_path: str):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    return text

def processTextInput(text: str, run_id: str = None):  
    text = StringIO(text).read()  
    CHUNK_SIZE=150
    chunks = [text[i:i + CHUNK_SIZE] for i in range(0, len(text), CHUNK_SIZE)]  

    df = pd.DataFrame.from_dict({'text': chunks}) 
    return df

def convertToList(df):
    df['col']=df[['text']].apply(lambda row: ' '.join(row.dropna().astype(str)), axis=1)
    seqOfStrings: Sequence[str]=df['col'].tolist()
    return seqOfStrings

def embed(Texts: Sequence[str]):
    res=co.embed(texts=Texts, model="small")
    return res.embeddings


options=st.selectbox("Input type", ["PDF","TEXT"])

if options=="PDF":
    pdf_file=st.file_uploader("Upload file", type=["pdf"])
    if pdf_file is not None:
        text=extractTextFromPdf(pdf_file)
    if text is not None:
        df=processTextInput(text)
elif options == "TEXT":  
    text = st.text_area("Paste the Document")  
    if text is not None:  
        df = processTextInput(text)

if text is not None:
    # st.write(embed(df))
    st.write(type(embed(convertToList(df))))

