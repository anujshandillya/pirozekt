import os
from io import StringIO
from typing import Sequence

import cohere
import numpy as np
import pandas as pd
import pdfplumber
import streamlit as st
from dotenv import load_dotenv
from numpy.linalg import norm

#  loaded local env
load_dotenv()

api=os.getenv('API_KEY')
base=os.getenv('BASE')
# default settings for generation of text
TEMPERATURE = 0.5
MAX_TOKENS = 200
text=""
result=None

co=cohere.Client(api)

# Get the text from pdf file.
def extractTextFromPdf(pdfPath: str):
    text = ""
    with pdfplumber.open(pdfPath) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    return text

# Creating a dataframe to break information into user defined Chunks.
def processTextInput(text: str, run_id: str = None):
    text = StringIO(text).read()
    CHUNK_SIZE=150
    chunks = [text[i:i + CHUNK_SIZE] for i in range(0, len(text), CHUNK_SIZE)]

    df = pd.DataFrame.from_dict({'text': chunks})
    return df

# Converting the dataframe to list of strings.
def convertToList(df):
    df['col']=df[['text']].apply(lambda row: ' '.join(row.dropna().astype(str)), axis=1)
    seqOfStrings: Sequence[str]=df['col'].tolist()
    return seqOfStrings

# Using the Cohere embed endpoint to embed the data into vector data.
def embed(Texts: Sequence[str]):
    res=co.embed(texts=Texts, model="small")
    return res.embeddings

# Finding K nearest neighbours to enhance the answer.
def topNNeighbours(promptEmbeddings: np.ndarray, storageEmbeddings: np.ndarray, df, k: int = 5):
	if isinstance(storageEmbeddings, list):
		storageEmbeddings = np.array(storageEmbeddings)
	if isinstance(promptEmbeddings, list):
		storageEmbeddings = np.array(promptEmbeddings)
	similarityMatrix = promptEmbeddings @ storageEmbeddings.T / np.outer(norm(promptEmbeddings, axis=-1), norm(storageEmbeddings, axis=-1))
	numNeighbours = min(similarityMatrix.shape[1], k)
	indices = np.argsort(similarityMatrix, axis=-1)[:, -numNeighbours:]
	listOfStr=df.values.tolist()
	neighbourValues:list=[]
	for idx in indices[0]:
		neighbourValues.append(listOfStr[idx])
	return neighbourValues

# Using the Cohere generate endpoint to return the answer into text data with additional options namely 'temperature' and 'max_tokens'.
def generate(promptt, tmp, maxTokens):
    res=co.generate(prompt=promptt, temperature=tmp, max_tokens=maxTokens)
    return res

# Using the streamlit library to create a user interface for better understanding.
options=st.selectbox("Input type", ["PDF","TEXT"])

if options=="PDF":
    pdfFile=st.file_uploader("Upload file", type=["pdf"])
    if pdfFile is not None:
        text=extractTextFromPdf(pdfFile)
    if text is not None:
        df=processTextInput(text)
elif options == "TEXT":
    text = st.text_area("Paste the Document")
    if text is not None:
        df = processTextInput(text)

if text!="":
    listOfText=convertToList(df)
    embeddings=embed(listOfText)

if df is not None:
    prompt=st.text_input('Ask a Question')
    advancedOpt=st.checkbox('Advanced Options')
    if advancedOpt is not None:
        TEMPERATURE=st.slider('Temperature', min_value=0.0, max_value=1.0, value=TEMPERATURE)
        MAX_TOKENS=st.slider('Max Tokens', min_value=1, max_value=5000, value=MAX_TOKENS)

if df is not None and prompt != "":
    basePrompt = base
    promptEmbeddings = embed([prompt])
    augPrompts = topNNeighbours(np.array(promptEmbeddings), embeddings, df)
    joinedPrompt = '\n'.join(str(neighbour) for neighbour in augPrompts) + '\n\n' + basePrompt + '\n' + prompt + '\n'
    result = generate(joinedPrompt,TEMPERATURE,MAX_TOKENS)

if result is not None:
    st.write(result.generations[0].text)
