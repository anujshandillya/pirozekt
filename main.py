import cohere
from io import StringIO
import numpy as np
import pandas as pd
import streamlit as st
import pdfplumber
from numpy.linalg import norm

co=cohere.Client('lxIvoNPeLLQy4d5QQq41zfro6076VpS2681QjziY')
CHUNK_SIZE=1024
TEMPERATURE = 0.5
MAX_TOKENS = 50

def extract_text_from_pdf(pdf_path: str):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    return text

def process_text_input(text: str, run_id: str = None):  
	text = StringIO(text).read()  
	chunks = [text[i:i + CHUNK_SIZE] for i in range(0, len(text), CHUNK_SIZE)]  
	df = pd.DataFrame.from_dict({'text': chunks}) 
	return df

def embed(list_of_texts: list):  
	response = co.embed(model='embed-english-v3.0', texts=list_of_texts, input_type="search_document")  
	return response.embeddings

def top_n_neighbors_indices(prompt_embedding: np.ndarray, storage_embeddings: np.ndarray, n: int):  
	if isinstance(storage_embeddings, list):  
		storage_embeddings = np.array(storage_embeddings)  
	if isinstance(prompt_embedding, list):  
		storage_embeddings = np.array(prompt_embedding)  
	similarity_matrix = prompt_embedding @ storage_embeddings.T / np.outer(norm(prompt_embedding, axis=-1), norm(storage_embeddings, axis=-1))  
	num_neighbors = min(similarity_matrix.shape[1], n)  
	indices = np.argsort(similarity_matrix, axis=-1)[:, -num_neighbors:]  
	return indices

def get_embeddings_from_df(df):
    texts=df['text'].tolist()
    embeddings=embed(texts)
    return np.array(embeddings)

def get_augmented_prompts(pr_embed: np.ndarray, storage_embed, dataFrame: list):
    storage_embedding=get_embeddings_from_df(dataFrame)
    indices=top_n_neighbors_indices(pr_embed,storage_embedding,5)
    prompt_texts = df.iloc[indices]['text'].tolist()
    augmented_prompt='\n'.join(prompt_texts)
    return augmented_prompt


option=st.selectbox("Input Type", ["PDF"])
df=None
if option=="PDF":
    pdf_file=st.file_uploader("Upload pdf file", type=["pdf"])
    text=extract_text_from_pdf(pdf_file)
    embeddings=None
    if text!="":
        df=process_text_input(text)

if df is not None:  
  prompt = st.text_input('Ask a question')  
  advanced_options = st.checkbox("Advanced options") 
  if advanced_options:  
    TEMPERATURE = st.slider('temperature', min_value=0.0, max_value=1.0, value=TEMPERATURE)  
    MAX_TOKENS = st.slider('max_tokens', min_value=1, max_value=1024, value=MAX_TOKENS)
    prompt_embedding = embed([prompt])
    embeddings = get_embeddings_from_df(df)

if df is not None and prompt != "":
    base_prompt = "Based on the passage above, answer the following question:"
    prompt_embedding = embed([prompt])
    prompt_embedding=np.array(prompt_embedding)
    # embeddings = get_embeddings_from_df(df)
    aug_prompts = get_augmented_prompts(prompt_embedding, embeddings, df)
    # prompt_texts = df.iloc[aug_prompts]['text'].tolist()
    # augmented_prompt='\n'.join(prompt_texts)
    new_prompt = '\n'.join(aug_prompts) + '\n\n' + base_prompt + '\n' + prompt + '\n'
    print(new_prompt)
    is_success = False
    while not is_success:
        try:
            response = co.generate(new_prompt)
            is_success = True
        except Exception:
            aug_prompts = aug_prompts[:-1]
            new_prompt = '\n'.join(aug_prompts) + '\n' + base_prompt + '\n' + prompt  + '\n'

    st.write(response.generations[0].text)
