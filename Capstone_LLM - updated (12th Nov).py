#!/usr/bin/env python
# coding: utf-8

# ## Setting Up the Environment

# In[13]:


import os
import pandas as pd
import matplotlib.pyplot as plt
from transformers import GPT2TokenizerFast
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain


# In[14]:


os.environ["OPENAI_API_KEY"] = "{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}"


# ## 1. Loading PDFs and chunking with LangChain

# In[ ]:


# Initialling


# In[15]:


#This basic example demostrate the LLM response and ChatModel Response

from langchain.llms import AzureOpenAI
from langchain.chat_models import AzureChatOpenAI
import openai
import os
from dotenv import load_dotenv, find_dotenv

#app.py

from langchain import PromptTemplate


# In[16]:


# Load environment variables from the .env file
load_dotenv(find_dotenv())

# Retrieve Azure OpenAI specific configuration from environment variables
OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
OPENAI_API_TYPE = "Azure"
OPENAI_API_BASE = "https://testopenaisaturday.openai.azure.com/"
OPENAI_API_VERSION = "2023-10-01-preview"

print(OPENAI_API_KEY, OPENAI_API_TYPE,OPENAI_API_BASE, OPENAI_API_VERSION)


# In[17]:


# Set the OpenAI library configuration using the retrieved environment variables
openai.api_type = "Azure"
openai.api_base = "https://testopenaisaturday.openai.azure.com/"
openai.api_version = "2023-10-01-preview"
openai.api_key = OPENAI_API_KEY


# In[18]:


#app.py

from langchain import PromptTemplate
import openai
from dotenv import load_dotenv, find_dotenv
import os
from langchain.chat_models import AzureChatOpenAI
from dotenv import find_dotenv, load_dotenv


# In[8]:


get_ipython().system('pip uninstall pydantic dataclasses-json joblib langchain unstructured-client -y')


# In[12]:


# Install compatible versions
get_ipython().system('pip install "pydantic<2,>=1"')
get_ipython().system('pip install "dataclasses-json<0.7.0,>=0.6.7"')
get_ipython().system('pip install "joblib~=1.1.0"')
get_ipython().system('pip install "langchain<0.4.0,>=0.3.6"')
get_ipython().system('pip install unstructured unstructured-client')


# In[19]:


from test_unstructured.unit_utils import assert_round_trips_through_JSON, example_doc_path
from unstructured.chunking.title import chunk_by_title
from unstructured.documents.elements import ElementType

#!pip install unstructured_inference
#!pip install -U langchain-unstructured


# ## PINECONE Initialising

# In[14]:


get_ipython().run_line_magic('pip', 'install -qU langchain-pinecone pinecone-notebooks')


# In[1]:


# PINECONE API.     1203aeba-36cc-4ede-9dc6-1f01153fbde8


# In[20]:


import getpass
import os
import time

from pinecone import Pinecone, ServerlessSpec

if not os.getenv("PINECONE_API_KEY"):
    os.environ["PINECONE_API_KEY"] = getpass.getpass("Enter your Pinecone API key: ")

pinecone_api_key = os.environ.get("PINECONE_API_KEY")

pc = Pinecone(api_key=pinecone_api_key)


# In[3]:


#Pinecone API key - pcsk_wpzoJ_CXs3QWXo2Q6BGnx84d4kyRVn7kDw5N9SvCEJXEWNehAafEiWvtHoW9X2qLdRThh


# In[ ]:


#Old key(rkesh@uic.edu)Openai API key - 
#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


# In[21]:


import time

index_name = "langchain-test-index"  # change if desired

existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

if index_name not in existing_indexes:
    pc.create_index(
        name=index_name,
        dimension=3072,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    while not pc.describe_index(index_name).status["ready"]:
        time.sleep(1)

index = pc.Index(index_name)


# ## Open AI Initialise

# In[5]:


pip install langchain_openai


# In[22]:


import getpass
import os  # Import os module

# Set the OpenAI API key using getpass
os.environ["OPENAI_API_KEY"] = getpass.getpass()

from langchain_openai import OpenAIEmbeddings

# Instantiate OpenAI embeddings model
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")


# In[ ]:


#Openai API key (gkramyashree@gmail.com):
#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


# In[4]:


get_ipython().system('pip install pinecone-client')


# In[23]:


from langchain.embeddings.openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")


# ## Text CHUNKING (semantic Chunking)

# In[24]:


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# In[10]:


pip install PyPDF2


# In[25]:


import PyPDF2
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document  # Import the Document class


# In[26]:


import PyPDF2
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document  # Import the Document class

# Step 1: Extract text from the PDF using PyPDF2
with open('s3-outposts.pdf', 'rb') as file:
    reader = PyPDF2.PdfReader(file)
    

    # Iterate through all the pages and extract text
    text = ''
    for page_num in range(len(reader.pages)):
        page = reader.pages[page_num]
        extracted_text = page.extract_text()
        if extracted_text:  # Ensure the text extraction is successful
            text += extracted_text

# Step 2: Create a Document object
document = Document(page_content=text)

from langchain.text_splitter import RecursiveCharacterTextSplitter

# Step 3: Use RecursiveCharacterTextSplitter for better splitting
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=200,
    separators=["\n\n", "\n", " "]
)

# Split the document text into smaller chunks
texts = text_splitter.split_documents([document])  # Pass in a list of Document objects

# Step 4: Print the number of chunks
print(f"Number of chunks: {len(texts)}")

# Step 5: Optionally, output the first few chunks to inspect the splitting
for i, chunk in enumerate(texts[:3]):  # Limit to first 3 chunks for display
    print(f"Chunk {i+1}:")
    print(chunk.page_content[:500])  # Print first 500 characters of the chunk
    print("\n" + "-"*80 + "\n")


# In[27]:


# Step 4: Output only the first 2-3 split text chunks with truncation to avoid large data output
for chunk in texts[125:127]:  # Display the first 3 chunks
    print(chunk.page_content[:500])  # Print only the first 500 characters of each chunk
    print("\n" + "-"*80 + "\n")


# ## Semantic Chunking

# 1. Install Necessary Libraries:

# In[6]:


get_ipython().system('pip install sentence-transformers scikit-learn')


# ## Revised Code for Semantic Chunking (for further improving the chunking better) :

# In[3]:


pip uninstall numpy -y


# In[4]:


pip install numpy==1.24.3


# In[2]:


pip uninstall scikit-learn -y


# In[3]:


pip install scikit-learn


# 1. Extract Text and Split into Paragraphs:

# In[28]:


import PyPDF2

# Extract text from PDF using PyPDF2
with open('s3-outposts.pdf', 'rb') as file:
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page_num in range(len(reader.pages)):
        page = reader.pages[page_num]
        extracted_text = page.extract_text()
        if extracted_text:  # Ensure text extraction was successful
            text += extracted_text

# Split text into paragraphs (using a simple heuristic here; you could use more advanced tools like spaCy)
paragraphs = [p.strip() for p in text.split('\n') if p.strip()]


# 2. Generate Paragraph Embeddings: 

# In[29]:


from sentence_transformers import SentenceTransformer

# Load a pre-trained model for generating embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate embeddings for each paragraph
paragraph_embeddings = model.encode(paragraphs)  # Removed convert_to_tensor=True to keep it as a NumPy array


# 3. Proceed with Clustering:

# In[30]:


import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering

# Compute Cosine Distance Matrix
cosine_sim_matrix = cosine_similarity(paragraph_embeddings)
cosine_dist_matrix = 1 - cosine_sim_matrix  # Cosine distance is (1 - cosine similarity)

# Cluster Paragraphs with Adjusted Parameters using Cosine Distance
clustering = AgglomerativeClustering(
    n_clusters=None,
    metric='precomputed',  # Use precomputed distance matrix
    linkage='average',
    distance_threshold=0.75
)
clustering.fit(cosine_dist_matrix)

# Group Paragraphs Based on Clustering Results
clusters = {}
for idx, label in enumerate(clustering.labels_):
    if label not in clusters:
        clusters[label] = []
    clusters[label].append(paragraphs[idx])

# Create Semantic Chunks from Clusters
semantic_chunks = [" ".join(cluster) for cluster in clusters.values()]

# Inspect the First Few Semantic Chunks
for i, chunk in enumerate(semantic_chunks[:3]):
    print(f"Chunk {i+1}:")
    print(chunk[:500])  # Print the first 500 characters of each chunk
    print("\n" + "-"*80 + "\n")


# Test - Extracting Middle Chunks:

# In[31]:


# Get the total number of chunks
total_chunks = len(semantic_chunks)

# Define the middle range to inspect, for example, extract 3 chunks from the middle
middle_start = total_chunks // 2 - 1  # Start from the middle, adjust -1 for indexing
middle_end = min(middle_start + 3, total_chunks)  # Extract 3 chunks or up to the last chunk if less available

# Extract and inspect the middle chunks
for i, chunk in enumerate(semantic_chunks[middle_start:middle_end], start=1):
    print(f"Middle Chunk {i}:")
    print(chunk[:500])  # Print the first 500 characters of each chunk to avoid excessive output
    print("\n" + "-" * 80 + "\n")


# ## NLP Chunking method (not semantic)

# In[32]:


# Advanced method - Split by chunk for S3 Outposts PDF

# Step 1: Convert PDF to text
import PyPDF2
import pandas as pd
import matplotlib.pyplot as plt
from transformers import GPT2TokenizerFast
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Extract text from PDF using PyPDF2
with open('./s3-outposts.pdf', 'rb') as file:
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page_num in range(len(reader.pages)):
        page = reader.pages[page_num]
        extracted_text = page.extract_text()
        if extracted_text:
            text += extracted_text

# Step 2: Save to .txt and reopen (helps prevent issues)
with open('s3-outposts.txt', 'w') as f:
    f.write(text)

with open('s3-outposts.txt', 'r') as f:
    text = f.read()

# Step 3: Create function to count tokens
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text))

# Step 4: Split text into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=24,
    length_function=count_tokens,
)

chunks = text_splitter.create_documents([text])

# Step 5: Quick data visualization to ensure chunking was successful
# Create a list of token counts
token_counts = [count_tokens(chunk.page_content) for chunk in chunks]

# Create a DataFrame from the token counts
df = pd.DataFrame({'Token Count': token_counts})

# Create a histogram of the token count distribution
df.hist(bins=40)

# Show the plot
plt.xlabel("Token Count")
plt.ylabel("Frequency")
plt.title("Histogram of Token Count Distribution")
plt.show()


# In[33]:


# Step 6: Inspect a couple of chunks
for i, chunk in enumerate(chunks[:2]):  # Display the first 2 chunks
    print(f"Chunk {i+1}:")
    print(chunk.page_content[:500])  # Print the first 500 characters of each chunk
    print("\n" + "-"*80 + "\n")


# In[34]:


# Step 6: Inspect middle chunks
middle_index = len(chunks) // 2
for i in range(middle_index - 1, middle_index + 1):  # Display two middle chunks
    print(f"Chunk {i+1}:")
    print(chunks[i].page_content[:500])  # Print the first 500 characters of each chunk
    print("\n" + "-"*80 + "\n")


# ## Downloading the chunks in json file

# In[35]:


# will write/run later
import json

# Assuming `chunks` is the list of chunked data you already have.
# Convert chunks to a list of dictionaries for JSON serialization
chunk_data = [{"chunk": chunk.page_content} for chunk in chunks]

# Save the chunks to a JSON file
with open("chunked_data.json", "w") as json_file:
    json.dump(chunk_data, json_file, indent=4)

print("Chunked data saved as chunked_data.json")


# ## Embed text and store embeddings

# This code is designed to generate better-quality embeddings, making it suitable for tasks like semantic search, document clustering, or question answering systems.

# In[36]:


from typing import Union
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import tiktoken  # Tokenizer library
import PyPDF2

# Configuration for the embedding model
# I have updated the model to a more advanced version for better embeddings.
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
DIMENSIONALITY = 768  # Updated dimensionality according to the new model
THRESHOLD = 0.8  # Threshold for semantic similarity splitting
MAX_TOKENS = 20000  # Max tokens per request
MAX_INSTANCES = 250  # Max instances per request (sentences)

# Load the Hugging Face model and tokenizer
# The model here is an updated version of the previous one; it gives better embeddings.
model = AutoModel.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Example usage
# The code below is just a placeholder to show how you might use the model and tokenizer later in your code.

def get_embeddings(text: str):
    """ Get embeddings for the given text using the model and tokenizer. """
    tokens = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        model_output = model(**tokens)
    # Depending on the model, we often use the mean of the last hidden state as the embedding
    embeddings = model_output.last_hidden_state.mean(dim=1)
    return embeddings

# Example text
example_text = "Amazon S3 on Outposts helps you extend S3 storage to your on-premises environments."
embedding = get_embeddings(example_text)
print(embedding)


# ## Generating metadata for text chunks using a LLM

# Function 1: generate_metadata_for_chunks(llm, chunks, max_tokens=2048)

# In[38]:


import json
import torch
from transformers import T5ForConditionalGeneration, AutoTokenizer

# Load T5 model and tokenizer
MODEL_NAME = "t5-small"
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Load the chunks from a saved JSON file
with open("chunked_data.json", "r") as file:
    chunk_data = json.load(file)

# Extract the chunk text from the loaded JSON data
chunks = [item['chunk'] for item in chunk_data]  # Correct key is 'chunk', not 'chunk_text'

# Function to generate and save metadata
def generate_and_save_metadata(model, tokenizer, chunks, output_file_path, max_tokens=2048):
    results = []
    for i, chunk in enumerate(chunks):
        # Truncate the chunk if it exceeds the max_tokens limit
        truncated_chunk = chunk[:max_tokens]

        # Define the prompt for metadata generation
        prompt = f"""
        The following text is a chunk from a larger document. Please generate the 2-3 most relevant topic headings and tags for this text:
        Text: "{truncated_chunk}"
        """

        # Tokenize and prepare input for the model
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True, max_length=max_tokens)

        # Use the model to generate content based on the prompt
        with torch.no_grad():
            output = model.generate(**inputs, max_length=max_tokens)
            response = tokenizer.decode(output[0], skip_special_tokens=True)

        # Store the response metadata (topic headings and tags)
        results.append({
            "chunk_index": i,
            "chunk_text": truncated_chunk,
            "metadata": response
        })

    # Save the results to a JSON file
    with open(output_file_path, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"Metadata saved to {output_file_path}")
    return results

# Use the function to generate metadata and save it
metadata_results = generate_and_save_metadata(model, tokenizer, chunks, "metadata_results_outposts.json")


# ## Metadata gen on chunks - sample to check

# In[39]:


import json

# Load the metadata results from the JSON file
with open("metadata_results_outposts.json", "r") as file:
    metadata_results = json.load(file)

# Function to print a couple of results to check
def print_metadata_results(metadata_results, num_results=3):
    for i, result in enumerate(metadata_results[:num_results]):
        print(f"Chunk Index: {result['chunk_index']}")
        print("Chunk Text (truncated):")
        print(result['chunk_text'][:500])  # Display only the first 500 characters of the chunk text
        print("\nGenerated Metadata:")
        print(result['metadata'])
        print("\n" + "-" * 80 + "\n")

# Print a few metadata results to verify
print_metadata_results(metadata_results)


# ## -- Updated Code for Metadata gen:

# In[40]:


import json
import torch
from transformers import T5ForConditionalGeneration, AutoTokenizer

# Load a larger and more powerful model
MODEL_NAME = "google/flan-t5-large"
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Function to generate and save metadata for each chunk
def generate_and_save_metadata(model, tokenizer, chunks, output_file_path, max_tokens=1024):
    results = []
    for i, chunk in enumerate(chunks):
        # Truncate the chunk if it exceeds the max_tokens limit
        truncated_chunk = chunk[:max_tokens]

        # Refine the prompt for better results
        prompt = f"""
        You are an AI assistant. Your task is to generate concise, meaningful topic headings and tags.
        Below is a chunk of text from a larger document. Please generate the 2-3 most relevant topic headings and tags that accurately describe the content of this text.
        
        Text: "{truncated_chunk}"

        Please return your response in the format:
        - Headings: [heading1, heading2, ...]
        - Tags: [tag1, tag2, ...]
        """

        # Tokenize and prepare input for the model
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True, max_length=max_tokens)

        # Use the model to generate content based on the prompt
        with torch.no_grad():
            output = model.generate(**inputs, max_length=max_tokens)
            response = tokenizer.decode(output[0], skip_special_tokens=True)

        # Store the response metadata (topic headings and tags)
        results.append({
            "chunk_index": i,
            "chunk_text": truncated_chunk,
            "metadata": response
        })

        # Print progress for every 10 chunks
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(chunks)} chunks...")

    # Save the results to a JSON file
    with open(output_file_path, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"Metadata saved to {output_file_path}")
    return results

# Assuming `chunks` is the list of chunked data you already have.
# Use the function to generate metadata and save it
metadata_results = generate_and_save_metadata(model, tokenizer, chunks, "metadata_results_outposts_improved.json")

# Function to print a couple of results to check the generated metadata
def print_metadata_results(metadata_results, num_results=3):
    for i, result in enumerate(metadata_results[:num_results]):
        print(f"Chunk Index: {result['chunk_index']}")
        print("Chunk Text (truncated):")
        print(result['chunk_text'][:500])  # Display only the first 500 characters of the chunk text
        print("\nGenerated Metadata:")
        print(result['metadata'])
        print("\n" + "-" * 80 + "\n")


# In[41]:


# Print a few metadata results to verify the improvement
print_metadata_results(metadata_results)


# ## still improvising code for Metadata Gen:

# In[43]:


import json
import torch
from transformers import T5ForConditionalGeneration, AutoTokenizer

# Load the model and tokenizer
MODEL_NAME = "google/flan-t5-large"  # Consider using a stronger model if available
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Function to generate and save metadata for each chunk
def generate_and_save_metadata(model, tokenizer, chunks, output_file_path, max_tokens=1024):
    results = []
    for i, chunk in enumerate(chunks):
        truncated_chunk = chunk[:max_tokens]

        # Refined prompt with an explicit example and instructions
        prompt = f"""
        You are an AI assistant. Your task is to generate concise, meaningful topic headings and tags.
        Below is a chunk of text from a larger document. Please generate the 2-3 most relevant topic headings and tags that accurately describe the content of this text.
        
        Example Output:
        - Headings: [Amazon S3 Bucket Policies, Creating S3 Endpoints]
        - Tags: [S3 Outposts, Bucket Policies, Endpoints]

        Text: "{truncated_chunk}"

        Please return your response in the format:
        - Headings: [heading1, heading2, ...]
        - Tags: [tag1, tag2, ...]
        """

        # Tokenize and prepare input for the model
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True, max_length=max_tokens)

        # Use the model to generate content based on the prompt
        with torch.no_grad():
            output = model.generate(**inputs, max_length=max_tokens)
            response = tokenizer.decode(output[0], skip_special_tokens=True)

        # Store the response metadata (topic headings and tags)
        results.append({
            "chunk_index": i,
            "chunk_text": truncated_chunk,
            "metadata": response
        })

        # Print progress for every 10 chunks
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(chunks)} chunks...")

    # Save the results to a JSON file
    with open(output_file_path, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"Metadata saved to {output_file_path}")
    return results

# Use the function to generate metadata and save it
metadata_results = generate_and_save_metadata(model, tokenizer, chunks, "metadata_results_outposts_refined_v2.json")

# Function to print a couple of results to check the generated metadata
def print_metadata_results(metadata_results, num_results=3):
    for i, result in enumerate(metadata_results[:num_results]):
        print(f"Chunk Index: {result['chunk_index']}")
        print("Chunk Text (truncated):")
        print(result['chunk_text'][:500])  # Display only the first 500 characters of the chunk text
        print("\nGenerated Metadata:")
        print(result['metadata'])
        print("\n" + "-" * 80 + "\n")

# Print a few metadata results to verify the improvement
print_metadata_results(metadata_results)


# In[44]:


## Okay Metadata Gen has some issues, so lets take different approach.


# # Approach 1

# Approach 1: Group Chunks with Common Tags (Clustering Similar Content)
# Instead of relying solely on metadata generation, we can try clustering or grouping chunks that share common themes or concepts. This can create more meaningful categories and help in organizing the content. Here’s a potential approach:
# 
# Embed the Chunks: Use a pre-trained embedding model (e.g., sentence-transformers) to create vector embeddings for each chunk.
# Cluster the Embeddings: Apply a clustering algorithm, such as K-means or Agglomerative Clustering, to group similar chunks together.
# Assign Tags: For each cluster, we can either manually assign a tag or leverage the LLM to generate a summary for that cluster, which will be more informative because the cluster contains multiple related chunks.

# In[46]:


from sklearn.cluster import AgglomerativeClustering
from sentence_transformers import SentenceTransformer

# Step 1: Generate embeddings for each chunk using SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
chunk_embeddings = model.encode(chunks)

# Step 2: Use Agglomerative Clustering to group similar chunks
# Updated: Replace 'affinity' with 'metric' as per the newer scikit-learn version
clustering_model = AgglomerativeClustering(
    n_clusters=None, 
    distance_threshold=1.5, 
    metric='euclidean',  # Updated from 'affinity' to 'metric'
    linkage='ward'
)
labels = clustering_model.fit_predict(chunk_embeddings)

# Step 3: Organize chunks into clusters based on labels
clusters = {}
for idx, label in enumerate(labels):
    if label not in clusters:
        clusters[label] = []
    clusters[label].append(chunks[idx])

# Step 4: Print the clusters to verify
for cluster_id, cluster_chunks in clusters.items():
    print(f"Cluster {cluster_id} contains {len(cluster_chunks)} chunks.")


# ## Approach 2

# Approach 2: Hierarchical Indexing
# Hierarchical Indexing involves creating a structured way to navigate through the document's content. Instead of focusing on generating accurate tags, we create a table of contents and hierarchical layers that help navigate through different levels of information. Here's how we could do it:
# 
# High-Level Summarization: Use an LLM to generate a general summary of the entire document first. This becomes the top-level hierarchy (e.g., main sections).
# Divide and Summarize: After that, we dive deeper into each section (e.g., clusters or groups of related chunks) and generate a sub-summary. This creates a tree-like structure.
# Create Indexing: Each chunk or summary is indexed hierarchically — for example:
# Chapter 1: Introduction
# Section 1.1: S3 on Outposts Overview
# Section 1.2: Bucket Creation and Management
# Chapter 2: Advanced Features
# Section 2.1: Access Points and Policies
# etc.

# In[48]:


from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

# Load T5 model and tokenizer
MODEL_NAME = "t5-small"  # You can also try "t5-base" or "t5-large" for better results
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)

# Your document summary prompt
document_summary_prompt = "summarize: User Guide for Amazon S3 on Outposts. This document explains the concepts, features, and setup instructions for using Amazon S3 on AWS Outposts, including how to create buckets, manage access points, and interact with the API."

# Tokenize the input text
inputs = tokenizer(document_summary_prompt, return_tensors="pt", truncation=True, padding=True)

# Generate summary using the model
with torch.no_grad():
    output = model.generate(**inputs, max_length=100)

# Decode the output to get the summary
document_summary = tokenizer.decode(output[0], skip_special_tokens=True)

# Print the summary
print("Document Summary:", document_summary)


# ## Approach 3: Keyword Extraction and Tagging

# Approach 3: Keyword Extraction and Tagging
# Instead of relying on LLMs entirely for generating metadata, you can also try keyword extraction to create more meaningful tags:
# 
# Use NLP Libraries: Libraries like RAKE, spaCy, or KeyBERT can extract keywords from each chunk.
# Group Related Keywords: Post-process these keywords to group similar concepts and create clusters of tags.
# Assign Tags to Chunks: Use these keyword groups to tag chunks more effectively.
# This approach can be especially effective when the LLM struggles to understand technical terms or repetitive content.

# In[50]:


get_ipython().system('pip install keybert')


# In[51]:


from keybert import KeyBERT

# Initialize KeyBERT for keyword extraction
kw_model = KeyBERT()

# Step 1: Extract keywords for each chunk
chunk_keywords = {}
for i, chunk in enumerate(chunks):
    keywords = kw_model.extract_keywords(chunk, keyphrase_ngram_range=(1, 2), stop_words='english', top_n=5)
    chunk_keywords[i] = [kw[0] for kw in keywords]

# Step 2: Group similar keywords or tag chunks
for chunk_id, keywords in chunk_keywords.items():
    print(f"Chunk {chunk_id} Keywords: {keywords}")


# ## Sanity check of Chunks

# In[52]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Calculate the average length of chunks (in characters)
chunk_lengths = [len(chunk) for chunk in chunks]
average_length = np.mean(chunk_lengths)
print(f"Average length of chunks (in characters): {average_length:.2f}")

# Step 2: Calculate the average number of tokens per chunk using the tokenizer
tokenizer = AutoTokenizer.from_pretrained("t5-small")  # Or any other tokenizer you've been using
chunk_token_counts = [len(tokenizer.encode(chunk, truncation=True)) for chunk in chunks]
average_token_count = np.mean(chunk_token_counts)
print(f"Average number of tokens per chunk: {average_token_count:.2f}")

# Step 3: Create a DataFrame from chunk lengths for analysis
df_chunks = pd.DataFrame({'Chunk Length': chunk_lengths, 'Token Count': chunk_token_counts})

# Step 4: Plot histograms of chunk length and token count distributions
plt.figure(figsize=(12, 6))

# Histogram for chunk lengths
plt.subplot(1, 2, 1)
plt.hist(df_chunks['Chunk Length'], bins=30, color='skyblue', edgecolor='black')
plt.title("Distribution of Chunk Lengths (Characters)")
plt.xlabel("Length (Characters)")
plt.ylabel("Frequency")

# Histogram for token counts
plt.subplot(1, 2, 2)
plt.hist(df_chunks['Token Count'], bins=30, color='lightgreen', edgecolor='black')
plt.title("Distribution of Token Counts")
plt.xlabel("Token Count")
plt.ylabel("Frequency")

plt.tight_layout()
plt.show()

# Step 5: Analyze keyword frequency across chunks
from collections import Counter

all_keywords = [kw for kws in chunk_keywords.values() for kw in kws]
keyword_frequency = Counter(all_keywords)

print("\nTop 10 Most Frequent Keywords Across All Chunks:")
for kw, freq in keyword_frequency.most_common(10):
    print(f"{kw}: {freq}")

# Step 6: Print statistics summary
print("\nSummary Statistics:")
print(df_chunks.describe())


# ## Chunking by index(contents of PDF)

# In[1]:


import PyPDF2
import re
import json

def extract_text_from_pdf(pdf_path):
    """Extracts text from each page of the PDF."""
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = [page.extract_text() for page in reader.pages]
    return text

def parse_index(index_text):
    """Parses the index to identify section titles and page numbers."""
    index_pattern = re.compile(r'^(.*)\s+(\d+)$')
    sections = []
    
    for line in index_text.splitlines():
        match = index_pattern.match(line)
        if match:
            section_title, page_number = match.groups()
            sections.append({
                "title": section_title.strip(),
                "start_page": int(page_number)
            })
    
    return sections

def create_chunks(text_by_page, sections):
    """Creates chunks by mapping sections from the index to content."""
    chunks = []
    
    for i, section in enumerate(sections):
        start_page = section['start_page'] - 1
        end_page = sections[i + 1]['start_page'] - 1 if i + 1 < len(sections) else len(text_by_page)
        content = " ".join(text_by_page[start_page:end_page])
        chunks.append({
            "title": section['title'],
            "content": content
        })
    
    return chunks


# ### Replace index for each pdf

# In[55]:


index_text = """

What is S3 on Outposts? ................................................................................................................. 1
How S3 on Outposts works ....................................................................................................................... 1
Regions ...................................................................................................................................................... 2
Buckets ....................................................................................................................................................... 2
Objects ....................................................................................................................................................... 3
Keys ............................................................................................................................................................ 3
S3 Versioning ........................................................................................................................................... 4
Version ID .................................................................................................................................................. 4
Storage class and encryption ............................................................................................................... 4
Bucket policy ............................................................................................................................................ 4
S3 on Outposts access points .............................................................................................................. 5
Features of S3 on Outposts ....................................................................................................................... 5
Access management ............................................................................................................................... 5
Storage logging and monitoring ......................................................................................................... 6
Strong consistency .................................................................................................................................. 6
Related services ............................................................................................................................................ 7
Accessing S3 on Outposts .......................................................................................................................... 7
AWS Management Console ................................................................................................................... 7
AWS Command Line Interface ............................................................................................................. 7
AWS SDKs ................................................................................................................................................. 8
Paying for S3 on Outposts ......................................................................................................................... 8
Next steps ...................................................................................................................................................... 8
Setting up your Outpost ............................................................................................................... 10
Order a new Outpost ................................................................................................................................ 10
How S3 on Outposts is diﬀerent .................................................................................................. 11
Specifications .............................................................................................................................................. 11
Supported API operations ........................................................................................................................ 12
Unsupported Amazon S3 features ......................................................................................................... 12
Network restrictions .................................................................................................................................. 13
Getting started with S3 on Outposts ........................................................................................... 14
Using the S3 console ................................................................................................................................ 14
Create a bucket, an access point, and an endpoint ....................................................................... 15
Next steps ............................................................................................................................................... 17
Using the AWS CLI and SDK for Java .................................................................................................... 17
API Version 2006-03-01 iii
Amazon S3 on Outposts User Guide
Step 1: Create a bucket ....................................................................................................................... 18
Step 2: Create an access point ........................................................................................................... 19
Step 3: Create an endpoint ................................................................................................................ 20
Step 4: Upload an object to an S3 on Outposts bucket ............................................................... 22
Networking for S3 on Outposts ................................................................................................... 23
Choosing your networking access type ................................................................................................. 23
Accessing your S3 on Outposts buckets and objects ......................................................................... 23
Managing connections using cross-account elastic network interfaces .......................................... 24
Working with S3 on Outposts buckets ........................................................................................ 25
Buckets ......................................................................................................................................................... 25
Access points ............................................................................................................................................... 25
Endpoints ..................................................................................................................................................... 26
API operations on S3 on Outposts ......................................................................................................... 26
Creating and managing S3 on Outposts buckets ................................................................................ 28
Creating a bucket ....................................................................................................................................... 28
Adding tags ................................................................................................................................................. 32
Using bucket policies ................................................................................................................................. 33
Adding a bucket policy ........................................................................................................................ 34
Viewing a bucket policy ...................................................................................................................... 36
Deleting a bucket policy ..................................................................................................................... 37
Bucket policy examples ....................................................................................................................... 38
Listing buckets ............................................................................................................................................ 42
Getting a bucket ......................................................................................................................................... 43
Deleting your bucket ................................................................................................................................. 45
Working with access points ..................................................................................................................... 46
Creating an access point ..................................................................................................................... 47
Using a bucket-style alias for your access point ............................................................................ 48
Viewing access point configuration .................................................................................................. 53
Listing access points ............................................................................................................................. 54
Deleting an access point ..................................................................................................................... 55
Adding an access point policy ............................................................................................................ 56
Viewing an access point policy .......................................................................................................... 58
Working with endpoints ........................................................................................................................... 59
Creating an endpoint ........................................................................................................................... 60
Listing endpoints .................................................................................................................................. 62
Deleting an endpoint ........................................................................................................................... 64
API Version 2006-03-01 iv
Amazon S3 on Outposts User Guide
Working with S3 on Outposts objects ......................................................................................... 66
Upload an object ........................................................................................................................................ 67
Copying an object ...................................................................................................................................... 69
Using the AWS SDK for Java .............................................................................................................. 70
Getting an object ....................................................................................................................................... 71
Listing objects ............................................................................................................................................. 74
Deleting objects .......................................................................................................................................... 77
Using HeadBucket ...................................................................................................................................... 81
Performing a multipart upload ............................................................................................................... 83
Perform a multipart upload of an object in an S3 on Outposts bucket .................................... 84
Copy a large object in an S3 on Outposts bucket by using multipart upload .......................... 86
List parts of an object in an S3 on Outposts bucket .................................................................... 88
Retrieve a list of in-progress multipart uploads in an S3 on Outposts bucket ......................... 90
Using presigned URLs ............................................................................................................................... 91
Limiting presigned URL capabilities .................................................................................................. 91
Who can create a presigned URL ...................................................................................................... 93
When does S3 on Outposts check the expiration date and time of a presigned URL? ............ 94
Sharing objects ...................................................................................................................................... 94
Uploading an object ............................................................................................................................. 99
Amazon S3 on Outposts with local Amazon EMR ............................................................................ 104
Creating an Amazon S3 on Outposts bucket ............................................................................... 105
Getting started using Amazon EMR with Amazon S3 on Outposts ......................................... 106
Authorization and authentication caching ......................................................................................... 111
Configuring the authorization and authentication cache .......................................................... 112
Validating SigV4A signing ................................................................................................................ 112
Security ........................................................................................................................................ 113
Setting up IAM ......................................................................................................................................... 113
Principals for S3 on Outposts policies ........................................................................................... 116
ARNs for S3 on Outposts ................................................................................................................. 116
Example policies for S3 on Outposts ............................................................................................. 118
Permissions for endpoints ................................................................................................................ 119
Service-linked roles for S3 on Outposts ........................................................................................ 121
Data encryption ........................................................................................................................................ 121
AWS PrivateLink for S3 on Outposts .................................................................................................. 122
Restrictions and limitations .............................................................................................................. 123
Accessing S3 on Outposts interface endpoints ............................................................................ 124
API Version 2006-03-01 v
Amazon S3 on Outposts User Guide
Updating an on-premises DNS configuration ............................................................................... 126
Creating a VPC endpoint .................................................................................................................. 126
Creating VPC endpoint policies and bucket policies ................................................................... 126
Signature Version 4 (SigV4) policy keys ............................................................................................. 128
Bucket policy examples that use Signature Version 4-related condition keys ....................... 130
AWS managed policies ........................................................................................................................... 132
AWSS3OnOutpostsServiceRolePolicy ............................................................................................. 132
Policy updates ..................................................................................................................................... 133
Using service-linked roles ...................................................................................................................... 133
Service-linked role permissions for S3 on Outposts ................................................................... 134
Creating a service-linked role for S3 on Outposts ...................................................................... 137
Editing a service-linked role for S3 on Outposts ......................................................................... 137
Deleting a service-linked role for S3 on Outposts ...................................................................... 137
Supported Regions for S3 on Outposts service-linked roles ..................................................... 138
Managing S3 on Outposts storage ............................................................................................. 139
Managing S3 Versioning ......................................................................................................................... 139
Creating and managing a lifecycle configuration ............................................................................. 141
Using the console ............................................................................................................................... 142
Using the AWS CLI and SDK for Java ............................................................................................. 145
Replicating objects for S3 on Outposts .............................................................................................. 149
Replication configuration .................................................................................................................. 150
Requirements for S3 Replication on Outposts ............................................................................. 151
What is replicated? ............................................................................................................................. 152
What isn't replicated? ........................................................................................................................ 152
What isn't supported by S3 Replication on Outposts? ............................................................... 153
Setting up replication ........................................................................................................................ 153
Managing your replication ................................................................................................................ 172
Sharing S3 on Outposts ......................................................................................................................... 180
Prerequisites ........................................................................................................................................ 180
Procedure .............................................................................................................................................. 181
Usage examples .................................................................................................................................. 182
Other services ........................................................................................................................................... 184
Monitoring S3 on Outposts ........................................................................................................ 186
CloudWatch metrics ................................................................................................................................ 186
CloudWatch metrics ........................................................................................................................... 187
Amazon CloudWatch Events .................................................................................................................. 188
API Version 2006-03-01 vi
Amazon S3 on Outposts User Guide
CloudTrail logs .......................................................................................................................................... 190
Enabling CloudTrail logging for S3 on Outposts objects ........................................................... 190
Amazon S3 on Outposts AWS CloudTrail log file entries ........................................................... 193
Developing with S3 on Outposts ................................................................................................ 196
S3 on Outposts APIs ............................................................................................................................... 196
Amazon S3 API operations for managing objects ....................................................................... 196
Amazon S3 Control API operations for managing buckets ....................................................... 197
S3 on Outposts API operations for managing Outposts ............................................................ 198
Configuring S3 control client ................................................................................................................ 199
Making requests over IPv6 ..................................................................................................................... 199
Getting started with IPv6 ................................................................................................................. 200
Making requests using dual-stack endpoints ............................................................................... 201
Using IPv6 addresses in IAM policies ............................................................................................. 201
Testing IP address compatibility ..................................................................................................... 202
Using IPv6 with AWS PrivateLink .................................................................................................... 203
Using dual-stack endpoints .............................................................................................................. 206


"""


# In[56]:


# Example Usage
pdfname = 's3-outposts'
pdf_path = f'{pdfname}.pdf'

# Extract and parse
text_by_page = extract_text_from_pdf(pdf_path)
sections = parse_index(index_text)
chunks = create_chunks(text_by_page, sections)

# Save chunks to JSON
with open(f"{pdfname}_chunks.json", "w") as f:
    json.dump(chunks, f, indent=4)
    
print("Chunks created and saved successfully.")


# In[57]:


#Sanity Check for chunking

import json
from collections import defaultdict
from statistics import mean

def analyze_json(file_path):
    """Reads a JSON file and prints out detailed statistics on its structure and key lengths."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        if isinstance(data, list):
            print(f"The JSON file contains a list with {len(data)} items.\n")
            
            # Identify unique headers and their types
            headers = set()
            for item in data:
                if isinstance(item, dict):
                    headers.update(item.keys())
            
            print(f"Unique headers found: {len(headers)}")
            for header in headers:
                print(f"- {header}")
            
            # Gather statistics on each header
            header_stats = defaultdict(list)
            for item in data:
                for header in headers:
                    if header in item:
                        header_stats[header].append(len(str(item[header])))
                    else:
                        header_stats[header].append(0)
            
            # Print average length and data type information
            print("\nDetailed statistics for each header:")
            for header, lengths in header_stats.items():
                avg_length = mean(lengths)
                sample_value = next((item[header] for item in data if header in item), None)
                value_type = type(sample_value).__name__ if sample_value is not None else "NoneType"
                print(f"- {header}:")
                print(f"  * Average length: {avg_length:.2f}")
                print(f"  * Sample data type: {value_type}")
                if sample_value:
                    print(f"  * Sample value: {sample_value}")
                print("")

        elif isinstance(data, dict):
            print("The JSON file contains a dictionary with the following keys:")
            for key, value in data.items():
                value_type = type(value).__name__
                print(f"- {key}:")
                print(f"  * Data type: {value_type}")
                if isinstance(value, (list, dict)):
                    print(f"  * Length: {len(value)}")
                print("")

        else:
            print("The JSON file contains a single item of an unrecognized structure.")

        print("\nAnalysis complete.")

    except UnicodeDecodeError as e:
        print("Failed to read the file due to encoding error:", e)


# Example usage
file_path = f"{pdfname}_chunks.json"  # Replace with your file path
analyze_json(file_path)


# ## Raw chunks created - add clean up code here before key word extraction

# In[ ]:


import json
import re

# Reduced stopwords for basic cleanup
reduced_stopwords = {"and", "the", "of", "is", "to", "a", "in", "on", "at", "for"}

# Function to check for boilerplate text or empty chunks
def contains_boilerplate_text(chunk):
    boilerplate_patterns = [
        r'^\s*page\s*\d+\s*$',  # Matches patterns like "page 1"
    ]
    return not chunk.strip() or any(re.match(pattern, chunk, re.IGNORECASE) for pattern in boilerplate_patterns)

# Function to remove reduced stopwords
def remove_reduced_stopwords(chunk):
    words = chunk.split()  # Tokenize text
    filtered_words = [word for word in words if word.lower() not in reduced_stopwords]
    return ' '.join(filtered_words)

# Function to clean a single chunk
def clean_chunk(chunk):
    if contains_boilerplate_text(chunk):
        return None, True  # Mark for deletion
    
    cleaned_chunk = remove_reduced_stopwords(chunk)
    was_edited = cleaned_chunk != chunk
    return cleaned_chunk, was_edited

# Function to clean a JSON file containing chunks
def clean_chunks_in_json(input_file_path, output_file_path):
    edited_count = 0
    sanity_check_count = 0
    
    with open(input_file_path, 'r') as file:
        data = json.load(file)
    
    cleaned_data = []
    
    for entry in data:
        cleaned_entry = entry.copy()  # Copy the original entry
        if 'content' in entry and entry['content']:  # Process non-empty content
            cleaned_content, was_edited = clean_chunk(entry['content'])
            cleaned_entry['content'] = cleaned_content if cleaned_content is not None else entry['content']
            if was_edited:
                edited_count += 1
        cleaned_data.append(cleaned_entry)
        sanity_check_count += 1

    with open(output_file_path, 'w') as file:
        json.dump(cleaned_data, file, indent=4)
    
    print(f"Sanity check: {sanity_check_count} chunks processed.")
    print(f"Edited: {edited_count} chunks.")
    print(f"Cleaned data saved to {output_file_path}")


# In[ ]:


# Example usage
input_file = "s3-outposts_chunks.json"  # Replace with your JSON file path
output_file = "s3-outposts_cleaned_chunks.json"  # Replace with your desired output file path
clean_chunks_in_json(input_file, output_file)


# In[ ]:





# ## Keyword extraction - Metadata finalized

# In[2]:


import json
from keybert import KeyBERT

# Load the chunks from the JSON file created in the previous step
pdfname = 's3'
input_file_path = f"{pdfname}_cleaned_chunks.json"
output_file_path = f"{pdfname}_cleaned_chunks_with_keywords.json"

# Step 1: Load the JSON file containing the chunks
with open(input_file_path, "r") as file:
    chunks = json.load(file)

# Step 2: Initialize KeyBERT for keyword extraction
kw_model = KeyBERT()

# Step 3: Extract keywords for each chunk
for chunk in chunks:
    content = chunk["content"]
    
    # Extract keywords
    keywords = kw_model.extract_keywords(
        content, 
        keyphrase_ngram_range=(1, 2), 
        stop_words='english', 
        top_n=5
    )
    chunk["keywords"] = [kw[0] for kw in keywords]  # Store extracted keywords in each chunk

# Step 4: Save the new JSON file with title, keywords, and content for each chunk
with open(output_file_path, "w") as f:
    json.dump(chunks, f, indent=4)

print(f"Chunks with keywords saved successfully to {output_file_path}")


# ### Discard empty/incoherent chunks

# In[4]:


import json

# Function to filter out chunks with empty content
def remove_empty_content_chunks(input_file_path, output_file_path):
    # Load the JSON data
    with open(input_file_path, 'r') as file:
        data = json.load(file)
    
    # Sanity check before filtering
    initial_count = len(data)
    
    # Filter out chunks with empty content
    cleaned_data = [chunk for chunk in data if chunk.get('content', '').strip()]
    
    # Sanity check after filtering
    final_count = len(cleaned_data)
    
    # Save the cleaned data back to a JSON file
    with open(output_file_path, 'w') as file:
        json.dump(cleaned_data, file, indent=4)
    
    # Print sanity check results
    print(f"Initial chunk count: {initial_count}")
    print(f"Final chunk count: {final_count}")
    print(f"Chunks removed: {initial_count - final_count}")
    print(f"Cleaned data saved to {output_file_path}")



# In[5]:


# Example usage
save_file_path = f"{pdfname}_cleaned_chunks_with_keywords_final.json"
remove_empty_content_chunks(output_file_path, save_file_path)


# In[ ]:





# ## Embedd to Pinecone

# In[25]:


import os
import time
import json
import torch
from transformers import AutoModel, AutoTokenizer
from pinecone import Pinecone, ServerlessSpec
import tiktoken  # Ensure tiktoken is installed if used for chunk tokenization

# Initialize Pinecone
pinecone_api_key = os.getenv("PINECONE_API_KEY") or input("Enter your Pinecone API key: ")
pc = Pinecone(api_key=pinecone_api_key)
index_name = "chunks-with-metadata-final"  # Change if desired
#index_name = "chunks-without-metadata-final"  # Change if desired

# Delete existing index if dimension mismatch
existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
if index_name in existing_indexes:
    pc.delete_index(index_name)

# Create index with the correct dimension (384 for avsolatorio/NoInstruct-small-Embedding-v0)
pc.create_index(
    name=index_name,
    dimension=384,
    metric="cosine",
    spec=ServerlessSpec(cloud="aws", region="us-east-1"),
)

# Wait until the index is ready
while not pc.describe_index(index_name).status["ready"]:
    time.sleep(1)

# Re-initialize the index after creation
index = pc.Index(index_name)


# In[8]:


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
model = AutoModel.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


# In[9]:


import json
import torch

# Assuming `model` and `tokenizer` are already loaded (e.g., SentenceTransformer, T5, etc.)

# Function to compute embeddings
def get_embedding(text, mode="sentence"):
    model.eval()
    inp = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        output = model(**inp)
    return output.last_hidden_state[:, 0, :].numpy()  # [CLS] token representation

# Function to index semantic chunks and metadata in Pinecone
def index_chunks_with_metadata(index, chunks_with_metadata):
    # Process each chunk, compute embeddings, and store it in Pinecone
    for i, chunk_data in enumerate(chunks_with_metadata):
        # Extract the content and metadata
        chunk = chunk_data["content"]
        keywords = chunk_data.get("keywords", [])
        title = chunk_data.get("title", "Unknown Title")

        # Generate embedding for the chunk
        embedding = get_embedding(chunk).flatten().tolist()

        # Prepare metadata for Pinecone
        doc_metadata = {
            "title": title,
            "keywords": keywords,
            "text": chunk  # Storing the text as part of metadata if desired
        }

        # Index each chunk in Pinecone
        index.upsert(
            [(f"chunk-{i}", embedding, doc_metadata)]
        )

        # Print progress every 100 chunks
        if (i + 1) % 100 == 0:
            print(f"Indexed {i + 1} chunks out of {len(chunks_with_metadata)}")

    print("All chunks and metadata have been successfully indexed in Pinecone.")


# In[10]:


# Load chunks with metadata from JSON file
with open("COMBINED_chunks_FINAL.json", "r") as file:
    chunks_with_metadata = json.load(file)

# Index the chunks with metadata
# Note: The `index` variable should be the Pinecone index instance that you have already created
index_chunks_with_metadata(index, chunks_with_metadata)


# In[16]:


import json
import torch

# Assuming `model` and `tokenizer` are already loaded (e.g., SentenceTransformer, T5, etc.)
# Function to compute embeddings
def get_embedding(text, mode="sentence"):
    model.eval()
    inp = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        output = model(**inp)
    return output.last_hidden_state[:, 0, :].numpy()  # [CLS] token representation

# Function to truncate metadata to fit Pinecone limits
def truncate_metadata(metadata, limit=40960):
    """Ensure metadata size is within the specified limit."""
    json_metadata = json.dumps(metadata)  # Convert metadata to JSON string
    if len(json_metadata.encode('utf-8')) > limit:  # Check size in bytes
        # Truncate the content field first, then title, and keywords if necessary
        if "text" in metadata:
            metadata["text"] = metadata["text"][:limit // 4]  # Approx. 25% of the limit
        if "title" in metadata:
            metadata["title"] = metadata["title"][:limit // 8]  # Approx. 12.5% of the limit
        if "keywords" in metadata:
            metadata["keywords"] = metadata["keywords"][:3]  # Keep only 3 keywords
    return metadata

# Function to index semantic chunks and metadata in Pinecone
def index_chunks_with_metadata(index, chunks_with_metadata):
    for i, chunk_data in enumerate(chunks_with_metadata):
        # Extract the content and metadata
        chunk = chunk_data.get("content", "")
        keywords = chunk_data.get("keywords", [])
        title = chunk_data.get("title", "Unknown Title")

        # Generate embedding for the chunk
        embedding = get_embedding(chunk).flatten().tolist()

        # Prepare metadata and truncate if needed
        doc_metadata = truncate_metadata({
            "title": title,
            "keywords": keywords,
            "text": chunk  # Store the text as part of metadata
        })

        # Index each chunk in Pinecone
        index.upsert(
            [(f"chunk-{i}", embedding, doc_metadata)]
        )

        # Print progress every 100 chunks
        if (i + 1) % 100 == 0:
            print(f"Indexed {i + 1} chunks out of {len(chunks_with_metadata)}")

    print("All chunks and metadata have been successfully indexed in Pinecone.")



# In[17]:


# Load chunks with metadata from JSON file
with open("COMBINED_chunks_NAIVE.json", "r") as file:
    chunks_with_metadata = json.load(file)

# Index the chunks with metadata
# Note: Replace `index` with your Pinecone index instance
index_chunks_with_metadata(index, chunks_with_metadata)


# In[ ]:





# In[18]:


import pinecone
import numpy as np

# Step 1: Fetch some sample chunks from the index
# Assume that we upserted data with ids in the format "chunk-0", "chunk-1", etc.
num_samples = 5  # Number of samples to fetch
sample_ids = [f"chunk-{i}" for i in range(30,35)]
response = index.fetch(ids=sample_ids)

# Step 2: Extract the chunks and metadata
sample_chunks = []
for item_id, item_data in response['vectors'].items():
    chunk_text = item_data['metadata'].get('text', '')
    sample_chunks.append(chunk_text)
    print(f"ID: {item_id}")
    print(f"Text (truncated to 500 characters): {chunk_text[:500]}")
    print("\nMetadata:")
    print(item_data['metadata'])
    print("\n" + "-" * 80 + "\n")

# Step 3: Compute Statistics for Chunk Lengths
# Fetch all metadata for the sanity check
fetch_response = index.describe_index_stats()
total_chunks = fetch_response['total_vector_count']

# Fetch chunks by iterating over index ids
# Assuming chunks are indexed as "chunk-0" to "chunk-(total_chunks-1)"
all_chunk_lengths = []
batch_size = 100  # Process chunks in batches to avoid overloading the system
for i in range(0, total_chunks, batch_size):
    ids = [f"chunk-{j}" for j in range(i, min(i + batch_size, total_chunks))]
    response = index.fetch(ids=ids)
    for item_data in response['vectors'].values():
        chunk_text = item_data['metadata'].get('text', '')
        all_chunk_lengths.append(len(chunk_text))

# Compute average length of chunks
average_length = np.mean(all_chunk_lengths)

# Print summary statistics
print(f"Total number of chunks: {total_chunks}")
print(f"Average length of chunks: {average_length:.2f} characters")


# ## Visualize Embeddings from a Vector Database (Pinecone)

# In[21]:


import matplotlib.pyplot as plt
import numpy as np

# Function to fetch data from Pinecone index
def fetch_pinecone_data(index, top_k=100, include_metadata=True):
    """
    Fetches top_k vectors from the Pinecone index using a dummy query vector.

    Args:
        index: The Pinecone index instance.
        top_k (int): Number of vectors to fetch.
        include_metadata (bool): Whether to include metadata in the results.

    Returns:
        List of tuples containing vector ID, embedding, and metadata.
    """
    dummy_query = [0.0] * 384  # Adjust dimension to match your model
    response = index.query(
        vector=dummy_query,
        top_k=top_k,
        include_values=True,
        include_metadata=include_metadata,
    )

    data = []
    for match in response["matches"]:
        data.append((match["id"], match["values"], match.get("metadata", {})))
    return data

# Function to visualize embeddings and metadata
def visualize_pinecone_data(data):
    """
    Visualize Pinecone data using matplotlib.

    Args:
        data: List of tuples (vector_id, embedding, metadata).
    """
    if not data:
        print("No data available for visualization.")
        return

    # Extract embeddings and metadata
    embeddings = np.array([entry[1] for entry in data])
    metadata = [entry[2] for entry in data]

    # Embedding distribution
    plt.figure(figsize=(10, 5))
    plt.hist(embeddings.flatten(), bins=50, alpha=0.7, label="Embedding Values")
    plt.xlabel("Embedding Value")
    plt.ylabel("Frequency")
    plt.title("Distribution of Embedding Values")
    plt.legend()
    plt.show()

    # Metadata analysis (if text metadata exists)
    if metadata and any("text" in meta for meta in metadata):
        word_counts = [
            len(meta.get("text", "").split()) for meta in metadata if "text" in meta
        ]
        plt.hist(word_counts, bins=10, alpha=0.7, label="Text Word Count")
        plt.xlabel("Word Count")
        plt.ylabel("Frequency")
        plt.title("Distribution of Text Word Counts")
        plt.legend()
        plt.show()

    # Metadata preview
    print("Sample Metadata:")
    for meta in metadata[:5]:
        print(json.dumps(meta, indent=2))

# Fetch and visualize data from Pinecone
top_k = 100  # Number of vectors to fetch
data = fetch_pinecone_data(index, top_k=top_k, include_metadata=True)
visualize_pinecone_data(data)


# In[22]:


import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# Function to fetch all vectors from the index (chunk by chunk)
def fetch_all_vectors(index, batch_size=100):
    """
    Fetch all vectors from the Pinecone index.

    Args:
        index: The Pinecone index instance.
        batch_size (int): Number of vectors to fetch per batch.

    Returns:
        embeddings (list): List of all embeddings.
        metadata (list): List of metadata corresponding to embeddings.
    """
    embeddings = []
    metadata = []

    # Pagination support (if required)
    next_token = None
    while True:
        response = index.query(
            vector=[0.0] * 384,  # Dummy vector to retrieve all vectors
            top_k=batch_size,
            include_values=True,
            include_metadata=True,
            filter=None,
            namespace=None,
            next_page_token=next_token,
        )
        for match in response["matches"]:
            embeddings.append(match["values"])
            metadata.append(match.get("metadata", {}))

        next_token = response.get("next_page_token", None)
        if not next_token:
            break

    return np.array(embeddings), metadata

# Function for dimensionality reduction and visualization
def visualize_embeddings(embeddings, metadata=None, method="tsne"):
    """
    Visualize embeddings using dimensionality reduction techniques.

    Args:
        embeddings (np.array): High-dimensional embeddings.
        metadata (list): Metadata corresponding to embeddings.
        method (str): Dimensionality reduction method ("tsne" or "pca").
    """
    if method == "tsne":
        reducer = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    elif method == "pca":
        reducer = PCA(n_components=2)
    else:
        raise ValueError("Invalid method. Choose 'tsne' or 'pca'.")

    reduced_embeddings = reducer.fit_transform(embeddings)

    # Plotting
    plt.figure(figsize=(10, 8))
    plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], alpha=0.6)
    plt.title(f"Vector Embeddings Visualization ({method.upper()})")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")

    # Optional: Annotate with metadata (e.g., titles or labels)
    if metadata and len(metadata) == len(reduced_embeddings):
        for i, meta in enumerate(metadata[:50]):  # Annotate only the first 50 points
            if "title" in meta:
                plt.text(
                    reduced_embeddings[i, 0],
                    reduced_embeddings[i, 1],
                    meta["title"][:20],  # Shorten title for clarity
                    fontsize=8,
                    alpha=0.7,
                )

    plt.show()

# Fetch all vectors from Pinecone
embeddings, metadata = fetch_all_vectors(index)

# Visualize embeddings
visualize_embeddings(embeddings, metadata=metadata, method="tsne")


# ## Retrive ranked chunks:

# In[36]:


from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def rerank_with_cosine_similarity(query_embedding, retrieved_docs, embedding_model):
    """
    Rerank retrieved documents based on cosine similarity to the query embedding.
    
    Args:
        query_embedding (list[float]): The embedding of the query.
        retrieved_docs (list[dict]): Retrieved documents with 'text' and 'embedding'.
        embedding_model: The embedding model to generate embeddings for documents if not present.
    
    Returns:
        list[dict]: Reranked documents.
    """
    # Prepare the embeddings and texts
    doc_embeddings = []
    doc_texts = []
    
    for doc in retrieved_docs:
        if "embedding" not in doc:  # If no embedding, generate it
            doc["embedding"] = embedding_model.encode(doc["text"]).tolist()
        doc_embeddings.append(doc["embedding"])
        doc_texts.append(doc["text"])

    # Convert query embedding to numpy array and calculate cosine similarities
    query_embedding = np.array(query_embedding).reshape(1, -1)
    doc_embeddings = np.array(doc_embeddings)
    similarities = cosine_similarity(query_embedding, doc_embeddings).flatten()

    # Combine texts and scores and sort by similarity
    reranked_docs = sorted(
        [{"text": text, "score": score} for text, score in zip(doc_texts, similarities)],
        key=lambda x: x["score"],
        reverse=True,
    )

    return reranked_docs

# Example usage:
query = "What are the steps to create a bucket, access point, and endpoint for Amazon S3 on Outposts?"
query_embedding = model.encode(query).tolist()  # Query embedding
retrieved_docs = [
    {"text": "Document 1 content about buckets."},
    {"text": "Document 2 content about endpoints."},
    {"text": "Unrelated document content."},
]

reranked_docs = rerank_with_cosine_similarity(query_embedding, retrieved_docs, model)

# Display reranked results
for i, doc in enumerate(reranked_docs, start=1):
    print(f"Rank {i}:")
    print(f"Score: {doc['score']:.4f}")
    print(f"Text: {doc['text'][:100]}")  # Truncate long texts
    print("-" * 50)


# In[ ]:


from rank_bm25 import BM25Okapi

def rerank_with_bm25(query, retrieved_docs):
    """
    Rerank retrieved documents using BM25.
    
    Args:
        query (str): The user query.
        retrieved_docs (list[dict]): Retrieved documents with 'text'.
    
    Returns:
        list[dict]: Reranked documents.
    """
    # Tokenize documents and query
    tokenized_docs = [doc["text"].split() for doc in retrieved_docs]
    tokenized_query = query.split()

    # Initialize BM25
    bm25 = BM25Okapi(tokenized_docs)

    # Get BM25 scores for the query
    scores = bm25.get_scores(tokenized_query)

    # Combine scores with texts and sort by score
    reranked_docs = sorted(
        [{"text": doc["text"], "score": score} for doc, score in zip(retrieved_docs, scores)],
        key=lambda x: x["score"],
        reverse=True,
    )

    return reranked_docs

# Example usage:
query = "What are the steps to create a bucket?"
retrieved_docs = [
    {"text": "Document 1: Steps to create a bucket."},
    {"text": "Document 2: Information about endpoints."},
    {"text": "Document 3: Steps for S3 bucket creation."},
]

reranked_docs = rerank_with_bm25(query, retrieved_docs)

# Display reranked results
for i, doc in enumerate(reranked_docs, start=1):
    print(f"Rank {i}:")
    print(f"Score: {doc['score']:.4f}")
    print(f"Text: {doc['text'][:100]}")
    print("-" * 50)


# In[ ]:





# ## Evaluation of the Model (Test Method)

# In[4]:


import os
import numpy as np
import matplotlib.pyplot as plt
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer

# Prompt user for Pinecone API key
pinecone_api_key = os.getenv("PINECONE_API_KEY") or input("Enter your Pinecone API key: ")

# Initialize Pinecone client
pc = Pinecone(api_key=pinecone_api_key)

# Define index names
naive_index_name = "chunks-without-metadata-final"
metadata_index_name = "chunks-with-metadata-final"

# Connect to vector databases
if naive_index_name not in [index["name"] for index in pc.list_indexes()]:
    raise ValueError(f"Index {naive_index_name} does not exist in Pinecone.")

if metadata_index_name not in [index["name"] for index in pc.list_indexes()]:
    raise ValueError(f"Index {metadata_index_name} does not exist in Pinecone.")

naive_index = pc.Index(naive_index_name)
metadata_index = pc.Index(metadata_index_name)

# Function to retrieve data from a Pinecone index
def retrieve_from_index(index, query_embedding, top_k=5):
    results = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
    return [match["metadata"]["text"] for match in results["matches"]]

# Metrics Calculation Functions
def calculate_recall(retrieved, ground_truth):
    """Recall: Proportion of ground truth retrieved."""
    true_positives = len(set(retrieved) & set(ground_truth))
    return true_positives / len(ground_truth) if ground_truth else 0

def calculate_completeness(generated, key_points):
    """Completeness: Key points covered in the response."""
    covered = sum(1 for kp in key_points if kp in generated)
    return covered / len(key_points) if key_points else 0

def calculate_hallucination(generated, key_points):
    """Hallucination: Irrelevant or incorrect information."""
    contradicts = sum(1 for gen in generated if gen not in key_points)
    return contradicts / len(generated) if generated else 0

def calculate_irrelevance(generated, key_points):
    """Irrelevance: Missing relevant information."""
    return 1 - calculate_completeness(generated, key_points) - calculate_hallucination(generated, key_points)

# Ground Truth and Key Points
ground_truth = ["ground_truth_chunk1", "ground_truth_chunk2", "ground_truth_chunk3"]
key_points = ["key_point1", "key_point2", "key_point3"]

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

def query_embedding(query):
    return model.encode(query).tolist()

# Compare both vector databases
queries = ["example_query1", "example_query2"]
results = {"naive": [], "metadata": []}

for query in queries:
    query_emb = query_embedding(query)
    
    naive_retrieval = retrieve_from_index(naive_index, query_emb)
    metadata_retrieval = retrieve_from_index(metadata_index, query_emb)
    
    for retrieval, db_type in zip([naive_retrieval, metadata_retrieval], ["naive", "metadata"]):
        recall = calculate_recall(retrieval, ground_truth)
        completeness = calculate_completeness(retrieval, key_points)
        hallucination = calculate_hallucination(retrieval, key_points)
        irrelevance = calculate_irrelevance(retrieval, key_points)
        
        results[db_type].append({"recall": recall, "completeness": completeness,
                                 "hallucination": hallucination, "irrelevance": irrelevance})


# In[5]:


# Visualization
metrics = ["recall", "completeness", "hallucination", "irrelevance"]
x = range(len(queries))

for metric in metrics:
    naive_scores = [result[metric] for result in results["naive"]]
    metadata_scores = [result[metric] for result in results["metadata"]]
    
    plt.figure()
    plt.plot(x, naive_scores, label="Naive DB", marker="o")
    plt.plot(x, metadata_scores, label="Metadata-Enriched DB", marker="s")
    plt.title(f"{metric.capitalize()} Comparison")
    plt.xlabel("Query")
    plt.ylabel(metric.capitalize())
    plt.legend()
    plt.show()


# In[6]:


import os
import numpy as np
import pandas as pd
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

# Initialize Pinecone
pinecone_api_key = os.getenv("PINECONE_API_KEY") or input("Enter your Pinecone API key: ")
pc = Pinecone(api_key=pinecone_api_key)

# Connect to vector databases
naive_index = pc.Index("chunks-without-metadata-final")
metadata_index = pc.Index("chunks-with-metadata-final")

# Define the embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

def query_embedding(query):
    """Generate embeddings for a given query."""
    return model.encode(query).tolist()

# Metrics Calculation Functions
def calculate_recall(retrieved, ground_truth):
    """Recall: Proportion of ground truth retrieved."""
    true_positives = len(set(retrieved) & set(ground_truth))
    return true_positives / len(ground_truth) if ground_truth else 0

def calculate_completeness(generated, key_points):
    """Completeness: Proportion of key factual points covered."""
    covered = sum(1 for kp in key_points if kp in generated)
    return covered / len(key_points) if key_points else 0

def calculate_hallucination(generated, key_points):
    """Hallucination: Proportion of irrelevant or incorrect information."""
    contradicts = sum(1 for gen in generated if gen not in key_points)
    return contradicts / len(generated) if generated else 0

def calculate_irrelevance(generated, key_points):
    """Irrelevance: Missing or unrelated key points."""
    return 1 - calculate_completeness(generated, key_points) - calculate_hallucination(generated, key_points)

def retrieve_from_index(index, query_embedding, top_k=5):
    """Retrieve data from a Pinecone index."""
    results = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
    return [match["metadata"]["text"] for match in results["matches"]]

# Input data for testing
queries = ["example_query1", "example_query2"]
ground_truth = ["ground_truth_chunk1", "ground_truth_chunk2", "ground_truth_chunk3"]
key_points = ["key_point1", "key_point2", "key_point3"]

# Initialize results storage
results = []

# Perform retrieval and compute metrics
for query in queries:
    query_emb = query_embedding(query)

    for index, index_name in [(naive_index, "Naive DB"), (metadata_index, "Metadata-Enriched DB")]:
        retrieved = retrieve_from_index(index, query_emb)
        
        recall = calculate_recall(retrieved, ground_truth)
        completeness = calculate_completeness(retrieved, key_points)
        hallucination = calculate_hallucination(retrieved, key_points)
        irrelevance = calculate_irrelevance(retrieved, key_points)
        
        results.append({
            "Query": query,
            "Index": index_name,
            "Recall": recall,
            "Completeness": completeness,
            "Hallucination": hallucination,
            "Irrelevance": irrelevance
        })

# Convert results to a pandas DataFrame
results_df = pd.DataFrame(results)

# Print results in tabular format
print(results_df)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ## Integrate LLM with chain and chat History - Implemented in Vertex AI

# In[38]:


import os
import torch
from transformers import AutoModel, AutoTokenizer
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import Pinecone as LangchainPinecone
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings.base import Embeddings
from langchain.llms.base import LLM
from typing import Optional, List

import vertexai
from vertexai.generative_models import GenerativeModel

# Initialize Vertex AI for Google Gemini model
PROJECT_ID = "ids-560-project-group-1-bosch"
vertexai.init(project=PROJECT_ID, location="us-central1")
gemini_model = GenerativeModel("gemini-1.5-flash")

# Custom wrapper for Google Gemini to make it compatible with LangChain
class RunnableGemini(LLM):
    def __init__(self, model: GenerativeModel):
        super().__init__()
        self._model = model

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        response = self._model.generate_content(prompt)
        return response.text

    @property
    def _llm_type(self) -> str:
        return "google_gemini"

# Instantiate the wrapped model
llm = RunnableGemini(gemini_model)

# Load the embedding model for the retriever function
MODEL_NAME = "avsolatorio/NoInstruct-small-Embedding-v0"
embedding_model = AutoModel.from_pretrained(MODEL_NAME)
embedding_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Custom LangChain Embedding wrapper
class HFEmbeddingWrapper(Embeddings):
    def embed_query(self, text: str) -> List[float]:
        return get_embedding(text).flatten().tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [get_embedding(text).flatten().tolist() for text in texts]

# Initialize the custom embedding wrapper
embedding = HFEmbeddingWrapper()

# Initialize Pinecone client
pinecone_api_key = os.getenv("PINECONE_API_KEY") or input("Enter your Pinecone API key: ")
pc = Pinecone(api_key=pinecone_api_key)
index_name = "llm-chatbot-project"

# Check if the index exists; if not, create it with dimension 384
if index_name not in [index_info["name"] for index_info in pc.list_indexes()]:
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
index = pc.Index(index_name)
print(f"Connected to index '{index_name}' in Pinecone.")

# Initialize LangChain Pinecone Retriever with embedding wrapper
vectorstore = LangchainPinecone(index=index, embedding=embedding)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

# Define the QA PromptTemplate for debugging and documentation assistance
QA_PROMPT = PromptTemplate.from_template("""
You are an expert code debugger and documentation assistant. Help answer technical queries based on the script provided, offering solutions, clarifications, or steps as needed. Use precise language and avoid assumptions. Refer to the documentation context for direct responses.

Documentation Context:
{context}

Question: {question}
Answer:
""")

# Set up the conversational retrieval chain with the custom Gemini wrapper
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    condense_question_prompt=QA_PROMPT,
    combine_docs_chain_kwargs={"prompt": QA_PROMPT},
    return_source_documents=True
)

# Function to maintain conversation history in the RAG chain
def ask_question_with_history(qa_chain, question, chat_history):
    result = qa_chain.invoke({"question": question, "chat_history": chat_history})
    print("Response:", result["answer"])
    chat_history.append((question, result["answer"]))
    return chat_history


# In[74]:


from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory

# Define the QA PromptTemplate with the 'context' and 'question' placeholders
QA_PROMPT = PromptTemplate.from_template("""
You are an assistant helping with error debugging and code understanding. Use the context provided to answer the user's question. If there is no context, let the user know more information is needed.

Context:
{context}

User's Question: {question}
Answer:
""")

# Initialize your LLM and retriever (already defined in your environment)

# Create the RetrievalQA chain with the custom prompt and specify the input key
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    input_key="question",  # Specify 'question' as the input key
    chain_type_kwargs={
        "prompt": QA_PROMPT,
    },
    return_source_documents=True,
)

# Initialize memory to store conversation history
memory = ConversationBufferMemory(memory_key="chat_history", input_key="question")

# Function to ask a question with context management
def ask_question_with_history(qa_chain, question, memory):
    # Retrieve the conversation history
    chat_history = memory.load_memory_variables({}).get("chat_history", "")

    # Prepare the inputs
    inputs = {
        "question": question,  # Use 'question' as the key
        "chat_history": chat_history,
    }

    # Call the QA chain with the inputs
    result = qa_chain(inputs)

    # Display the result and update memory
    print("Response:", result["result"])
    memory.save_context({"question": question}, {"result": result["result"]})
    return memory

# Example usage
question = "What are the steps to create a bucket, access point, and endpoint for Amazon S3 on Outposts?"
memory = ask_question_with_history(qa_chain, question, memory)

question = "Can you explain how Amazon S3 on Outposts is different from regular Amazon S3?"
memory = ask_question_with_history(qa_chain, question, memory)

question = "What are the supported API operations for Amazon S3 on Outposts, and how do they differ from standard Amazon S3?"
memory = ask_question_with_history(qa_chain, question, memory)


# In[ ]:




