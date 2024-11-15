import os
import streamlit as st
import time
import langchain
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from pinecone import Pinecone, ServerlessSpec
import time
from langchain.embeddings.openai import OpenAIEmbeddings
import torch
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer
from numpy.linalg import norm
import API_KEY_FOLDER.api_keys

# Function to inject CSS file
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Call the function with the CSS file path
local_css("CSS/styles.css")
# Initialize OpenAI API key
os.environ["OPENAI_API_KEY"] = API_KEY_FOLDER.api_keys.OPENAI_API_KEY
    
arr = np.load("embedding_model-e5-large-999rows.npy")
# Initialize Pinecone client
pinecone_api_key = API_KEY_FOLDER.api_keys.PINECONE_API_KEY

# Initialize ServerlessSpec
def retrieve_vectordb_index():
    spec = ServerlessSpec(cloud="aws", region="us-west-2")
    pc = Pinecone(api_key = pinecone_api_key)
    index_name = 'pinecone-final'
    existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

    # Check if the index exists, create it if not
    if index_name not in existing_indexes:
        pc.create_index(
            index_name,
            dimension=1536,  # Dimensionality of ada 002
            metric='dotproduct',
            spec=spec
        )
        while not pc.describe_index(index_name).status['ready']:
            time.sleep(1)
    index = pc.Index(index_name)
    
    return index

def retrieve_db_results(query, k):
    text_field = "comments"
    # import statement here looks weird. But without this we get some config error and somehow the below statement solves that
    from langchain.vectorstores import Pinecone
    index = retrieve_vectordb_index()
    embed_model = OpenAIEmbeddings(model="text-embedding-ada-002")
    vectorstore = Pinecone(
        index, embed_model.embed_query, text_field
    )
    results = vectorstore.similarity_search(query, k)
    source_knowledge = "\n".join([x.page_content for x in results])
    return source_knowledge

def ADA_augment_prompt(query: str, k):
    source_knowledge = retrieve_db_results(query, k)
    augmented_prompt = f"""As a bot gathering public opinions from YouTube comments,
    analyze the comments related to Pierre Poilievre and Justin Trudeau based on the given query.
    Provide a summary of the comments extracted from videos. Summarize the overall sentiment in 2-3 sentences,
    and indicate the overall sentiment using one word. Positive, Negative or Neutral.

    Comments:
    {source_knowledge}

    Query: {query}"""
    return augmented_prompt

def ask_chatbot(num_comments, question):
    messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="Hi AI, how are you today?"),
    AIMessage(content="I'm great thank you. How can I help you?"),
    HumanMessage(content="I'd like to understand string theory.")
    ]    

    chat = ChatOpenAI(
        openai_api_key = os.environ["OPENAI_API_KEY"],
        model = 'gpt-3.5-turbo'
    )     
    prompt = HumanMessage(
    content = ADA_augment_prompt(question, num_comments)
    )
        
    messages.append(prompt)
    res = chat(messages)    
    return res.content

def embed(docs: list[str]) -> list[list[float]]:

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_id = "intfloat/e5-large-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).to(device)
    model.eval()

    docs = [f"passage: {d}" for d in docs]
    # Tokenize
    tokens = tokenizer(
        docs, padding=True, max_length=512, truncation=True, return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        # Process with model for token-level embeddings
        out = model(**tokens)
        # Mask padding tokens
        last_hidden = out.last_hidden_state.masked_fill(
            ~tokens["attention_mask"][..., None].bool(), 0.0
        )
        # Create mean pooled embeddings
        doc_embeds = last_hidden.sum(dim=1) / \
            tokens["attention_mask"].sum(dim=1)[..., None

]
    return doc_embeds.cpu().numpy()

def query(text: str, top_k) -> list[str]:

    comments_df = pd.read_csv('Data/translations_data.csv', engine='python')
    comments_df = comments_df[comments_df['Comment'].str.len() > 50]
    comments_df['comments'] = comments_df['channelTitle'] + ' ' + comments_df['Comment']
    comments_df.drop(columns=['Comment', 'channelTitle'], inplace=True)
    comments_df['index'] = comments_df.index

    # Select a subset of data for processing
    process_df = comments_df[1:1000]

    # Convert comments to numpy array for efficient indexing
    chunks = process_df["comments"]
    chunk_arr = np.array(chunks)

    # Embed query text
    xq = embed([f"query: {text}"])[0]

    # Calculate cosine similarities
    sim = np.dot(arr, xq.T) / (norm(arr, axis=1)*norm(xq.T))

    # Get indices of top_k records
    idx = np.argpartition(sim, -top_k)[-top_k:]

    # Get top_k documents
    docs = chunk_arr[idx]

    return docs.tolist()

def intfloat_augment_prompt(source_knowledge, query_text):

    augmented_prompt = f"""Using the Youtube comments below, answer the query.
    It should be like reflecting people opinion. Do not give the comments directly, summarize the content and give the overall opinion in 2 or 3 sentences.
    Also mention the overall sentiment (Positive, Negative, Neutral) at the end like Overall Sentiment:.

    Comments:
    {source_knowledge}

    Query: {query_text}"""
    
    return augmented_prompt

def intfloat_chatbot_response(num_comments, question):
    # Initialize OpenAI chat model
    chat = ChatOpenAI(
        openai_api_key=os.environ["OPENAI_API_KEY"],
        model='gpt-3.5-turbo'
    )

    source_knowledge = query(question, num_comments)
    prompt = HumanMessage(content=intfloat_augment_prompt(source_knowledge, question))
    res = chat([SystemMessage(content="You are a helpful assistant."), prompt])
    return res.content

def main():
    
    st.markdown(
        """
        <div class="container">
            <h1 class="jumping-title">📢 Public Pulse 📢</h1>
        </div>
        """,
        unsafe_allow_html=True
    )
  
    # Create the number input field
    num_comments = st.number_input("Number of relevant comments to fetch", min_value=3, max_value=10, value=10)

    question = st.text_input("What do you want to ask?", placeholder="Enter your question here")
 
    # Mocked responses
    ada_response = ""
    intfloat_response = ""
    if  st.button("Go"):
        ada_response = ask_chatbot(num_comments, question)
        intfloat_response = intfloat_chatbot_response(num_comments, question)
        with st.spinner("Generating Response..."):
            for i in range(10):  # Simulating response generation steps
                time.sleep(1)   # Simulating delay
                st.spinner(f"Generating Response ({i+1}/10)")
    # Display responses
    if ada_response and intfloat_response:
    # Displaying responses side by side
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("From text-ada-embedding-002 embeddings:")
            st.markdown(f'<div class="response" style="height: 450px;width: 300px;"><p>{ada_response}</p></div>', unsafe_allow_html=True)

        with col2:
            st.subheader("From intfloat/e5-large-v2 embeddings:")
            st.markdown(f'<div class="response" style="height: 450px;width: 300px;"><p>{intfloat_response}</p></div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()