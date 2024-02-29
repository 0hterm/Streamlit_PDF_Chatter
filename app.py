import streamlit as st
from openai import OpenAI
import langchain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts.chat import ChatPromptTemplate
from langchain.docstore.document import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
import pdfplumber
import nltk
import os
import shutil
# Below needed to run on Streamlit Cloud
__import__ ('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

if 'db' not in st.session_state:
    st.session_state['db'] = ''

def handleChange():
    st.session_state['db'] = ''

with st.sidebar:
    openai_api_key = st.text_input('OpenAI API Key',key='chatbot_api_key',type='password')

    uploaded_files = st.file_uploader('Upload Your Documents', type=('pdf'), accept_multiple_files=True, on_change=handleChange)

    if uploaded_files and '' == st.session_state['db']:
        # Extract text from documents
        raw_text_docs = []
        for file in uploaded_files:
            raw_text = ''
            with pdfplumber.open(file) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()
                    raw_text += text + ' '
                doc =  Document(page_content=raw_text, metadata={"source": "local"})
                raw_text_docs.append(doc)

        # Preprocess text
        def preprocess_text(docs):
            with st.spinner('Preprocessing Documents...'):
                preprocessed_docs = [*docs]
                for document in preprocessed_docs:
                    document.page_content = document.page_content.replace('\n', ' ')
                    document.page_content = document.page_content.lower()
                return preprocessed_docs

        processed_texts = preprocess_text(raw_text_docs)

        # Create chunks
        def split_text(docs):
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=100,
                length_function = len,
                add_start_index=True
            )
            chunks = text_splitter.split_documents(docs)
            return chunks

        chunked_texts = split_text(processed_texts)

        # Have to declare db before actually saving so we can force clear it.
        embeddings = HuggingFaceEmbeddings(model_name="All-MiniLM-L6-v2")
        db = Chroma.from_documents(chunked_texts[:3],embeddings)
        db.delete_collection()

        # Save chunks to Chroma.
        with st.spinner('Saving Chunks...'):
            db = Chroma.from_documents(chunked_texts,embeddings)
        st.session_state['db'] = db



os.environ['OPENAI_API_KEY'] = openai_api_key

st.title('PDF Chatter 📄🔴')
st.caption('Upload your PDF files and chat with them. Try it out!\n\nWARNING: The greater the size of documents, the longer it will take to process and the higher embedding costs will be.')

if st.session_state['db'] != '':
    db = st.session_state['db']
    
    
# Ensure files are uploaded and API key is enetered before running chat functionalities.
if uploaded_files and openai_api_key:

    prompt_template = ChatPromptTemplate.from_template(
        """
        Answer the question based on the following context:\n
        {context}
        ---
        Answer the question based on the above context (do not mention the context): {question}
        """
    )

    # Display intro message if no messages have been sent by the user.
    if 'messages' not in st.session_state:
        st.session_state['messages'] = [{'role': 'assistant', 'content': 'Ask away!'}]

    # Write each message to the screen
    for msg in st.session_state.messages:
        st.chat_message(msg['role']).write(msg['content'])

    if prompt := st.chat_input():
        llm = OpenAI(api_key=openai_api_key)
        # Search for relevant context.
        relevant_chunks = db.similarity_search_with_relevance_scores(prompt,k=3)
        context_text = '\n\n---\n\n'.join([doc.page_content for doc, _score in relevant_chunks])
        # Format prompt with context and question.
        formatted_prompt = [{'role': 'user', 'content': prompt_template.format(context=context_text, question=prompt)}]
        # Save response
        response = llm.chat.completions.create(model='gpt-3.5-turbo-1106', messages=formatted_prompt, temperature=0.3)
        # Insert unformatted prompt into the session states messages.
        st.session_state.messages.append({'role': 'user', 'content': prompt})
        st.chat_message('user').write(prompt)
        # Insert final response to the session.
        msg = response.choices[0].message.content
        st.session_state.messages.append({'role': 'assistant', 'content': msg})
        st.chat_message('assistant').write(msg)

else:
    st.session_state.clear()
    st.info('Please upload your documents and add your OpenAI API key in the sidebar.')
