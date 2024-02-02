import streamlit as st
from openai import OpenAI
import langchain
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.prompts.chat import ChatPromptTemplate
from langchain.docstore.document import Document
import pdfplumber
import nltk
import os
import shutil
__import__ ('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

with st.sidebar:
    openai_api_key = st.text_input('OpenAI API Key',key='chatbot_api_key',type='password')

os.environ['OPENAI_API_KEY'] = openai_api_key

st.title('PDF Chatter 📄🔴')
st.caption('Upload your PDF files and chat with them. Try it out!')

uploaded_files = st.file_uploader('Upload Your Documents', type=('pdf'), accept_multiple_files=True)

if uploaded_files and openai_api_key:
    raw_text_docs = []
    for file in uploaded_files:
        raw_text = ''
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                raw_text += text + ' '
            doc =  Document(page_content=raw_text, metadata={"source": "local"})
            raw_text_docs.append(doc)

    def preprocess_text(docs):
        with st.spinner('Preprocessing Documents...'):
            preprocessed_docs = [*docs]
            for document in preprocessed_docs:
                document.page_content = document.page_content.replace('\n', ' ')
                document.page_content = document.page_content.lower()
            return preprocessed_docs

    processed_texts = preprocess_text(raw_text_docs)

    def split_text(docs):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=500,
            length_function = len,
            add_start_index=True
        )
        chunks = text_splitter.split_documents(docs)
        print(f'Split {len(docs)} documents into {len(chunks)} chunks.')
        return chunks

    chunked_texts = split_text(processed_texts)

    # Have to declare db before actually saving so we can force clear it.
    db = Chroma.from_documents([Document(page_content='delete')],OpenAIEmbeddings())

    db.delete_collection()
    with st.spinner('Saving Chunks...'):
        db = Chroma.from_documents(chunked_texts,OpenAIEmbeddings())
    print(f'Chroma saved {len(chunked_texts)} chunks.')


    prompt_template = ChatPromptTemplate.from_template(
        """
        Answer the question based on the following context:\n
        {context}
        ---
        Answer the question based on the above context (do not mention the context): {question}
        """
    )

    if 'messages' not in st.session_state:
        st.session_state['messages'] = [{'role': 'assistant', 'content': 'Ask away!'}]

    for msg in st.session_state.messages:
        st.chat_message(msg['role']).write(msg['content'])

    if prompt := st.chat_input():
        llm = OpenAI(api_key=openai_api_key)
        relevant_chunks = db.similarity_search_with_relevance_scores(prompt,k=3)
        context_text = '\n\n---\n\n'.join([doc.page_content for doc, _score in relevant_chunks])
        print(context_text)
        formatted_prompt = [{'role': 'user', 'content': prompt_template.format(context=context_text, question=prompt)}]
        response = llm.chat.completions.create(model='gpt-3.5-turbo-1106', messages=formatted_prompt, temperature=0.4)
        st.session_state.messages.append({'role': 'user', 'content': prompt})
        st.chat_message('user').write(prompt)
        msg = response.choices[0].message.content
        st.session_state.messages.append({'role': 'assistant', 'content': msg})
        st.chat_message('assistant').write(msg)

else:
    st.session_state.clear()
    st.info('Please upload your documents and add your OpenAI API key in the sidebar.')