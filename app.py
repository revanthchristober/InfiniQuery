# from langchain_community.document_loaders.pdf import PyPDFLoader
# from langchain_text_splitters import NLTKTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough
import streamlit as st
from PIL import Image

favicon_image = Image.open('app_icon.jpeg')

st.set_page_config(
    page_title='InfiniQuery',
    page_icon=favicon_image,
    layout='wide'
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Setting up the retriever with the path to our local data
# document = PyPDFLoader(file_path='2404.07143v1.pdf', extract_images=True)

# data = document.load()

# text_splitter = NLTKTextSplitter(chunk_size=500, chunk_overlap=100)

with open('api_key.txt') as f:
    api_key = f.read()

chat_template = ChatPromptTemplate.from_messages([
    # System Message Prompt Template
    SystemMessage(content="""You are a Helpful AI Bot. 
    You take the context and question from user. Your answer should be based on the specific context."""),
    # Human Message Prompt Template
    HumanMessagePromptTemplate.from_template("""Aswer the question based on the given context.
    Context:
    {context}
    
    Question: 
    {question}
    
    Answer: """)
])

chat_model = ChatGoogleGenerativeAI(google_api_key=api_key, 
                                   model="gemini-1.5-pro-latest", stream=True)
output_parser = StrOutputParser()

# chunks = text_splitter.split_documents(data)

embedding_model = GoogleGenerativeAIEmbeddings(google_api_key=api_key, model='models/embedding-001')

# db = Chroma.from_documents(chunks, embedding_model, persist_directory="./chroma_db_")

db_connection = Chroma(persist_directory="./chroma_db_", embedding_function=embedding_model)

retriever = db_connection.as_retriever(search_kwargs={"k": 5})

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | chat_template
    | chat_model
    | output_parser
)

def main():
    try:
        st.title('InfiniQuery: The Infinite Context QA Companion 📜📄')
        st.subheader('An advanced AI contextual question-answering system based on the groundbreaking ‘Leave No Context Behind’ research paper.')

        # User Input: Question
        user_question = st.text_input('Ask your question based on the paper:')

        if st.button('ASK 📩'):
            if user_question:
                response = rag_chain.invoke(user_question)

                if response:
                    st.subheader('Response 🤖: ')
                    st.write(response)
                else: st.warning('Can\'t generate the reponse given the question.')
            
            elif user_question is '':
                st.warning('Please ask a question!')

    except Exception as e: st.error(f'Error Occured: {e}')

if __name__ == '__main__':
    main()

# print(response)