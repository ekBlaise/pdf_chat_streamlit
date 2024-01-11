import streamlit as st
from dotenv import load_dotenv

from PyPDF2 import PdfMerger
import numpy as np
import tempfile
from progressbar import ProgressBar
import fitz
import pytesseract
from pdf2image import convert_from_path
import os
import io

# from transformers import GPT2TokenizerFast
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import CharacterTextSplitter

from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.llms import HuggingFaceHub
# from langchain_community.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
# from langchain_community.vectorstores import FAISS

from langchain.chat_models import ChatOpenAI
# from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI

from htmlTemplate import css, bot_template, user_template



# Classify Pdf as Image or Text
def classifier(pdf_file):
    with open(pdf_file,"rb") as f:
        pdf = fitz.open(f)
        res = 0
        for page in pdf:
            image_area = 0.0
            text_area = 0.0
            for b in page.get_text("blocks"):
                if '<image:' in b[4]:
                    r = fitz.Rect(b[:4])
                    image_area = image_area + abs(r)
                else:
                    r = fitz.Rect(b[:4])
                    text_area = text_area + abs(r)
            if image_area == 0.0 and text_area != 0.0:
                res += 1
            if text_area == 0.0 and image_area != 0.0:
                res += 0
        total_area = text_area + image_area
        text_percentage = np.round((text_area/total_area) * 100, 2)
        return text_percentage
    
# Make Searchable
def create_searchable_pdf(images: list, output_path: str):
    """Generate a searchable PDF from document images.
    """
    # Decide here whether to clean image header or not.
    merger = PdfMerger()
    pbar = ProgressBar()
    
    for page_index, image in enumerate(pbar(images)):
        pdf_page = pytesseract.image_to_pdf_or_hocr(image, extension='pdf')
        
        temp_dir = tempfile.gettempdir()
        pdf_page_path = os.path.join(temp_dir, f"{page_index}.pdf") 
        with open(pdf_page_path, "wb") as f:
            f.write(pdf_page)
        merger.append(pdf_page_path)
        # os.remove(pdf_page_path)
        
    merger.write(output_path)
    return output_path

# Split the scanned pdf into pages
def split_pdf_scan(pdf_path, output_folder):
    output_files = []  # List to store output file paths
    try:
        for uploaded_file in pdf_path:
            pdf_content = uploaded_file.read()
            pdf_document = fitz.open(stream=io.BytesIO(pdf_content))
            page_count = len(pdf_document)

            if not os.path.exists(output_folder):
                os.makedirs(output_folder)

            for page_number in range(1, page_count + 1):
                try:
                    pdf_output = fitz.open()
                    pdf_output.insert_pdf(pdf_document, from_page=page_number - 1, to_page=page_number - 1)

                    output_file = os.path.join(output_folder, f"page_{page_number}.pdf")
                    pdf_output.save(output_file)
                    output_files.append(output_file)  # Append output file path to list
                    
                except Exception as e:
                    print(f"An error occurred while processing page {page_number}: {str(e)}")
                

            pdf_document.close()
            return output_files
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

# Extract text from a page
def extract_text(pdf_path):
    text = ""
    for uploaded_file in pdf_path:
        pdf_content = uploaded_file.read()
        pdf_document = fitz.open(stream=io.BytesIO(pdf_content))
        
        for page_number in range(pdf_document.page_count):
            page = pdf_document[page_number]
            text += page.get_text("text") + " "
        
        pdf_document.close()
    # page_number = 0  # Extracting text from the first (and only) page
    # page = pdf_document.load_page(page_number)
    # text = page.get_text("text")
    # pdf_document.close()
    return text.replace('\n', ' ')

# Get the chunks of the extracted text
def get_text_chunks(text):
    # Option 1 - use for simple cases, faster and mostly used is text extraction is complex
    text_splitter = CharacterTextSplitter(
        separator = "\n",
        chunk_size = 1000,
        chunk_overlap = 200,
        length_function = len
    )
    
    chunks = text_splitter.split_text(text)
    
    # # Create function to count tokens
    # tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

    # def count_tokens(text: str) -> int:
    #     return len(tokenizer.encode(text))

    # # Step 4: Split text into chunks
    # text_splitter = RecursiveCharacterTextSplitter(
    #     # Set a really small chunk size, just to show.
    #     chunk_size = 512,
    #     chunk_overlap  = 24,
    #     length_function = count_tokens,
    # )

    # chunks = text_splitter.create_documents([text])
    return chunks

# Create a vector store embeddings
def get_vectors_store(chunk):
    embeddings = OpenAIEmbeddings()
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=chunk, embedding=embeddings)
    return vectorstore

# Create a conversational retrieval using memory and retrieval from langchain
def get_conversational_chains(vectorstore):
    llm = ChatOpenAI()
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

# Function to handle the user interaction
def handle_userinput(user_question):
    response  = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']
    
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)


def main():
    load_dotenv()
    output_folder = "./output_folder"
    classification_threshold = 10
    
    st.set_page_config(page_title="Chat with multiple PDF files!", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)
    
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
        
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    
    st.header("Chat with multiple PDF :books:")
    # Allow users to ask questions
    user_question = st.text_input("Please enter your question about your documents:")
    
    if user_question:
        handle_userinput(user_question)
        
    # Write the messages from user and chat
    # st.write(user_template.replace("{{MSG}}", "Hello robot"), unsafe_allow_html=True)
    # st.write(bot_template.replace("{{MSG}}", "Hello human"), unsafe_allow_html=True)
    
    # Add sidebar to enable user to add doc
    # use with to add things into sidebar
    with st.sidebar:
        st.subheader("Upload your documents")
        pdf_docs = st.file_uploader("Upload PDF here...", 
                         accept_multiple_files=True,
                         type=["pdf"],
                        #  max_upload_size=10 * 1024 * 1024  # 10 MB in bytes
                         )
        # Check if a file is uploaded
        # if pdf_docs is not None:
            # pdf_document = fitz.open(pdf_docs)
            # print(pdf_docs)
        # Check if the file is scanned pdf or not using functions above
        
        # Check if the process is clicked then perform actions
        if st.button("Process"):
            with st.spinner("Processing.."):
                # Get pdf text
                raw_text = extract_text(pdf_docs)
                
                # Get the text chuncks
                get_chunks = get_text_chunks(raw_text)
                
                # Create vector store, embeddings to store in database so users can easily query
                # Paid - OpenAI
                vectorstore = get_vectors_store(get_chunks)
                # st.write(vectorstore)
                # cerate conversation chain
                st.session_state.conversation = get_conversational_chains(vectorstore)
                
                # Free version - instructor - embeddings (slower but faster on GPU)
    

if __name__ == '__main__':
    main()