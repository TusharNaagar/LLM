
from dotenv import load_dotenv
load_dotenv()
import re
import streamlit as st
import os
import google.generativeai as genai 
import io
import PIL.Image
import time
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
#from langchain.vectorstores import FAISS
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate



path = 'Dataset path'
files = os.listdir(path)



genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model=genai.GenerativeModel("gemini-pro")
model_pro=genai.GenerativeModel("gemini-pro-vision")
chat=model.start_chat(history=[])
def get_gemini_response(question):
    response = chat.send_message(question)
    return response.text

def get_gemini_pro_response(question):
    response = model_pro.generate_content(question)
    return response.text

def read_image(file):
    image = PIL.Image.open(file)
    return image

def display_image(image):
    st.image(image, caption='Uploaded Image')

def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader=PdfReader(pdf)
        for page in pdf_reader.pages:
          text+=page.extract_text()  
    return text

def get_text_chunks(text):
   text_splitter= RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
   chunks=text_splitter.split_text(text)
   return chunks

def get_vector_store(text_chunks):
   embeddings=GoogleGenerativeAIEmbeddings(model="models/embedding-001")
   vector_store=FAISS.from_texts(text_chunks,embedding=embeddings)
   vector_store.save_local("faiss_index")


def autovector(text):        # Function to create a vector store
    raw_text=get_pdf_text(text)
    text_chunks=get_text_chunks(raw_text)
    get_vector_store(text_chunks)

autovector(files)


def get_conversational_chain():             #Creates Conversational Chain , Any prompt engineering tells the model how it should percieve the vector store date
      prompt_template="""
Answer the question as detailed as possible from the provided context,try to give answer in steps,make sure you provide summary of the answer, make sure to provide all the details, if the answer is not in
    provided context just say, "I didn't have relavent answer, I can connect you with agent", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:"""
      model = ChatGoogleGenerativeAI(model="gemini-pro",temperature=0.4)
      prompt=PromptTemplate(template= prompt_template,input_variables=["context","question"])
      chain=load_qa_chain(model,chain_type="stuff",prompt=prompt)

      return chain

def user_input(user_question):                     # Runs Similarity search between user prompt and vector store
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    
    # new_db = FAISS.load_local("faiss_index", embeddings)
    new_db = FAISS.load_local("faiss_index", embeddings,allow_dangerous_deserialization=True)

    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    
    response = chain(
        {"input_documents":docs, "question": user_question}
        , return_only_outputs=True)

    return response["output_text"]

def stream_data(text):                 # Streams output a word at a time
    for word in text.split():
        yield word + " "
        time.sleep(0.02)


st.set_page_config(page_title="Team Skynet")           # Streamlit UI Starts here
st.header("Hi John Doe, How can I help you today?")

if 'chat_history' not in st.session_state:
    st.session_state['chat_history']=[]

input = st.chat_input("Say something")
with st.sidebar:                                            #Sidebar starts here
        st.title("Image")
        image_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        if image_file is not None:
             image = read_image(image_file)
             display_image(image)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
             check_prompt ="Is the image about an electronic device? answer in a word."
             if image_file is not None:
                 response =get_gemini_pro_response([check_prompt,image])
                 if (re.search("Yes",response)):
                     #Prompt engineering here will guide the foundational model for object detection
                     response =get_gemini_pro_response(["""list all                            
                                                        the electronic objects you see in the
                                                         image? what the build and make
                                                         of the device. Do not mention
                                                        what the device is made of. List the 
                                                        damages if you see any.
                                                        start your response 'with I See'
                                                        """, image])
                     st.session_state['chat_history'].append(("Bot",response))
                     st.session_state['chat_history'].append(("Bot","You can ask me questions regarding this product"))
                     st.success("Done")
                 else:
                     st.error("Image is not an electronic object")




if input:
    response = user_input(input)
    st.session_state['chat_history'].append(("Mike",input))
    st.session_state['chat_history'].append(("Bot",response))
    #st.subheader("The Response is")
    



c=1

for role, text in st.session_state['chat_history']:
    i=len(st.session_state['chat_history'])
    if (re.search("Bot",role)) and c==i:
        with st.chat_message(role):
            st.write_stream(stream_data(text))

    if role and c!=i:
        with st.chat_message(role):
            st.write(text)
    c=c+1


#Streamlit ends here
