import os
import sqlite3
import hashlib
import streamlit as st
import google.generativeai as genai

from PyPDF2 import PdfReader
from streamlit_option_menu import option_menu
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from email_validator import validate_email, EmailNotValidError
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv


UPLOAD_DIR = "user_uploaded_files"
DB_NAME = "classroom.db"

load_dotenv()
genai.configure(api_key = os.getenv('GOOGLE_API_KEY'))

gemini = genai.GenerativeModel("gemini-2.0-flash")


def initialize_database():
    with sqlite3.connect(DB_NAME) as conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS students(
            student_id INTEGER PRIMARY KEY AUTOINCREMENT,
            first_name TEXT NOT NULL,
            last_name TEXT NOT NULL,
            date_of_birth TEXT NOT NULL,
            email TEXT NOT NULL UNIQUE,
            password TEXT NOT NULL                                         
        )
        """)

        conn.execute("""
        CREATE TABLE IF NOT EXISTS files(
            file_id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER NOT NULL,
            file_name TEXT NOT NULL,
            file_path TEXT NOT NULL,
            FOREIGN KEY (student_id) REFERENCES students(student_id)
        )
        """) 
        
    print("Database and Tables initialized")  
    
def hash_password(password):
     return hashlib.sha256(password.encode()).hexdigest()

def email_check(email):
    if email:
        try:
            st.write(email)
            validate_email(email,check_deliverability=False)
            return True
        except EmailNotValidError:
            return # The above code simply contains the Python keyword `False` and some comment lines
            # denoted by `
            False
    else:
        return False   
        

def sign_up(first_name, last_name, date_of_birth, email, password):
    with sqlite3.connect(DB_NAME) as conn:
        try:
            conn.execute("""
            INSERT INTO students(first_name, last_name, date_of_birth, email, password)
            VALUES (?, ?, ?, ?, ?)                    
            """, (first_name, last_name, date_of_birth, email, hash_password(password)))
            conn.commit()
            return True, "Dear Student! Your account has been created Successfully, You're all set to Login"
        
        except sqlite3.IntegrityError:
            return False, "This email {email} is already registered, Please try logging in."
        

def login(email, password):
    with sqlite3.connect(DB_NAME) as conn:
        user = conn.execute("""
        SELECT student_id, first_name, last_name FROM students WHERE email = ? AND password = ?                            
        """, (email, hash_password(password))).fetchone()
        return user if user else None
    

def save_file(student_id,  file_name, file_path):
    with sqlite3.connect(DB_NAME) as conn:
        conn.execute("""
        INSERT INTO files (student_id, file_name, file_path)
        VALUES (?, ?, ?)
        """, (student_id, file_name, file_path))
        conn.commit()

def get_user_files(student_id):
    with sqlite3.connect(DB_NAME) as conn:
        files = conn.execute("""
        SELECT file_name, file_path FROM files where student_id = ? 
        """, (student_id,)).fetchall()
        return files

def delete_file(student_id,  file_name):
    with sqlite3.connect(DB_NAME) as conn:
        conn.execute("""
        DELETE FROM files WHERE (student_id, file_name) = (?, ?)
        """, (student_id,  file_name)).fetchall()
        conn.commit()

initialize_database()


#embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

def get_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size = 10000, chunk_overlap = 200)
    chunks = splitter.split_text(text)
    return chunks

def get_vector_store(chunks):
    db = FAISS.from_texts(chunks, embeddings)
    return db

def get_rel_text(user_query, db):
    rel_text = db.similarity_search(user_query, k=1)
    return rel_text[0].page_content if rel_text else "No relevant information found."

def bot_response(model, query, relevant_texts, history):
    context = " ".join(relevant_texts)
    prompt = f"""

    This is the context of the document
    Context : {context}
    And this is the User Query
    User : {query}
    And this is the history of the conversation
    History : {history}

    Please generate a response to the user query based on the context and the history of the conversation.
    The questions might be asked related to the provided context, and may also be in terms of the medical field, diseases, biology, etc.
    Answer the question with respect to the context provided, you can also use your additional knowledge too, but do not ignore the content of the provided medical report.
    Answer the following queries like a professional doctor, having a lot of knowledge on the basis of report context.

    Bot :

    """

    response = model.generate_content(
        prompt,
        generation_config = genai.GenerationConfig(
            temperature = 0.68
        )
    )

    return response.text

def get_value(i, lst):
    for pair in lst:
        if pair[0] == i:
            return pair[1]
    return None

st.set_page_config(page_title="ClassRoomMate", page_icon="📖", layout="wide")

def home_description():
    description = """
    
**Learn better. Ask anything. Anytime.**
    
Classroom Bot is your all-in-one learning assistant designed to make studying easy, fun, and effective! Whether you're stuck on a tricky math problem, curious about a science concept, exploring history, or brushing up on your English grammar — Classroom Bot is here to help, 24/7.

With support for Maths, Science, Social Studies, and English, our smart bot can:

- Explain topics clearly and simply

- Help with homework and assignments

- Provide step-by-step solutions

- Offer quizzes and practice questions

- Encourage independent learning

💡 Why struggle alone when Classroom Bot is ready to guide you every step of the way?

Start learning smarter today!
    
    """
    
    return description


if 'messages' not in st.session_state:
    st.session_state.messages = {}

with st.sidebar:    
    selected = option_menu(
        "Menu", ["Home", "Student Login/Register", "Science Bot", "Maths Bot","Social Bot", "English Bot",  "Student Peformance Record Bot"],
        icons = ["house", "person", "chat-dots",  "chat-dots",  "chat-dots", "chat-dots", "report-card"],
        menu_icon = "cast", default_index=0
    )


if selected == "Home":
    st.header ("Welcome to Classroom Bot – Your Smart Study Companion!")
    st.markdown(home_description())


if selected == "Student Login/Register":
    st.header("Student Login/Register")

    if "student_id" in st.session_state:
        st.info(f"You are logged in as {st.session_state['first_name']} {st.session_state['last_name']}")
        if st.button("Logout"):
            st.session_state.clear()
            st.success("Logged Out Successfully!")

    else:
        action = st.selectbox("Select an action", ['Login', 'Sign Up'])

        if action == "Sign Up":
            st.subheader("Sign Up")
            first_name = st.text_input("First Name")
            last_name = st.text_input("Last Name")
            dob = st.date_input("Date Of Birth")
            email = st.text_input("Email")
            password = st.text_input("Password", type='password')

            if st.button("Sign Up"):
                if email_check(email):
                    success, msg = sign_up(first_name, last_name, dob, email, password)
                    if success:
                        st.success(msg)
                    else:
                        st.error(msg)
                else:
                    st.error("Email address is not provided or Invalid! Please check.")

        
        elif action == "Login":
            st.subheader("Login")
            email = st.text_input("Email")
            password = st.text_input("Password", type='password')
            
            if st.button("Login"):
                user = login(email, password)
                if user:
                    st.session_state['student_id'], st.session_state['first_name'], st.session_state['last_name'] = user
                    st.session_state['science'] = "".join(['s',user[1],user[2]])
                    st.session_state['maths'] = "".join(['m',user[1],user[2]])
                    st.session_state['social'] = "".join(['ss',user[1],user[2]])
                    st.session_state['english'] = "".join(['e',user[1],user[2]])
                    st.session_state['rag']  = "".join(['rag',user[1],user[2]])
                    st.success(f"Logged in as: {user[1]} {user[2]}!")
                    st.session_state.messages[st.session_state['student_id']] = []
                    st.session_state.messages[st.session_state['science']] = []
                    st.session_state.messages[st.session_state['maths']] = []
                    st.session_state.messages[st.session_state['social']] = []
                    st.session_state.messages[st.session_state['english']] = []
                    st.session_state.messages[st.session_state['rag']] = []
                else:
                    st.error("Invalid Email or Password")

def get_model(subject):
    match subject:
        case "science":
            instructions =  """"

            Your name is "VignanBot" and you are a teach who teaches science subject and let the student know the details of the question with examples for the question provided.

            Your Roles:
            1) you are a VignanBot, who is intelligent to understand the question and analyze, gather appropriate info with examples and provide the student.
            2) you are a highly talented science teacher who has very good knoweldge of physics, chemistry and biology. And you know class room teaching methodologies and explain the concepts with ease in a simple understanable language and make learning an exciting adventure.
            3) you have knowledge of Traditional science and modern science. You should not teach any concepts of maths or any other subjects. your scope is limited to science subject.
            4) If possible you can also give the examples and theories of the relavent topic of the modern science.  
            
            Points to remember:
            1) You should engage with student like a classroom teacher, mentor and coach, and give the user proper reply for his queries
            2) The conentration and the gist of the conversation no need to be completely based on query provided, your flow of chat should be more like a human conversation.
            3) If the conversation goes way too out of subject or concept or if the user input is abusive, let the user know that the content is abusive and we cannot tolerate suh inputs.
            4) The important part is that you should not anywhere mention, "You should enroll into a instructor led tranings for further details"

            """
        case "maths":
            instructions = """"

            Your name is "Ramanujan Bot" and you are a teach who teaches Mathematics subject and let the student know the details of the question with examples for the question provided.
            Your Roles:
            1) you are a Ramanujan Bot, who is intelligent to understand the question and analyze, gather appropriate info with examples and provide the student.
            2) you are a highly talented Maths teacher, know class room teaching methodologies and explain the concepts with ease in a simple understanable language and make learning an exciting adventure.
            3) you have to solve the math problem and provide step and step details and explanation. You should not teach any concepts of science or any other subjects. your scope is limited to Maths subject.
            4) If possible you can also give the examples and applicable concepts of theorems with proofs and keep it clear and precise to the student.  
            
            Points to remember:
            1) You should engage with student like a classroom teacher, mentor and coach, and give the user proper reply for his queries
            2) The conentration and the gist of the conversation no need to be completely based on query provided, your flow of chat should be more like a human conversation.
            3) If the conversation goes way too out of subject or concept or if the user input is abusive, let the user know that the content is abusive and we cannot tolerate such inputs.
            4) The important part is that you should not anywhere mention, "You should enroll into a instructor led tranings for further details"

            """
        case "social":
            instructions = """"

            Your name is "VishwamBot" and you are the one teach who teaches Social studies subject and let the student know the details of the question with examples for the question provided.
            Your Roles:
            1) you are a VishwamBot, who is intelligent to understand the question and analyze, gather appropriate info with examples and provide the student.
            2) you are a highly talented Social teacher, know class room teaching methodologies and explain the concepts with ease in a simple understanable language and make learning an exciting adventure.
            3) You have enoromus knowledge of World Geographly, Civics, History and Economics. You should not teach any concepts of other subjects except Socical. your scope is limited to Georgraphy, History, Civics and Economics subject.
            4) If possible you can also give the examples when mentioning the concepts. keep it clear and precise to the student.  
            
            Points to remember:
            1) You should engage with student like a classroom teacher, mentor and coach, and give the user proper reply for his queries
            2) The conentration and the gist of the conversation no need to be completely based on query provided, your flow of chat should be more like a human conversation.
            3) If the conversation goes way too out of subject or concept or if the user input is abusive, let the user know that the content is abusive and we cannot tolerate such inputs.
            4) The important part is that you should not anywhere mention, "You should enroll into a instructor led tranings for further details"

            """
            
        case "english":
            instructions = """"

            Your name is "ShakespeareBot" and you are a teach who teaches English subject and let the student know the details of the question with examples for the question provided.
            Your Roles:
            1) you are a ShakespeareBot, who is intelligent to understand the question and analyze, gather appropriate info with examples and provide the student.
            2) you are a highly Proficient English teacher, know class room teaching methodologies and explain the concepts with ease in a simple understanable language and make learning an exciting adventure.
            3) Explain the grammer, words, sentences, peoms and all english related concepts. You should not teach any concepts of any other subjects except english. your scope is limited to English subject.
            4) If possible you can also give the examples and applicable concepts of theorems with proofs and keep it clear and precise to the student.  
            
            Points to remember:
            1) You should engage with student like a classroom teacher, mentor and coach, and give the user proper reply for his queries
            2) The conentration and the gist of the conversation no need to be completely based on query provided, your flow of chat should be more like a human conversation.
            3) If the conversation goes way too out of subject or concept or if the user input is abusive, let the user know that the content is abusive and we cannot tolerate suh inputs.
            4) The important part is that you should not anywhere mention, "You should enroll into a instructor led tranings for further details"

            """
    model = genai.GenerativeModel(model_name="gemini-2.0-flash",
    system_instruction   = instructions
    )
    return model
            
def classroom_bots(subject, bot_name):
    if 'student_id' not in st.session_state:
        st.warning(f"Please login to access the {bot_name} Bot")

    else:
        st.info(f"Welcome {st.session_state['first_name']} !!")
        st.write(f"I'm your {subject} guide. Feel free to ask me any questions related to  {subject} subject.")
        
        chat_history = st.session_state.messages.get(st.session_state[subject], [])

        chat_bot = model.start_chat(
            history = chat_history
        )

        for message in chat_history:
            if message['role'] == 'user':
                st.chat_message(message['role']).markdown(message['parts'][0]['text'])
            else:
                st.chat_message(message['role']).markdown(message['parts'][0]['text'])

        user_question = st.chat_input("Type your message here: ")

        if user_question:
            st.chat_message("user").markdown(user_question)
            chat_history.append(
                {'role': 'user',
                 'parts': [{'text' : user_question}]
                }
            )

            with st.spinner("Cooking..."):
                response = chat_bot.send_message(user_question)

                st.chat_message("assistant").markdown(response.text)

                chat_history.append(
                    {
                        'role':'assistant',
                        'parts': [{'text': response.text}]
                    }
                )

            st.session_state.messages[st.session_state[subject]] = chat_history

if selected == "Science Bot":
    model = get_model('science')
    st.subheader("Chat with Vignan Bot")
    classroom_bots('science','Vignan')
    
elif selected == "Maths Bot":
    model = get_model('maths')
    st.subheader("Chat with Ramanujan Bot")
    classroom_bots('maths','Ramanujan'),
    
elif selected == "Social Bot":
    model = get_model('social')
    st.subheader("Chat with Vishwam Bot")
    classroom_bots('social','Vishwam')

elif selected == "English Bot":
    model = get_model('english')
    st.subheader("Chat with Shakespeare Bot")
    classroom_bots('english','Shakespeare')
else:
    if selected == "Student Peformance Record Bot":
        st.subheader("Student Peformance Report Reader")

        if 'student_id' not in st.session_state:
            st.warning("Please login to access the Student Performance Record Bot")

        else:
            with st.expander("Select the feature ", expanded = True):
                choice = st.radio(
                    label = "Select an option",
                    options = ["Upload Performance Report", "Chat with Performance Report Bot"]
                )

            st.info(f"Welcome {st.session_state['first_name']} !!")

            if choice == "Upload Performance Report":
                file = st.file_uploader(label = "Upload your report card", type = 'pdf')

                if file:
                    file_name = file.name
                    file_path = os.path.join(UPLOAD_DIR, f"{st.session_state['student_id']}_{file_name}")
                    os.makedirs(UPLOAD_DIR, exist_ok=True)

                    if not os.path.exists(file_path):                    
                        with open(file_path, 'wb') as f:
                            f.write(file.getbuffer())            

                    if st.button("Save file"):
                        save_file(st.session_state['student_id'], file_name, file_path)
                        st.success(f"File {file_name} saved successfully!")


                st.subheader("Your Uploaded Files")
                files = get_user_files(st.session_state['student_id'])
                if files:
                    for file_name, file_path in files:
                        st.markdown(f"- {file_name}")
                        if st.button(f"Delete {file_name}"):
                            delete_file(st.session_state['student_id'], file_name)
                            if os.path.exists(file_path):
                                os.remove(file_path)
                            st.success(f"File {file_name} deleted successfully!")

                    st.subheader("File Content Viewer")
                    s_file = st.selectbox(label = "Select the file", options = [i for i,v in files])

                    def get_value(i, lst):
                        for pair in lst:
                            if pair[0] == i:
                                return pair[1]
                        return None
                    
                    if s_file:
                        file_path = get_value(s_file, files)
                        script_path = os.path.abspath(__file__)
                        script_directory = os.path.dirname(script_path)
                        fullfile_path = "".join([script_directory,"\\",file_path])
                        st.write(fullfile_path)
                        if st.button("View Content"):
                            with st.spinner("Giving the details"):
                                st.write(file_path)
                                if os.path.exists(file_path):
                                    if os.path.getsize(fullfile_path) > 0:
                                        text = ''
                                        pdf_reader = PdfReader(fullfile_path)
                                        for page in pdf_reader.pages:
                                            text += page.extract_text()
                                        st.subheader(f"The content of {file_name}")
                                        st.markdown(text)
                                    else:
                                        st.error("The uploaded PDF file is empty.")
                                else:
                                    st.error("File not found at the specified path.")
                else:
                    st.info("No files uploaded yet.")
            if choice == "Chat with Performance Report Bot":
            #################################################
            ## RAG system - my quota exceded , cannot check this functionality fully. Other than this rest of the functiaonly is working fine
            ####################################################
                files = get_user_files(st.session_state['student_id'])
                s_file = st.selectbox(label = "Select the file", options = [i for i,v in files])
                text =''
                if s_file:
                    file_path = get_value(s_file, files)
                    pdf_reader = PdfReader(file_path)
                    for page in pdf_reader.pages:
                        text += page.extract_text()
                    chunks = get_chunks(text)
                    vector_db = get_vector_store(chunks)
                    
                    rag_model = genai.GenerativeModel(
                        model_name = 'gemini-2.0-flash',
                        system_instruction = """
                        You are a very experienced answer provider,
                        Based on the relevant content given to you, you have the ability to easily answer the query asked by the user.

                        """
                    )
                    
                chat_history = st.session_state.messages.get(st.session_state['rag'], [])
                chat_bot = rag_model.start_chat(
                    history = chat_history
                )

                for message in chat_history:
                    if message['role'] == 'user':
                        st.chat_message(message['role']).markdown(message['parts'][0]['text'])
                    else:
                        st.chat_message(message['role']).markdown(message['parts'][0]['text'])

                user_question = st.chat_input("Type your message here: ")

                if user_question:
                    st.chat_message("user").markdown(user_question)
                    chat_history.append(
                        {'role': 'user',
                        'parts': [{'text' : user_question}]
                        }
                    )
                    rel_text = get_rel_text(user_question,vector_db)

                    with st.spinner("Cooking..."):
                        response = bot_response(rag_model,user_question,rel_text,chat_history)
                        st.chat_message("assistant").markdown(response)
                        chat_history.append(
                        {
                        'role':'assistant',
                        'parts': [{'text': response}]
                        }
                         )
                        st.session_state.messages[st.session_state['rag']] = chat_history
                
                
            






 


    
    
    
