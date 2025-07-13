import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from streamlit_chat import message

import streamlit as st
import sqlite3
import base64

st.markdown(f'<h1 style="color:#8d1b92;text-align: center;font-size:36px;">{"Developing an AI based interactive chatbot or virtual assistant on department of justice website"}</h1>', unsafe_allow_html=True)


st.write("---------------------------------------------------------------------------------")

st.markdown(f'<h1 style="color:#000000;font-size:24px;text-align:center;font-family:canvat;">{"Login Here !!!"}</h1>', unsafe_allow_html=True)



def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
add_bg_from_local('b1.jpg')

# Load dataset from the Excel file
df = pd.read_excel("Dataset.xlsx")

# Check that your dataset contains the expected columns
if 'Question' not in df.columns or 'Answer' not in df.columns:
    raise ValueError("The dataset must contain 'Questions' and 'Answers' columns")

# Function to get the answer based on the user's question
def get_answer(user_question, df):
    # Initialize a TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(stop_words='english')
    
    # Fit the vectorizer on the dataset's question column
    tfidf_matrix = vectorizer.fit_transform(df['Question'])
    
    # Transform the user's input question
    user_tfidf = vectorizer.transform([user_question])
    
    # Calculate cosine similarity between the user's question and dataset questions
    similarity = cosine_similarity(user_tfidf, tfidf_matrix)
    
    # Get the index of the most similar question
    most_similar_idx = similarity.argmax()
    
    # Retrieve and return the corresponding answer
    return df.iloc[most_similar_idx]['Answer']

# Streamlit app layout
# st.title("Chatbot")

# Initialize session state variables to store conversation history if they don't exist
if 'user_input' not in st.session_state:
    st.session_state['user_input'] = []

if 'openai_response' not in st.session_state:
    st.session_state['openai_response'] = []

# Function to get the user input text box
def get_text():
    input_text = st.text_input("Type your question here", key="input")
    return input_text

# Function to handle user interaction and conversation flow
def chat():
    user_input = get_text()

    # If the user enters 'thank you', stop the conversation and reset the session
    if user_input and user_input.lower() in ["thank you", "thanks", "goodbye"]:
        st.session_state['user_input'].append("Thank you! Goodbye.")
        st.session_state['openai_response'].append("You're welcome! Have a great day!")
        return

    # If the user asks a question, get the response
    if user_input:
        answer = get_answer(user_input, df)
        # Store the user input and bot response in the session state
        st.session_state.openai_response.append(answer)
        st.session_state.user_input.append(user_input)

# Call the chat function to handle user input and response
chat()

# Display the message history
if st.session_state['user_input']:
    for i in range(len(st.session_state['user_input']) - 1, -1, -1):
        # Display the user's message
        message(st.session_state["user_input"][i], 
                key=str(i), avatar_style="icons")
        # Display the chatbot's response
        message(st.session_state['openai_response'][i], 
                avatar_style="miniavs", is_user=True,
                key=str(i) + 'data_by_user')

# Button to start a new conversation
if len(st.session_state['user_input']) > 0 and (st.session_state['user_input'][-1].lower() in ["thank you", "thanks", "goodbye"]):
    if st.button("Start a new conversation"):
        # Clear the previous conversation history to start fresh
        st.session_state['user_input'] = []
        st.session_state['openai_response'] = []
