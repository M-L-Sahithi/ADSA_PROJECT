import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset from the Excel file
df = pd.read_excel("Dataset.xlsx")

# Check that your dataset contains the expected columns
print(df.head())

# Function to get the answer based on the user's question
def get_answer(user_question, df):
    # Ensure the column names are correct
    if 'Question' not in df.columns or 'Answer' not in df.columns:
        raise ValueError("The dataset must contain 'Questions' and 'Answers' columns")

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

# Example usage
user_question = input("Enter your question: ")
answer = get_answer(user_question, df)
print("Answer:", answer)



# import streamlit as st
# import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity

# # Load the dataset from the Excel file
# df = pd.read_excel("Dataset.xlsx")

# # Check if the dataset has the expected columns
# if 'Question' not in df.columns or 'Answer' not in df.columns:
#     st.error("The dataset must contain 'Questions' and 'Answers' columns")
#     st.stop()

# # Function to get the answer based on the user's question
# def get_answer(user_question, df):
#     vectorizer = TfidfVectorizer(stop_words='english')
#     tfidf_matrix = vectorizer.fit_transform(df['Question'])
#     user_tfidf = vectorizer.transform([user_question])
    
#     similarity = cosine_similarity(user_tfidf, tfidf_matrix)
#     most_similar_idx = similarity.argmax()
    
#     return df.iloc[most_similar_idx]['Answer']

# # Streamlit app title
# st.title("Question Answering System")

# # Introduction text
# st.write("Ask me anything related to the dataset. Type 'thank you' to exit.")

# # Session state to store user input history
# if "question_history" not in st.session_state:
#     st.session_state.question_history = []

# # Display the question history if available
# if len(st.session_state.question_history) > 0:
#     st.write("Your Previous Questions:")
#     for idx, item in enumerate(st.session_state.question_history):
#         st.write(f"{idx+1}. {item['question']} - Answer: {item['answer']}")

# # Text input for asking questions
# user_question = st.text_input("Enter your question:")

# # If the user entered a question
# if user_question:
#     if user_question.lower() == "thank you":
#         st.write("You're welcome! Have a great day!")
#         st.session_state.question_history.clear()  # Clear history if exit message
#     else:
#         # Get the answer from the dataset
#         answer = get_answer(user_question, df)

#         # Save the current question and answer to history
#         st.session_state.question_history.append({"question": user_question, "answer": answer})

#         # Display the answer
#         st.write("Answer:", answer)


# import streamlit as st
# import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity

# # Load the dataset from the Excel file
# df = pd.read_excel("Dataset.xlsx")

# # Check if the dataset has the expected columns
# if 'Question' not in df.columns or 'Answer' not in df.columns:
#     st.error("The dataset must contain 'Question' and 'Answer' columns")
#     st.stop()

# # Function to get the answer based on the user's question
# def get_answer(user_question, df):
#     vectorizer = TfidfVectorizer(stop_words='english')
#     tfidf_matrix = vectorizer.fit_transform(df['Question'])
#     user_tfidf = vectorizer.transform([user_question])
    
#     similarity = cosine_similarity(user_tfidf, tfidf_matrix)
#     most_similar_idx = similarity.argmax()
    
#     return df.iloc[most_similar_idx]['Answer']

# # Streamlit app title
# st.title("Question Answering System")

# # Introduction text
# st.write("Ask me anything related to the dataset. Type 'thank you' to exit.")

# # Session state to store user input history
# if "question_history" not in st.session_state:
#     st.session_state.question_history = []
# if "waiting_for_answer" not in st.session_state:
#     st.session_state.waiting_for_answer = False

# # Display the question history if available
# if len(st.session_state.question_history) > 0:
#     st.write("Your Previous Questions:")
#     for idx, item in enumerate(st.session_state.question_history):
#         st.write(f"{idx+1}. {item['question']} - Answer: {item['answer']}")

# # Text input for asking questions
# if not st.session_state.waiting_for_answer:
#     user_question = st.text_input("Enter your question:")

#     # If the user entered a question
#     if user_question:
#         if user_question.lower() == "thank you":
#             st.write("You're welcome! Have a great day!")
#             st.session_state.question_history.clear()  # Clear history if exit message
#             st.session_state.waiting_for_answer = False
#         else:
#             # Get the answer from the dataset
#             answer = get_answer(user_question, df)

#             # Save the current question and answer to history
#             st.session_state.question_history.append({"question": user_question, "answer": answer})

#             # Display the answer
#             st.write("Answer:", answer)

#             # Set the flag to wait for the next question
#             st.session_state.waiting_for_answer = True
# else:
#     # Automatically ask for the next question after an answer
#     st.write("Please ask the next question.")
#     st.session_state.waiting_for_answer = False







# import streamlit as st
# import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity

# # Load the dataset from the Excel file
# df = pd.read_excel("Dataset.xlsx")

# # Check if the dataset has the expected columns
# if 'Question' not in df.columns or 'Answer' not in df.columns:
#     st.error("The dataset must contain 'Question' and 'Answer' columns")
#     st.stop()

# # Function to get the answer based on the user's question
# def get_answer(user_question, df):
#     vectorizer = TfidfVectorizer(stop_words='english')
#     tfidf_matrix = vectorizer.fit_transform(df['Question'])
#     user_tfidf = vectorizer.transform([user_question])
    
#     similarity = cosine_similarity(user_tfidf, tfidf_matrix)
#     most_similar_idx = similarity.argmax()
    
#     return df.iloc[most_similar_idx]['Answer']

# # Streamlit app title
# st.title("Question Answering System")

# # Introduction text
# st.write("Ask me up to 5 questions. Type 'thank you' to exit early.")

# # Session state to store user input history and count questions
# if "question_history" not in st.session_state:
#     st.session_state.question_history = []

# if "question_count" not in st.session_state:
#     st.session_state.question_count = 0  # To count the number of questions asked

# # Display the question history if available
# if len(st.session_state.question_history) > 0:
#     st.write("Your Previous Questions:")
#     for idx, item in enumerate(st.session_state.question_history):
#         st.write(f"{idx+1}. {item['question']} - Answer: {item['answer']}")

# # Limit to 5 questions
# if st.session_state.question_count < 5:
#     # Text input for asking questions
#     user_question = st.text_input(f"Enter question {st.session_state.question_count + 1}:")

#     # If the user entered a question
#     if user_question:
#         if user_question.lower() == "thank you":
#             st.write("You're welcome! Have a great day!")
#             st.session_state.question_history.clear()  # Clear history if exit message
#             st.session_state.question_count = 5  # Stop asking questions
#         else:
#             # Get the answer from the dataset
#             answer = get_answer(user_question, df)

#             # Save the current question and answer to history
#             st.session_state.question_history.append({"question": user_question, "answer": answer})

#             # Display the answer
#             st.write("Answer:", answer)

#             # Increment the question count
#             st.session_state.question_count += 1

# else:
#     # Stop asking questions after 5 questions
#     st.write("You've asked 5 questions. Thank you for using the service!")




import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the dataset from the Excel file
df = pd.read_excel("Dataset.xlsx")

# Check if the dataset has the expected columns
if 'Question' not in df.columns or 'Answer' not in df.columns:
    st.error("The dataset must contain 'Question' and 'Answer' columns")
    st.stop()

# Function to get the answer based on the user's question
def get_answer(user_question, df):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(df['Question'])
    user_tfidf = vectorizer.transform([user_question])
    
    similarity = cosine_similarity(user_tfidf, tfidf_matrix)
    most_similar_idx = similarity.argmax()
    
    return df.iloc[most_similar_idx]['Answer']

# Streamlit app title
st.title("Question Answering System")

# Introduction text
st.write("Ask me up to 5 questions. Type 'thank you' to exit early.")

# Session state to store user input history and count questions
if "question_history" not in st.session_state:
    st.session_state.question_history = []

if "question_count" not in st.session_state:
    st.session_state.question_count = 0  # To count the number of questions asked

# Display the question history if available
if len(st.session_state.question_history) > 0:
    st.write("Your Previous Questions:")
    for idx, item in enumerate(st.session_state.question_history):
        st.write(f"{idx+1}. {item['question']} - Answer: {item['answer']}")

# Limit to 5 questions
if st.session_state.question_count < 5:
    # Get the current question from the dataset
    current_question = df.iloc[st.session_state.question_count]['Question']
    
    # Ask the current question
    user_question = st.text_input(f"Question {st.session_state.question_count + 1}: {current_question}")

    # If the user entered a question
    if user_question:
        if user_question.lower() == "thank you":
            st.write("You're welcome! Have a great day!")
            st.session_state.question_history.clear()  # Clear history if exit message
            st.session_state.question_count = 5  # Stop asking questions
        else:
            # Get the answer from the dataset
            answer = get_answer(user_question, df)

            # Save the current question and answer to history
            st.session_state.question_history.append({"question": user_question, "answer": answer})

            # Display the answer
            st.write("Answer:", answer)

            # Increment the question count
            st.session_state.question_count += 1

else:
    # Stop asking questions after 5 questions
    st.write("You've asked 5 questions. Thank you for using the service!")














