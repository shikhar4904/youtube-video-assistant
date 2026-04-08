import streamlit as st
from main_backend import reply, extract_video_id, extract_transcript, retrieve_documents


#################  Session State  #################

if 'retriever' not in st.session_state:
    st.session_state['retriever'] = None

if 'video_id' not in st.session_state:
    st.session_state['video_id'] = None

if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []


#################  Sidebar  #################

st.sidebar.title("Enter youtube video link")
video_url = st.sidebar.text_input("URL")
st.session_state['video_id'] = extract_video_id(video_url)
load_button = st.sidebar.button("Load")

if load_button:
    if st.session_state['video_id']:
        transcript = extract_transcript(video_url)
        if transcript:
            st.session_state['retriever'] = retrieve_documents(video_url)
            st.sidebar.success("Video loaded successfully")
            st.sidebar.button(st.session_state['video_id'])
        else:
            st.sidebar.error("No transcript available for this video")
    else:
        st.sidebar.error("Please enter the video url")


#################  User input and AI response  #################

user_query = st.chat_input("Enter your question here")

for i in st.session_state['message_history']:
    with st.chat_message(i['role']):
        st.text(i['content'])

if user_query:
    
    st.session_state['message_history'].append({'role': 'user', 'content': user_query})
    with st.chat_message('user'):
        st.text(user_query)

    ai_response = reply(user_query, st.session_state['retriever'])
    st.session_state['message_history'].append({'role': 'ai', 'content': ai_response})
    with st.chat_message('ai'):
        st.text(ai_response)