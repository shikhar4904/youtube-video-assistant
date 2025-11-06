from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
import os

load_dotenv()

url = input("Enter the url of the video: ")
video_id = ''
for i in range(32, len(url)):
    if url[i] == '&':
        break
    video_id += url[i]

youtube_api = YouTubeTranscriptApi()

try:
    transcript_list = YouTubeTranscriptApi.fetch(self=youtube_api, video_id=video_id, languages=["en"])
    transcript = ''
    for i in transcript_list.snippets:
        transcript += ' ' + i.text

except TranscriptsDisabled:
    print("Transcript for this video is not available")

text_spiltter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)

chunks = text_spiltter.create_documents([transcript])

embeddings = GoogleGenerativeAIEmbeddings(
        google_api_key = os.environ['GOOGLE_API_KEY'],
        model = 'gemini-embedding-001'
    )

vector_store = FAISS.from_documents(
        documents = chunks,
        embedding = embeddings
    )

retriever = vector_store.as_retriever(
        search_type = 'similarity',
        search_kwargs = {'k':4}
    )

llm = ChatGoogleGenerativeAI(model = 'gemini-2.5-flash')

prompt = PromptTemplate(
    template="""
      You are a helpful assistant.
      Answer ONLY from the provided transcript context.
      If the context is insufficient, just say you don't know.

      {context}
      Question: {question}
    """,
    input_variables = ['context', 'question']
)

question = input("Enter your question: ")

retrieved_docs = retriever.invoke(question)

context = '\n'.join(doc.page_content for doc in retrieved_docs)

final_prompt = prompt.invoke({'context': context, 'question' : question})

print(llm.invoke(final_prompt).content)