import os
import pinecone
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain


directory = 'data'

pine_api_key = os.getenv("pine_api_key")

def load_docs(directory):
   loader = DirectoryLoader(directory)
   documents = loader.load()
   return documents


documents = load_docs(directory)
len(documents)
print(len(documents))


def split_docs(documents, chunk_size=1000, chunk_overlap=20):
   text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
   docs = text_splitter.split_documents(documents)
   return docs


docs = split_docs(documents)
# print(a)
print(len(docs))
#print(docs)


embeddings = OpenAIEmbeddings(model="ada")

# query_result = embeddings.embed_query("Hello world")
# len(query_result)
# print(query_result)
# print(len(query_result))

pinecone.init(
    api_key=pine_api_key,
    environment="us-west4-gcp"
)

index_name = "langchain-own-project"

index = Pinecone.from_documents(docs, embeddings, index_name=index_name)


def get_similar_docs(query, k=8, score=False):
    if score:
        similar_docs = index.similarity_search_with_score(query, k=k)
    else:
        similar_docs = index.similarity_search(query, k=k)
        return similar_docs


query = "what is tree"

answer = get_similar_docs(query, k=8, score=False)
print(answer)


model = "text-davinci-003"
# model_name = "gpt-3.5-turbo"
# model_name = "gpt-4"
llm = OpenAI(model=model)

chain = load_qa_chain(llm, chain_type="stuff")


def get_answer(query):
    similar_docs = get_similar_docs(query)
    answer = chain.run(input_documents=similar_docs, question=query)
    return answer


#query = "what is universe?"
#answer = get_answer(query)
#print(answer)


query = "what is tree "
answer = get_answer(query)
print(answer)
query_result = embeddings.embed_query(query)
print(query_result)
print(len(query_result))