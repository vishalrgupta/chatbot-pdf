# File: utils.py
import os
from operator import itemgetter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferWindowMemory
from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv

load_dotenv()
memory = ConversationBufferWindowMemory(k=5, return_messages=True)

def load_azure_openai():
    model_openai = AzureChatOpenAI(
        azure_deployment="gpt-35-turbo",
        openai_api_version=os.getenv('OPENAI_API_VERSION'),
        verbose=True,
        openai_api_key=os.getenv('OPENAI_API_KEY'),
        azure_endpoint=os.getenv('AZURE_ENDPOINT'))
    return model_openai
# Instantiate QA object

def setup_dbqa(query):

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': 'cpu'})

    vectordb = FAISS.load_local(folder_path='../../db_faiss', embeddings=embeddings)
    model=load_azure_openai()
    retriever = vectordb.as_retriever(search_type="similarity_score_threshold", search_kwargs={"k": 2, "score_threshold": .5})

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    contextualize_q_system_prompt = """Given a chat history and the latest user question \
    which might reference context in the chat history, formulate a standalone question \
    which can be understood without the chat history. Do NOT answer the question, \
    just reformulate it if needed and otherwise return it as is."""

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
    )
    contextualize_q_chain = contextualize_q_prompt | model | StrOutputParser()

    qa_system_prompt = """You are an assistant for question-answering tasks.\
    You are a technical helpdesk bot for KIA. KIA is car manufacturer.\
    
        Use the following pieces of retrieved context i.e. Technical Sevice Bulletin to answer the question. \
        Always try to answer question with required parts numbers to be replaced if required.\
        If answer to user's question is not present in the below context, simply say I do not know the answer AND i such case do not return any source document.\

        {context}
        
        """

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
    )

    def contextualized_question(input: dict):
        if input.get("chat_history"):
            return contextualize_q_chain
        else:
            return input["question"]


    rag_chain = (
            RunnablePassthrough.assign(
                chat_history=RunnableLambda(memory.load_memory_variables) | itemgetter("history")
            ).assign(source_documents=contextualized_question | retriever)
            | RunnablePassthrough.assign(
        context=lambda inputs: format_docs(inputs["source_documents"]) if inputs["source_documents"] else "")
            | RunnablePassthrough.assign(prompt=qa_prompt)
            | RunnablePassthrough.assign(response=lambda inputs: model(inputs["prompt"].messages))
    )

    response = rag_chain.invoke({"question": query})
    memory.save_context({"input": query}, {"output": response['response'].content})
    return response['response'].content, response['source_documents']


#
# if __name__ == "__main__":
#     response, source_document = setup_dbqa(query='What is the weather ?')
#     print(response)
#     print(source_document)

