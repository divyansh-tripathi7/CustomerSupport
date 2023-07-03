from langchain.document_loaders import CSVLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
import os

os.environ["OPENAI_API_KEY"] = "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"

loader = CSVLoader(file_path='NOW.csv')

index_creator = VectorstoreIndexCreator()
docsearch = index_creator.from_loaders([loader])

# Create a question-answering chain using the index
chain = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=docsearch.vectorstore.as_retriever(), input_key="question")

# just need chain rn
# query = input("whats up?:  ")

# response = chain({"question": query})
# print(response['result'])
