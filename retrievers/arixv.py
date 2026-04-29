from langchain_community.retrievers import ArxivRetriever

# create the retriever
retriever = ArxivRetriever(
    load_max_docs=3,
    load_all_available_meta=True,
)

# query arxiv
# keyword matching
docs = retriever.invoke("Large Language Models")

for i, doc in enumerate(docs):
    print(f"Document {i+1}:")
    print(f"Title: {doc.metadata['Title']}")
    print(f"Authors: {doc.metadata['Authors']}")
    print(f"Summary: {doc.page_content[:500]}")