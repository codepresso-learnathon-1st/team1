# MCP Server 시작 지점
import sys
import os
from langchain.retrievers.ensemble import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

from agent.agent1 import repo_to_rag
from modules.rag import DocumentEmbedder


from agent.agent1 import repo_to_rag
from modules.rag import DocumentEmbedder

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from mcp.server.fastmcp import FastMCP

"""
MCP 도구를 사용하기 위해서는 다음과 같이 질문하시면 됩니다:

GitHub 저장소를 검색하고 싶으시다면 (agent1):

"https://github.com/사용자명/저장소명 저장소에 대해 알려주세요"
"https://github.com/tensorflow/tensorflow 코드를 분석해주세요"


저장된 저장소에 대해 질문하고 싶으시다면 (agent2):

"이 코드는 어떤 기능을 하나요?"
"이 라이브러리의 주요 특징은 무엇인가요?"
"이 코드에서 ~부분은 어떻게 작동하나요?"
"""

embedder = DocumentEmbedder()
rag = embedder.get_vectorstore()

mcp = FastMCP(
    name="tema1-leanathon-lst",
    version="0.0.1",
    description="-"
)

@mcp.tool()
async def agent1(repo_url: str) -> str:
    """
    GITHUB Repository Clone ⇒ Embedding and Store in VectorDB
    Clones the given GitHub repository, embeds the source code, and stores it in a VectorDB.
    This process is used for enabling vector-based code search and retrieval-based question answering.

    Parameters:
        repo_url: URL of the GitHub repository
    """

    try:
        return repo_to_rag(repo_url)
    except Exception as e:
        return f"An error occurred while processing the repository: {str(e)}"

@mcp.tool()
async def agent2(query: str, top_k: int = 5) -> str:
    """
    Embedding Search ⇒ Generate Answer
    Receives a question, performs embedding-based similarity search, and generates a response
    based on the most relevant documents from the VectorDB.

    Parameters:
        query: The user's input question
        top_k: Number of top documents to retrieve
    """

    try:
        # bm25_retriever = BM25Retriever.from_documents(split_docs, k=top_k)
        # dense_retriever = db.as_retriever(search_kwargs={"k": top_k})
        # retriever = EnsembleRetriever(
        #     retrievers=[bm25_retriever, dense_retriever],
        #     weights=[0.5, 0.5]
        # )
        # return retriever.get_relevant_documents(query)
        # return "agnet2 check"
        dense_retriever = rag.as_retriever(search_kwargs={"k": top_k})
        results = dense_retriever.get_relevant_documents(query)

        # 결과를 문자열로 변환해서 리턴
        return "\n\n".join([doc.page_content for doc in results])
    except Exception as e:
        return f"An error occurred while generating the response: {str(e)}"

if __name__ == "__main__":
    mcp.run()