# MCP Server 시작 지점
import sys
import os
from langchain_core.documents import Document
from typing import List, Dict, Any 

from agent.agent1 import repository_clone
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
top_k = 5

mcp = FastMCP(
    name="tema1-leanathon-lst",
    version="0.0.1",
    description="-"
)


def format_repo_context(repo_info: Dict[str, Any]) -> str:
    """
    Format repository analysis as markdown.

    Args:
        analysis: Dictionary containing repository analysis
    """
    markdown_result = f"""
    ## 저장소 분석 결과

    ### 기본 정보
    - **저장소 URL**: {repo_info['repository_url']}
    - **총 파일 수**: {repo_info['summary']['total_files']}
    - **총 청크 수**: {repo_info['summary']['document_chunks']}

    ### 저장소 구조
    - **디렉토리 수**: {len(repo_info['structure']['directories'])}
    - **사용 언어**: {', '.join(repo_info['structure']['languages'])}

    ### README 내용
    {repo_info['readme']}
    """
    return markdown_result


def format_search_results(docs: List[Document]) -> str:
    """
    Format search results as markdown.

    Args:
        docs: List of documents to format

    Returns:
        Markdown formatted search results

    """

    if not docs:
        return "No relevant information found."

    markdown_results = "## Search Results\n\n"

    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "Unknown source")
        page = doc.metadata.get("page", None)
        page_info = f" (Page: {page + 1})" if page is not None else ""

        markdown_results += f"### Result {i}{page_info}\n\n"
        markdown_results += f"{doc.page_content}\n\n"
        markdown_results += f"Source: {source}\n\n"
        markdown_results += "---\n\n"

    return markdown_results


@mcp.tool()
async def repo_to_rag(repo_url: str) -> str:
    """
    GITHUB Repository Clone ⇒ Embedding and Store in VectorDB
    Clones the given GitHub repository, embeds the source code, and stores it in a VectorDB.
    This process is used for enabling vector-based code search and retrieval-based question answering.

    Parameters:
        repo_url: URL of the GitHub repository
    """

    try:
        repo_info = await repository_clone(repo_url)
        return format_repo_context(repo_info)
    except Exception as e:
        return f"An error occurred while processing the repository: {str(e)}"

@mcp.tool()
async def rag_to_context(query: str) -> str:
    """
    Embedding Search ⇒ Generate Answer
    Receives a question, performs embedding-based similarity search, and generates a response
    based on the most relevant documents from the VectorDB.

    Parameters:
        query: The user's input question
    """

    try:
        retriever = rag.as_retriever(search_kwargs={"k": top_k})
        results = retriever.get_relevant_documents(query)
        return format_search_results(results)
    except Exception as e:
        return f"An error occurred while generating the response: {str(e)}"

if __name__ == "__main__":
    mcp.run(transport="sse")