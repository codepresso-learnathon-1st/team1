import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.code_loaders import MultiLanguageDocumentLoader
from modules.code_splitter import MultiLanguageDocumentSplitter
from modules.rag import DocumentEmbedder
from modules.repo_manage import clone_repo_url, remove_repository

import uuid

embedder = DocumentEmbedder()
rag = embedder.get_vectorstore()

def repo_to_rag(repo_url: str):
    # 1. 리포지토리 클론
    # repo_path = f"./repo_data/{uuid.uuid4()}"
    repo_path = f"/tmp/repo_data/{uuid.uuid4()}" # 읽기 전용 디렉토리 이슈 해결 코드
    repo = clone_repo_url(repo_url, repo_path)
    if repo is None:
        raise RuntimeError(f"Failed to clone repository: returned None :: ❌ 클론 실패: {repo_url}")
    # 2. 리포지토리 내 모든 파일 로드
    loader = MultiLanguageDocumentLoader(repo.working_dir)
    documents = loader.load_documents()
    # 3. 파일 분할
    splitter = MultiLanguageDocumentSplitter()
    chunks = splitter.split_documents(documents)
    # 4. 분할된 파일 임베딩
    embedder.add_documents(chunks)
    # 5. 레포지토리 클론 데이터 삭제
    remove_repository(repo_path)

    return None


# if __name__ == "__main__":
#     repo_to_rag("https://github.com/openai/openai-python")






