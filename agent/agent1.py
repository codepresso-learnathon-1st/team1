import sys
import os
from typing import Dict, Any

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.code_loaders import MultiLanguageDocumentLoader
from modules.code_splitter import MultiLanguageDocumentSplitter
from modules.rag import DocumentEmbedder
from modules.repo_manage import clone_repo_url, remove_repository

import uuid

embedder = DocumentEmbedder()
rag = embedder.get_vectorstore()

def repository_clone(repo_url: str) -> Dict[str, Any]:
    """GitHub 레포지토리를 RAG에 저장하고 분석 결과를 반환합니다."""
    # 1. 리포지토리 클론
    repo_path = f"/tmp/repo_data/{uuid.uuid4()}"
    repo = clone_repo_url(repo_url, repo_path)
    if repo is None:
        raise RuntimeError(f"Failed to clone repository: returned None :: ❌ 클론 실패: {repo_url}")
    
    try:
        # 2. 리포지토리 분석
        analysis = {
            "repository_url": repo_url,
            "structure": analyze_repository(repo.working_dir),
            "readme": get_readme_content(repo.working_dir),
            "summary": {
                "total_files": 0,
                "languages": set(),
                "main_directories": []
            }
        }
        
        # 3. 리포지토리 내 모든 파일 로드
        loader = MultiLanguageDocumentLoader(repo.working_dir)
        documents = loader.load_documents()
        
        # 4. 파일 분할
        splitter = MultiLanguageDocumentSplitter()
        chunks = splitter.split_documents(documents)
        
        # 5. 분할된 파일 임베딩
        embedder.add_documents(chunks)
        
        # 6. 통계 정보 업데이트
        analysis["summary"]["total_files"] = len(documents)
        analysis["summary"]["document_chunks"] = len(chunks)
        
        return analysis
        
    finally:
        # 7. 레포지토리 클론 데이터 삭제
        remove_repository(repo_path)

def analyze_repository(repo_path: str) -> Dict[str, Any]:
    """레포지토리의 구조를 분석합니다."""
    structure = {
        "directories": [],
        "languages": set(),
        "file_count": 0
    }
    
    for root, dirs, files in os.walk(repo_path):
        if '.git' in dirs:
            dirs.remove('.git')
        
        rel_path = os.path.relpath(root, repo_path)
        if rel_path != '.':
            structure["directories"].append(rel_path)
        
        for file in files:
            structure["file_count"] += 1
            ext = os.path.splitext(file)[1].lower()
            if ext:
                structure["languages"].add(ext[1:])  # 점(.) 제거
    
    structure["languages"] = list(structure["languages"])
    return structure

def get_readme_content(repo_path: str) -> str:
    """README 파일의 내용을 반환합니다."""
    readme_paths = ['README.md', 'README.rst', 'README.txt', 'README']
    
    for path in readme_paths:
        full_path = os.path.join(repo_path, path)
        if os.path.exists(full_path):
            with open(full_path, 'r', encoding='utf-8') as f:
                return f.read()
    
    return "README not found"

# if __name__ == "__main__":
#     repo_to_rag("https://github.com/openai/openai-python")






