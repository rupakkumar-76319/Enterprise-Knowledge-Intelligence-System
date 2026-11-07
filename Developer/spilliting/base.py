import json
from typing import Iterator
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document

class FmjlLoader(BaseLoader):
    def __init__(self, file_path):
        self.file_path = file_path

    def lazy_load(self) -> Iterator[Document]:
        page, text = None, []
        with open(self.file_path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                e = json.loads(line)
                if e.get("type") not in ("paragraph", "heading") or not e.get("md"):
                    continue
                if e.get("page") != page and text:
                    yield Document(page_content="\n\n".join(text), metadata={"page": page})
                    text = []
                page = e.get("page")
                text.append(e["md"])
        if text:
            yield Document(page_content="\n\n".join(text), metadata={"page": page})