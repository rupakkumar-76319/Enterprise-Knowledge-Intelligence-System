import json
from typing import Iterator
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document

class FmjlLoader(BaseLoader):
    def __init__(self, file_path):
        self.file_path = file_path

    def lazy_load(self) -> Iterator[Document]:
        chapter, page, text = None, None, []
        with open(self.file_path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                e = json.loads(line)
                if e.get("type") not in ("paragraph", "heading") or not e.get("md"):
                    continue
                key = e["id"] if e["type"] == "heading" else e.get("parent")
                if key is None:                      # front matter (before Chapter I)
                    continue
                if key != chapter and text:
                    yield Document(page_content="\n\n".join(text), metadata={"chapter": chapter, "page": page})
                    text = []
                if not text:
                    page = e.get("page")             # page where the chapter starts
                chapter = key
                if e.get("continues") and text:      # paragraph split by a page break
                    prev = text[-1]
                    text[-1] = prev[:-1] + e["md"] if prev.endswith("-") else prev + " " + e["md"]
                else:
                    text.append(e["md"])
        if text:
            yield Document(page_content="\n\n".join(text), metadata={"chapter": chapter, "page": page})
