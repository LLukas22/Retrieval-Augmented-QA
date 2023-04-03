from typing import List
from haystack.schema import Document
from haystack.document_stores import BaseDocumentStore
from schemas.query import FilterRequest
from ._router  import BaseRouter

class DocumentRouter(BaseRouter):
    def __init__(self,document_store:BaseDocumentStore):
        super().__init__("/documents")
        self.document_store = document_store
        self.router.add_api_route("/get", self.get_documents, methods=["POST"], response_model=List[Document], response_model_exclude_none=True)
        self.router.add_api_route("/delete", self.delete_documents, methods=["POST"], response_model=bool)

    def get_documents(self,filters: FilterRequest):
        """
        This endpoint allows you to retrieve documents contained in your document store.
        You can filter the documents to retrieve by metadata (like the document's name),
        or provide an empty JSON object to clear the document store.

        Example of filters:
        `'{"filters": {{"name": ["some", "more"], "category": ["only_one"]}}'`

        To get all documents you should provide an empty dict, like:
        `'{"filters": {}}'`
        """
        docs = self.document_store.get_all_documents(filters=filters.filters)
        for doc in docs:
            doc.embedding = None
        return docs


    def delete_documents(self,filters: FilterRequest):
        """
        This endpoint allows you to delete documents contained in your document store.
        You can filter the documents to delete by metadata (like the document's name),
        or provide an empty JSON object to clear the document store.

        Example of filters:
        `'{"filters": {{"name": ["some", "more"], "category": ["only_one"]}}'`

        To get all documents you should provide an empty dict, like:
        `'{"filters": {}}'`
        """
        self.document_store.delete_documents(filters=filters.filters)
        return True
