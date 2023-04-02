from haystack.nodes.base import BaseComponent
from typing import Optional,List,Dict,Tuple,Any,Union
from haystack.schema import MultiLabel, Document

class DocumentTaggingNode(BaseComponent):
    """
    A node to add a value to the meta of a document
    """
    outgoing_edges = 1
    def __init__(self,name:str,value:str):
        super().__init__()
        self._name = name
        self._value=value
        
    def run(
        self,
        query: Optional[str] = None,
        file_paths: Optional[List[str]] = None,
        labels: Optional[MultiLabel] = None,
        documents: Optional[List[Document]] = None,
        meta: Optional[dict] = None,
    ) -> Tuple[Dict, str]:
        
        if documents:
            for doc in documents:
                doc.meta[self._name]=self._value
                
        output = {
            "query":query,
            "file_paths":file_paths,
            "labels":labels,
            "documents":documents,
            "meta":meta,
        }
        return output, "output_1"
    
    
    def run_batch(
        self,
        queries: Optional[Union[str, List[str]]] = None,
        file_paths: Optional[List[str]] = None,
        labels: Optional[Union[MultiLabel, List[MultiLabel]]] = None,
        documents: Optional[Union[List[Document], List[List[Document]]]] = None,
        meta: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
        params: Optional[dict] = None,
        debug: Optional[bool] = None,
    ):
        
        if documents:
            if len(documents)>0:
                #Check if the first element is a docuemnt
                first = documents[0]
                if isinstance(first,Document):
                    for doc in documents:
                        doc.meta[self._name]=self._value
                else:
                    for doc_list in documents:
                        for doc in doc_list:
                            doc.meta[self._name]=self._value
                            
        output = {
            "queries":queries,
            "file_paths":file_paths,
            "labels":labels,
            "documents":documents,
            "meta":meta,
            "params":params,
            "debug":debug,
        }
        
        return output, "output_1"