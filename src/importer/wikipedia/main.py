from  typing import List
import os
import requests
from pathlib import Path
import bz2
from tqdm  import tqdm
import wikitextparser
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from tqdm.contrib.concurrent import process_map,thread_map
from haystack.schema import Document
from haystack.nodes import PreProcessor
from haystack.document_stores import ElasticsearchDocumentStore
   
if __name__ == "__main__":
    #Download and  extract
    print("Disclaimer: This script is not intended for production use. It could take a long time to run and it is not optimized for performance.")
    elatic_host = os.getenv("ELASTIC_HOST","localhost")
    elastic_port = os.getenv("ELASTIC_PORT",9200)
    dim = os.getenv("ELASTIC_EMBEDDING_DIM",384)
    cache_dir = Path(os.getenv("CACHE_DIR","./importer_cache"))
    url=os.getenv("WIKI_URL","https://dumps.wikimedia.org/simplewiki/20230401/simplewiki-20230401-pages-articles-multistream.xml.bz2")
    
    filename = url.split("/")[-1]
    cache_dir.mkdir(parents=True, exist_ok=True)
    pages_dir  = cache_dir /  "pages"
    document_dir  = cache_dir /  "documents"
    path = os.path.join(cache_dir,filename)
    decompressed  = cache_dir / filename.replace(".bz2", "")
    if not os.path.exists(path):
        with open(path, 'wb') as f:
            print("Downloading file...")
            f.write(requests.get(url).content)

    if not os.path.exists(decompressed):        
        with open(decompressed, 'wb') as new_file, bz2.BZ2File(path, 'rb') as file:
            for data in tqdm(iter(lambda : file.read(100 * 1024), b''),desc="Decompressing file..."):
                new_file.write(data)    
    
    #Extract Pages
    if not os.path.exists(pages_dir):
        pages_dir.mkdir(exist_ok=True)
        namespaces = {"":'http://www.mediawiki.org/xml/export-0.10/',"xsi":'http://www.w3.org/2001/XMLSchema-instance'}
        tree = ET.parse(decompressed)
        root = tree.getroot()
        page_elements = root.findall("./page",namespaces)
        for page_element in tqdm(page_elements,"Writing pages..."):
            id = page_element.find("./id",namespaces).text
            
            with open(pages_dir/f"{id}.xml","w") as f:
                f.write(ET.tostring(page_element).decode("utf-8"))
        del tree
        del root
        del page_elements


    #Load pages and create documents
    if not os.path.exists(document_dir):
        page_file = list(pages_dir.iterdir())
        
        
        def clean_sample(path)->Document:
            namespaces = {"ns0":'http://www.mediawiki.org/xml/export-0.10/'}
            base_url  = "https://en.wikipedia.org/wiki/"
            
            tree = ET.parse(path)
            root = tree.getroot()
            title_element = root.find("./ns0:title",namespaces)
            id_element = root.find("./ns0:id",namespaces)
            text_element = root.find(".//ns0:text",namespaces)
            if text_element is None or text_element.text is None:
                return None
            
            parsed = wikitextparser.parse(text_element.text)
            sections = [s.plain_text() for  s in parsed.sections if s.title is None or s.title.lower() not in ["references","external links","see also"]]
            joined_sections = "\n\n".join(sections)
            #Only accept documents with at least 150 letters
            if len(joined_sections)<150:
                return None
            
            return Document(content=joined_sections,meta={"title":title_element.text,"wiki_id":id_element.text,"url":base_url+title_element.text})
        
        clean_samples:List[Document] = []
        clean_samples=thread_map(clean_sample,page_file,desc="Cleaning samples...",chunksize=25)
        clean_samples=[sample for sample in clean_samples if sample is not None]
        document_dir.mkdir(exist_ok=True)
        for sample in tqdm(clean_samples,"Writing documents..."):
            with open(document_dir/f"{sample.meta['wiki_id']}.json","w") as f:
                f.write(sample.to_json())
    
    
    document_files = list(document_dir.iterdir())
    documents:List[Document] =[]
    for document_file in tqdm(document_files,"Loading documents..."):
        with open(document_file,"r") as f:
            documents.append(Document.from_json(f.read()))
            
    preprocessor = PreProcessor(clean_empty_lines=True,
                        clean_whitespace=True,
                        split_by="word",
                        split_length = 250,
                        split_overlap = 0,
                        split_respect_sentence_boundary=True
                    )
    
    
    preprocessed_docs =  preprocessor.process(documents)
    document_store = ElasticsearchDocumentStore(host=elatic_host,port=elastic_port,embedding_dim=dim)
    print(f"Found {len(preprocessed_docs)} documents. Beginning to write to document store...")
    document_store.write_documents(preprocessed_docs)
    
    