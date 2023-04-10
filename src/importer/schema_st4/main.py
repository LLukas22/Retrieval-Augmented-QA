from pathlib import Path
import os
import glob
from typing import Optional,List
from haystack.schema import Document
from haystack.nodes import PreProcessor
from haystack.document_stores import ElasticsearchDocumentStore
import logging
from schema_st4_parser import parse, St4Entry
import xml.etree.ElementTree as ET
import pandas as pd


def process_node(node:ET.Element,join_char="\n")->str:
    cleaned_texts = []
    
    def process_children()->List[str]:
        texts=[]
        for child in node:
            texts.append(process_node(child))
        return texts
                
    match node.tag:
        case "p" | "emphasis" | "sub" | "i" | "b":
            if node.text:
                cleaned_texts.append(node.text.strip())
            cleaned_texts += process_children()    
        
        case "link":
            cleaned_texts.append(f"(link to {node.attrib.get('linkid')})")
            cleaned_texts += process_children()  
        case "sup":
            return f"(see {node.text})"
        case "indexentry":
            return ""
        case "ul":
            cleaned_texts.append("Listing:")
            listings = process_children()
            for listing in listings:
                cleaned_texts.append(f"- {listing}")
        case "safety":
            severity = node.attrib.get("severity")
            if node.attrib.get('type') == "notice":
                consequence = node.find("consequence")
                cleaned_texts.append(f"Safety notice: \"{consequence.text}\"")
            else:
                cause = node.find("cause").text
                consequences = node.findall("consequence")
                instructions = node.findall(".//safety_instruction")
                cleaned_texts.append(f"Safety Notice: \"{cause}\"")
                cleaned_texts.append(f"Severity: {severity}")
                cleaned_texts.append(f"Consequences:")
                for consequence in consequences:
                    cleaned_texts.append(f"- {consequence.text}")
                cleaned_texts.append(f"Instructions to fix:")
                for instruction in instructions:
                    cleaned_texts.append(f"- {instruction.text}")
        
        case "image-container":
            image_title_node = node.find("image-title")
            if image_title_node:
                cleaned_texts.append(f"Image: {image_title_node.text}")
            table_container = node.find("table-container")
            if table_container:
                process_node(table_container)
                
        case "table-container":
            table = node.find("table")
            if table:
                type = node.attrib.get('type')
                if type != "header":
                    cleaned_texts.append(f"Table Type: \"{type}\"")
                else:
                    cleaned_texts.append(f"Table:")
                cleaned_texts.append(process_node(table))
                 
        case "table":
            df = pd.read_html(ET.tostring(node))[0]
            columns = df.columns
            has_header = False
            for column in columns:
                if not str(column).isnumeric():
                    has_header = True
                    break
                
            df_string = df.to_string(index=False,header=has_header)
            cleaned_texts.append("=== Table START ===")
            cleaned_texts.append(df_string)
            cleaned_texts.append("=== Table END ===")
            
        case "procedural-instructions":
            instructions = node.findall(".//instruction")
            if len(instructions) > 0:
                cleaned_texts.append("Instructions:")
                for i,instruction in enumerate(instructions):
                    cleaned_texts.append(f"{i+1}. {instruction.text}")
                    
        case "img":
            return f"(image of \"{node.attrib.get('src')}\""
        case _:
            if node.tag not in ["td","th","li"]:
                logging.info(f"Unknown tag: {node.tag}") 
            cleaned_texts += process_children()
            
    cleaned_texts = [text for text in cleaned_texts if text and text != "" and text != "\n"]            
    return join_char.join(cleaned_texts)

def process_entry(entry:St4Entry,filename:str)->Optional[Document]:
    if "en" not in entry.languages:
        return None
    
    if len(entry.content) == 0 or "en" not in entry.content:
        return None
    
    try:
        content = entry.content["en"]
        cleaned_contents = []
        et = ET.fromstring(content)
        for element in et:
            cleaned_contents.append(process_node(element))
            
        joined_contents = "\n".join(cleaned_contents)
        if entry.titles and "en" in entry.titles:
            joined_contents = f"{entry.titles['en']}\n{joined_contents}"
            
        return Document(joined_contents,meta={"filename":filename,"type":entry.type,"label":entry.label,"title":entry.titles.get("en","")})
    
    except Exception as e:
        logging.error(f"Error processing entry {entry.label} in file {filename}!")
        logging.error(e)
        return None
    
    
    

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s: %(message)s')
    elatic_host = os.getenv("ELASTIC_HOST","localhost")
    elastic_port = os.getenv("ELASTIC_PORT",9200)
    dim = os.getenv("ELASTIC_EMBEDDING_DIM",384)
    
    st4_files_folder = Path(os.getenv("ST4_FOLDER","./.st4_files")).absolute()
    cache_folder = st4_files_folder / "cache"
    cache_folder.mkdir(parents=True, exist_ok=True)
    
    logging.info(f"ST4 files folder: {st4_files_folder}")
    
    st4_files = st4_files_folder.glob("*.xml")
    
    logging.info(f"Starting to process Files ..")
    to_upload={}
    for file in st4_files:
        fielname = file.stem
        export_folder = cache_folder / fielname
        export_folder.mkdir(parents=True, exist_ok=True)
        already_converted=export_folder.glob("*.json")
        if len(list(already_converted)) > 0:
            logging.info(f"Skipping conversion of {file} as it has already been processed!")
            to_upload[fielname]=already_converted
            continue
        
        logging.info(f"Processing file: {file}")
        entries = parse(file)
        logging.info(f"Found {len(entries)} entries!")
        processed_documents = [process_entry(doc,fielname) for doc in entries]
        processed_documents = [doc for doc in processed_documents if doc]
        logging.info(f"Processed {len(processed_documents)} documents!")
        
        for doc in processed_documents:
            with open(export_folder / f"{doc.meta['label']}.json","w") as f:
                f.write(doc.to_json())
                
        to_upload[fielname]=export_folder.glob("*.json")
    
    documentstore = ElasticsearchDocumentStore(host=elatic_host, port=elastic_port, embedding_dim=dim)
    
    for file,document_files in to_upload.items():
        logging.info(f"Uploading Documents for {file} ...")
        
        documents = []
        for file in document_files:
            with open(file,"r") as f:
                documents.append(Document.from_json(f.read()))
        
        documentstore.write_documents(documents)        
        logging.info(f"Finished uploading Documents for {file}!")
        
        
        

        
    
    
