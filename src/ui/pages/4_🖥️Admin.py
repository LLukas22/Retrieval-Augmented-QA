import os
import sys
from pathlib import Path
#Ensure the  modules are in the path
root = str(Path(__file__).parent.parent.parent)
if root not in sys.path:
    sys.path.insert(0, root)
    
    

import streamlit as st
from ui.utils import sidebar_footer
from ui.api_connector import get_api_connector


st.set_page_config(page_title="Admin", page_icon="🖥️", layout="wide")

def render():
    connector = get_api_connector()
    enabled = os.getenv("ENABLE_ADMIN",False)
    
    st.title("🖥️  Admin area")
    sidebar_footer()
    
    if not enabled:
        st.warning("Sorry but the admin area is disabled. You can enable it by setting the environment variable `ENABLE_ADMIN` to `True`.🧐")
        return 
    st.write(
        """Here are some information about the system and the modules. If you dont know what you are doing better dont touch anything here.🧐"""
    )
    st.button("↻ Refresh")
    if not connector.healtcheck():
        st.write("🤔The API-Node seams to be offline.")
        return
    
    st.write("## Node Information")
    info = connector.system_usage()
    versions = connector.versions()
    if versions:
        with st.expander("Versions",expanded=False):
            for key, value in versions.items():
                st.write(f"**{key}**: {value}")
    if info:
        with st.expander("Usage",expanded=True):
            st.write(f"CPU: {info.cpu}%")
            st.write(f"Memory: {info.memory}%")
            st.write(f"GPU Acceleration: {len(info.gpus) > 0}")
            if len(info.gpus) > 0:
                st.markdown("""---""")
                for gpu in info.gpus:
                    st.markdown(f"#### {gpu.name}")
                    st.write(f"Index: {gpu.index}")
                    st.write(f"Memory: {gpu.usage.memory_total} MB; Used: {gpu.usage.memory_used} MB ({round((gpu.usage.memory_used/gpu.usage.memory_total)*100,2)}%)")
                    st.write(f"Core Usage: {gpu.usage.kernel_usage}%")
                    st.markdown("""---""")
            else:
                st.write("⚠️The API-Node has no GPU acceleration!⚠️")
         
    st.write("## ✳️Embeddings")
    st.text("Recalculate all embeddings in the database.⚠️ If the node is not GPU accelerated this could take some time!")
    update_all_embeddings = st.checkbox(value=False,label="Recalculate all embeddings")
    should_update_embeddings = st.button("↻ Update Embeddings")

    result=None
    with st.spinner("⌛️ &nbsp;&nbsp; Embeddings are updating..."):
        if should_update_embeddings:
            result = connector.reindex(update_all_embeddings)
    if result:
        st.write("👀 Updated Embeddings sucessfully!")

    st.write("## 🪠Pipelines")

    with st.spinner("⌛️ &nbsp;&nbsp; Getting Pipelines..."):
        pipelines = connector.pipelines()
    
    if pipelines:
        for pipeline in pipelines.pipelines:

            st.markdown(f"### {pipeline.name}")
            st.graphviz_chart(pipeline.graph,use_container_width=True)
            
            with st.expander("Pipeline Components",expanded=False):
                for component in pipeline.components:
                    st.markdown(f"### {component.name}")
                    code = []
                    for param,value in component.params.items():
                        code.append(f"{param}={value}")
                    st.code("\n".join(code),language="python")
            
        st.markdown("---")        
        
render()