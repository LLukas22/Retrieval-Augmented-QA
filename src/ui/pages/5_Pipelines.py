import streamlit as st

from typing import Optional
from implementation.ui.utils import system_info,haystack_status,sidebar_footer,get
from implementation.api.schemas.pipelines import PipelinesResponse,PipelineDescription,ComponentDescription

def get_pipelines():
    url="/get_pipelines"
    try:
        response = get(url)
        pipelines = PipelinesResponse.parse_obj(response.json())
        return pipelines
    except Exception as e:
        st.exception(e)

def render():
    st.set_page_config(page_title="Pipelines", page_icon="🪠", layout="wide")

    st.markdown("# 🪠Pipelines")
    sidebar_footer()
    st.write(
        """Here the pipelines of the system with their parameters are displayed."""
    )
    pipelines=None
    with st.spinner("⌛️ &nbsp;&nbsp; Getting Pipelines..."):
        pipelines = get_pipelines()

    if pipelines:
        for pipeline in pipelines.pipelines:

            st.markdown(f"## {pipeline.name}")
            st.graphviz_chart(pipeline.graph,use_container_width=True)
            
            with st.expander("Show Components"):
                for component in pipeline.components:
                    st.markdown(f"### {component.name}")
                    code = []
                    for param,value in component.params.items():
                        code.append(f"{param}={value}")
                    st.code("\n".join(code),language="python")
                
            st.markdown("---")     
render()