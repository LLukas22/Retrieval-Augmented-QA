import streamlit as st

from ui.utils import system_info,haystack_status,sidebar_footer

st.set_page_config(page_title="System Information", page_icon="🖥️", layout="wide")

st.markdown("# System Information")
sidebar_footer()
st.write(
    """Here the current state of the system executing the API is displayed."""
)


system_column, modules_column = st.columns(2,gap="medium")
system_column.header("System")
modules_column.header("Modules")

info = system_info()

if info:
    system_column.write(f"CPU Usage: {info.cpu}%")
    system_column.write(f"Memory Usage: {info.memory}%")
    system_column.write(f"Haystack Version: {info.version}")
    if len(info.gpus) > 0:
        system_column.write(f"GPUs: {len(info.gpus)}")
        system_column.markdown("""---""")
        for gpu in info.gpus:
            system_column.markdown(f"### {gpu.name}")
            system_column.write(f"Index: {gpu.index}")
            system_column.write(f"Memory: {gpu.usage.memory_total} MB; Used: {gpu.usage.memory_used} MB ({round((gpu.usage.memory_used/gpu.usage.memory_total)*100,2)}%)")
            system_column.write(f"Core Usage: {gpu.usage.kernel_usage}%")
            system_column.markdown("""---""")
    else:
        system_column.write("⚠️The API-Node has no GPU acceleration!⚠️")
else:
    system_column.write(f"⚠️Error getting the system information! Is the API offline?")
    
initialized = haystack_status()
 
if initialized:
    for module,state in initialized.items():
        modules_column.markdown(f"**{module}**: {'✔️' if state else '❌'}")
else:
    modules_column.write(f"⚠️Error getting the module status! Is the API offline?")
     
st.button("↻ Refresh")
        
