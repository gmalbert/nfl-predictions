import streamlit as st
import requests
import re
import pandas as pd

st.title("nflfastR Field Descriptions Table")
import os
# Display NFL logo
logo_path = os.path.join("data_files", "NFL-Logo.png")
if os.path.exists(logo_path):
    st.image(logo_path, width=150)

url = "https://raw.githubusercontent.com/nflverse/nflfastR/master/vignettes/field_descriptions.Rmd"
md_filename = "nflfastR_fields.md"

@st.cache_data
def get_fields():
    response = requests.get(url)
    response.raise_for_status()
    md_text = response.text
    pattern = r"\|\s*([\w_]+)\s*\|\s*(.*?)\s*\|"
    fields = re.findall(pattern, md_text)
    return fields

fields = get_fields()

# Display as DataFrame
field_df = pd.DataFrame(fields, columns=["Field Name", "Description"])
st.dataframe(field_df, use_container_width=True)

# Markdown table string
md_table = "| Field Name | Description |\n|------------|-------------|\n" + "\n".join([
    f"| {field} | {desc.replace('|', '\\|')} |" for field, desc in fields
])

st.download_button(
    label="Download Markdown Table",
    data=md_table,
    file_name=md_filename,
    mime="text/markdown"
)

st.write("Showing all nflfastR play-by-play field descriptions. Data source: ", url)
