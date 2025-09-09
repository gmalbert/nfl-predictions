import requests
import re

# URL of the official nflfastR field descriptions markdown
url = "https://raw.githubusercontent.com/nflverse/nflfastR/master/vignettes/field_descriptions.Rmd"
md_filename = "nflfastR_fields.md"

response = requests.get(url)
response.raise_for_status()
md_text = response.text

# Regex to extract table rows: | field_name | description |
pattern = r"\|\s*([\w_]+)\s*\|\s*(.*?)\s*\|"
fields = re.findall(pattern, md_text)

with open(md_filename, "w", encoding="utf-8") as f:
    f.write("| Field Name | Description |\n")
    f.write("|------------|-------------|\n")
    for field, desc in fields:
        # Escape pipes in description
        desc = desc.replace("|", "\\|")
        f.write(f"| {field} | {desc} |\n")

print(f"Markdown table with {len(fields)} fields written to {md_filename}")
