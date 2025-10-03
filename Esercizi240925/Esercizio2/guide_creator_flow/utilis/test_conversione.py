"""
Le installazioni da fare sono:
    - pip install markdown-it-py
    - pip install selenium

"""

import base64

from markdown_it import MarkdownIt
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

def markdown_to_html(md_path, html_path):
    print("Entrato in md to html")
    # Read Markdown
    with open(md_path, "r", encoding="utf-8") as f:
        md_text = f.read()

    # Convert to HTML
    md = MarkdownIt()
    html_body = md.render(md_text)

    # Wrap in HTML
    html_full = f"""
    <html>
    <head>
        <meta charset="utf-8"/>
        <style>
            body {{ font-family: Arial, sans-serif; padding: 20px; }}
            h1 {{ font-size: 24pt; margin-top: 24px; }}
            h2 {{ font-size: 20pt; margin-top: 18px; }}
            h3 {{ font-size: 16pt; margin-top: 14px; }}
            p {{ margin: 6px 0; }}
        </style>
    </head>
    <body>
        {html_body}
    </body>
    </html>
    """

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_full)
    print(f"HTML saved to {html_path}")

def html_to_pdf(html_path, pdf_path):
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    import os

    options = Options()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")  # sometimes needed on Windows/Linux
    options.add_argument("--disable-software-rasterizer")
    options.add_argument("--disable-dev-shm-usage")

    driver = webdriver.Chrome(options=options)

    driver.get(f"file:///{os.path.abspath(html_path)}")
    print("Conversione in corso")
    pdf_data = driver.execute_cdp_cmd("Page.printToPDF", {"printBackground": True})
    driver.quit()
    print("Conversione finita")
    return base64.b64decode(pdf_data['data'])

    


# if __name__ == "__main__":
#     md_file = r"C:\Users\XT286AX\OneDrive - EY\Desktop\report.md"
#     html_file = r"C:\Users\XT286AX\OneDrive - EY\Desktop\report.html"
#     pdf_file = r"C:\Users\XT286AX\OneDrive - EY\Desktop\report.pdf"

#     print("creating the html file ------")
#     markdown_to_html(md_file, html_file)
#     print("creating the pdf file-----")
#     html_to_pdf(html_file, pdf_file)

