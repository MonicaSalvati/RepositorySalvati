import pymupdf # PyMuPDF
import colorsys
from azure.core.credentials import AzureKeyCredential
from azure.ai.textanalytics import TextAnalyticsClient
import requests
from dotenv import load_dotenv
import os

load_dotenv()

class Checker():
    def __init__(self, file_path, threshold_font_size=6):
        self.file_path = file_path
        self.threshold_font_size = threshold_font_size

    def is_similar_to_background(self, rgb, threshold=0.9):
        r, g, b = rgb
        h, l, s = colorsys.rgb_to_hls(r/255, g/255, b/255)
        return l > threshold

    def extract_pdf_text(self):
        set_size = []
        doc = pymupdf.open(self.file_path)
        text = ""

        for page_num in range(len(doc)):
            page = doc[page_num]
            blocks = page.get_text("dict")["blocks"]
            for block in blocks:
                if "lines" in block:
                    for line in block["lines"]:
                        for span in line["spans"]:
                            font_size = span["size"]
                            set_size.append(span["size"])
                            color = span["color"]
                            rgb = ((color >> 16) & 255, (color >> 8) & 255, color & 255)

                            if font_size > self.threshold_font_size and not self.is_similar_to_background(rgb):
                                text += span["text"] + " "  
        return text
    
    def pii_detection(self, text):
        # Configura il client
        
        text_analytics_client = TextAnalyticsClient(
            endpoint=os.getenv("AZURE_API_BASE"),
            credential=AzureKeyCredential(os.getenv("AZURE_API_KEY"))
        )
        
        # Chiamata API per riconoscere PII e redigere
        risultati = text_analytics_client.recognize_pii_entities(
            [text],
            language="en",
            categories_filter=["InternationalBankingAccountNumber"]
        )
        # "Person","PhoneNumber", "Email", "Address"
        return risultati[0].redacted_text
    
    def prompt_shield(self, text, query=False):

        api_version = "2024-09-01"

        documents = [
            text
        ]
        
        if not query:
            body = {
                "userPrompt": "",
                "documents": documents
            }
        else:
            body = {
                "userPrompt": text,
                "documents": []
            }

        # Set up the API request
        url = f"{os.getenv("AZURE_API_BASE")}/contentsafety/text:shieldPrompt?api-version={api_version}"

        headers = {
            "Content-Type": "application/json",
            "Ocp-Apim-Subscription-Key": os.getenv("AZURE_API_KEY")
        }

        # Send the API request
        response = requests.post(url, headers=headers, json=body)
        response = response.json()
        return response["documentsAnalysis"][0]["attackDetected"]


    def decode(self, text):
        # Ottieni solo i byte UTF-8 validi
        utf8_bytes = text.encode("utf-8", errors="ignore")
        # Torna a stringa pulita (solo caratteri validi)
        s_pulita = utf8_bytes.decode("utf-8")
        return s_pulita
    







 