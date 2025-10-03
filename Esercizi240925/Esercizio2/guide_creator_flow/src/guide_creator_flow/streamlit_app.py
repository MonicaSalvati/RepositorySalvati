"""Streamlit user interface for the Travel Guide Assistant.

This module implements a simple Streamlit application that provides two
screens: a home page and an "Esegui Ricerca" page which runs the
``PoemFlow`` defined in :mod:`guide_creator_flow.main`.

The UI offers a textbox for the user query, checkboxes to view intermediate
results and controls to run the flow and generate/download a PDF. The PDF
generation uses an optional selenium-based pipeline if available or falls
back to `PyMuPDF` (``fitz``).

Public functions
----------------
home_page()
    Renders the application home page and team info.

_markdown_to_pdf_bytes(md_text)
    Internal helper that converts markdown text to PDF bytes. It tries a
    high-quality selenium + browser pipeline if available and falls back to
    a simple PyMuPDF rendering when selenium is not installed.

Notes
-----
This module is intended to be run as a Streamlit application. It imports the
``PoemFlow`` object and runs it programmatically; therefore running the app
requires the external dependencies used by the flow (CrewAI, Opik, etc.).
"""

import streamlit as st
from streamlit_option_menu import option_menu
import io
import os
import tempfile
from datetime import datetime
import sys
import warnings
import io

warnings.filterwarnings("ignore")

try:
    import fitz  # PyMuPDF per fallback semplice
except ImportError:  # se manca useremo solo selenium route
    fitz = None

# Importa le utility di conversione markdown->HTML->PDF (chromium/selenium)
try:
    from guide_creator_flow.utilis.test_conversione import markdown_to_html, html_to_pdf  # type: ignore
except Exception:
    markdown_to_html = None  # type: ignore
    html_to_pdf = None  # type: ignore

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from guide_creator_flow.main import PoemFlow

st.set_page_config(page_title="Travel Guide Assistant", layout="wide")

st.title("Travel Guide Assistant ✈️")

default_topic = "Turismo/Vacanze/Viaggi"

team_members = [
    {"name": "Jacopo Bonanno", "city": "📍Roma"},
    {"name": "Pietro Montresori", "city": "📍Milano"},
    {"name": "Roberto Parodo", "city": "📍Cagliari"},
    {"name": "Luca Sangiovanni", "city": "📍Milano"},
    {"name": "Monica Salvati", "city": "📍Roma"}
]

with st.sidebar:
    selected = option_menu(
        menu_title="Menu",
        options=["Home", "Esegui Ricerca"],  # aggiunta nuova voce
        icons=["house-fill", "rocket-takeoff-fill"],  # icona per Home e Ricerca
        menu_icon="cast",
        default_index=0,
        orientation="vertical"
    )

@st.cache_data
def home_page():
    st.divider()
    st.markdown(
        """
        Stai pensando a una nuova meta e non sai da dove iniziare?  
        Questo progetto è il tuo **compagno di viaggio digitale**, capace di raccogliere, selezionare e organizzare le migliori informazioni per creare una guida su misura per te.

        ✨ **Come funziona:**
        1. Tu scrivi la tua destinazione o una domanda sul viaggio che vuoi organizzare.  
        2. L’assistente analizza la richiesta e raccoglie idee e contenuti rilevanti.  
        3. Vengono recuperate informazioni da documenti interni e dalle fonti web più affidabili (solo siti selezionati come LonelyPlanet, TripAdvisor, Booking, Expedia, ecc.).  
        4. Tutto viene integrato e trasformato in un **PDF personalizzato** con consigli, itinerari e spunti utili.  

        🚀 In questo modo non avrai più bisogno di passare ore a cercare informazioni sparse: riceverai una **guida completa, chiara e pronta all’uso**, perfetta per iniziare subito a pianificare la tua prossima avventura.  

        ✈️ Che sia un weekend in Europa o un viaggio dall’altra parte del mondo, il tuo **Assistente di Viaggio con CrewAI** è pronto a darti ispirazione e organizzazione in un unico strumento.
        """
    )         
    st.divider()

    st.subheader("Il Team\n")
    cols = st.columns(len(team_members))
    for col, member in zip(cols, team_members):
        col.markdown(f"**{member['name']}**")
        col.markdown(f"{member['city']}")

    st.divider()
    col1, col_center, col3 = st.columns([5, 2, 5])  # col_center = 50% approx

    with col_center:
        img1_col, img2_col = st.columns(2)
        with img1_col:
            st.image(
                "https://docs.crewai.com/images/crew_only_logo.png",
                use_container_width=True
            )
        with img2_col:
            st.image(
                "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTcojeoCRATOAp-WA4hao-x0enVJacE9ket4w&s",
                use_container_width=True
            )


if selected == "Home":
    home_page()

elif selected == "Esegui Ricerca":
    st.divider()

    st.markdown("Scrivi la tua destinazione o una domanda sul viaggio che vuoi organizzare. Il nostro assistente AI valuterà la rilevanza, cercherà informazioni da fonti affidabili e genererà una guida completa su viaggi, itinerari e consigli pratici.")
    user_query = st.text_area("La tua domanda di viaggio", height=140, placeholder="Es: Vorrei organizzare un viaggio di 5 giorni a Londra con budget medio, suggerimenti?", key="user_query_input")
    
    show_docs = st.checkbox("Mostra documenti vettoriali", value=False)
    show_web = st.checkbox("Mostra risultati web", value=False)

    st.info("⚠️ Nota: Il testo generato è prodotto dall'AI. Verifica sempre le informazioni prima di prenderle come definitive.")
    run = st.button("Esegui Flow")

    st.divider()


    output_container = st.empty()

    def _markdown_to_pdf_bytes(md_text: str) -> bytes:
        """Genera un PDF a partire da markdown.

        Strategia:
        1. Se disponibili le funzioni selenium (markdown_to_html + html_to_pdf) le usa per qualità migliore.
        Ritorna i bytes del PDF.
        """
        # Percorso 1: pipeline selenium se disponibile
        if markdown_to_html and html_to_pdf:
            try:
                with tempfile.TemporaryDirectory() as tmpdir:
                    md_path = os.path.join(tmpdir, "output.md")
                    html_path = os.path.join(tmpdir, "output.html")
                    pdf_path = os.path.join(tmpdir, "output.pdf")
                    with open(md_path, "w", encoding="utf-8") as f:
                        f.write(md_text)
                    markdown_to_html(md_path, html_path)
                    return html_to_pdf(html_path, pdf_path)
                    # with open(pdf_path, "rb") as fpdf:
                    #     return fpdf.read()
            except Exception as e:
                if not fitz:
                    raise RuntimeError(f"Errore conversione Selenium e nessun fallback disponibile: {e}")

        # Percorso 2: fallback PyMuPDF
        if not fitz:
            raise RuntimeError("Nessun motore disponibile per generare PDF (installa selenium + browser oppure pymupdf)")

        doc = fitz.open()
        page = doc.new_page()
        margin_x, margin_y = 40, 40
        y = margin_y
        line_gap = 4
        font_size = 12
        for line in md_text.splitlines():
            if not line.strip():
                y += font_size + line_gap
                continue
            if y > page.rect.height - margin_y:
                page = doc.new_page()
                y = margin_y
            clean = line.replace("**", "").replace("__", "").replace("*", "").replace("_", "")
            page.insert_text((margin_x, y), clean, fontsize=font_size, fontname="helv")
            y += font_size + line_gap
        buf = io.BytesIO()
        doc.save(buf)
        doc.close()
        return buf.getvalue()


    if run:
        if not user_query.strip():
            st.warning("Per favore inserisci una domanda prima di eseguire il flow.")
            st.stop()

        with st.spinner("🌍 Il tuo assistente di viaggio è al lavoro… tra poco avrai la tua guida pronta per esplorare nuove destinazioni!"):
            flow = PoemFlow()
            # Pre-popola lo stato per evitare l'input() nella fase get_user_query
            flow.state.user_query = user_query.strip()
            flow.state.topic = default_topic
            try:
                flow.plot()
                result = flow.kickoff()
            except Exception as e:
                st.error(f"Errore durante l'esecuzione del flow: {e}")
                st.stop()

        st.success("Flow completato")

        if isinstance(result, dict):
            # Output principale
            output_text = result.get("output") or "(Nessun output generato)"
            # Salva nello stato per permettere download senza riesecuzione immediata
            st.session_state["last_output_markdown"] = output_text
            st.subheader("Output Finale")
            st.markdown(output_text)

            # Pulsante download PDF
            col1, col2 = st.columns([1,3])
            with col1:
                if st.button("Genera PDF", help="Converte l'output markdown in PDF (usa Selenium se disponibile, altrimenti fallback)"):
                    try:
                        pdf_bytes = _markdown_to_pdf_bytes(output_text)
                        st.session_state['pdf_bytes'] = pdf_bytes
                        st.success("PDF generato. Usa il pulsante di download a destra.")
                    except Exception as e:
                        st.error(f"Errore nella generazione PDF: {e}")
            with col2:
                if 'pdf_bytes' in st.session_state and st.session_state['pdf_bytes']:
                    st.download_button(
                        label="Scarica PDF",
                        data=st.session_state['pdf_bytes'],
                        file_name=f"output_flow_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf"
                    )

            if show_docs:
                st.subheader("Documenti Vettoriali")
                docs = result.get("documents", "")
                st.text(docs if docs else "(Nessun documento)")

            if show_web:
                st.subheader("Documenti Web")
                web_docs = result.get("web_documents", "")
                st.text(web_docs if web_docs else "(Nessun documento web)")
        else:
            st.info("È stato riscontrato un problema con l'input dell'utente.")

    else:
        # Se non è stato appena eseguito il flow ma esiste output precedente, mostra opzioni di download
        if 'last_output_markdown' in st.session_state and st.session_state['last_output_markdown']:
            with st.expander("Ultimo Output Generato (sessione)"):
                st.markdown(st.session_state['last_output_markdown'])
                gen, dl = st.columns(2)
                with gen:
                    if st.button("Genera PDF da ultimo output"):
                        try:
                            st.session_state['pdf_bytes'] = _markdown_to_pdf_bytes(st.session_state['last_output_markdown'])
                            st.success("PDF generato.")
                        except Exception as e:
                            st.error(f"Errore nella generazione PDF: {e}")
                with dl:
                    if 'pdf_bytes' in st.session_state and st.session_state['pdf_bytes']:
                        st.download_button(
                            label="Scarica PDF",
                            data=st.session_state['pdf_bytes'],
                            file_name=f"output_flow_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                            mime="application/pdf"
                        )
