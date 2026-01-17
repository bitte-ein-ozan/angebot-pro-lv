import streamlit as st
import pandas as pd
import sqlite3
import re
import os
from dotenv import load_dotenv
from openai import AzureOpenAI, APIStatusError
import pdfplumber
import json
from fpdf import FPDF
from datetime import datetime
from thefuzz import process, fuzz

# Load environment variables
load_dotenv()

# --- Configuration ---
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
DB_PATH = 'data/prices.db'
HISTORY_DB_PATH = 'data/history.db'

# --- Initialisation ---
st.set_page_config(page_title="Angebot Pro", layout="wide", page_icon="🏗️")

# Initialize Azure OpenAI client
if all([AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_DEPLOYMENT_NAME]):
    try:
        client = AzureOpenAI(
            api_key=AZURE_OPENAI_API_KEY,
            api_version="2024-02-01",
            azure_endpoint=AZURE_OPENAI_ENDPOINT
        )
        st.session_state.ai_enabled = True
    except Exception as e:
        st.error(f"Fehler bei der Initialisierung des Azure OpenAI Clients: {e}")
        st.session_state.ai_enabled = False
else:
    st.session_state.ai_enabled = False

# Initialize session state for the wizard
if 'step' not in st.session_state:
    st.session_state.step = 1
if 'extracted_items' not in st.session_state:
    st.session_state.extracted_items = []
if 'results_df' not in st.session_state:
    st.session_state.results_df = pd.DataFrame()
if 'file_name' not in st.session_state:
    st.session_state.file_name = ""
if 'project_name' not in st.session_state:
    st.session_state.project_name = ""
if 'recipient_address' not in st.session_state:
    st.session_state.recipient_address = ""
if 'current_pdf_text' not in st.session_state:
    st.session_state.current_pdf_text = ""
if 'processing_log' not in st.session_state:
    st.session_state.processing_log = []

# --- Database Functions ---
def get_db_connection(db_path):
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    return sqlite3.connect(db_path)

def load_price_list():
    conn = get_db_connection(DB_PATH)
    try:
        df = pd.read_sql_query("SELECT * FROM prices", conn)
    except pd.io.sql.DatabaseError:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS prices (
                id INTEGER PRIMARY KEY,
                description TEXT,
                unit TEXT,
                price_min REAL,
                price_max REAL,
                category TEXT
            )
        """)
        return pd.DataFrame(columns=['id', 'description', 'unit', 'price_min', 'price_max', 'category'])
    conn.close()
    return df

def save_to_history(df, file_name, total_price):
    conn = get_db_connection(HISTORY_DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS angebote (
            id INTEGER PRIMARY KEY,
            file_name TEXT,
            total_price REAL,
            timestamp TEXT
        )
    """)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn.execute("INSERT INTO angebote (file_name, total_price, timestamp) VALUES (?, ?, ?)",
                 (file_name, total_price, timestamp))
    conn.commit()
    conn.close()
    return len(df)

# --- Core Logic: Extraction & Matching ---

def extract_metadata_with_ai(first_page_text):
    if not st.session_state.ai_enabled:
        return "", ""

    system_prompt = """
    Extrahiere aus dem folgenden Text eines Leistungsverzeichnisses (erste Seite) den Namen/Anschrift des Auftraggebers (Empfänger) und den Projektnamen/Bauvorhaben.
    Gib das Ergebnis als JSON zurück.
    Beispiel: {"recipient": "Musterbau GmbH\\nMusterstraße 1\\n12345 Musterstadt", "project_name": "Neubau Wohnanlage West"}
    Wenn Informationen fehlen, lasse das Feld leer.
    """

    try:
        response = client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": first_page_text}
            ],
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        content = response.choices[0].message.content
        data = json.loads(content)
        return data.get('recipient', ''), data.get('project_name', '')
    except Exception as e:
        print(f"Metadata extraction failed: {e}")
        return "", ""

def identify_excel_columns_with_ai(df_head):
    """
    Asks AI to identify which columns correspond to description, price, and unit.
    Returns: JSON dict {'description': 'ColA', 'price': 'ColB', 'unit': 'ColC'}
    """
    if not st.session_state.ai_enabled:
        return None

    csv_preview = df_head.to_csv(index=False)
    system_prompt = """
    You are a data import assistant. Analyze this CSV snippet of a price list.
    Identify the column names that correspond to:
    1. 'description' (Artikel, Text, Bezeichnung, Leistung)
    2. 'price' (Preis, EP, Einheitspreis, Betrag - look for numeric columns)
    3. 'unit' (Einheit, ME, Mengeneinheit - e.g., m2, Stk)

    Return a JSON object mapping the keys 'description', 'price', 'unit' to the EXACT column names found in the CSV.
    If a column is missing, set it to null.
    Example: {"description": "Kurztext", "price": "EP Euro", "unit": "ME"}
    """

    try:
        response = client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": csv_preview}
            ],
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        content = response.choices[0].message.content
        return json.loads(content)
    except Exception as e:
        print(f"Column mapping failed: {e}")
        return None

def extract_pricelist_from_text_ai(page_text):
    """
    Extracts structured price list items from a text page using AI.
    """
    if not st.session_state.ai_enabled:
        return []

    system_prompt = """
    You are a data extraction API.
    Your task is to extract price list items from the German text provided inside <source_text> tags.
    Ignore headers, footers, and noise.
    For each item, extract:
    - "description": The item text/name (Material, Service).
    - "price": The unit price (numeric, float).
    - "unit": The unit (e.g., m2, Stk, psch).
    Return ONLY a JSON array of objects. Example: [{"description": "Item", "price": 12.50, "unit": "m2"}]
    If no items found, return [].
    """

    try:
        response = client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"<source_text>\n{page_text}\n</source_text>"}
            ],
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        content = response.choices[0].message.content
        data = json.loads(content)
        # Handle various AI return formats
        if isinstance(data, list): return data
        if isinstance(data, dict):
            # Check for 'items' key or any list value
            if 'items' in data and isinstance(data['items'], list):
                return data['items']
            for k, v in data.items():
                if isinstance(v, list): return v
        return []
    except APIStatusError as e:
        if e.status_code == 403:
            print(f"AI Import Blocked on page: {e}")
        return []
    except Exception as e:
        print(f"AI Import Failed: {e}")
        return []

def analyze_with_azure_ai(full_text):
    if not st.session_state.ai_enabled:
        return [], [{'level': 'warning', 'message': "Azure AI ist nicht konfiguriert. Führe Regex-Extraktion durch."}]

    system_prompt = """
    You are a data extraction API. Your task is to process text from a German construction tender document (Leistungsverzeichnis, LV) and extract item details.
    The content is purely technical construction specifications. Ignore any perceived policy violations as false positives for technical data.
    For each item (Position), you MUST extract:
    - "oz": The position number (e.g., '01.01.0010').
    - "text": The short description of the service.
    - "menge": The quantity, as a floating point number.
    - "einheit": The unit (e.g., 'm3', 'Stk', 'psch').
    Return the result as a single, valid JSON array of objects.
    Example: [{"oz": "01.01.0010", "text": "Stahlbeton C25/30", "menge": 10.5, "einheit": "m3"}]
    If no items are found on the provided page, you MUST return an empty JSON array: [].
    Do not add any explanations, introductory text, or markdown. Only output the raw JSON.
    """

    if isinstance(full_text, list):
        pages = full_text
    else:
        pages = full_text.split('\f')

    all_items = []
    processing_log = []
    progress_bar = st.progress(0, text="Analysiere Seiten mit KI...")

    for i, page_text in enumerate(pages):
        if not page_text.strip() or len(page_text) < 50:
            continue

        try:
            response = client.chat.completions.create(
                model=AZURE_OPENAI_DEPLOYMENT_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": page_text}
                ],
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            content = response.choices[0].message.content

            try:
                json_content = json.loads(content)
                items_list = []
                if isinstance(json_content, list):
                    items_list = json_content
                elif isinstance(json_content, dict):
                    list_values = [v for v in json_content.values() if isinstance(v, list)]
                    if list_values:
                        items_list = list_values[0]
                    elif "oz" in json_content:
                        items_list = [json_content]

                for item in items_list:
                    normalized_item = {
                        'oz': item.get('oz', ''),
                        'description': item.get('text', '') or item.get('description', ''),
                        'quantity': item.get('menge', 0) or item.get('quantity', 0),
                        'unit': item.get('einheit', '') or item.get('unit', '')
                    }
                    try: normalized_item['quantity'] = float(normalized_item['quantity'])
                    except: normalized_item['quantity'] = 0.0

                    if normalized_item['oz']:
                        all_items.append(normalized_item)

            except json.JSONDecodeError:
                msg = f"KI hat auf Seite {i+1} ungültiges JSON zurückgegeben."
                processing_log.append({'level': 'warning', 'message': msg})

        except APIStatusError as e:
            msg = f"Seite {i+1}: API Fehler {e}"
            processing_log.append({'level': 'warning' if e.status_code == 403 else 'error', 'message': msg})
        except Exception as e:
            msg = f"Seite {i+1}: Unerwarteter Fehler: {e}"
            processing_log.append({'level': 'error', 'message': msg})

        progress_bar.progress(min((i + 1) / len(pages), 1.0), text=f"Seite {i+1}/{len(pages)} analysiert.")

    progress_bar.empty()
    return all_items, processing_log

def extract_lv_items(text):
    # Regex fallback
    oz_pattern = r'^(\d{2,}\.\d{2,}\.\d{2,}(?:\.\d{2,})?\.?)\s+(.*)'
    qty_line_pattern = r'^\s*([\d\.,]+)\s*([a-zA-Z²³]+|m2|m3|Stk|psch|lfm|h|t)\s*(\.{2,}|Nur Einh\.|Einh\.-Pr\.)?'

    if isinstance(text, list):
        text = "\n".join(text)

    items = []
    lines = text.split('\n')
    current_item = None

    for line in lines:
        line = line.strip()
        if not line: continue

        match_oz = re.match(oz_pattern, line)
        if match_oz:
            if current_item and current_item['quantity'] > 0:
                items.append(current_item)
            current_item = {'oz': match_oz.group(1), 'description': match_oz.group(2), 'quantity': 0.0, 'unit': ''}
            continue

        if current_item:
            match_qty = re.search(qty_line_pattern, line)
            if match_qty and "von" not in line:
                try:
                    current_item['quantity'] = float(match_qty.group(1).replace('.', '').replace(',', '.'))
                    current_item['unit'] = match_qty.group(2)
                    items.append(current_item)
                    current_item = None
                except: pass
            else:
                if "Datum:" not in line and "Projekt:" not in line:
                    current_item['description'] += " " + line

    if current_item and current_item['quantity'] > 0:
        items.append(current_item)
    return items

def find_best_match(item_text, price_db):
    if price_db.empty:
        return {'price': 0.0, 'description': "--- KEIN TREFFER ---", 'unit': '', 'score': 0, 'price_id': -1}

    choices = price_db['description'].tolist()
    best_match = process.extractOne(item_text, choices, scorer=fuzz.partial_token_sort_ratio)

    if best_match:
        match_text, score = best_match
        if score < 50:
             return {'price': 0.0, 'description': "--- KEIN TREFFER ---", 'unit': '', 'score': score, 'price_id': -1}

        row = price_db[price_db['description'] == match_text].iloc[0]
        
        # Safely get ID
        try:
            p_id = int(row['id']) if pd.notna(row['id']) else -1
        except:
            p_id = -1
            
        # Safely get Price (prevent None > 0 error)
        try:
            price_val = float(row['price_min']) if pd.notna(row['price_min']) else 0.0
        except:
            price_val = 0.0

        return {'price': price_val, 'description': row['description'], 'unit': row['unit'], 'score': score, 'price_id': p_id}

    return {'price': 0.0, 'description': "--- KEIN TREFFER ---", 'unit': '', 'score': 0, 'price_id': -1}

def prepare_dataframe_for_display(extracted_items, price_df):
    if not extracted_items:
        return pd.DataFrame()

    results = []
    for item in extracted_items:
        desc = item.get('description') or item.get('text', '')
        match = find_best_match(desc, price_df)

        status_icon = "🔴"
        if match and match.get('price', 0) > 0:
            status_icon = "🟢" if match.get('score', 0) > 90 else "🟡"

        results.append({
            'Status': status_icon,
            'OZ': item['oz'],
            'Beschreibung (LV)': desc,
            'Menge': item.get('quantity', 0),
            'Einheit (LV)': item.get('unit', ''),
            'Zugeordneter Artikel': match.get('description', '--- KEIN TREFFER ---') if match else '--- KEIN TREFFER ---',
            'Preis (€)': match.get('price', 0.0) if match else 0.0,
            'Gesamt (€)': item.get('quantity', 0) * (match.get('price', 0.0) if match else 0.0)
        })

    return pd.DataFrame(results)

# --- PDF Generation ---
class PDF(FPDF):
    def __init__(self, project_name="", recipient_address=""):
        super().__init__()
        self.project_name = project_name
        self.recipient_address = recipient_address

    def header(self):
        self.set_y(15)
        # Title "ANGEBOT"
        self.set_font('Arial', 'B', 20)
        self.set_text_color(0, 0, 0)
        self.cell(0, 10, 'ANGEBOT', 0, 1, 'R')

        # Date
        self.set_font('Arial', '', 10)
        self.set_text_color(100, 100, 100)
        self.cell(0, 6, datetime.now().strftime("%d.%m.%Y"), 0, 1, 'R')
        self.ln(10)

        # Recipient Address (Left)
        if self.recipient_address:
            self.set_font('Arial', '', 11)
            self.set_text_color(0, 0, 0)
            self.multi_cell(100, 6, self.recipient_address)
            self.ln(10)

        # Project Name
        if self.project_name:
            self.set_font('Arial', 'B', 14)
            self.cell(0, 10, f'Projekt: {self.project_name}', 0, 1, 'L')
            self.ln(5)

        # Line Separator
        self.set_draw_color(200, 200, 200)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(5)

    def footer(self):
        self.set_y(-20)
        self.set_draw_color(200, 200, 200)
        self.line(10, self.get_y(), 200, self.get_y())
        self.set_y(-15)
        self.set_font('Arial', '', 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f'Seite {self.page_no()}/{{nb}}', 0, 0, 'C')

def generate_pdf(df, project_name, total_price, recipient_address=""):
    pdf = PDF(project_name=project_name, recipient_address=recipient_address)
    pdf.alias_nb_pages()
    pdf.add_page()

    # Table Header
    pdf.set_font('Arial', 'B', 10)
    pdf.set_text_color(0, 0, 0)
    pdf.set_draw_color(180, 180, 180)

    w_oz = 25
    w_desc = 85
    w_qty = 20
    w_unit = 20
    w_price = 20
    w_total = 20

    pdf.cell(w_oz, 10, 'OZ', 'B', 0, 'L')
    pdf.cell(w_desc, 10, 'Beschreibung', 'B', 0, 'L')
    pdf.cell(w_qty, 10, 'Menge', 'B', 0, 'R')
    pdf.cell(w_unit, 10, 'Einh.', 'B', 0, 'C')
    pdf.cell(w_price, 10, 'Preis', 'B', 0, 'R')
    pdf.cell(w_total, 10, 'Gesamt', 'B', 1, 'R')
    pdf.ln(12)

    # Table Rows
    pdf.set_font('Arial', '', 9)
    for index, row in df.iterrows():
        desc_text = str(row.get('Beschreibung (LV)', ''))
        # Clean desc text to avoid latin-1 errors
        desc_text = desc_text.encode('latin-1', 'replace').decode('latin-1')
        desc = desc_text[:45] + "..." if len(desc_text) > 45 else desc_text

        pdf.cell(w_oz, 8, str(row.get('OZ', '')), 'B', 0)
        pdf.cell(w_desc, 8, desc, 'B', 0)
        pdf.cell(w_qty, 8, f"{float(row.get('Menge', 0)):.2f}", 'B', 0, 'R')
        pdf.cell(w_unit, 8, str(row.get('Einheit (LV)', '')), 'B', 0, 'C')
        pdf.cell(w_price, 8, f"{float(row.get('Preis (€)', 0)):.2f}", 'B', 0, 'R')
        pdf.cell(w_total, 8, f"{float(row.get('Gesamt (€)', 0)):.2f}", 'B', 1, 'R')

    # Total
    pdf.ln(10)
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(150, 10, 'Gesamtsumme (Netto):', 0, 0, 'R')
    pdf.cell(40, 10, f"{total_price:,.2f} EUR", 0, 1, 'R')

    return bytes(pdf.output())

# --- UI Functions ---
def display_sidebar():
    with st.sidebar:
        st.title("Angebot Pro")
        st.info("KI-gestützte Angebotskalkulation")
        st.markdown("---")
        # Logo Placeholder
        st.markdown("""
        <div style="border: 2px dashed #ccc; padding: 20px; text-align: center; border-radius: 10px; margin-bottom: 20px; background-color: #f9f9f9;">
            <h4 style="color: #666; margin: 0;">hier könnte dein logo für 10.000 € stehen</h4>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("---")

        if st.session_state.ai_enabled:
            st.success("Azure AI verbunden", icon="✅")
        else:
            st.warning("Azure AI deaktiviert", icon="⚠️")

def tab_angebot_erstellen():
    def next_step(): st.session_state.step += 1
    def prev_step(): st.session_state.step -= 1
    def reset_wizard():
        st.session_state.step = 1
        st.session_state.extracted_items = []
        st.session_state.results_df = pd.DataFrame()
        st.session_state.file_name = ""
        st.session_state.project_name = ""
        st.session_state.recipient_address = ""
        st.session_state.processing_log = []

    # STEP 1
    if st.session_state.step == 1:
        st.header("Schritt 1: LV hochladen und analysieren")
        uploaded_file = st.file_uploader("Wählen Sie eine LV-Datei (PDF)", type="pdf")
        use_ai = st.checkbox("🧠 Azure AI für Analyse nutzen", value=True, disabled=not st.session_state.ai_enabled)

        if st.button("LV verarbeiten", type="primary", use_container_width=True):
            if uploaded_file:
                with st.spinner('Datei wird gelesen...'):
                    with pdfplumber.open(uploaded_file) as pdf:
                        pages_content = [page.extract_text() or "" for page in pdf.pages]
                        st.session_state.current_pdf_text = "\n".join(pages_content)

                # Metadata Extraction
                if use_ai and st.session_state.ai_enabled and pages_content:
                    with st.spinner('Extrahiere Projektdaten...'):
                        recip, proj = extract_metadata_with_ai(pages_content[0])
                        st.session_state.recipient_address = recip
                        st.session_state.project_name = proj

                with st.spinner('Positionen werden extrahiert...'):
                    extracted_items = []
                    processing_log = []
                    if use_ai and st.session_state.ai_enabled:
                        extracted_items, processing_log = analyze_with_azure_ai(pages_content)
                        if not extracted_items:
                            msg = "KI hat keine Positionen gefunden. Starte Fallback auf Regex..."
                            st.warning(msg)
                            processing_log.append({'level': 'warning', 'message': msg})
                            extracted_items = extract_lv_items(pages_content)
                    else:
                        extracted_items = extract_lv_items(pages_content)

                    st.session_state.processing_log = processing_log

                    if not extracted_items:
                        st.warning("Keine Positionen gefunden.")
                    else:
                        price_df = load_price_list()
                        st.session_state.file_name = uploaded_file.name
                        st.session_state.extracted_items = extracted_items
                        st.session_state.results_df = prepare_dataframe_for_display(extracted_items, price_df)
                        next_step()
                        st.rerun()
            else:
                st.warning("Bitte Datei hochladen.")

    # STEP 2
    elif st.session_state.step == 2:
        st.header("Schritt 2: Daten prüfen")

        # Logs
        if st.session_state.processing_log:
            with st.expander("⚠️ Analyse-Protokoll", expanded=False):
                for log in st.session_state.processing_log:
                    if log['level'] == 'warning': st.warning(log['message'])
                    else: st.error(log['message'])

        if not st.session_state.results_df.empty:
            st.subheader("Projektdaten")
            c1, c2 = st.columns(2)
            with c1: st.session_state.project_name = st.text_input("Projekt", value=st.session_state.project_name)
            with c2: st.session_state.recipient_address = st.text_area("Empfänger", value=st.session_state.recipient_address, height=100)

            st.markdown("---")

            price_db = load_price_list()
            price_options = ["--- KEIN TREFFER ---"] + sorted(price_db['description'].unique().tolist())

            edited_df = st.data_editor(
                st.session_state.results_df,
                column_config={
                    "Status": st.column_config.TextColumn(width="small"),
                    "OZ": st.column_config.TextColumn(width="small"),
                    "Beschreibung (LV)": st.column_config.TextColumn(width="large", disabled=True),
                    "Menge": st.column_config.NumberColumn(format="%.2f", width="small"),
                    "Einheit (LV)": st.column_config.TextColumn(width="small"),
                    "Zugeordneter Artikel": st.column_config.SelectboxColumn(options=price_options, width="large"),
                    "Preis (€)": st.column_config.NumberColumn(format="%.2f"),
                    "Gesamt (€)": st.column_config.NumberColumn(format="%.2f", disabled=True)
                },
                use_container_width=True,
                hide_index=True,
                num_rows="dynamic"
            )

            # Recalculate
            edited_df['Gesamt (€)'] = edited_df['Menge'] * edited_df['Preis (€)']
            st.session_state.results_df = edited_df

            c1, c2 = st.columns([1, 1])
            with c1: st.button("⬅️ Zurück", on_click=prev_step, use_container_width=True)
            with c2: st.button("Weiter zum Export ➡️", on_click=next_step, type="primary", use_container_width=True)

    # STEP 3
    elif st.session_state.step == 3:
        st.header("Schritt 3: Export")
        final_df = st.session_state.results_df
        total = final_df['Gesamt (€)'].sum()
        st.metric("Gesamtsumme", f"{total:,.2f} €")

        c1, c2 = st.columns(2)
        with c1:
            try:
                pdf_bytes = generate_pdf(final_df, st.session_state.project_name or st.session_state.file_name, total, st.session_state.recipient_address)
                st.download_button("📄 PDF herunterladen", pdf_bytes, file_name="angebot.pdf", mime="application/pdf", type="primary", use_container_width=True)
            except Exception as e:
                st.error(f"PDF Fehler: {e}")

        with c2:
            st.download_button("📊 CSV herunterladen", final_df.to_csv(index=False).encode('utf-8'), "angebot.csv", "text/csv", use_container_width=True)

        if st.button("💾 Speichern", use_container_width=True):
            save_to_history(final_df, st.session_state.file_name, total)
            st.success("Gespeichert!")

        st.markdown("---")
        st.button("Neues Angebot", on_click=reset_wizard)
        st.button("Zurück", on_click=prev_step)

def tab_datenbank_verwalten():
    st.header("Datenbank")

    with st.expander("📥 Preisliste importieren (Excel/PDF)", expanded=False):
        uploaded_file = st.file_uploader("Datei auswählen (.xlsx, .xls, .pdf)", type=['xlsx', 'xls', 'pdf'])
        use_ai_import = st.checkbox("🧠 Intelligenten AI-Import nutzen", value=st.session_state.ai_enabled, disabled=not st.session_state.ai_enabled, help="Analysiert die Datei intelligent, falls Spaltennamen nicht exakt übereinstimmen.")

        if uploaded_file is not None:
            if st.button("Import starten"):
                try:
                    conn = get_db_connection(DB_PATH)
                    new_items_count = 0

                    # Excel Import
                    if uploaded_file.name.endswith(('.xlsx', '.xls')):
                        df_import = pd.read_excel(uploaded_file)

                        mapped_data = []

                        # AI Path
                        if use_ai_import and st.session_state.ai_enabled:
                            with st.spinner("AI analysiert Excel-Struktur..."):
                                mapping = identify_excel_columns_with_ai(df_import.head(10))

                            if mapping:
                                col_desc = mapping.get('description')
                                col_price = mapping.get('price')
                                col_unit = mapping.get('unit')

                                if col_desc and col_price:
                                    st.success(f"Spalten erkannt: {col_desc} (Text), {col_price} (Preis), {col_unit or 'N/A'} (Einheit)")
                                    for _, row in df_import.iterrows():
                                        try:
                                            price_val = row[col_price]
                                            # Clean price
                                            if isinstance(price_val, str):
                                                price_val = float(price_val.replace('€', '').replace('.', '').replace(',', '.').strip())
                                            else:
                                                price_val = float(price_val)

                                            mapped_data.append({
                                                'description': str(row[col_desc]),
                                                'unit': str(row[col_unit]) if col_unit and pd.notna(row[col_unit]) else '',
                                                'price_min': price_val,
                                                'price_max': price_val,
                                                'category': 'AI Excel Import'
                                            })
                                        except: pass
                                else:
                                    st.warning("AI konnte keine passenden Spalten identifizieren. Versuche Standard-Import.")

                        # Standard/Fallback Path
                        if not mapped_data:
                            # Normalize columns (basic mapping)
                            df_import.columns = [str(c).lower().strip() for c in df_import.columns]
                            for _, row in df_import.iterrows():
                                # Try to find description
                                desc = row.get('beschreibung') or row.get('description') or row.get('text') or row.get('artikel')
                                # Try to find price
                                price = row.get('preis') or row.get('price') or row.get('betrag') or row.get('ep')
                                # Try to find unit
                                unit = row.get('einheit') or row.get('unit') or row.get('mengeneinheit') or row.get('me')

                                if desc and pd.notna(price):
                                    try:
                                        price_val = float(str(price).replace(',', '.').replace('€', '').strip())
                                    except: price_val = 0.0

                                    mapped_data.append({
                                        'description': str(desc),
                                        'unit': str(unit) if unit else '',
                                        'price_min': price_val,
                                        'price_max': price_val, # Assume single price
                                        'category': 'Import'
                                    })

                        if mapped_data:
                            df_db = pd.DataFrame(mapped_data)
                            df_db.to_sql('prices', conn, if_exists='append', index=False)
                            new_items_count = len(df_db)

                    # PDF Import
                    elif uploaded_file.name.endswith('.pdf'):
                        full_text = ""
                        with pdfplumber.open(uploaded_file) as pdf:
                            pages_text = [page.extract_text() or "" for page in pdf.pages]
                            full_text = "\n".join(pages_text)
                    
                        # DEBUG: Show what we read
                        with st.expander("🔍 Debug: Extrahierter PDF-Text (Vorschau)", expanded=False):
                            st.text(full_text[:1000] + "..." if len(full_text) > 1000 else full_text)
                    
                        mapped_data = []
                        ai_failed = False

                        # 1. Try AI Import (if enabled)
                        if use_ai_import and st.session_state.ai_enabled:
                            progress_bar = st.progress(0, text="AI analysiert PDF-Seiten...")
                            for i, page_text in enumerate(pages_text):
                                if len(page_text) > 50:
                                    # Try to extract
                                    items = extract_pricelist_from_text_ai(page_text)
                                    
                                    # If AI returns nothing or blocked, mark as failed to trigger fallback
                                    if not items:
                                        # We don't stop immediately to try other pages, but we track failure
                                        pass
                                    
                                    for item in items:
                                        try:
                                            mapped_data.append({
                                                'description': item.get('description', ''),
                                                'unit': item.get('unit', ''),
                                                'price_min': float(item.get('price', 0)),
                                                'price_max': float(item.get('price', 0)),
                                                'category': 'AI PDF Import'
                                            })
                                        except: pass
                                progress_bar.progress((i + 1) / len(pages_text))
                            progress_bar.empty()
                            
                            if not mapped_data:
                                ai_failed = True
                                st.warning("KI-Import blockiert oder keine Daten erkannt. Wechsle zum Standard-Import...")

                        # 2. Regex Fallback (if AI disabled or failed)
                        if not mapped_data or ai_failed or not use_ai_import:
                            # Robust Regex for Price Lists
                            # Looks for lines ending in a price: "Some Text 12,50" or "Some Text 12.50"
                            # Optional: Unit and Currency
                            lines = full_text.split('\n')
                            for line in lines:
                                line = line.strip()
                                if not line or len(line) < 5: continue
                                
                                # Pattern: Description (greedy) + Space + Price (number with , or .) + Optional Unit/Currency
                                # Example: "Beton C25/30 145,00" -> Desc: "Beton C25/30", Price: "145,00"
                                match = re.search(r'^(.*?)\s+(\d+[\.,]\d{2})\s*([a-zA-Z€/].*)?$', line)
                                
                                if match:
                                    desc = match.group(1).strip()
                                    price_str = match.group(2).replace('.', '').replace(',', '.')
                                    unit_raw = match.group(3) if match.group(3) else ""
                                    
                                    # Clean unit
                                    unit = unit_raw.replace('€', '').replace('/', '').strip()
                                    
                                    try:
                                        price_val = float(price_str)
                                        # Filter out likely page numbers or dates (too small price or too short text)
                                        if len(desc) > 3 and 0.1 < price_val < 100000:
                                            mapped_data.append({
                                                'description': desc,
                                                'unit': unit,
                                                'price_min': price_val,
                                                'price_max': price_val,
                                                'category': 'PDF Standard Import'
                                            })
                                    except: pass

                        if mapped_data:
                            df_db = pd.DataFrame(mapped_data)
                            df_db.to_sql('prices', conn, if_exists='append', index=False)
                            new_items_count = len(df_db)
                            st.success(f"{new_items_count} Artikel erfolgreich importiert!")
                            st.rerun()
                        else:
                            st.error("Konnte keine Daten importieren. Bitte prüfen Sie, ob das PDF Text enthält (kein reiner Scan).")
                except Exception as e:
                    st.error(f"Fehler beim Import: {e}")

    price_db = load_price_list()
    edited = st.data_editor(price_db, use_container_width=True, num_rows="dynamic")
    
    col_save, col_delete = st.columns([2, 1])
    
    with col_save:
        if st.button("💾 Änderungen speichern", key="save_db_btn", type="primary", use_container_width=True):
            try:
                conn = get_db_connection(DB_PATH)
                edited.to_sql('prices', conn, if_exists='replace', index=False)
                conn.commit()
                st.success("Datenbank erfolgreich gespeichert!")
            except Exception as e:
                st.error(f"Fehler beim Speichern: {e}")
            finally:
                if 'conn' in locals(): conn.close()

    with col_delete:
        if st.button("🗑️ Alles löschen", key="delete_db_btn", type="secondary", use_container_width=True):
            try:
                conn = get_db_connection(DB_PATH)
                conn.execute("DELETE FROM prices")
                # Try to clean up sequence if it exists, ignore if not
                try: conn.execute("DELETE FROM sqlite_sequence WHERE name='prices'")
                except: pass
                conn.commit()
                st.warning("Datenbank wurde vollständig geleert.")
                st.rerun()
            except Exception as e:
                st.error(f"Fehler beim Löschen: {e}")
            finally:
                if 'conn' in locals(): conn.close()

def tab_verlauf():
    st.header("Verlauf")
    try:
        conn = get_db_connection(HISTORY_DB_PATH)
        df = pd.read_sql("SELECT * FROM angebote ORDER BY timestamp DESC", conn)
        conn.close()
        st.dataframe(df, use_container_width=True)
    except: st.info("Leer")

def setup_premium_design():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        :root {
            --primary: #3b82f6; /* Modern Tech Blue */
            --primary-dark: #2563eb;
            --background: #f8fafc;
            --surface: #ffffff;
            --text-main: #0f172a;
            --text-sub: #64748b;
        }

        /* --- GLOBAL APP CONTAINER --- */
        .stApp {
            background: radial-gradient(circle at top center, #f1f5f9, #f8fafc) !important;
            font-family: 'Inter', sans-serif !important;
            color: var(--text-main) !important;
        }

        /* --- HEADERS (Gradient Text) --- */
        h1 {
            font-weight: 800 !important;
            font-size: 2.5rem !important;
            letter-spacing: -0.03em !important;
            background: linear-gradient(135deg, #1e293b 0%, #3b82f6 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            padding-bottom: 1.5rem !important;
            border-bottom: 1px solid rgba(0,0,0,0.05);
            margin-bottom: 2rem !important;
        }

        h2, h3 {
            font-weight: 700 !important;
            color: #334155 !important;
            letter-spacing: -0.02em !important;
        }

        /* --- MODERN SIDEBAR (Floating Glass) --- */
        section[data-testid="stSidebar"] {
            background-color: rgba(255, 255, 255, 0.7) !important;
            backdrop-filter: blur(12px);
            border-right: 1px solid rgba(255, 255, 255, 0.5);
            box-shadow: 20px 0 40px rgba(0,0,0,0.02);
        }
        
        section[data-testid="stSidebar"] h1 {
            background: none;
            -webkit-text-fill-color: var(--text-main);
            text-align: left;
            font-size: 1.5rem !important;
        }

        /* --- CARDS & CONTAINERS --- */
        [data-testid="stMetric"], [data-testid="stExpander"] {
            background: var(--surface) !important;
            border: 1px solid rgba(255,255,255,0.6);
            border-radius: 20px !important;
            box-shadow: 0 10px 30px -10px rgba(0,0,0,0.05) !important;
            padding: 24px !important;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }

        [data-testid="stMetric"]:hover {
            transform: translateY(-5px);
            box-shadow: 0 20px 40px -10px rgba(59, 130, 246, 0.15) !important;
            border-color: rgba(59, 130, 246, 0.3);
        }

        [data-testid="stMetricLabel"] {
            color: var(--text-sub) !important;
            font-weight: 600 !important;
            text-transform: uppercase;
            font-size: 0.8rem !important;
            letter-spacing: 0.05em;
        }

        [data-testid="stMetricValue"] {
            color: var(--primary-dark) !important;
            font-size: 2.2rem !important;
            font-weight: 800 !important;
        }

        /* --- BUTTONS (Pill Shaped, High Tech) --- */
        div.stButton > button[type="primary"] {
            background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%) !important;
            color: white !important;
            border: none;
            border-radius: 9999px !important; /* Full Pill */
            padding: 0.75rem 2rem !important;
            font-weight: 600 !important;
            letter-spacing: 0.02em;
            box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3) !important;
            transition: all 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275) !important; /* Bouncy */
        }

        div.stButton > button[type="primary"]:hover {
            transform: scale(1.05);
            box-shadow: 0 8px 20px rgba(37, 99, 235, 0.4) !important;
        }

        div.stButton > button[type="secondary"] {
            background: white !important;
            color: var(--text-main) !important;
            border: 1px solid #e2e8f0 !important;
            border-radius: 9999px !important;
            font-weight: 500 !important;
        }
        
        div.stButton > button[type="secondary"]:hover {
            background: #f8fafc !important;
            border-color: var(--primary) !important;
            color: var(--primary) !important;
        }

        /* --- DATAFRAME (Clean & Spacious) --- */
        [data-testid="stDataFrame"] {
            border: none !important;
            border-radius: 16px !important;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05) !important;
            background: white !important;
            padding: 5px;
        }

        /* --- TABS (Floating Pills) --- */
        .stTabs [data-baseweb="tab-list"] {
            gap: 12px;
            background-color: rgba(255,255,255,0.5);
            padding: 8px;
            border-radius: 9999px;
            border: 1px solid rgba(0,0,0,0.05);
            display: inline-flex;
            margin-bottom: 2rem;
        }

        .stTabs [data-baseweb="tab"] {
            background-color: transparent;
            border: none;
            border-radius: 9999px;
            padding: 8px 24px;
            font-weight: 600;
            color: var(--text-sub);
            transition: all 0.2s;
        }

        .stTabs [aria-selected="true"] {
            background-color: white !important;
            color: var(--primary) !important;
            box-shadow: 0 4px 12px rgba(0,0,0,0.05) !important;
        }

        /* --- ALERTS --- */
        .stAlert {
            border-radius: 16px !important;
            border: none !important;
            box-shadow: 0 4px 12px rgba(0,0,0,0.03) !important;
        }

        /* --- FILE UPLOADER (Idiot-Proof Dropzone) --- */
        section[data-testid="stFileUploaderDropzone"] {
            border: 2px dashed var(--primary) !important;
            background-color: rgba(59, 130, 246, 0.05) !important;
            border-radius: 20px !important;
            padding: 30px !important;
            transition: all 0.3s ease;
        }

        section[data-testid="stFileUploaderDropzone"]:hover {
            background-color: rgba(59, 130, 246, 0.1) !important;
            border-color: var(--primary-dark) !important;
            transform: scale(1.01);
        }
        
        /* Make the instruction text larger */
        section[data-testid="stFileUploaderDropzone"] div {
            font-size: 1.1rem !important;
            color: var(--primary-dark) !important;
            font-weight: 600 !important;
        }

        /* --- MOBILE --- */
        @media (max-width: 768px) {
            h1 { font-size: 2rem !important; text-align: left; }
            .stTabs [data-baseweb="tab-list"] { width: 100%; border-radius: 16px; overflow-x: auto; }
            [data-testid="stMetric"] { margin-bottom: 1rem; }
        }
    </style>
    """, unsafe_allow_html=True)

def main():
    setup_premium_design()
    display_sidebar()
    t1, t2, t3 = st.tabs(["Angebot", "Datenbank", "Verlauf"])
    with t1: tab_angebot_erstellen()
    with t2: tab_datenbank_verwalten()
    with t3: tab_verlauf()

if __name__ == "__main__":
    main()
