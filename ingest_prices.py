import sqlite3
import pandas as pd
from docx import Document
import re
import os
import glob

# Configuration
DB_PATH = 'data/prices.db'
SOURCE_DIR = '/Users/ozan/Desktop/Rahmenvereinbarung Kopie'

def init_db():
    """Initialize the SQLite database"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS prices (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_file TEXT,
            category TEXT,
            description TEXT,
            price_min REAL,
            price_max REAL,
            unit TEXT,
            raw_text TEXT
        )
    ''')

    # Create History Table
    c.execute('''
        CREATE TABLE IF NOT EXISTS offers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_name TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            total_amount REAL,
            item_count INTEGER,
            pdf_name TEXT
        )
    ''')

    # Create Learning Table (The "Brain")
    c.execute('''
        CREATE TABLE IF NOT EXISTS learning_mappings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            lv_text_hash TEXT, -- Hash of the description to identify it quickly
            lv_text_raw TEXT,  -- The original text from PDF
            mapped_price_id INTEGER, -- The ID from our internal price list
            confirmed_count INTEGER DEFAULT 1, -- How often was this mapping confirmed?
            FOREIGN KEY(mapped_price_id) REFERENCES prices(id)
        )
    ''')

    conn.commit()
    return conn

def parse_price_line(text):
    """
    Parses a line like "Perimeterdämmung einbauen: 3,00 €/m²"
    Returns a dict with description, price_min, price_max, unit
    """
    # Pattern for "Text... : 12,34 €/Einheit" or "Text... 12,34-15,00 €/Einheit"
    # This is a heuristic - real world data is messy!
    pattern = r'^(.*?)(?::|\s+)(\d+(?:,\d+)?)(?:\s*-\s*(\d+(?:,\d+)?))?\s*€\s*/\s*([a-zA-Z²³]+)'

    match = re.search(pattern, text)
    if match:
        desc = match.group(1).strip()
        p1 = float(match.group(2).replace(',', '.'))
        p2 = float(match.group(3).replace(',', '.')) if match.group(3) else p1
        unit = match.group(4).strip()
        return {
            'description': desc,
            'price_min': p1,
            'price_max': p2,
            'unit': unit,
            'raw_text': text
        }
    return None

def extract_from_docx(filepath, conn):
    """Extracts prices from a DOCX file and saves to DB"""
    try:
        doc = Document(filepath)
        print(f"Processing {filepath}...")

        cursor = conn.cursor()
        count = 0

        for para in doc.paragraphs:
            text = para.text.strip()
            if not text:
                continue

            parsed = parse_price_line(text)
            if parsed:
                cursor.execute('''
                    INSERT INTO prices (source_file, category, description, price_min, price_max, unit, raw_text)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    os.path.basename(filepath),
                    'General', # Future: Try to detect headers for category
                    parsed['description'],
                    parsed['price_min'],
                    parsed['price_max'],
                    parsed['unit'],
                    parsed['raw_text']
                ))
                count += 1

        conn.commit()
        print(f"  -> Extracted {count} items.")

    except Exception as e:
        print(f"Error reading {filepath}: {e}")

def main():
    if not os.path.exists('data'):
        os.makedirs('data')

    conn = init_db()

    # 1. Process "Interne Preisliste.docx" explicitly first as requested
    main_file = os.path.join(SOURCE_DIR, 'Interne Preisliste.docx')
    if os.path.exists(main_file):
        extract_from_docx(main_file, conn)
    else:
        print(f"Warning: Main file not found at {main_file}")

    # 2. (Optional) Process other DOCX files in the folder
    # docx_files = glob.glob(os.path.join(SOURCE_DIR, '*.docx'))
    # for f in docx_files:
    #     if f != main_file:
    #         extract_from_docx(f, conn)

    conn.close()
    print("\nDone! Database created at data/prices.db")

if __name__ == "__main__":
    main()
