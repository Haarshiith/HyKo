import json
from openai import OpenAI

from ocr_module import ocr_all_pdfs_to_token_lists, preload_ocr_models

directory = r"D:\VS Code Env\OCR IMPROVED"

def main():
    print("Starte Hauptprogramm...")

    # 1. OCR Modelle laden (optional, wenn du es separat willst)
    preload_ocr_models()

    # 2. Die Funktion aus der anderen Datei aufrufen
    # Die Rückgabewerte landen direkt in diesen Variablen:
    print("Führe OCR durch...")
    all_rec_texts_list, all_rec_scores_list = ocr_all_pdfs_to_token_lists(directory)

    # 3. Jetzt hast du die Daten in der Main und kannst weiterarbeiten
    print(f"Erhaltene Daten: {len(all_rec_texts_list)} Seiten.")
    return all_rec_texts_list, all_rec_scores_list

    # ... Hier kommt dein Excel/Vendor Code hin ...
    # run_pipeline_page_by_page(all_rec_texts_list, all_rec_scores_list)

# Setup
client = OpenAI(base_url='http://localhost:11434/v1', api_key='ollama')
MODEL = "llama3.1" 

def extract_with_stream(text, keyword):
    prompt = f"""
    Du bist ein präziser Daten-Extraktions-Algorithmus.
    
    DEINE AUFGABE:
    Durchsuche den folgenden Text nach dem Keyword "{keyword}" (oder Synonymen wie Umsatz, Erlös, Revenue).
    Extrahiere jeden Fund in eine strikte JSON-Struktur.
    
    REGELN:
    1. "Value": Der exakte Zahlenwert (z.B. "4,530,303.00").
    2. "Context": Eine Zusammenfassung des Umfelds (ca. 25 Wörter davor/danach).
    3. "Keyword": Welches Wort hast du genau gefunden? (z.B. "Sales" oder "Umsatz").
    4. Gib mir eine LISTE von Objekten zurück, unter dem Key "entries".
    5. Antworte NUR mit JSON. Kein Gelaber davor oder danach.

    TEXT: {text}
    """

    stream = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        stream=True,
        # Optional: Hilft dem Modell, valides JSON zu schreiben
        response_format={"type": "json_object"} 
    )

    full_response = ""
    print("Modell denkt...", end="", flush=True)

    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            text_chunk = chunk.choices[0].delta.content
            print(text_chunk, end="", flush=True) 
            full_response += text_chunk
            
    print("\n") # Neue Zeile nach dem Stream
    return full_response

# --- TEST ---
text_beispiel = """
Der Geschäftsbericht 2024 zeigt interessante Zahlen. 
Auf Seite 5 sehen wir, dass der Sales im Bereich Nord 4.530.303,00 EUR betrug. 
Das war ein Anstieg. Hingegen war der Umsatz im Süden nur 200.000 EUR.
"""

if __name__ == "__main__":
    text_result, text_scores = main()

    print(text_scores, text_result)

    # 1. Den RAW TEXT holen (String)
    json_string = extract_with_stream(text_beispiel, "Sales")

    # 2. FIX: Den String in echtes Python-Objekt umwandeln
    try:
        # Manchmal packt das Modell Markdown ```json davor, das entfernen wir sicherheitshalber
        clean_json = json_string.replace("```json", "").replace("```", "").strip()
        
        data = json.loads(clean_json)
        
        # Wir holen uns die Liste unter dem Key "entries" (falls sie fehlt, leere Liste)
        echte_liste = data.get("entries", [])

        print("\n--- ERGEBNISSE ---")
        
        # 3. Jetzt können wir loopen!
        for eintrag in echte_liste:
            # .get() benutzen, damit es nicht crasht, falls Keys klein geschrieben sind
            val = eintrag.get('Value', eintrag.get('value', 'N/A'))
            context = eintrag.get('Context', eintrag.get('context', 'N/A'))
            
            print(f"💰 WERT: {val}")
            print(f"🔍 KONTEXT: {context}")
            print("-" * 20)

    except json.JSONDecodeError:
        print("❌ FEHLER: Das Modell hat kein gültiges JSON geliefert.")
    except Exception as e:
        print(f"❌ ERROR: {e}")