import os
import json
import io
import base64
import numpy as np
import streamlit as st
from PIL import Image
from transformers import pipeline

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & CUSTOM CSS
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="FUNDGRUBE KATHARINEUM",
    page_icon="🏫",
    layout="wide",
    initial_sidebar_state="collapsed"
)

custom_css = """
<style>
    .stApp {
        background-color: #F8F9FA;
        color: #212121;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    .curved-header {
        text-align: center;
        font-weight: 900;
        font-size: 2.2rem;
        color: #D32F2F;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 0.2rem;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
    }
    
    .sub-header-logo {
        text-align: center;
        font-size: 2.8rem;
        margin-bottom: 1rem;
    }

    div.stButton > button:first-child {
        background-color: #D32F2F !important;
        color: white !important;
        border-radius: 12px !important;
        border: none !important;
        font-weight: bold !important;
        padding: 0.5rem 1rem !important;
        transition: all 0.3s ease;
    }
    div.stButton > button:first-child:hover {
        background-color: #B71C1C !important;
        box-shadow: 0 4px 8px rgba(211, 47, 47, 0.3);
    }

    .card {
        background-color: #FFFFFF;
        border-radius: 16px;
        padding: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        border: 1px solid #E0E0E0;
        margin-bottom: 15px;
        transition: transform 0.2s;
    }
    .card:hover {
        transform: translateY(-3px);
    }
    
    .card-title {
        font-weight: 700;
        font-size: 1.1rem;
        color: #212121;
        margin-top: 8px;
    }
    
    .card-tags {
        font-size: 0.85rem;
        color: #757575;
        margin-top: 2px;
    }

    .lost-badge {
        background-color: #D32F2F;
        color: white;
        padding: 4px 8px;
        border-radius: 6px;
        font-weight: bold;
        font-size: 0.8rem;
    }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. PERSISTENT STORAGE (GERÄTEÜBERGREIFENDE DATENBANK)
# -----------------------------------------------------------------------------
DB_FILE = "items_db.json"

def load_db():
    """Lädt die Fundstücke aus der JSON-Datei auf dem Server."""
    if os.path.exists(DB_FILE):
        try:
            with open(DB_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []

def save_db(items):
    """Speichert die Fundstücke dauerhaft in der JSON-Datei."""
    try:
        with open(DB_FILE, "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=2)
    except Exception as e:
        st.error(f"Fehler beim Speichern der Datenbank: {e}")

def image_to_base64(pil_img: Image.Image) -> str:
    """Konvertiert ein PIL-Bild in einen Base64-String für JSON-Speicherung."""
    buffered = io.BytesIO()
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")
    pil_img.save(buffered, format="JPEG", quality=85)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

# -----------------------------------------------------------------------------
# 3. HUGGING FACE KI & FARBANALYSE
# -----------------------------------------------------------------------------
CATEGORIES = ["T-Shirt", "Pullover", "Mütze", "Flasche", "Brotdose", "Fahrradhelm", "Sonstiges"]

@st.cache_resource
def load_hf_classifier():
    """Lädt das Bildklassifikations-Modell von Hugging Face."""
    try:
        return pipeline("image-classification", model="google/vit-base-patch16-224")
    except Exception as e:
        st.error(f"Fehler beim Laden des Hugging Face Modells: {e}")
        return None

classifier = load_hf_classifier()

def detect_color_name(pil_img: Image.Image) -> str:
    """Ermittelt automatisch die Hauptfarbe des hochgeladenen Bildes."""
    img = pil_img.copy().resize((50, 50))
    arr = np.array(img)
    if arr.shape[-1] == 4:
        arr = arr[:, :, :3]
    
    mean_rgb = arr.mean(axis=(0, 1))
    r, g, b = mean_rgb[0], mean_rgb[1], mean_rgb[2]
    
    if r < 50 and g < 50 and b < 50:
        return "Schwarz"
    elif r > 200 and g > 200 and b > 200:
        return "Weiß"
    elif abs(r - g) < 20 and abs(g - b) < 20 and abs(r - b) < 20:
        return "Grau"
    elif r > g + 30 and r > b + 30:
        return "Rot"
    elif g > r + 20 and g > b + 20:
        return "Grün"
    elif b > r + 20 and b > g + 20:
        return "Blau"
    elif r > 180 and g > 150 and b < 100:
        return "Gelb"
    elif r > 120 and g > 70 and b < 50:
        return "Braun"
    else:
        return "Bunt"

def map_label_to_category(raw_label: str) -> str:
    """Mappt das englische Hugging Face Ergebnis auf die vorgegebenen Kategorien."""
    lbl = raw_label.lower()
    
    if any(k in lbl for k in ["t-shirt", "jersey", "polo", "shirt"]):
        return "T-Shirt"
    elif any(k in lbl for k in ["sweater", "cardigan", "sweatshirt", "hoodie", "jacket", "coat"]):
        return "Pullover"
    elif any(k in lbl for k in ["hat", "cap", "beanie", "bonnet", "beret"]):
        return "Mütze"
    elif any(k in lbl for k in ["bottle", "flask", "canteen", "thermos", "water bottle"]):
        return "Flasche"
    elif any(k in lbl for k in ["box", "container", "lunchbox", "bento"]):
        return "Brotdose"
    elif any(k in lbl for k in ["helmet", "crash helmet"]):
        return "Fahrradhelm"
    else:
        return "Sonstiges"

def classify_and_generate_tags(image: Image.Image):
    """Analysiert das Bild, wählt die Kategorie und generiert automatische Tags."""
    category = "Sonstiges"
    confidence = 0.0
    
    if classifier is not None:
        if image.mode != "RGB":
            image = image.convert("RGB")
        results = classifier(image)
        top_result = results[0]
        raw_label = top_result['label']
        confidence = float(top_result['score'])
        category = map_label_to_category(raw_label)
        
    color = detect_color_name(image)
    auto_tags = f"{category}, {color}"
    
    return category, color, auto_tags, confidence

# -----------------------------------------------------------------------------
# 4. SESSION STATE INITIALIZATION
# -----------------------------------------------------------------------------
if "current_screen" not in st.session_state:
    st.session_state.current_screen = "Suchen"

if "selected_item_id" not in st.session_state:
    st.session_state.selected_item_id = None

# Always sync session state with the JSON database file
st.session_state.items_db = load_db()

# -----------------------------------------------------------------------------
# 5. HEADER & NAVIGATION
# -----------------------------------------------------------------------------
def render_header(title_override=None, show_back=False):
    col_back, col_title, col_settings = st.columns([1, 4, 1])
    
    with col_back:
        if show_back:
            if st.button("← Zurück", key="btn_back_header"):
                st.session_state.current_screen = "Suchen"
                st.session_state.selected_item_id = None
                st.rerun()

    with col_title:
        title = title_override if title_override else "FUNDGRUBE KATHARINEUM"
        st.markdown(f"<div class='curved-header'>{title}</div>", unsafe_allow_html=True)
        if not title_override:
            st.markdown("<div class='sub-header-logo'>🏫</div>", unsafe_allow_html=True)

    with col_settings:
        if st.button("⚙️", key="btn_settings_header"):
            st.session_state.current_screen = "Einstellungen"
            st.rerun()

    st.markdown("---")

def render_bottom_nav():
    st.markdown("<br><br><br>", unsafe_allow_html=True)
    
    nav_container = st.container()
    with nav_container:
        st.markdown("---")
        c1, c2, c3 = st.columns(3)
        current = st.session_state.current_screen
        
        with c1:
            lbl1 = "🔍 Suchen (Aktiv)" if current == "Suchen" else "🔍 Suchen"
            if st.button(lbl1, width="stretch", key="nav_suchen"):
                st.session_state.current_screen = "Suchen"
                st.session_state.selected_item_id = None
                st.rerun()
                
        with c2:
            lbl2 = "➕ HINZUFÜGEN (Aktiv)" if current == "Hinzufügen" else "➕ Hinzufügen"
            if st.button(lbl2, width="stretch", key="nav_hinzufuegen"):
                st.session_state.current_screen = "Hinzufügen"
                st.rerun()
                
        with c3:
            lbl3 = "🏷️ Vermisst (Aktiv)" if current == "Vermisst" else "🏷️ Vermisst [LOST]"
            if st.button(lbl3, width="stretch", key="nav_vermisst"):
                st.session_state.current_screen = "Vermisst"
                st.rerun()

# -----------------------------------------------------------------------------
# 6. SCREEN 1: SUCHEN & FILTERN
# -----------------------------------------------------------------------------
def screen_suchen():
    render_header()
    
    # Reload latest DB state from disk
    st.session_state.items_db = load_db()
    
    search_query = st.text_input("🔍 Suchleiste (Eingeben und Enter drücken)", placeholder="z. B. Mütze, Rot, Schulhof...")
    
    with st.expander("Filter hinzufügen ▽", expanded=False):
        col_cat, col_col, col_loc = st.columns(3)
        
        with col_cat:
            filter_cat = st.selectbox("Kategorie Filter", ["Alle"] + CATEGORIES)
        with col_col:
            filter_color = st.selectbox("Farbe Filter", ["Alle", "Schwarz", "Weiß", "Grau", "Rot", "Grün", "Blau", "Gelb", "Braun", "Bunt"])
        with col_loc:
            filter_loc = st.text_input("Ort Filter", placeholder="z.B. Pausenhof, Mensa")

    st.markdown("### Fundstücke Galerie")
    
    filtered_items = st.session_state.items_db
    
    # Text-Suche (Funktioniert direkt bei Enter)
    if search_query.strip():
        q = search_query.lower().strip()
        filtered_items = [
            item for item in filtered_items 
            if q in item.get('title', '').lower() 
            or q in item.get('tags', '').lower() 
            or q in item.get('category', '').lower()
            or q in item.get('color', '').lower()
            or q in item.get('location', '').lower()
        ]
        
    # Dropdown- & Orts-Filter
    if 'filter_cat' in locals() and filter_cat != "Alle":
        cat_q = filter_cat.lower()
        filtered_items = [
            item for item in filtered_items 
            if cat_q == item.get('category', '').lower() or cat_q in item.get('tags', '').lower()
        ]
        
    if 'filter_color' in locals() and filter_color != "Alle":
        col_q = filter_color.lower()
        filtered_items = [
            item for item in filtered_items 
            if col_q in item.get('color', '').lower() or col_q in item.get('tags', '').lower()
        ]

    if 'filter_loc' in locals() and filter_loc.strip():
        l_q = filter_loc.lower().strip()
        filtered_items = [
            item for item in filtered_items 
            if l_q in item.get('location', '').lower() or l_q in item.get('tags', '').lower()
        ]

    if not filtered_items:
        if len(st.session_state.items_db) == 0:
            st.info("Aktuell sind keine Fundstücke in der Datenbank vorhanden.")
        else:
            st.info("Keine Fundstücke gefunden, die auf deine Suchanfrage passen.")
        return

    cols = st.columns(3)
    for idx, item in enumerate(filtered_items):
        col = cols[idx % 3]
        with col:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            if item.get("image_b64"):
                img_bytes = base64.b64decode(item["image_b64"])
                st.image(img_bytes, width="stretch")
            
            st.markdown(f"<div class='card-title'>{item['title']}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='card-tags'>Tags: {item['tags']}</div>", unsafe_allow_html=True)
            
            if st.button("Details anzeigen", key=f"btn_item_{item['id']}"):
                st.session_state.selected_item_id = item['id']
                st.session_state.current_screen = "Detail"
                st.rerun()
                
            st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 7. SCREEN 2: DETAILANSICHT
# -----------------------------------------------------------------------------
def screen_detail():
    render_header(title_override="FUNDSTÜCK", show_back=True)
    
    st.session_state.items_db = load_db()
    item = next((i for i in st.session_state.items_db if i['id'] == st.session_state.selected_item_id), None)
    
    if not item:
        st.error("Fundstück nicht gefunden.")
        return

    col_img, col_info = st.columns([1, 1])
    
    with col_img:
        st.markdown("### BILD")
        if item.get("image_b64"):
            img_bytes = base64.b64decode(item["image_b64"])
            st.image(img_bytes, width="stretch")
            
    with col_info:
        st.markdown(f"## {item['title']}")
        st.markdown(f"**🏷️ Tags (Kategorie & Farbe):** {item['tags']}")
        st.markdown(f"**📍 Findungsort:** {item['location']}")
        st.markdown(f"**📦 Ort der Aufbewahrung:** {item['storage_location']}")
        st.markdown(f"**👤 Finder:** {item['finder_name']}")
        
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Als abgeholt / zurückgegeben markieren"):
            st.session_state.items_db = [i for i in st.session_state.items_db if i['id'] != item['id']]
            save_db(st.session_state.items_db)
            st.success("Gegenstand wurde abgeholt und aus der Datenbank gelöscht!")
            st.session_state.current_screen = "Suchen"
            st.rerun()

# -----------------------------------------------------------------------------
# 8. SCREEN 3: FUNDSTÜCK MELDEN (HINZUFÜGEN)
# -----------------------------------------------------------------------------
def screen_hinzufuegen():
    render_header(title_override="HINZUFÜGEN", show_back=True)
    
    st.markdown("### Foto aufnehmen oder hochladen")
    upload_option = st.radio("Upload-Quelle wählen:", ["Datei hochladen", "Kamera nutzen"], horizontal=True)
    
    uploaded_image = None
    if upload_option == "Datei hochladen":
        file = st.file_uploader("Bild auswählen", type=["jpg", "jpeg", "png"])
        if file:
            uploaded_image = Image.open(file)
    else:
        camera_file = st.camera_input("Foto aufnehmen")
        if camera_file:
            uploaded_image = Image.open(camera_file)

    auto_cat = "Sonstiges"
    auto_color = "Unbekannt"
    auto_tags = ""
    confidence = 0.0

    if uploaded_image:
        st.image(uploaded_image, caption="Vorschau", width=250)
        
        with st.spinner("KI analysiert Bild & Farbe..."):
            auto_cat, auto_color, auto_tags, confidence = classify_and_generate_tags(uploaded_image)
            
        st.success(f"🤖 **KI-Erkennung:** {auto_cat} | **Farbe:** {auto_color}")

    st.markdown("---")
    st.markdown("### Fund-Informationen eintragen")
    
    with st.form("form_add_item"):
        # Vorausgewählte Werte aus der KI
        category_index = CATEGORIES.index(auto_cat) if auto_cat in CATEGORIES else CATEGORIES.index("Sonstiges")
        selected_category = st.selectbox("Kategorie", CATEGORIES, index=category_index)
        
        title_input = st.text_input("Titel / Gegenstand", value=f"{selected_category} ({auto_color})" if uploaded_image else "")
        tags_input = st.text_input("Tags (Automatisch aus Farbe & Kategorie)", value=auto_tags)
        location_input = st.text_input("Findungsort", placeholder="z. B. Schulhof, Turnhalle, Raum 204")
        storage_input = st.text_input("Ort der Aufbewahrung", placeholder="z. B. Sekretariat, Hausmeister")
        finder_input = st.text_input("Name vom Finder (Optional)", placeholder="Dein Name / Klasse")
        
        submit = st.form_submit_button("Fundstück speichern 🚀")
        
        if submit:
            if not uploaded_image:
                st.error("Bitte lade zuerst ein Bild des Gegenstands hoch!")
            elif not title_input or not location_input:
                st.error("Bitte mindestens Titel und Findungsort ausfüllen!")
            else:
                current_items = load_db()
                new_id = max([i['id'] for i in current_items], default=0) + 1
                
                new_item = {
                    "id": new_id,
                    "title": title_input,
                    "category": selected_category,
                    "color": auto_color,
                    "tags": tags_input,
                    "location": location_input,
                    "storage_location": storage_input if storage_input else "Sekretariat",
                    "finder_name": finder_input if finder_input else "Anonym",
                    "image_b64": image_to_base64(uploaded_image)
                }
                
                current_items.append(new_item)
                save_db(current_items)
                st.session_state.items_db = current_items
                
                st.success("Fundstück erfolgreich geräteübergreifend gespeichert!")
                st.session_state.current_screen = "Suchen"
                st.rerun()

# -----------------------------------------------------------------------------
# 9. SCREEN 4: VERMISST MELDEN & KI MATCHING
# -----------------------------------------------------------------------------
def screen_vermisst():
    render_header(title_override="VERMISST [LOST]", show_back=True)
    st.session_state.items_db = load_db()
    
    st.markdown("<span class='lost-badge'>HUGGING FACE SMART-MATCH</span>", unsafe_allow_html=True)
    st.write("Lade ein Foto deines verlorenen Gegenstands hoch. Die KI vergleicht es direkt mit der Fund-Datenbank.")
    
    file = st.file_uploader("Bild deines verlorenen Gegenstands hochladen", type=["jpg", "jpeg", "png"], key="lost_uploader")
    
    if file:
        img = Image.open(file)
        st.image(img, width=220, caption="Dein Such-Bild")
        
        with st.spinner("KI vergleicht Gegenstand..."):
            cat, col, tags, conf = classify_and_generate_tags(img)
            
        st.info(f"Erkannte Kategorie: **{cat}** | Farbe: **{col}**")
        st.markdown("### 🔍 MÖGLICHE TREFFER IN DER DATENBANK")
        
        matches = [
            i for i in st.session_state.items_db 
            if cat.lower() in i.get('category', '').lower() or cat.lower() in i.get('tags', '').lower()
        ]
        
        if matches:
            cols = st.columns(min(len(matches), 3))
            for idx, match in enumerate(matches):
                with cols[idx % 3]:
                    if match.get("image_b64"):
                        st.image(base64.b64decode(match["image_b64"]), width="stretch")
                    st.caption(f"**{match['title']}**\nOrt: {match['location']}")
        else:
            st.warning("Aktuell kein passender Gegenstand in der Datenbank gefunden.")

# -----------------------------------------------------------------------------
# 10. SCREEN 5: EINSTELLUNGEN
# -----------------------------------------------------------------------------
def screen_einstellungen():
    render_header(title_override="EINSTELLUNGEN", show_back=True)
    
    settings_options = [
        "🔔 Benachrichtigungen & Push-Service",
        "👤 Mein Profil / Kontaktdaten",
        "🏫 Schule / Standort (Katharineum)",
        "🔒 Datenschutz & Nutzungsbedingungen",
        "ℹ️ App-Info & Version (v2.1.0 Persistence & Hugging Face)"
    ]
    
    for opt in settings_options:
        col_txt, col_arrow = st.columns([5, 1])
        with col_txt:
            st.markdown(f"**{opt}**")
        with col_arrow:
            st.button("→", key=f"btn_opt_{opt}")
        st.markdown("---")

# -----------------------------------------------------------------------------
# 11. MAIN ROUTER
# -----------------------------------------------------------------------------
def main():
    screen = st.session_state.current_screen
    
    if screen == "Suchen":
        screen_suchen()
    elif screen == "Detail":
        screen_detail()
    elif screen == "Hinzufügen":
        screen_hinzufuegen()
    elif screen == "Vermisst":
        screen_vermisst()
    elif screen == "Einstellungen":
        screen_einstellungen()
        
    render_bottom_nav()

if __name__ == "__main__":
    main()
