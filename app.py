import os
import streamlit as st
from PIL import Image
import numpy as np
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
# 2. HUGGING FACE MODEL LOADING & PREDICTION
# -----------------------------------------------------------------------------
@st.cache_resource
def load_hf_classifier():
    """Lädt ein vortrainiertes Vision Transformer (ViT) Modell von Hugging Face."""
    try:
        # Schnelles und präzises Bildklassifikations-Modell
        return pipeline("image-classification", model="google/vit-base-patch16-224")
    except Exception as e:
        st.error(f"Fehler beim Laden des Hugging Face Modells: {e}")
        return None

classifier = load_hf_classifier()

def classify_image(image: Image.Image):
    """Klassifiziert ein Bild mit Hugging Face."""
    if classifier is not None:
        if image.mode != "RGB":
            image = image.convert("RGB")
            
        results = classifier(image)
        top_result = results[0]
        
        raw_label = top_result['label']
        confidence = float(top_result['score'])
        
        # Bereinigung des Labels (z. B. "backpack, back pack" -> "Backpack")
        clean_label = raw_label.split(',')[0].strip().title()
        return clean_label, confidence
    return "Fundstück", 0.0

# -----------------------------------------------------------------------------
# 3. SESSION STATE INITIALIZATION (LEERE DATENBANK)
# -----------------------------------------------------------------------------
if "current_screen" not in st.session_state:
    st.session_state.current_screen = "Suchen"

if "selected_item_id" not in st.session_state:
    st.session_state.selected_item_id = None

# Komplett leere Datenbank wie gewünscht
if "items_db" not in st.session_state:
    st.session_state.items_db = []

# -----------------------------------------------------------------------------
# 4. HEADER COMPONENT
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

# -----------------------------------------------------------------------------
# 5. BOTTOM NAVIGATION BAR
# -----------------------------------------------------------------------------
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
# 6. SCREEN 1: DASHBOARD & SCHNELLSUCHE (MIT OPTIMIERTEM TAG-FILTER)
# -----------------------------------------------------------------------------
def screen_suchen():
    render_header()
    
    search_query = st.text_input("🔍 Gegenstand oder Tag suchen...", placeholder="z. B. Schlüssel, Tasche, Rot...")
    
    with st.expander("Filter hinzufügen ▽", expanded=False):
        col_cat, col_col, col_brand, col_loc = st.columns(4)
        
        with col_cat:
            filter_cat = st.selectbox("Fundstück-Kategorie", ["Alle", "Schlüssel", "Taschen", "Flaschen", "Elektronik", "Kleidung"])
        with col_col:
            filter_color = st.selectbox("Farbe Suchen", ["Alle", "Blau", "Rot", "Grün", "Schwarz", "Silber"])
        with col_brand:
            filter_brand = st.text_input("Marke Suchen", placeholder="z.B. Nike, Mepal")
        with col_loc:
            filter_loc = st.text_input("Ort Suchen", placeholder="z.B. Pausenhof, Mensa")

    st.markdown("### Fundstücke Galerie")
    
    filtered_items = st.session_state.items_db
    
    # 1. Haupt-Suchfeld (Prüft Titel, Tags, Kategorie, Farbe, Ort)
    if search_query:
        q = search_query.lower().strip()
        filtered_items = [
            item for item in filtered_items 
            if q in item['title'].lower() 
            or q in item['tags'].lower() 
            or q in item.get('category', '').lower()
            or q in item.get('color', '').lower()
            or q in item.get('location', '').lower()
        ]
        
    # 2. Spezifische Filter (Prüfen das jeweilige Feld ODER die Tags)
    if 'filter_cat' in locals() and filter_cat != "Alle":
        cat_q = filter_cat.lower()
        filtered_items = [
            item for item in filtered_items 
            if cat_q in item.get('category', '').lower() or cat_q in item.get('tags', '').lower()
        ]
        
    if 'filter_color' in locals() and filter_color != "Alle":
        col_q = filter_color.lower()
        filtered_items = [
            item for item in filtered_items 
            if col_q in item.get('color', '').lower() or col_q in item.get('tags', '').lower()
        ]

    if 'filter_brand' in locals() and filter_brand.strip():
        b_q = filter_brand.lower().strip()
        filtered_items = [
            item for item in filtered_items 
            if b_q in item.get('brand', '').lower() or b_q in item.get('tags', '').lower()
        ]

    if 'filter_loc' in locals() and filter_loc.strip():
        l_q = filter_loc.lower().strip()
        filtered_items = [
            item for item in filtered_items 
            if l_q in item.get('location', '').lower() or l_q in item.get('tags', '').lower()
        ]

    if not filtered_items:
        if len(st.session_state.items_db) == 0:
            st.info("Aktuell sind keine Fundstücke registriert. Füge ein neues Fundstück über '➕ Hinzufügen' hinzu.")
        else:
            st.info("Keine Fundstücke gefunden, die auf deine Suchkriterien passen.")
        return

    cols = st.columns(3)
    for idx, item in enumerate(filtered_items):
        col = cols[idx % 3]
        with col:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            if item.get("image_data"):
                st.image(item["image_data"], width="stretch")
            elif item.get("image_url"):
                st.image(item["image_url"], width="stretch")
            
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
    
    item = next((i for i in st.session_state.items_db if i['id'] == st.session_state.selected_item_id), None)
    
    if not item:
        st.error("Fundstück nicht gefunden.")
        return

    col_img, col_info = st.columns([1, 1])
    
    with col_img:
        st.markdown("### BILDER")
        if item.get("image_data"):
            st.image(item["image_data"], width="stretch")
        elif item.get("image_url"):
            st.image(item["image_url"], width="stretch")
            
    with col_info:
        st.markdown(f"## {item['title']}")
        st.markdown(f"**🏷️ Tags:** {item['tags']}")
        st.markdown(f"**📍 Findungsort:** {item['location']}")
        st.markdown(f"**📦 Ort der Aufbewahrung:** {item['storage_location']}")
        st.markdown(f"**👤 Finder:** {item['finder_name']}")
        
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Als abgeholt / zurückgegeben markieren"):
            st.session_state.items_db = [i for i in st.session_state.items_db if i['id'] != item['id']]
            st.success("Gegenstand wurde aus der Datenbank entfernt.")
            st.session_state.current_screen = "Suchen"
            st.rerun()

# -----------------------------------------------------------------------------
# 8. SCREEN 3: FUNDSTÜCK MELDEN (HUGGING FACE KI INTEGRATION)
# -----------------------------------------------------------------------------
def screen_hinzufuegen():
    render_header(title_override="HINZUFÜGEN", show_back=True)
    
    st.markdown("### Bild hochladen oder Foto aufnehmen")
    
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

    detected_category = ""
    confidence = 0.0

    if uploaded_image:
        st.image(uploaded_image, caption="Hochgeladenes Bild", width=300)
        
        with st.spinner("Hugging Face KI analysiert das Bild..."):
            detected_category, confidence = classify_image(uploaded_image)
            
        st.success(f"🤖 **Hugging Face KI-Erkennung:** {detected_category} (Sicherheit: {confidence*100:.1f}%)")

    st.markdown("---")
    st.markdown("### Fund-Informationen vervollständigen")
    
    with st.form("form_add_item"):
        title_input = st.text_input("Vorgeschlagener KI-Titel (Anpassbar)", value=detected_category if detected_category else "")
        tags_input = st.text_input("Tags (Kommagetrennt für Such-Filter)", value=f"{detected_category}, Fundstück, Katharineum" if detected_category else "Fundstück, Katharineum")
        location_input = st.text_input("Findungsort", placeholder="z. B. Schulhof, Turnhalle, Raum 204")
        storage_input = st.text_input("Ort der Aufbewahrung", placeholder="z. B. Sekretariat, Hausmeister")
        finder_input = st.text_input("Name vom Finder", placeholder="Dein Name / Klasse (Optional)")
        
        submit = st.form_submit_button("Fundstück veröffentlichen 🚀")
        
        if submit:
            if not title_input or not location_input:
                st.error("Bitte mindestens Titel und Findungsort ausfüllen!")
            else:
                new_id = max([i['id'] for i in st.session_state.items_db], default=0) + 1
                new_item = {
                    "id": new_id,
                    "title": title_input,
                    "tags": tags_input,
                    "category": detected_category if detected_category else "Sonstiges",
                    "color": "Unbekannt",
                    "brand": "Unbekannt",
                    "location": location_input,
                    "storage_location": storage_input if storage_input else "Sekretariat",
                    "finder_name": finder_input if finder_input else "Anonym",
                    "image_data": uploaded_image if uploaded_image else None
                }
                st.session_state.items_db.append(new_item)
                st.success("Fundstück erfolgreich registriert!")
                st.session_state.current_screen = "Suchen"
                st.rerun()

# -----------------------------------------------------------------------------
# 9. SCREEN 4: VERMISST MELDEN & KI MATCHING
# -----------------------------------------------------------------------------
def screen_vermisst():
    render_header(title_override="VERMISST [LOST]", show_back=True)
    
    st.markdown("<span class='lost-badge'>HUGGING FACE SMART-MATCH</span>", unsafe_allow_html=True)
    st.write("Lade ein Foto deines verloren gegangenen Gegenstands hoch. Die KI vergleicht es direkt mit der Fund-Datenbank.")
    
    file = st.file_uploader("Bild deines verlorenen Gegenstands hochladen", type=["jpg", "jpeg", "png"], key="lost_uploader")
    
    if file:
        img = Image.open(file)
        st.image(img, width=250, caption="Dein Such-Bild")
        
        with st.spinner("Hugging Face KI vergleicht Gegenstand..."):
            predicted_cat, conf = classify_image(img)
            
        st.info(f"Erkannte Kategorie: **{predicted_cat}**")
        st.markdown("### 🔍 ÄHNLICHE BILDER IN DER DATENBANK")
        
        p_q = predicted_cat.lower()
        matches = [
            i for i in st.session_state.items_db 
            if p_q in i['title'].lower() or p_q in i['tags'].lower()
        ]
        
        if matches:
            cols = st.columns(min(len(matches), 3))
            for idx, match in enumerate(matches):
                with cols[idx % 3]:
                    st.image(match.get('image_data') or match.get('image_url'), width="stretch")
                    st.caption(f"**{match['title']}**\nOrt: {match['location']}")
        else:
            st.warning("Aktuell kein passender Gegenstand in der Datenbank vorhanden.")

    st.markdown("---")
    st.markdown("### Such-Auftrag erstellen")
    st.text_input("Titel hinzufügen", placeholder="z. B. Meine blaue Jacke")
    st.text_input("Tags hinzufügen", placeholder="z. B. Jacke, Blau, XL, Adidas")
    
    if st.button("Vermisst-Meldung speichern"):
        st.success("Such-Auftrag gespeichert!")

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
        "ℹ️ App-Info & Version (v2.0.0 Hugging Face Edition)"
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
