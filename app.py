import os
import json
import io
import base64
import numpy as np
import streamlit as st
from PIL import Image
from transformers import pipeline

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & EXACT PDF STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Fundgrube Katharineum",
    page_icon="🏫",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS basierend auf der exakten Farb- und Typografieanalyse aus dem PDF
custom_css = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Fredoka:wght@600;700&family=Inter:wght@400;600;800&display=swap');

    .stApp {
        background-color: #FFFFFF;
        color: #111111;
        font-family: 'Inter', sans-serif;
    }
    
    /* Gebogener Header-Schriftzug mit dunkler Outline */
    .arched-header-container {
        display: flex;
        justify-content: center;
        align-items: center;
        margin-top: 10px;
        margin-bottom: 5px;
    }
    
    .arched-header {
        font-family: 'Fredoka', cursive, sans-serif;
        font-size: 2.2rem;
        font-weight: 700;
        color: #FFFFFF;
        text-transform: uppercase;
        letter-spacing: 2px;
        -webkit-text-stroke: 2.5px #1B382B;
        text-shadow: 2px 2px 0px #1B382B;
        transform: perspective(300px) rotateX(10deg);
        text-align: center;
    }

    /* Stempel Badges (LOST & EXAMPLE) */
    .stamp-badge {
        display: inline-block;
        border: 2px solid #FFFFFF;
        outline: 2px solid #D11D32;
        background-color: #D11D32;
        color: white;
        font-weight: 800;
        font-size: 0.75rem;
        padding: 2px 6px;
        border-radius: 3px;
        transform: rotate(-3deg);
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* Input-Felder & Abgerundete Container */
    .stTextInput > div > div > input {
        border-radius: 20px !important;
        border: 2px solid #111111 !important;
        padding: 8px 15px !important;
    }

    /* Custom Cards im Grid */
    .item-card {
        background: #FFFFFF;
        border-radius: 12px;
        padding: 8px;
        text-align: center;
    }
    
    .item-title {
        font-family: 'Fredoka', sans-serif;
        font-weight: 700;
        font-size: 1.1rem;
        color: #000000;
        text-transform: uppercase;
        margin-top: 6px;
    }

    .item-tags {
        font-size: 0.8rem;
        color: #555555;
    }

    /* Bildeinfassung im Comic-Look */
    .img-placeholder-container {
        border-radius: 12px;
        overflow: hidden;
        border: 2px solid #111111;
    }

    /* Bottom Navigation Styling */
    div.stButton > button {
        border-radius: 25px !important;
        border: none !important;
        font-weight: 700 !important;
        height: 50px !important;
    }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. DATENBANK (JSON PERSISTENZ)
# -----------------------------------------------------------------------------
DB_FILE = "items_db.json"

def load_db():
    if os.path.exists(DB_FILE):
        try:
            with open(DB_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []

def save_db(items):
    try:
        with open(DB_FILE, "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=2)
    except Exception as e:
        st.error(f"Fehler beim Speichern: {e}")

def image_to_base64(pil_img: Image.Image) -> str:
    buffered = io.BytesIO()
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")
    pil_img.save(buffered, format="JPEG", quality=85)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

# -----------------------------------------------------------------------------
# 3. HUGGING FACE KI & AUTOMATISCHE ERKENNUNG
# -----------------------------------------------------------------------------
CATEGORIES = ["T-Shirt", "Pullover", "Mütze", "Flasche", "Brotdose", "Fahrradhelm", "Sonstiges"]

@st.cache_resource
def load_hf_classifier():
    try:
        return pipeline("image-classification", model="google/vit-base-patch16-224")
    except Exception:
        return None

classifier = load_hf_classifier()

def detect_color_name(pil_img: Image.Image) -> str:
    img = pil_img.copy().resize((50, 50))
    arr = np.array(img)
    if arr.shape[-1] == 4:
        arr = arr[:, :, :3]
    mean_rgb = arr.mean(axis=(0, 1))
    r, g, b = mean_rgb[0], mean_rgb[1], mean_rgb[2]
    
    if r < 50 and g < 50 and b < 50: return "Schwarz"
    elif r > 200 and g > 200 and b > 200: return "Weiß"
    elif abs(r - g) < 20 and abs(g - b) < 20 and abs(r - b) < 20: return "Grau"
    elif r > g + 30 and r > b + 30: return "Rot"
    elif g > r + 20 and g > b + 20: return "Grün"
    elif b > r + 20 and b > g + 20: return "Blau"
    elif r > 180 and g > 150 and b < 100: return "Gelb"
    elif r > 120 and g > 70 and b < 50: return "Braun"
    else: return "Bunt"

def map_label_to_category(raw_label: str) -> str:
    lbl = raw_label.lower()
    if any(k in lbl for k in ["t-shirt", "jersey", "polo", "shirt"]): return "T-Shirt"
    elif any(k in lbl for k in ["sweater", "cardigan", "sweatshirt", "hoodie", "jacket", "coat"]): return "Pullover"
    elif any(k in lbl for k in ["hat", "cap", "beanie", "bonnet", "beret"]): return "Mütze"
    elif any(k in lbl for k in ["bottle", "flask", "canteen", "thermos"]): return "Flasche"
    elif any(k in lbl for k in ["box", "container", "lunchbox", "bento"]): return "Brotdose"
    elif any(k in lbl for k in ["helmet", "crash helmet"]): return "Fahrradhelm"
    else: return "Sonstiges"

def classify_and_generate_tags(image: Image.Image):
    category = "Sonstiges"
    if classifier is not None:
        if image.mode != "RGB": image = image.convert("RGB")
        results = classifier(image)
        category = map_label_to_category(results[0]['label'])
    color = detect_color_name(image)
    return category, color, f"{category}, {color}"

# -----------------------------------------------------------------------------
# 4. INITIALISIERUNG
# -----------------------------------------------------------------------------
if "current_screen" not in st.session_state:
    st.session_state.current_screen = "Suchen"

if "selected_item_id" not in st.session_state:
    st.session_state.selected_item_id = None

st.session_state.items_db = load_db()

# -----------------------------------------------------------------------------
# 5. HEADER & BOTTOM NAVIGATION (PDF 1:1)
# -----------------------------------------------------------------------------
def render_header(title_text="FUNDGRUBE KATHARINEUM", show_logo=False, show_back=False):
    col_back, col_title, col_gear = st.columns([1, 4, 1])
    
    with col_back:
        if show_back:
            if st.button("← Zurück", key="hdr_back"):
                st.session_state.current_screen = "Suchen"
                st.session_state.selected_item_id = None
                st.rerun()

    with col_title:
        st.markdown(f"<div class='arched-header-container'><div class='arched-header'>{title_text}</div></div>", unsafe_allow_html=True)
        if show_logo:
            st.markdown("<div style='text-align:center; font-size: 2.2rem; margin-bottom: 5px;'>🏫</div>", unsafe_allow_html=True)

    with col_gear:
        if st.button("⚙️", key="hdr_gear"):
            st.session_state.current_screen = "Einstellungen"
            st.rerun()

def render_bottom_nav():
    st.markdown("<br><br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    curr = st.session_state.current_screen

    # Aktiver Button kriegt Signalrot (#D11D32), inaktive kriegen Neutralgrau (#737373)
    style_suchen = "background-color: #D11D32 !important; color: white !important;" if curr == "Suchen" else "background-color: #737373 !important; color: white !important;"
    style_add = "background-color: #D11D32 !important; color: white !important;" if curr == "Hinzufügen" else "background-color: #737373 !important; color: white !important;"
    style_lost = "background-color: #D11D32 !important; color: white !important;" if curr == "Vermisst" else "background-color: #737373 !important; color: white !important;"

    with c1:
        st.markdown(f"<style>div.element-container:has(#nav_btn_suchen) + div button {{ {style_suchen} }}</style>", unsafe_allow_html=True)
        st.markdown("<span id='nav_btn_suchen'></span>", unsafe_allow_html=True)
        if st.button("🔍  Suchen", width="stretch", key="btn_nav_suchen"):
            st.session_state.current_screen = "Suchen"
            st.session_state.selected_item_id = None
            st.rerun()

    with c2:
        st.markdown(f"<style>div.element-container:has(#nav_btn_add) + div button {{ {style_add} }}</style>", unsafe_allow_html=True)
        st.markdown("<span id='nav_btn_add'></span>", unsafe_allow_html=True)
        if st.button("➕  Hinzufügen", width="stretch", key="btn_nav_add"):
            st.session_state.current_screen = "Hinzufügen"
            st.rerun()

    with c3:
        st.markdown(f"<style>div.element-container:has(#nav_btn_lost) + div button {{ {style_lost} }}</style>", unsafe_allow_html=True)
        st.markdown("<span id='nav_btn_lost'></span>", unsafe_allow_html=True)
        if st.button("🏷️  Vermisst", width="stretch", key="btn_nav_lost"):
            st.session_state.current_screen = "Vermisst"
            st.rerun()

# -----------------------------------------------------------------------------
# 6. SCREEN 1: STARTBILDSCHIRM & SUCHEN
# -----------------------------------------------------------------------------
def screen_suchen():
    render_header("FUNDGRUBE KATHARINEUM", show_logo=True)
    st.session_state.items_db = load_db()

    search_query = st.text_input("", placeholder="Suchen Q", key="search_bar_input")

    with st.expander("Tags hinzufügen ∇", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("<span class='stamp-badge'>EXAMPLE</span> **Fundstück**", unsafe_allow_html=True)
            filter_cat = st.selectbox("", ["Alle"] + CATEGORIES, key="f_cat")
            st.markdown("<span class='stamp-badge'>EXAMPLE</span> **Farbe**", unsafe_allow_html=True)
            filter_color = st.selectbox("", ["Alle", "Schwarz", "Weiß", "Grau", "Rot", "Grün", "Blau", "Gelb", "Braun", "Bunt"], key="f_col")
        with c2:
            st.markdown("<span class='stamp-badge'>EXAMPLE</span> **Ort**", unsafe_allow_html=True)
            filter_loc = st.text_input("", placeholder="Ort suchen...", key="f_loc")

    st.markdown("---")

    filtered = st.session_state.items_db

    if search_query.strip():
        q = search_query.lower().strip()
        filtered = [i for i in filtered if q in i.get('title','').lower() or q in i.get('tags','').lower() or q in i.get('location','').lower()]

    if 'filter_cat' in locals() and filter_cat != "Alle":
        filtered = [i for i in filtered if filter_cat.lower() == i.get('category','').lower()]

    if 'filter_color' in locals() and filter_color != "Alle":
        filtered = [i for i in filtered if filter_color.lower() in i.get('color','').lower()]

    if not filtered:
        st.info("Keine Fundstücke gefunden.")
        return

    cols = st.columns(3)
    for idx, item in enumerate(filtered):
        with cols[idx % 3]:
            st.markdown("<div class='item-card'>", unsafe_allow_html=True)
            if item.get("image_b64"):
                st.image(base64.b64decode(item["image_b64"]), use_container_width=True)
            st.markdown(f"<div class='item-title'>{item['title']}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='item-tags'>Tags: {item['tags']}</div>", unsafe_allow_html=True)
            if st.button("Details", key=f"btn_det_{item['id']}"):
                st.session_state.selected_item_id = item['id']
                st.session_state.current_screen = "Detail"
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 7. SCREEN 2: HINZUFÜGEN
# -----------------------------------------------------------------------------
def screen_hinzufuegen():
    render_header("HINZUFÜGEN", show_back=True)

    c_upload, c_preview = st.columns([1.2, 1])

    uploaded_img = None
    with c_upload:
        st.markdown("### Bild hochladen")
        f = st.file_uploader("Bild aus Dateien hier hochladen", type=["jpg", "png", "jpeg"])
        if f:
            uploaded_img = Image.open(f)

    auto_cat, auto_col, auto_tags = "Sonstiges", "Unbekannt", ""
    if uploaded_img:
        with c_preview:
            st.markdown("### HOCHGELADENE BILDER")
            st.image(uploaded_img, width=200)
            auto_cat, auto_col, auto_tags = classify_and_generate_tags(uploaded_img)

    st.markdown("---")
    with st.form("form_add"):
        cat_idx = CATEGORIES.index(auto_cat) if auto_cat in CATEGORIES else 6
        sel_cat = st.selectbox("Kategorie", CATEGORIES, index=cat_idx)
        
        titel = st.text_input("Vorgeschlagener KI Titel ✏️", value=f"{sel_cat} ({auto_col})" if uploaded_img else "")
        tags = st.text_input("Vorgeschlagende KI Tags ✏️", value=auto_tags)
        ort = st.text_input("Findungsort", placeholder="z. B. Schulhof")
        aufbewahrung = st.text_input("Ort der Aufbewahrung", placeholder="z. B. Sekretariat")
        finder = st.text_input("Name vom Finder", placeholder="Dein Name")

        if st.form_submit_button("Fundstück speichern"):
            if not uploaded_img:
                st.error("Bitte lade ein Bild hoch!")
            elif not titel or not ort:
                st.error("Titel und Findungsort ausfüllen!")
            else:
                db = load_db()
                new_id = max([i['id'] for i in db], default=0) + 1
                new_item = {
                    "id": new_id,
                    "title": titel,
                    "category": sel_cat,
                    "color": auto_col,
                    "tags": tags,
                    "location": ort,
                    "storage_location": aufbewahrung if aufbewahrung else "Sekretariat",
                    "finder_name": finder if finder else "Anonym",
                    "image_b64": image_to_base64(uploaded_img)
                }
                db.append(new_item)
                save_db(db)
                st.session_state.current_screen = "Suchen"
                st.rerun()

# -----------------------------------------------------------------------------
# 8. SCREEN 3: VERMISST [LOST]
# -----------------------------------------------------------------------------
def screen_vermisst():
    render_header("VERMISST", show_back=True)
    st.session_state.items_db = load_db()

    f = st.file_uploader("Bild aus Dateien hier hochladen", type=["jpg", "png", "jpeg"])
    
    if f:
        img = Image.open(f)
        st.image(img, width=180)
        cat, col, tags = classify_and_generate_tags(img)

        st.markdown("### ÄHNLICHE BILDER")
        matches = [i for i in st.session_state.items_db if cat.lower() in i.get('category','').lower()]

        if matches:
            cols = st.columns(3)
            for idx, m in enumerate(matches):
                with cols[idx % 3]:
                    if m.get("image_b64"):
                        st.image(base64.b64decode(m["image_b64"]), use_container_width=True)
                    st.caption(m['title'])
        else:
            st.info("Keine ähnlichen Bilder in der Datenbank.")

    st.text_input("Titel hinzufügen")
    st.text_input("Tags hinzufügen")
    
    st.toggle("Benachrichtigung", value=True)

# -----------------------------------------------------------------------------
# 9. SCREEN 4: FUNDSTÜCK DETAILANSICHT
# -----------------------------------------------------------------------------
def screen_detail():
    render_header("FUNDSTÜCK", show_back=True)
    st.session_state.items_db = load_db()
    item = next((i for i in st.session_state.items_db if i['id'] == st.session_state.selected_item_id), None)

    if not item:
        st.error("Gegenstand nicht gefunden.")
        return

    st.markdown("### BILDER")
    if item.get("image_b64"):
        st.image(base64.b64decode(item["image_b64"]), width=300)

    st.markdown(f"## {item['title']}")
    st.markdown(f"**Tags:** {item['tags']}")
    st.markdown(f"**Findungsort:** {item['location']}")
    st.markdown(f"**Ort der Aufbewahrung:** {item['storage_location']}")
    st.markdown(f"**Name vom Finder:** {item['finder_name']}")

    if st.button("Als abgeholt markieren"):
        st.session_state.items_db = [i for i in st.session_state.items_db if i['id'] != item['id']]
        save_db(st.session_state.items_db)
        st.session_state.current_screen = "Suchen"
        st.rerun()

# -----------------------------------------------------------------------------
# 10. SCREEN 5: EINSTELLUNGEN
# -----------------------------------------------------------------------------
def screen_einstellungen():
    render_header("EINSTELLUNGEN", show_back=True)

    for i in range(5):
        c_txt, c_arr = st.columns([5, 1])
        with c_txt:
            st.write("verschiedene Einstellungen")
        with c_arr:
            st.button("→", key=f"einst_arrow_{i}")
        st.markdown("---")

# -----------------------------------------------------------------------------
# 11. MAIN ROUTER
# -----------------------------------------------------------------------------
def main():
    scr = st.session_state.current_screen
    if scr == "Suchen": screen_suchen()
    elif scr == "Hinzufügen": screen_hinzufuegen()
    elif scr == "Vermisst": screen_vermisst()
    elif scr == "Detail": screen_detail()
    elif scr == "Einstellungen": screen_einstellungen()

    render_bottom_nav()

if __name__ == "__main__":
    main()
