import os
import json
import io
import base64
import numpy as np
import streamlit as st
from PIL import Image
from transformers import pipeline

# -----------------------------------------------------------------------------
# 1. PAGE CONFIG & CUSTOM CSS (PERFEKTES MATCH DER PDF-VORLAGE)
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Fundgrube Katharineum",
    page_icon="🏫",
    layout="wide",
    initial_sidebar_state="collapsed"
)

custom_css = """
<style>
    /* Hintergrund & Hauptschrift */
    .stApp {
        background-color: #FFFFFF;
        color: #111111;
        font-family: 'Segoe UI', Arial, sans-serif;
    }
    
    /* Header Container */
    .header-container {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 5px 0 15px 0;
    }
    
    /* Red Gear Icon */
    .gear-btn {
        background-color: transparent;
        border: none;
        color: #D11D32;
        font-size: 2rem;
        cursor: pointer;
    }
    
    /* Search Bar Round Styling */
    .stTextInput input {
        border-radius: 20px !important;
        border: 2px solid #333333 !important;
        padding: 8px 15px !important;
        font-weight: 500;
    }

    /* Red Stamp Badges [LOST] & [EXAMPLE] */
    .stamp-badge {
        display: inline-block;
        background-color: #D11D32;
        color: #FFFFFF;
        font-weight: 900;
        font-size: 0.8rem;
        padding: 2px 8px;
        border: 2px solid #FFFFFF;
        box-shadow: 0 0 0 2px #D11D32;
        border-radius: 4px;
        letter-spacing: 1px;
        transform: rotate(-2deg);
        margin: 2px;
    }
    
    .stamp-example {
        background-color: #FFFFFF;
        color: #D11D32;
        border: 2px solid #D11D32;
        font-weight: 900;
        font-size: 0.75rem;
        padding: 1px 6px;
        border-radius: 3px;
        letter-spacing: 1px;
        display: inline-block;
        margin-right: 5px;
    }

    /* Card Layouts */
    .card-item {
        background-color: #FFFFFF;
        border-radius: 12px;
        padding: 10px;
        text-align: left;
        margin-bottom: 15px;
    }
    
    .card-title-bold {
        font-weight: 900;
        font-size: 1.1rem;
        color: #000000;
        text-transform: uppercase;
        margin-top: 6px;
        letter-spacing: 0.5px;
    }
    
    .card-tags-text {
        font-size: 0.85rem;
        color: #555555;
    }

    /* Bottom Navigation Bar (Floating Pills) */
    .bottom-nav-container {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background-color: #FFFFFF;
        border-top: 1px solid #E0E0E0;
        padding: 10px 20px;
        z-index: 9999;
    }

    /* Buttons override */
    div.stButton > button {
        border-radius: 25px !important;
        font-weight: bold !important;
        border: none !important;
        transition: all 0.2s !important;
    }
    
    /* Custom Green Toggle Simulation */
    .toggle-active {
        background-color: #00D54F;
        color: white;
        padding: 6px 16px;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
    }
    
    /* Settings Row Line */
    .settings-row {
        border-bottom: 1px solid #111111;
        padding: 12px 0;
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-weight: 600;
        font-size: 1.05rem;
    }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. PERSISTENT STORAGE (JSON SERVER DB)
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
        st.error(f"Fehler beim Speichern der Datenbank: {e}")

def image_to_base64(pil_img: Image.Image) -> str:
    buffered = io.BytesIO()
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")
    pil_img.save(buffered, format="JPEG", quality=85)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

# Default SVG Landscape Placeholder (Himmelblau #BCE3F7, Wiese #82A70C, Wolke)
DEFAULT_PLACEHOLDER_SVG = """
<svg viewBox="0 0 300 220" width="100%" height="180px" xmlns="http://www.w3.org/2000/svg" style="border-radius:8px;">
  <rect width="300" height="220" fill="#BCE3F7"/>
  <path d="M 70 80 Q 80 50 110 60 Q 130 40 160 60 Q 180 50 190 70 Q 200 90 170 100 L 70 100 Z" fill="#FFFFFF"/>
  <path d="M 0 140 Q 150 100 300 160 L 300 220 L 0 220 Z" fill="#82A70C"/>
</svg>
"""

# -----------------------------------------------------------------------------
# 3. HUGGING FACE KI & FARBANALYSE
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
    lbl = raw_label.lower()
    if any(k in lbl for k in ["t-shirt", "jersey", "polo", "shirt"]):
        return "T-Shirt"
    elif any(k in lbl for k in ["sweater", "cardigan", "sweatshirt", "hoodie", "jacket"]):
        return "Pullover"
    elif any(k in lbl for k in ["hat", "cap", "beanie", "bonnet"]):
        return "Mütze"
    elif any(k in lbl for k in ["bottle", "flask", "canteen", "thermos"]):
        return "Flasche"
    elif any(k in lbl for k in ["box", "container", "lunchbox"]):
        return "Brotdose"
    elif any(k in lbl for k in ["helmet", "crash helmet"]):
        return "Fahrradhelm"
    else:
        return "Sonstiges"

def classify_and_generate_tags(image: Image.Image):
    category = "Sonstiges"
    if classifier is not None:
        if image.mode != "RGB":
            image = image.convert("RGB")
        results = classifier(image)
        category = map_label_to_category(results[0]['label'])
        
    color = detect_color_name(image)
    auto_tags = f"{category}, {color}"
    return category, color, auto_tags

# -----------------------------------------------------------------------------
# 4. SESSION STATE INITIALISIERUNG
# -----------------------------------------------------------------------------
if "current_screen" not in st.session_state:
    st.session_state.current_screen = "Suchen"

if "selected_item_id" not in st.session_state:
    st.session_state.selected_item_id = None

st.session_state.items_db = load_db()

# -----------------------------------------------------------------------------
# 5. HEADER HELPER (EXAKTE BOGEN-SCHRIFT & LOGO)
# -----------------------------------------------------------------------------
def render_curved_header(title_text, show_back=False, show_logo=False):
    col_back, col_title, col_gear = st.columns([1, 4, 1])
    
    with col_back:
        if show_back:
            if st.button("← Zurück", key=f"btn_back_{title_text}"):
                st.session_state.current_screen = "Suchen"
                st.session_state.selected_item_id = None
                st.rerun()

    with col_title:
        # SVG Arched Path Curved Header Text
        svg_header = f"""
        <svg viewBox="0 0 500 90" width="100%" height="75px" xmlns="http://www.w3.org/2000/svg">
          <path id="curve" fill="transparent" d="M 20,75 Q 250,10 480,75" />
          <text style="font-size: 34px; font-weight: 900; font-family: 'Impact', 'Arial Black', sans-serif; letter-spacing: 2px;">
            <textPath href="#curve" startOffset="50%" text-anchor="middle" fill="#FFFFFF" stroke="#1B382B" stroke-width="3px" paint-order="stroke fill">
              {title_text}
            </textPath>
          </text>
        </svg>
        """
        st.markdown(svg_header, unsafe_allow_html=True)
        
        if show_logo:
            # Schul-Logo Badge im Startbildschirm
            st.markdown(
                """
                <div style="text-align: center; margin-top: -15px; margin-bottom: 10px;">
                    <span style="background-color:#003366; color:white; padding: 6px 18px; border-radius:20px; font-weight:900; font-size:0.9rem; letter-spacing:1px; border:2px solid #82A70C;">
                        LOGO
                    </span>
                </div>
                """, 
                unsafe_allow_html=True
            )

    with col_gear:
        if st.button("⚙️", key=f"gear_{title_text}"):
            st.session_state.current_screen = "Einstellungen"
            st.rerun()

    st.markdown("<hr style='margin: 5px 0 15px 0; border: 0.5px solid #EEEEEE;'>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 6. BOTTOM NAVIGATION BAR (PILL BUTTONS)
# -----------------------------------------------------------------------------
def render_bottom_nav():
    st.markdown("<br><br><br>", unsafe_allow_html=True)
    
    st.markdown("<div class='bottom-nav-container'>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    curr = st.session_state.current_screen

    with c1:
        # Active vs Inactive Red #D11D32 vs Grey #737373
        is_active = (curr == "Suchen")
        bg_col = "#D11D32" if is_active else "#737373"
        st.markdown(f"<style>div.stButton > button#btn_nav_suchen {{ background-color: {bg_col} !important; color: white !important; width: 100%; }}</style>", unsafe_allow_html=True)
        if st.button("🔍  Suchen", key="btn_nav_suchen"):
            st.session_state.current_screen = "Suchen"
            st.session_state.selected_item_id = None
            st.rerun()

    with c2:
        is_active = (curr == "Hinzufügen")
        bg_col = "#D11D32" if is_active else "#737373"
        st.markdown(f"<style>div.stButton > button#btn_nav_add {{ background-color: {bg_col} !important; color: white !important; width: 100%; }}</style>", unsafe_allow_html=True)
        if st.button("➕  Hinzufügen", key="btn_nav_add"):
            st.session_state.current_screen = "Hinzufügen"
            st.rerun()

    with c3:
        is_active = (curr == "Vermisst")
        bg_col = "#D11D32" if is_active else "#737373"
        st.markdown(f"<style>div.stButton > button#btn_nav_lost {{ background-color: {bg_col} !important; color: white !important; width: 100%; }}</style>", unsafe_allow_html=True)
        if st.button("🏷️ LOST  Vermisst", key="btn_nav_lost"):
            st.session_state.current_screen = "Vermisst"
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 7. SCREEN 1 & 3: SUCHEN / GALERIE
# -----------------------------------------------------------------------------
def screen_suchen():
    render_curved_header("FUNDGRUBE KATHARINEUM", show_logo=True)
    st.session_state.items_db = load_db()

    # Rounded Search Input
    search_query = st.text_input("", placeholder="Suchen Q", key="main_search_input")

    # Filter Accordion ("Tags hinzufügen ∇")
    with st.expander("Tags hinzufügen ∇", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            filter_cat = st.selectbox("Kategorie", ["Alle"] + CATEGORIES)
            filter_color = st.selectbox("Farbe", ["Alle", "Schwarz", "Weiß", "Grau", "Rot", "Grün", "Blau", "Gelb", "Braun", "Bunt"])
        with col2:
            filter_loc = st.text_input("Ort", placeholder="z. B. Pausenhof")
            filter_brand = st.text_input("Marke", placeholder="z. B. Adidas")

        # Category Rows with [EXAMPLE] Stamps
        st.markdown("<br>", unsafe_allow_html=True)
        for cat_label in ["Fundstück", "Farbe", "Marke", "Ort"]:
            st.markdown(
                f"""
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                    <span style="font-weight:bold;">{cat_label}</span>
                    <div>
                        <span class="stamp-example">EXAMPLE</span>
                        <span class="stamp-example">EXAMPLE</span>
                        <span style="font-size:0.85rem; text-decoration:underline; cursor:pointer;">Suchen</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # Filter Logic
    items = st.session_state.items_db
    if search_query.strip():
        q = search_query.lower().strip()
        items = [i for i in items if q in i.get('title', '').lower() or q in i.get('tags', '').lower() or q in i.get('location', '').lower()]

    if 'filter_cat' in locals() and filter_cat != "Alle":
        items = [i for i in items if filter_cat.lower() in i.get('category', '').lower() or filter_cat.lower() in i.get('tags', '').lower()]

    # 3-Column Card Grid
    if not items:
        # Fallback Placeholder Grid when empty
        cols = st.columns(3)
        for idx in range(6):
            with cols[idx % 3]:
                st.markdown("<div class='card-item'>", unsafe_allow_html=True)
                st.markdown(DEFAULT_PLACEHOLDER_SVG, unsafe_allow_html=True)
                st.markdown("<div class='card-title-bold'>TITEL</div>", unsafe_allow_html=True)
                st.markdown("<div class='card-tags-text'>Tags...</div>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
    else:
        cols = st.columns(3)
        for idx, item in enumerate(items):
            with cols[idx % 3]:
                st.markdown("<div class='card-item'>", unsafe_allow_html=True)
                if item.get("image_b64"):
                    img_bytes = base64.b64decode(item["image_b64"])
                    st.image(img_bytes, use_container_width=True)
                else:
                    st.markdown(DEFAULT_PLACEHOLDER_SVG, unsafe_allow_html=True)

                st.markdown(f"<div class='card-title-bold'>{item['title']}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='card-tags-text'>{item['tags']}</div>", unsafe_allow_html=True)
                
                if st.button("Details", key=f"btn_det_{item['id']}"):
                    st.session_state.selected_item_id = item['id']
                    st.session_state.current_screen = "Detail"
                    st.rerun()
                st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 8. SCREEN 2: HINZUFÜGEN (FUNDSTÜCK MELDEN)
# -----------------------------------------------------------------------------
def screen_hinzufuegen():
    render_curved_header("HINZUFÜGEN", show_back=True)

    col_upload, col_preview = st.columns([1.2, 1])

    uploaded_image = None
    with col_upload:
        st.markdown("### Bild hochladen")
        file = st.file_uploader("", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
        
        if not file:
            # Sky & Hill Placeholder with Big + Symbol
            upload_box_svg = """
            <svg viewBox="0 0 300 220" width="100%" height="200px" xmlns="http://www.w3.org/2000/svg" style="border-radius:12px; cursor:pointer;">
              <rect width="300" height="220" fill="#BCE3F7"/>
              <path d="M 70 80 Q 80 50 110 60 Q 130 40 160 60 Q 180 50 190 70 Q 200 90 170 100 L 70 100 Z" fill="#FFFFFF"/>
              <path d="M 0 140 Q 150 100 300 160 L 300 220 L 0 220 Z" fill="#82A70C"/>
              <line x1="150" y1="60" x2="150" y2="120" stroke="#111111" stroke-width="12" stroke-linecap="round"/>
              <line x1="120" y1="90" x2="180" y2="90" stroke="#111111" stroke-width="12" stroke-linecap="round"/>
            </svg>
            """
            st.markdown(upload_box_svg, unsafe_allow_html=True)
            st.caption("Bild aus Dateien hier hochladen")
        else:
            uploaded_image = Image.open(file)
            st.image(uploaded_image, use_container_width=True)

    with col_preview:
        st.markdown("<h4 style='font-weight:900; font-family:Impact, sans-serif;'>HOCHGELADENE BILDER</h4>", unsafe_allow_html=True)
        if uploaded_image:
            st.image(uploaded_image, width=140)
        else:
            st.markdown(DEFAULT_PLACEHOLDER_SVG, unsafe_allow_html=True)

    auto_cat = "Sonstiges"
    auto_color = "Unbekannt"
    auto_tags = ""

    if uploaded_image:
        with st.spinner("KI analysiert Bild..."):
            auto_cat, auto_color, auto_tags = classify_and_generate_tags(uploaded_image)

    st.markdown("<br>", unsafe_allow_html=True)

    # Formular
    with st.form("form_add"):
        c_title, c_edit = st.columns([4, 1])
        with c_title:
            cat_idx = CATEGORIES.index(auto_cat) if auto_cat in CATEGORIES else 6
            selected_cat = st.selectbox("Kategorie", CATEGORIES, index=cat_idx)
            title_in = st.text_input("Vorgeschlagener KI Titel", value=f"{selected_cat} ({auto_color})" if uploaded_image else "")
        with c_edit:
            st.markdown("<br><br>✏️ Bearbeiten", unsafe_allow_html=True)

        tags_in = st.text_input("Vorgeschlagende KI Tags", value=auto_tags)
        loc_in = st.text_input("Findungsort", placeholder="z. B. Schulhof, Sporthalle")
        storage_in = st.text_input("Ort der Aufbewahrung", placeholder="z. B. Sekretariat")

        submit = st.form_submit_button("Fundstück Speichern 🚀")

        if submit:
            if not uploaded_image:
                st.error("Bitte lade zuerst ein Bild hoch!")
            elif not loc_in:
                st.error("Bitte gib den Findungsort an!")
            else:
                db = load_db()
                new_id = max([i['id'] for i in db], default=0) + 1
                new_item = {
                    "id": new_id,
                    "title": title_in,
                    "category": selected_cat,
                    "color": auto_color,
                    "tags": tags_in,
                    "location": loc_in,
                    "storage_location": storage_in if storage_in else "Sekretariat",
                    "finder_name": "Anonym",
                    "image_b64": image_to_base64(uploaded_image)
                }
                db.append(new_item)
                save_db(db)
                st.success("Erfolgreich gespeichert!")
                st.session_state.current_screen = "Suchen"
                st.rerun()

# -----------------------------------------------------------------------------
# 9. SCREEN 4: FUNDSTÜCK DETAILANSICHT
# -----------------------------------------------------------------------------
def screen_detail():
    render_curved_header("FUNDSTÜCK", show_back=True)
    st.session_state.items_db = load_db()
    
    item = next((i for i in st.session_state.items_db if i['id'] == st.session_state.selected_item_id), None)
    if not item:
        st.error("Gegenstand nicht gefunden.")
        return

    c_img1, c_img2 = st.columns([1.5, 1])
    with c_img1:
        st.markdown("<h4 style='font-weight:900;'>BILDER</h4>", unsafe_allow_html=True)
        if item.get("image_b64"):
            st.image(base64.b64decode(item["image_b64"]), use_container_width=True)
        else:
            st.markdown(DEFAULT_PLACEHOLDER_SVG, unsafe_allow_html=True)

    with c_img2:
        if item.get("image_b64"):
            st.image(base64.b64decode(item["image_b64"]), width=120)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"### **Titel:** {item['title']}")
    st.markdown(f"**Tags:** {item['tags']}")
    st.markdown(f"**Findungsort:** {item['location']}")
    st.markdown(f"**Ort der Aufbewahrung:** {item['storage_location']}")
    st.markdown(f"**Name vom Finder:** {item.get('finder_name', 'Anonym')}")

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Als abgeholt / erledigt löschen"):
        st.session_state.items_db = [i for i in st.session_state.items_db if i['id'] != item['id']]
        save_db(st.session_state.items_db)
        st.success("Gegenstand gelöscht.")
        st.session_state.current_screen = "Suchen"
        st.rerun()

# -----------------------------------------------------------------------------
# 10. SCREEN 5: VERMISST (LOST SMART MATCH)
# -----------------------------------------------------------------------------
def screen_vermisst():
    render_curved_header("VERMISST", show_back=True)
    st.session_state.items_db = load_db()

    st.markdown("<span class='stamp-badge'>LOST</span>", unsafe_allow_html=True)
    st.markdown("<br><br>", unsafe_allow_html=True)

    c_up, c_sim = st.columns([1.2, 1])

    lost_file = None
    with c_up:
        file = st.file_uploader("Bild deines verlorenen Gegenstands", type=["jpg", "jpeg", "png"], key="lost_file_up")
        if file:
            lost_file = Image.open(file)
            st.image(lost_file, use_container_width=True)
        else:
            upload_box_svg = """
            <svg viewBox="0 0 300 220" width="100%" height="180px" xmlns="http://www.w3.org/2000/svg" style="border-radius:12px;">
              <rect width="300" height="220" fill="#BCE3F7"/>
              <path d="M 70 80 Q 80 50 110 60 Q 130 40 160 60 Q 180 50 190 70 Q 200 90 170 100 L 70 100 Z" fill="#FFFFFF"/>
              <path d="M 0 140 Q 150 100 300 160 L 300 220 L 0 220 Z" fill="#82A70C"/>
              <line x1="150" y1="60" x2="150" y2="120" stroke="#111111" stroke-width="12" stroke-linecap="round"/>
              <line x1="120" y1="90" x2="180" y2="90" stroke="#111111" stroke-width="12" stroke-linecap="round"/>
            </svg>
            """
            st.markdown(upload_box_svg, unsafe_allow_html=True)
            st.caption("Bild aus Dateien hier hochladen")

    with c_sim:
        st.markdown("<h4 style='font-weight:900;'>ÄHNLICHE BILDER</h4>", unsafe_allow_html=True)
        st.markdown(DEFAULT_PLACEHOLDER_SVG, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.text_input("Titel hinzufügen", placeholder="z. B. Roter Hoodie")
    st.text_input("Tags hinzufügen", placeholder="z. B. Pullover, Rot, Nike")

    # Custom Neon-Green Active Toggle Switch
    st.markdown(
        """
        <div style="display:flex; justify-content:space-between; align-items:center; margin-top:15px;">
            <span style="font-weight:bold; font-size:1.1rem;">Benachrichtigung</span>
            <div class="toggle-active">AN</div>
        </div>
        """,
        unsafe_allow_html=True
    )

# -----------------------------------------------------------------------------
# 11. SCREEN 6: EINSTELLUNGEN
# -----------------------------------------------------------------------------
def screen_einstellungen():
    render_curved_header("EINSTELLUNGEN", show_back=True)

    rows = [
        "verschiedene Einstellungen",
        "verschiedene Einstellungen",
        "verschiedene Einstellungen",
        "verschiedene Einstellungen",
        "verschiedene Einstellungen",
        "verschiedene Einstellungen"
    ]

    for item in rows:
        st.markdown(
            f"""
            <div class="settings-row">
                <span>{item}</span>
                <span>──→</span>
            </div>
            """,
            unsafe_allow_html=True
        )

# -----------------------------------------------------------------------------
# 12. ROUTER & MAIN EXECUTION
# -----------------------------------------------------------------------------
def main():
    screen = st.session_state.current_screen
    
    if screen == "Suchen":
        screen_suchen()
    elif screen == "Hinzufügen":
        screen_hinzufuegen()
    elif screen == "Detail":
        screen_detail()
    elif screen == "Vermisst":
        screen_vermisst()
    elif screen == "Einstellungen":
        screen_einstellungen()

    render_bottom_nav()

if __name__ == "__main__":
    main()
