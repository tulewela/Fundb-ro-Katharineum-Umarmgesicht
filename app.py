import os
import json
import io
import base64
import numpy as np
import streamlit as st
from PIL import Image
from transformers import pipeline

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Fundgrube Katharineum",
    page_icon="🏫",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Erzeugung von sanft gebogenem, gut lesbarem Text + Logo darunter
def generate_curved_header_svg(title_text: str) -> str:
    svg = f"""
    <svg width="420" height="135" viewBox="0 0 420 135" fill="none" xmlns="http://www.w3.org/2000/svg" style="display:block; margin:auto;">
      <defs>
        <path id="gentleArc" d="M 20 45 Q 210 15 400 45" />
        <linearGradient id="brandGrad" x1="0%" y1="0%" x2="100%" y2="0%">
          <stop offset="0%" stop-color="#B71C1C" />
          <stop offset="100%" stop-color="#D11D32" />
        </linearGradient>
      </defs>
      
      <text font-family="'Poppins', sans-serif" font-weight="800" font-size="22" fill="url(#brandGrad)" text-anchor="middle" letter-spacing="1.5">
        <textPath href="#gentleArc" startOffset="50%">{title_text}</textPath>
      </text>

      <g transform="translate(180, 48)">
        <path d="M 5 22 L 30 10 L 55 22 L 55 52 L 30 62 L 5 52 Z" fill="#D11D32" fill-opacity="0.1" stroke="#D11D32" stroke-width="2.5" stroke-linejoin="round"/>
        <path d="M 5 22 L 30 32 L 55 22" stroke="#D11D32" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>
        <path d="M 30 32 L 30 62" stroke="#D11D32" stroke-width="2.5" opacity="0.5"/>
        <path d="M 18 18 C 18 12, 42 12, 42 18 L 48 25 L 42 28 L 40 40 L 20 40 L 18 28 L 12 25 Z" fill="#D11D32" stroke="#D11D32" stroke-width="1.5" stroke-linejoin="round"/>
        <path d="M 26 18 L 30 24 L 34 18" stroke="#FFFFFF" stroke-width="1.5" stroke-linecap="round" fill="none"/>
      </g>
    </svg>
    """
    b64 = base64.b64encode(svg.encode('utf-8')).decode('utf-8')
    return f"data:image/svg+xml;base64,{b64}"

custom_css = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&display=swap');

    .stApp {
        background-color: #FAFAFA;
        color: #212121;
        font-family: 'Poppins', sans-serif;
    }

    .header-svg-container {
        display: flex;
        justify-content: center;
        align-items: center;
        margin-top: -5px;
        margin-bottom: 10px;
    }

    /* Streamlit Buttons */
    div.stButton > button {
        border-radius: 12px !important;
        border: 2px solid #D11D32 !important;
        background-color: #FFFFFF !important;
        color: #D11D32 !important;
        font-weight: 600 !important;
        padding: 0.4rem 0.8rem !important;
        transition: all 0.2s ease-in-out !important;
        box-shadow: 0 2px 5px rgba(209, 29, 50, 0.08) !important;
    }

    div.stButton > button:hover {
        background-color: #D11D32 !important;
        color: #FFFFFF !important;
        transform: translateY(-2px);
        box-shadow: 0 4px 10px rgba(209, 29, 50, 0.22) !important;
    }

    /* Input Felder */
    .stTextInput > div > div > input, .stSelectbox > div > div {
        border-radius: 12px !important;
        border: 1.5px solid #E0E0E0 !important;
        padding: 8px 12px !important;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #D11D32 !important;
    }

    /* BÜNDIGE BIBLIOTHEK-KARTEN */
    .item-card {
        background: #FFFFFF;
        border-radius: 16px;
        padding: 12px;
        border: 1px solid #EEEEEE;
        box-shadow: 0 4px 12px rgba(0,0,0,0.03);
        margin-bottom: 15px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        transition: transform 0.2s ease;
    }
    
    .item-card:hover {
        transform: translateY(-3px);
    }

    .item-card-img {
        width: 100%;
        height: 210px;
        object-fit: cover;
        border-radius: 10px;
    }

    .item-card-content {
        padding: 8px 2px 2px 2px;
        text-align: center;
    }
    
    .item-title {
        font-weight: 600;
        font-size: 1rem;
        color: #212121;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }

    .item-tags {
        font-size: 0.8rem;
        color: #757575;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        margin-top: 2px;
    }

    /* GROSSES HAUPTBILD BEIM HOCHLADEN (3-4x größer) */
    .main-upload-preview {
        width: 100%;
        max-width: 260px;
        height: 240px;
        object-fit: cover;
        border-radius: 14px;
        border: 2px solid #D11D32;
        box-shadow: 0 4px 12px rgba(209, 29, 50, 0.12);
        margin-bottom: 10px;
        display: block;
    }

    /* KLEINERE ZUSÄTZLICHE THUMBNAILS DARUNTER */
    .upload-thumb {
        width: 70px;
        height: 70px;
        object-fit: cover;
        border-radius: 8px;
        border: 1.5px solid #CCCCCC;
        margin-right: 8px;
        margin-bottom: 8px;
    }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. DATENBANK HILFSFUNKTIONEN
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

def images_to_base64_list(pil_images) -> list:
    encoded_list = []
    for pil_img in pil_images:
        buffered = io.BytesIO()
        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")
        pil_img.save(buffered, format="JPEG", quality=85)
        encoded_list.append(base64.b64encode(buffered.getvalue()).decode("utf-8"))
    return encoded_list

# -----------------------------------------------------------------------------
# 3. HUGGING FACE KI ERKENNUNG
# -----------------------------------------------------------------------------
CATEGORIES = ["T-Shirt", "Pullover", "Mütze", "Flasche", "Brotdose", "Fahrradhelm", "Sonstiges"]
COLORS = ["Schwarz", "Weiß", "Grau", "Rot", "Grün", "Blau", "Gelb", "Braun", "Bunt"]

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
# 5. HEADER & NAVIGATION
# -----------------------------------------------------------------------------
def render_header(title_text="FUNDGRUBE KATHARINEUM", show_back=False):
    col_back, col_title, col_gear = st.columns([1, 5, 1])
    
    with col_back:
        if show_back:
            if st.button("← Zurück", key="hdr_back"):
                st.session_state.current_screen = "Suchen"
                st.session_state.selected_item_id = None
                st.rerun()

    with col_title:
        svg_url = generate_curved_header_svg(title_text)
        st.markdown(
            f"<div class='header-svg-container'><img src='{svg_url}' style='width:380px; max-width:100%;'/></div>", 
            unsafe_allow_html=True
        )

    with col_gear:
        if st.button("⚙️", key="hdr_gear"):
            st.session_state.current_screen = "Einstellungen"
            st.rerun()

def render_bottom_nav():
    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    curr = st.session_state.current_screen

    style_active = "background-color: #D11D32 !important; color: white !important; border: 2px solid #D11D32 !important;"
    style_inactive = "background-color: #FFFFFF !important; color: #D11D32 !important; border: 2px solid #D11D32 !important;"

    with c1:
        st.markdown(f"<style>div.element-container:has(#nav_btn_suchen) + div button {{ {style_active if curr == 'Suchen' else style_inactive} }}</style>", unsafe_allow_html=True)
        st.markdown("<span id='nav_btn_suchen'></span>", unsafe_allow_html=True)
        if st.button("🔍  Suchen", width="stretch", key="btn_nav_suchen"):
            st.session_state.current_screen = "Suchen"
            st.session_state.selected_item_id = None
            st.rerun()

    with c2:
        st.markdown(f"<style>div.element-container:has(#nav_btn_add) + div button {{ {style_active if curr == 'Hinzufügen' else style_inactive} }}</style>", unsafe_allow_html=True)
        st.markdown("<span id='nav_btn_add'></span>", unsafe_allow_html=True)
        if st.button("➕  Hinzufügen", width="stretch", key="btn_nav_add"):
            st.session_state.current_screen = "Hinzufügen"
            st.rerun()

    with c3:
        st.markdown(f"<style>div.element-container:has(#nav_btn_lost) + div button {{ {style_active if curr == 'Vermisst' else style_inactive} }}</style>", unsafe_allow_html=True)
        st.markdown("<span id='nav_btn_lost'></span>", unsafe_allow_html=True)
        if st.button("🏷️  Vermisst", width="stretch", key="btn_nav_lost"):
            st.session_state.current_screen = "Vermisst"
            st.rerun()

# -----------------------------------------------------------------------------
# 6. SCREEN 1: SUCHEN & BIBLIOTHEK
# -----------------------------------------------------------------------------
def screen_suchen():
    render_header("FUNDGRUBE KATHARINEUM")
    st.session_state.items_db = load_db()

    search_query = st.text_input("", placeholder="Suchen nach Gegenstand, Farbe, Ort...", key="search_bar_input")

    with st.expander("Filter & Tags auswählen ∇", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            selected_cats = st.multiselect("Kategorie auswählen", CATEGORIES, key="f_cats")
        with c2:
            selected_colors = st.multiselect("Farbe auswählen", COLORS, key="f_colors")
        
        filter_loc = st.text_input("Ort filtern", placeholder="z.B. Schulhof, Sporthalle", key="f_loc")

    st.markdown("---")

    filtered = st.session_state.items_db

    if search_query.strip():
        q = search_query.lower().strip()
        filtered = [i for i in filtered if q in i.get('title','').lower() or q in i.get('tags','').lower() or q in i.get('location','').lower()]

    if 'selected_cats' in locals() and selected_cats:
        filtered = [i for i in filtered if any(c.lower() == i.get('category','').lower() for c in selected_cats)]

    if 'selected_colors' in locals() and selected_colors:
        filtered = [i for i in filtered if any(col.lower() in i.get('color','').lower() or col.lower() in i.get('tags','').lower() for col in selected_colors)]

    if 'filter_loc' in locals() and filter_loc.strip():
        l_q = filter_loc.lower().strip()
        filtered = [i for i in filtered if l_q in i.get('location','').lower()]

    if not filtered:
        st.info("Keine Fundstücke gefunden.")
        return

    cols = st.columns(3)
    for idx, item in enumerate(filtered):
        with cols[idx % 3]:
            images = item.get("images_b64", [])
            if not images and item.get("image_b64"):
                images = [item["image_b64"]]
                
            img_src = f"data:image/jpeg;base64,{images[0]}" if images else "https://via.placeholder.com/200x210"

            st.markdown(
                f"""
                <div class='item-card'>
                    <img src='{img_src}' class='item-card-img'/>
                    <div class='item-card-content'>
                        <div class='item-title'>{item['title']}</div>
                        <div class='item-tags'>Tags: {item['tags']}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
            if st.button("Details anzeigen", key=f"btn_det_{item['id']}", use_container_width=True):
                st.session_state.selected_item_id = item['id']
                st.session_state.current_screen = "Detail"
                st.rerun()

# -----------------------------------------------------------------------------
# 7. SCREEN 2: HINZUFÜGEN (PROMINENTES HAUPTBILD)
# -----------------------------------------------------------------------------
def screen_hinzufuegen():
    render_header("HINZUFÜGEN", show_back=True)

    st.markdown("### Bilder hochladen")
    files = st.file_uploader("Bilder aus Dateien hier hochladen", type=["jpg", "png", "jpeg"], accept_multiple_files=True)
    
    uploaded_imgs = []
    if files:
        uploaded_imgs = [Image.open(f) for f in files]

    auto_cat, auto_col, auto_tags = "Sonstiges", "Unbekannt", ""
    
    if uploaded_imgs:
        b64_list = images_to_base64_list(uploaded_imgs)
        
        # Prominente Anzeige des Hauptbildes + kleine Thumbnails für weitere Bilder
        preview_html = f"""
        <div style='margin-bottom:15px;'>
            <div style='font-size:0.85rem; font-weight:600; color:#555; margin-bottom:6px;'>Ausgewähltes Hauptbild:</div>
            <img src='data:image/jpeg;base64,{b64_list[0]}' class='main-upload-preview'/>
        """
        
        if len(b64_list) > 1:
            preview_html += "<div style='font-size:0.85rem; font-weight:600; color:#555; margin-top:10px; margin-bottom:6px;'>Weitere hochgeladene Bilder:</div><div style='display:flex; flex-wrap:wrap;'>"
            for b64_extra in b64_list[1:]:
                preview_html += f"<img src='data:image/jpeg;base64,{b64_extra}' class='upload-thumb'/>"
            preview_html += "</div>"
            
        preview_html += "</div>"
        st.markdown(preview_html, unsafe_allow_html=True)
        
        auto_cat, auto_col, auto_tags = classify_and_generate_tags(uploaded_imgs[0])

    st.markdown("---")
    with st.form("form_add"):
        cat_idx = CATEGORIES.index(auto_cat) if auto_cat in CATEGORIES else 6
        sel_cat = st.selectbox("Kategorie", CATEGORIES, index=cat_idx)
        
        titel = st.text_input("Vorgeschlagener KI Titel", value=f"{sel_cat} ({auto_col})" if uploaded_imgs else "")
        tags = st.text_input("Vorgeschlagende KI Tags", value=auto_tags)
        ort = st.text_input("Findungsort", placeholder="z. B. Schulhof, Mensa")
        aufbewahrung = st.text_input("Ort der Aufbewahrung", placeholder="z. B. Sekretariat, Hausmeister")
        finder = st.text_input("Name vom Finder", placeholder="Dein Name / Klasse")

        if st.form_submit_button("Fundstück speichern 🚀"):
            if not uploaded_imgs:
                st.error("Bitte lade mindestens ein Bild hoch!")
            elif not titel or not ort:
                st.error("Titel und Findungsort bitte ausfüllen!")
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
                    "images_b64": images_to_base64_list(uploaded_imgs)
                }
                db.append(new_item)
                save_db(db)
                st.session_state.current_screen = "Suchen"
                st.rerun()

# -----------------------------------------------------------------------------
# 8. SCREEN 3: VERMISST
# -----------------------------------------------------------------------------
def screen_vermisst():
    render_header("VERMISST", show_back=True)
    st.session_state.items_db = load_db()

    st.markdown("### Foto deines verlorenen Gegenstands hochladen")
    f = st.file_uploader("Bild auswählen", type=["jpg", "png", "jpeg"], key="lost_file")
    
    if f:
        img = Image.open(f)
        b64 = images_to_base64_list([img])[0]
        st.markdown(f"<img src='data:image/jpeg;base64,{b64}' class='main-upload-preview' style='max-width:200px; height:180px;'/>", unsafe_allow_html=True)
        
        cat, col, tags = classify_and_generate_tags(img)

        st.markdown("### ÄHNLICHE BILDER AUS DER DATENBANK")
        matches = [i for i in st.session_state.items_db if cat.lower() in i.get('category','').lower()]

        if matches:
            cols = st.columns(3)
            for idx, m in enumerate(matches):
                with cols[idx % 3]:
                    imgs = m.get("images_b64", [])
                    if not imgs and m.get("image_b64"): imgs = [m["image_b64"]]
                    if imgs:
                        st.image(base64.b64decode(imgs[0]), use_container_width=True)
                    st.caption(m['title'])
        else:
            st.info("Keine ähnlichen Bilder in der Datenbank gefunden.")

    st.text_input("Titel hinzufügen")
    st.text_input("Tags hinzufügen")
    st.toggle("Bei Match benachrichtigen", value=True)

# -----------------------------------------------------------------------------
# 9. SCREEN 4: DETAILANSICHT
# -----------------------------------------------------------------------------
def screen_detail():
    render_header("FUNDSTÜCK", show_back=True)
    st.session_state.items_db = load_db()
    item = next((i for i in st.session_state.items_db if i['id'] == st.session_state.selected_item_id), None)

    if not item:
        st.error("Gegenstand nicht gefunden.")
        return

    st.markdown("### GALERIE")
    images = item.get("images_b64", [])
    if not images and item.get("image_b64"):
        images = [item["image_b64"]]

    if images:
        cols = st.columns(min(len(images), 3))
        for idx, b64_img in enumerate(images):
            with cols[idx % 3]:
                st.image(base64.b64decode(b64_img), use_container_width=True)

    st.markdown(f"## {item['title']}")
    st.markdown(f"**🏷️ Tags:** {item['tags']}")
    st.markdown(f"**📍 Findungsort:** {item['location']}")
    st.markdown(f"**📦 Ort der Aufbewahrung:** {item['storage_location']}")
    st.markdown(f"**👤 Name vom Finder:** {item['finder_name']}")

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Als abgeholt / zurückgegeben markieren"):
        st.session_state.items_db = [i for i in st.session_state.items_db if i['id'] != item['id']]
        save_db(st.session_state.items_db)
        st.session_state.current_screen = "Suchen"
        st.rerun()

# -----------------------------------------------------------------------------
# 10. SCREEN 5: EINSTELLUNGEN
# -----------------------------------------------------------------------------
def screen_einstellungen():
    render_header("EINSTELLUNGEN", show_back=True)

    settings_list = [
        ("🔔 Push-Benachrichtigungen & Match-Alerts", "Aktiviert"),
        ("👤 Mein Profil & Kontaktdaten", "Klasse 9b"),
        ("🏫 Schulstandort", "Katharineum zu Lübeck"),
        ("🔒 Datenschutz & Nutzungsbedingungen", "Eingesehen"),
        ("ℹ️ App-Version & Systeminfo", "v3.4.0 (Final Release)")
    ]

    for title, sub in settings_list:
        c_txt, c_btn = st.columns([4, 1])
        with c_txt:
            st.markdown(f"**{title}**  \n<small style='color:#757575;'>{sub}</small>", unsafe_allow_html=True)
        with c_btn:
            st.button("Anpassen", key=f"einst_btn_{title}")
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
