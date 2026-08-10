"""
ระบบวิเคราะห์ข้อความจองทริปท่องเที่ยว (Trip Booking Text Analyzer)
เทคนิคที่ใช้: Regex & Cleansing, Tokenization, POS Tagging, Named Entity Recognition (NER)
สกัดข้อมูล: สถานที่เที่ยว, โรงแรม/ที่พัก, จำนวนวัน/คืน, งบประมาณ

ดีไซน์: ธีม "ตั๋วเดินทาง / Boarding Pass" — พัฒนาด้วย Streamlit + PyThaiNLP
"""

import re
import pandas as pd
import streamlit as st
from pythainlp.tokenize import word_tokenize
from pythainlp.tag import pos_tag, NER

# ------------------------------------------------------------------
# ตั้งค่าหน้าเว็บ
# ------------------------------------------------------------------
st.set_page_config(
    page_title="วิเคราะห์ข้อความจองทริปท่องเที่ยว",
    page_icon="🧳",
    layout="wide",
)

HOTEL_KEYWORDS = ["โรงแรม", "รีสอร์ท", "resort", "hotel", "โฮสเทล", "hostel", "เกสต์เฮาส์", "guesthouse", "ที่พัก"]

# สีสำหรับ badge ใน POS / NER table (ใช้กับ pandas Styler)
NER_COLOR_MAP = {
    "LOCATION": "#2FA89B3D",
    "ORGANIZATION": "#F3663F3D",
    "MONEY": "#E7B54B55",
    "DATE": "#7C9B934D",
    "TIME": "#7C9B934D",
    "PERSON": "#8E7CC34D",
}
POS_COLOR_MAP = {
    "NOUN": "#2FA89B3D",
    "PROPN": "#2FA89B3D",
    "VERB": "#F3663F3D",
    "NUM": "#E7B54B55",
    "ADJ": "#8E7CC34D",
}


# ------------------------------------------------------------------
# ธีม / CSS — คอนเซ็ปต์ "ตั๋วเดินทาง (Boarding Pass)"
# ------------------------------------------------------------------
def inject_theme():
    st.markdown(
        """
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link href="https://fonts.googleapis.com/css2?family=Chonburi&family=IBM+Plex+Sans+Thai:wght@400;500;600;700&family=IBM+Plex+Mono:wght@500;600&display=swap" rel="stylesheet">
        <style>

        :root{
            --bg-deep:#0E3B36;
            --bg-deep-2:#123F3A;
            --paper:#FBF3E1;
            --paper-2:#F5EAD2;
            --ink:#12312B;
            --gold:#E7B54B;
            --coral:#F3663F;
            --teal:#2FA89B;
            --muted:#7C9B93;
        }

        html, body, [class*="css"] { font-family: 'IBM Plex Sans Thai', sans-serif; }

        /* พื้นหลังหลัก: ท้องฟ้ายามเย็นโทนเขียวมรกต */
        [data-testid="stAppViewContainer"]{
            background:
                radial-gradient(1100px 500px at 15% -10%, #175048 0%, transparent 60%),
                radial-gradient(900px 500px at 100% 0%, #0F423C 0%, transparent 55%),
                var(--bg-deep);
        }
        [data-testid="stHeader"]{ background: transparent; }

        /* Sidebar: แผงเข้มเข้าชุดกัน มีเส้นทองคั่น */
        [data-testid="stSidebar"]{
            background: var(--bg-deep-2);
            border-right: 1px dashed rgba(231,181,75,0.35);
        }
        [data-testid="stSidebar"] * { color: #EFE7D4 !important; }
        [data-testid="stSidebar"] .stButton>button{
            background: rgba(251,243,225,0.06);
            border: 1px solid rgba(231,181,75,0.4);
            color: #F6EFDC !important;
            border-radius: 999px;
            font-size: 0.85rem;
            transition: background .15s ease, transform .15s ease;
        }
        [data-testid="stSidebar"] .stButton>button:hover{
            background: rgba(231,181,75,0.18);
            transform: translateY(-1px);
        }

        /* หัวเรื่อง: eyebrow แบบรหัสเที่ยวบิน + ชื่อระบบตัวใหญ่ */
        .ticket-eyebrow{
            font-family:'IBM Plex Mono', monospace;
            letter-spacing:.18em;
            font-size:0.75rem;
            color: var(--gold);
            text-transform: uppercase;
            display:flex; align-items:center; gap:.6rem;
            margin-bottom:.35rem;
        }
        .ticket-eyebrow .dot{ width:6px; height:6px; border-radius:50%; background: var(--coral); display:inline-block; }
        .hero-title{
            font-family:'Chonburi', serif;
            color:#FBF3E1;
            font-size: clamp(1.9rem, 3.6vw, 3rem);
            line-height:1.15;
            margin: 0 0 .5rem 0;
        }
        .hero-sub{
            color:#CFE3DE;
            font-size:0.98rem;
            max-width: 780px;
            line-height:1.6;
            margin-bottom: 0;
        }

        /* เส้นปรุแบบตั๋ว */
        .perforation{
            border: none;
            border-top: 2px dashed rgba(231,181,75,0.55);
            margin: 1.6rem 0 1.4rem 0;
        }

        /* กล่องข้อความ (พื้นที่วางข้อความ) ให้ดูเหมือนกระดาษตั๋ว */
        [data-testid="stTextArea"] textarea{
            background: var(--paper) !important;
            color: var(--ink) !important;
            border-radius: 14px !important;
            border: 1px solid rgba(18,49,43,0.15) !important;
            font-size: 1rem !important;
            padding: 1rem !important;
            box-shadow: 0 10px 26px rgba(0,0,0,0.18);
        }
        [data-testid="stTextArea"] textarea:focus{
            outline: none !important;
            box-shadow: 0 0 0 3px rgba(231,181,75,0.55), 0 10px 26px rgba(0,0,0,0.18) !important;
        }
        [data-testid="stTextArea"] label{ color:#EFE7D4 !important; font-weight:500; }

        /* ปุ่มหลัก: coral pill ทรงตั๋ว */
        .stButton>button[kind="primary"]{
            background: linear-gradient(135deg, var(--coral), #E24E2C);
            border: none;
            border-radius: 999px;
            padding: 0.6rem 1.6rem;
            font-weight: 600;
            box-shadow: 0 8px 18px rgba(243,102,63,0.35);
            transition: transform .15s ease, box-shadow .15s ease;
        }
        .stButton>button[kind="primary"]:hover{
            transform: translateY(-2px);
            box-shadow: 0 12px 22px rgba(243,102,63,0.45);
        }

        /* การ์ดสรุปแบบ "ตั๋วฉีก" มีรอยบากวงกลมสองข้าง */
        .stub-row{ display:flex; gap:1rem; flex-wrap:wrap; margin: 1.4rem 0; }
        .stub{
            position:relative;
            flex:1 1 190px;
            background: var(--paper);
            border-radius: 16px;
            padding: 1.1rem 1.2rem;
            box-shadow: 0 10px 24px rgba(0,0,0,0.22);
            overflow:visible;
        }
        .stub::before, .stub::after{
            content:"";
            position:absolute;
            width:22px; height:22px;
            background: var(--bg-deep-2);
            border-radius:50%;
            top:50%; transform: translateY(-50%);
        }
        .stub::before{ left:-11px; }
        .stub::after{ right:-11px; }
        .stub .stub-icon{ font-size:1.3rem; }
        .stub .stub-label{
            font-family:'IBM Plex Mono', monospace;
            font-size:0.72rem; letter-spacing:.08em;
            color: var(--muted); text-transform:uppercase; margin-top:.35rem;
        }
        .stub .stub-value{
            font-size:1.9rem; font-weight:700; color: var(--ink); line-height:1.1; margin-top:.15rem;
        }

        /* การ์ดผลลัพธ์หลัก (ครึ่งซ้าย-ขวาของตั๋ว) */
        .board-panel{
            background: var(--paper);
            border-radius: 18px;
            padding: 1.4rem 1.5rem;
            box-shadow: 0 12px 28px rgba(0,0,0,0.22);
            height: 100%;
        }
        .board-panel h4{
            font-family:'IBM Plex Mono', monospace;
            font-size:0.78rem; letter-spacing:.1em; text-transform:uppercase;
            color: var(--coral); margin:0 0 .8rem 0; display:flex; align-items:center; gap:.5rem;
        }
        .chip-list{ display:flex; flex-direction:column; gap:.5rem; }
        .chip{
            background: var(--paper-2);
            border-left: 3px solid var(--teal);
            border-radius: 8px;
            padding: .55rem .8rem;
            color: var(--ink);
            font-size:0.95rem;
        }
        .chip.gold{ border-left-color: var(--gold); }
        .chip-empty{ color: var(--muted); font-style: italic; font-size:0.9rem; }

        /* แท็บ POS / NER ให้ดูเป็นแท็บ "ประตูขึ้นเครื่อง" */
        [data-testid="stTabs"] button{
            font-family:'IBM Plex Mono', monospace;
            color:#EFE7D4 !important;
            font-size:0.85rem;
        }
        [data-testid="stTabs"] [aria-selected="true"]{
            color: var(--gold) !important;
            border-bottom-color: var(--gold) !important;
        }

        [data-testid="stDataFrame"]{
            border-radius: 12px; overflow:hidden;
            border: 1px solid rgba(231,181,75,0.25);
        }

        .empty-ticket{
            border: 1.5px dashed rgba(231,181,75,0.5);
            border-radius: 18px;
            padding: 2.2rem 1.5rem;
            text-align:center;
            color:#CFE3DE;
        }
        .empty-ticket .big{ font-size:2rem; margin-bottom:.4rem; }

        @media (prefers-reduced-motion: reduce){
            * { transition:none !important; animation:none !important; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def stat_stub(icon: str, label: str, value) -> str:
    return f"""
    <div class="stub">
        <div class="stub-icon">{icon}</div>
        <div class="stub-label">{label}</div>
        <div class="stub-value">{value}</div>
    </div>
    """


def chip_list_html(items, gold: bool = False) -> str:
    if not items:
        return '<div class="chip-empty">ไม่พบข้อมูลในข้อความนี้</div>'
    cls = "chip gold" if gold else "chip"
    return '<div class="chip-list">' + "".join(f'<div class="{cls}">{it}</div>' for it in items) + "</div>"


# ------------------------------------------------------------------
# โหลดโมเดล NER แบบ cache (โหลดครั้งเดียว ไม่ต้องโหลดใหม่ทุกครั้งที่กดปุ่ม)
# ------------------------------------------------------------------
@st.cache_resource(show_spinner="กำลังโหลดโมเดล NER ครั้งแรก (รอสักครู่)...")
def load_ner_engine():
    return NER(engine="thainer")


# ------------------------------------------------------------------
# 1) Regex & Cleansing
# ------------------------------------------------------------------
def clean_text(text: str) -> str:
    """ลบลิงก์และเบอร์โทรศัพท์ที่ไม่เกี่ยวกับการวิเคราะห์ แล้วลบช่องว่างส่วนเกิน"""
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)          # ลบลิงก์
    text = re.sub(r"0\d{1,2}-?\d{3}-?\d{4}", " ", text)          # ลบเบอร์โทร
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_days_nights(text: str):
    """ใช้ Regex ดึงจำนวนวัน/คืน เช่น '3 วัน 2 คืน' หรือ '5 days 4 nights'"""
    pattern = r"(\d+)\s*(วัน|คืน|day|days|night|nights)"
    return re.findall(pattern, text, flags=re.IGNORECASE)


# ------------------------------------------------------------------
# 2) Tokenization + 3) POS Tagging
# ------------------------------------------------------------------
def run_pos_pipeline(text: str):
    tokens = word_tokenize(text, engine="newmm")
    tokens = [t for t in tokens if t.strip() != ""]
    tags = pos_tag(tokens, corpus="pud")  # Universal POS tagset อ่านง่าย เช่น NOUN, PROPN, NUM
    return tags


# ------------------------------------------------------------------
# 4) Named Entity Recognition (NER)
# ------------------------------------------------------------------
def run_ner_pipeline(text: str, ner_engine: NER):
    return ner_engine.tag(text)


def merge_bio_entities(ner_tags):
    """แปลง BIO tag (B-LOCATION, I-LOCATION, O, ...) ให้กลายเป็น list ของ entity
    โดยเก็บตำแหน่ง (index) ของคำไว้ด้วย เพื่อใช้เช็คคำแวดล้อม (context) ภายหลัง
    เช่น คำว่า "โรงแรม" มักถูกตัดคำแยกออกจากชื่อโรงแรม (เป็นคนละ entity หรือไม่ถูกแท็กเลย)
    จึงต้องดูคำที่อยู่ *ก่อนหน้า* entity ประกอบด้วย ไม่ใช่ดูแค่ข้อความในตัว entity เอง
    """
    words = [w for w, _ in ner_tags]
    entities = []
    current_word, current_type, start_idx = "", None, None

    def flush(end_idx):
        if current_type and current_word.strip():
            entities.append(
                {"type": current_type, "text": current_word.strip(), "start": start_idx, "end": end_idx}
            )

    for i, (word, tag) in enumerate(ner_tags):
        if tag == "O":
            flush(i)
            current_word, current_type, start_idx = "", None, None
            continue
        prefix, ent_type = tag.split("-", 1)
        if prefix == "B" or ent_type != current_type:
            flush(i)
            current_word, current_type, start_idx = word, ent_type, i
        else:  # prefix == "I" ต่อเนื่องจาก entity เดิม
            current_word += word
    flush(len(ner_tags))
    return entities, words


def classify_hotels_locations(entities: list, words: list, context_window: int = 3):
    """แยก entity ประเภท LOCATION/ORGANIZATION ว่าเป็น 'โรงแรม/ที่พัก' หรือ 'สถานที่เที่ยว'
    โดยเช็คทั้งในชื่อ entity เอง และคำ 2-3 คำก่อนหน้า entity (เผื่อคำว่า "โรงแรม"/"รีสอร์ท"
    ถูกตัดแยกออกไปเป็นคนละคำ เช่น "โรงแรม" + "ดุสิตดีทู")
    """
    hotels, places = [], []
    for ent in entities:
        if ent["type"] not in ("LOCATION", "ORGANIZATION"):
            continue
        context_before = "".join(words[max(0, ent["start"] - context_window): ent["start"]]).lower()
        text_lower = ent["text"].lower()
        is_hotel = any(k.lower() in text_lower or k.lower() in context_before for k in HOTEL_KEYWORDS)
        (hotels if is_hotel else places).append(ent["text"])

    # ตัดค่าซ้ำ (รักษาลำดับเดิม)
    hotels = list(dict.fromkeys(hotels))
    places = list(dict.fromkeys(places))
    return places, hotels


def style_by_keyword(df: pd.DataFrame, column: str, color_map: dict):
    def colorize(val):
        for key, color in color_map.items():
            if key in str(val):
                return f"background-color:{color};"
        return ""
    return df.style.map(colorize, subset=[column])


# ------------------------------------------------------------------
# UI
# ------------------------------------------------------------------
inject_theme()

st.markdown(
    """
    <div class="ticket-eyebrow"><span class="dot"></span>TRIP · TEXT · ANALYSIS · NLP-TH01</div>
    <div class="hero-title">🧳 ระบบวิเคราะห์ข้อความจองทริปท่องเที่ยว</div>
    <div class="hero-sub">
        สกัด <b>สถานที่เที่ยว / โรงแรม / จำนวนวัน-คืน / งบประมาณ</b> จากข้อความรีวิวหรือแชทจองทริป
        ด้วยเทคนิค Regex, Tokenization, POS Tagging และ Named Entity Recognition (PyThaiNLP)
    </div>
    <hr class="perforation" />
    """,
    unsafe_allow_html=True,
)

EXAMPLES = [
    "อยากไปเที่ยวเชียงใหม่ 3 วัน 2 คืน พักที่โรงแรมดุสิตดีทู เชียงใหม่ งบประมาณรวมประมาณ 15,000 บาท ไปเดือนธันวาคมนี้",
    "แพลนทริปภูเก็ต 4 วัน 3 คืน จองรีสอร์ทคาทาธานี ภูเก็ต บีช รีสอร์ท งบต่อคนไม่เกิน 8000 บาท",
    "สนใจไปเที่ยวเกาะเสม็ด 2 วัน 1 คืน พักเกสต์เฮาส์ริมทะเล งบประมาณ 3,500 บาทต่อคน โทร 081-234-5678",
]

with st.sidebar:
    st.markdown("### 🎫 ตัวอย่างข้อความ")
    st.caption("กดปุ่มเพื่อลองใช้งานตัวอย่าง")
    for i, ex in enumerate(EXAMPLES, start=1):
        if st.button(f"ตัวอย่างที่ {i}", use_container_width=True):
            st.session_state["trip_text"] = ex
    st.markdown('<hr style="border-top:1px dashed rgba(231,181,75,0.35);">', unsafe_allow_html=True)
    st.markdown(
        "**เทคนิค NLP ที่ใช้**\n"
        "- Regex & Cleansing\n"
        "- Tokenization (newmm)\n"
        "- POS Tagging (Universal POS)\n"
        "- Named Entity Recognition (thainer)"
    )

text_input = st.text_area(
    "วางข้อความจองทริป / รีวิวทริปท่องเที่ยว (ภาษาไทย)",
    height=150,
    key="trip_text",
    placeholder="เช่น อยากไปเที่ยวเชียงใหม่ 3 วัน 2 คืน พักโรงแรมดุสิตดีทู งบประมาณ 15000 บาท",
)

analyze = st.button("🔍 วิเคราะห์ข้อความ", type="primary")

if analyze:
    if not text_input or not text_input.strip():
        st.warning("กรุณาใส่ข้อความก่อนกดวิเคราะห์")
    else:
        with st.spinner("กำลังประมวลผล..."):
            cleaned = clean_text(text_input)
            pos_tags = run_pos_pipeline(cleaned)
            ner_engine = load_ner_engine()
            ner_tags = run_ner_pipeline(cleaned, ner_engine)
            entities, ner_words = merge_bio_entities(ner_tags)
            places, hotels = classify_hotels_locations(entities, ner_words)
            budgets = list(dict.fromkeys(e["text"] for e in entities if e["type"] == "MONEY"))
            days = extract_days_nights(cleaned)

        st.markdown(
            '<div style="color:#B7F2C2; font-weight:600; margin-bottom:.5rem;">✅ วิเคราะห์เสร็จแล้ว</div>',
            unsafe_allow_html=True,
        )

        # ---------------- สรุปผลลัพธ์หลัก (ตั๋วฉีก 4 ใบ) ----------------
        st.markdown(
            '<div class="stub-row">'
            + stat_stub("📍", "สถานที่เที่ยว", len(places))
            + stat_stub("🏨", "โรงแรม/ที่พัก", len(hotels))
            + stat_stub("💰", "งบประมาณที่พบ", len(budgets))
            + stat_stub("📅", "จำนวนวัน/คืน", len(days))
            + "</div>",
            unsafe_allow_html=True,
        )

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                '<div class="board-panel">'
                '<h4>📍 สถานที่เที่ยว &amp; 🏨 โรงแรม/ที่พัก</h4>'
                + chip_list_html(places)
                + '<div style="height:.6rem;"></div>'
                + chip_list_html(hotels, gold=True)
                + "</div>",
                unsafe_allow_html=True,
            )
        with col2:
            days_display = [f"{num} {unit}" for num, unit in days]
            st.markdown(
                '<div class="board-panel">'
                '<h4>💰 งบประมาณ &amp; 📅 จำนวนวัน/คืน</h4>'
                + chip_list_html(budgets, gold=True)
                + '<div style="height:.6rem;"></div>'
                + chip_list_html(days_display)
                + "</div>",
                unsafe_allow_html=True,
            )

        st.markdown('<hr class="perforation" />', unsafe_allow_html=True)

        tab1, tab2 = st.tabs(["🔎 POS TAGGING", "🏷️ NER (BIO TAG)"])
        with tab1:
            st.caption("ผลลัพธ์การตัดคำและกำหนดชนิดคำ (Universal POS Tagset)")
            pos_df = pd.DataFrame(pos_tags, columns=["คำ", "ชนิดคำ (POS)"])
            st.dataframe(
                style_by_keyword(pos_df, "ชนิดคำ (POS)", POS_COLOR_MAP),
                use_container_width=True,
                height=350,
            )
        with tab2:
            st.caption("ผลลัพธ์ Named Entity Recognition แบบ BIO Tagging (B-เริ่มต้น, I-ต่อเนื่อง, O-ไม่ใช่เอนทิตี)")
            ner_df = pd.DataFrame(ner_tags, columns=["คำ", "NER Tag"])
            st.dataframe(
                style_by_keyword(ner_df, "NER Tag", NER_COLOR_MAP),
                use_container_width=True,
                height=350,
            )
else:
    st.markdown(
        """
        <div class="empty-ticket">
            <div class="big">🎫</div>
            <b>ยังไม่มีตั๋วให้ตรวจ</b><br/>
            พิมพ์หรือวางข้อความด้านบน แล้วกด "วิเคราะห์ข้อความ"<br/>
            หรือลองกดตัวอย่างในแถบด้านซ้ายดูก่อนก็ได้
        </div>
        """,
        unsafe_allow_html=True,
    )
