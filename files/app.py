"""
ระบบวิเคราะห์ข้อความจองทริปท่องเที่ยว (Trip Booking Text Analyzer)
เทคนิคที่ใช้: Regex & Cleansing, Tokenization, POS Tagging, Named Entity Recognition (NER)
สกัดข้อมูล: สถานที่เที่ยว, โรงแรม/ที่พัก, จำนวนวัน/คืน, งบประมาณ

พัฒนาด้วย Streamlit + PyThaiNLP
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
    """แปลง BIO tag (B-LOCATION, I-LOCATION, O, ...) ให้กลายเป็น dict {ประเภท: [คำที่รวมแล้ว]}"""
    entities = {}
    current_word, current_type = "", None

    def flush():
        if current_type and current_word.strip():
            entities.setdefault(current_type, []).append(current_word.strip())

    for word, tag in ner_tags:
        if tag == "O":
            flush()
            current_word, current_type = "", None
            continue
        prefix, ent_type = tag.split("-", 1)
        if prefix == "B" or ent_type != current_type:
            flush()
            current_word, current_type = word, ent_type
        else:  # prefix == "I" ต่อเนื่องจาก entity เดิม
            current_word += word
    flush()
    return entities


def classify_hotels_locations(entities: dict):
    locations = entities.get("LOCATION", [])
    orgs = entities.get("ORGANIZATION", [])

    hotels = [w for w in (locations + orgs) if any(k.lower() in w.lower() for k in HOTEL_KEYWORDS)]
    places = [w for w in locations if w not in hotels]
    return places, list(dict.fromkeys(hotels))  # ตัดค่าซ้ำ


# ------------------------------------------------------------------
# UI
# ------------------------------------------------------------------
st.title("🧳 ระบบวิเคราะห์ข้อความจองทริปท่องเที่ยว")
st.caption(
    "สกัด **สถานที่เที่ยว / โรงแรม / จำนวนวัน-คืน / งบประมาณ** จากข้อความรีวิวหรือแชทจองทริป "
    "ด้วยเทคนิค Regex, Tokenization, POS Tagging และ Named Entity Recognition (PyThaiNLP)"
)

EXAMPLES = [
    "อยากไปเที่ยวเชียงใหม่ 3 วัน 2 คืน พักที่โรงแรมดุสิตดีทู เชียงใหม่ งบประมาณรวมประมาณ 15,000 บาท ไปเดือนธันวาคมนี้",
    "แพลนทริปภูเก็ต 4 วัน 3 คืน จองรีสอร์ทคาทาธานี ภูเก็ต บีช รีสอร์ท งบต่อคนไม่เกิน 8000 บาท",
    "สนใจไปเที่ยวเกาะเสม็ด 2 วัน 1 คืน พักเกสต์เฮาส์ริมทะเล งบประมาณ 3,500 บาทต่อคน โทร 081-234-5678",
]

with st.sidebar:
    st.header("📋 ตัวอย่างข้อความ")
    st.caption("กดปุ่มเพื่อลองใช้งานตัวอย่าง")
    for i, ex in enumerate(EXAMPLES, start=1):
        if st.button(f"ตัวอย่างที่ {i}", use_container_width=True):
            st.session_state["trip_text"] = ex
    st.divider()
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
            entities = merge_bio_entities(ner_tags)
            places, hotels = classify_hotels_locations(entities)
            budgets = entities.get("MONEY", [])
            days = extract_days_nights(cleaned)

        st.success("วิเคราะห์เสร็จแล้ว ✅")

        # ---------------- สรุปผลลัพธ์หลัก ----------------
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("📍 สถานที่เที่ยว", len(places))
        c2.metric("🏨 โรงแรม/ที่พัก", len(hotels))
        c3.metric("💰 งบประมาณที่พบ", len(budgets))
        c4.metric("📅 จำนวนวัน/คืน", len(days))

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📍 สถานที่เที่ยว")
            st.write("\n".join(f"- {p}" for p in places) if places else "_ไม่พบ_")

            st.subheader("🏨 โรงแรม / ที่พัก")
            st.write("\n".join(f"- {h}" for h in hotels) if hotels else "_ไม่พบ_")

        with col2:
            st.subheader("💰 งบประมาณ")
            st.write("\n".join(f"- {b}" for b in budgets) if budgets else "_ไม่พบ_")

            st.subheader("📅 จำนวนวัน/คืน")
            if days:
                st.write("\n".join(f"- {num} {unit}" for num, unit in days))
            else:
                st.write("_ไม่พบ_")

        st.divider()

        tab1, tab2 = st.tabs(["🔎 POS Tagging", "🏷️ NER (BIO Tag)"])
        with tab1:
            st.caption("ผลลัพธ์การตัดคำและกำหนดชนิดคำ (Universal POS Tagset)")
            st.dataframe(
                pd.DataFrame(pos_tags, columns=["คำ", "ชนิดคำ (POS)"]),
                use_container_width=True,
                height=350,
            )
        with tab2:
            st.caption("ผลลัพธ์ Named Entity Recognition แบบ BIO Tagging (B-เริ่มต้น, I-ต่อเนื่อง, O-ไม่ใช่เอนทิตี)")
            st.dataframe(
                pd.DataFrame(ner_tags, columns=["คำ", "NER Tag"]),
                use_container_width=True,
                height=350,
            )
else:
    st.info("⬆️ พิมพ์หรือวางข้อความ แล้วกด 'วิเคราะห์ข้อความ' หรือลองกดตัวอย่างในแถบด้านซ้าย")
