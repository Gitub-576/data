
import streamlit as st
import pandas as pd
import json
import os

# ==========================================
# 1. 기본 설정
# ==========================================

st.set_page_config(
    page_title="뇌졸중 예측 실습실",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

DATA_URL = (
    "https://raw.githubusercontent.com/"
    "greatsong/modudata/main/data/stroke.csv"
)

SAVE_DIR = "내_기록"
SAVE_FILE = os.path.join(SAVE_DIR, "stroke_notes.json")

os.makedirs(SAVE_DIR, exist_ok=True)


# ==========================================
# 2. JSON 저장 및 불러오기
# ==========================================

def load_notes():
    """저장된 학습 기록 불러오기"""

    if os.path.exists(SAVE_FILE):
        try:
            with open(SAVE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass

    return {
        "meanings": {},
        "source": ""
    }


def save_notes(meanings, source):
    """학습 기록을 JSON 파일로 저장"""

    notes = {
        "meanings": meanings,
        "source": source
    }

    with open(SAVE_FILE, "w", encoding="utf-8") as f:
        json.dump(
            notes,
            f,
            ensure_ascii=False,
            indent=4
        )


# ==========================================
# 3. 데이터 불러오기
# ==========================================

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8")
    return df


try:
    df = load_data()

except Exception as e:
    st.error("데이터를 불러오지 못했습니다.")
    st.error(f"오류 내용: {e}")
    st.stop()


# ==========================================
# 4. 저장된 기록 불러오기
# ==========================================

if "notes" not in st.session_state:
    st.session_state.notes = load_notes()

if "meaning_editor" not in st.session_state:
    saved_meanings = st.session_state.notes.get(
        "meanings", {}
    )

    st.session_state.meaning_editor = pd.DataFrame({
        "열 이름": df.columns,
        "우리말 뜻": [
            saved_meanings.get(col, "")
            for col in df.columns
        ],
        "값의 종류": [
            (
                "숫자형"
                if pd.api.types.is_numeric_dtype(df[col])
                else "범주형"
            )
            for col in df.columns
        ],
        "빈 값 개수": [
            int(df[col].isna().sum())
            for col in df.columns
        ]
    })

if "source_text" not in st.session_state:
    st.session_state.source_text = (
        st.session_state.notes.get("source", "")
    )


# ==========================================
# 5. 화면 제목
# ==========================================

st.title("🧠 뇌졸중 예측 실습실")

st.markdown(
    """
    **데이터를 이해하고, 분석하고, 예측하는 공간입니다.**

    뇌졸중 관련 데이터를 살펴보며 데이터 분석과
    머신러닝의 기초를 학습해 봅시다.
    """
)

st.divider()


# ==========================================
# 6. 데이터 소개
# ==========================================

st.header("📊 1. 데이터 한눈에 보기")

st.markdown(
    """
    이 데이터는 여러 사람의 건강 상태와 생활 습관,
    그리고 뇌졸중 발생 여부를 기록한 데이터입니다.

    각 사람의 나이, 성별, 고혈압 여부, 심장병 여부,
    평균 혈당 수치, BMI, 흡연 상태 등의 정보를 포함합니다.

    마지막 열인 `stroke`는 뇌졸중 발생 여부를 나타냅니다.
    """
)

# 통계 계산
total_people = len(df)
total_columns = len(df.columns)
stroke_count = int((df["stroke"] == 1).sum())
stroke_ratio = (
    stroke_count / total_people * 100
    if total_people > 0 else 0
)

# 숫자 카드 4개
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="👥 전체 사람 수",
        value=f"{total_people:,}명"
    )

with col2:
    st.metric(
        label="📋 전체 열 개수",
        value=f"{total_columns}개"
    )

with col3:
    st.metric(
        label="🧠 뇌졸중 발생 인원",
        value=f"{stroke_count:,}명"
    )

with col4:
    st.metric(
        label="📈 뇌졸중 발생 비율",
        value=f"{stroke_ratio:.2f}%"
    )


st.divider()


# ==========================================
# 7. 열 이름과 데이터 설명 표
# ==========================================

st.header("📝 2. 데이터 열 살펴보기")

st.markdown(
    """
    아래 표를 살펴보고 각 열이 어떤 의미인지 생각해 봅시다.

    **우리말 뜻** 칸을 직접 클릭하여 교재를 참고해 입력하세요.
    입력한 내용은 자동으로 저장됩니다.
    """
)

edited_df = st.data_editor(
    st.session_state.meaning_editor,
    use_container_width=True,
    hide_index=True,
    key="meaning_table",
    disabled=[
        "열 이름",
        "값의 종류",
        "빈 값 개수"
    ],
    column_config={
        "열 이름": st.column_config.TextColumn(
            "열 이름",
            width="medium"
        ),
        "우리말 뜻": st.column_config.TextColumn(
            "우리말 뜻",
            width="large",
            help="교재를 참고하여 직접 입력하세요."
        ),
        "값의 종류": st.column_config.TextColumn(
            "값의 종류",
            width="medium"
        ),
        "빈 값 개수": st.column_config.NumberColumn(
            "빈 값 개수",
            width="small"
        )
    }
)

# 수정 내용 저장
meanings = {}

for _, row in edited_df.iterrows():
    meanings[row["열 이름"]] = (
        str(row["우리말 뜻"])
        if pd.notna(row["우리말 뜻"])
        else ""
    )

save_notes(
    meanings,
    st.session_state.source_text
)

st.session_state.meaning_editor = edited_df


# ==========================================
# 8. 데이터 처음 다섯 줄
# ==========================================

st.divider()

st.header("🔎 3. 데이터 미리보기")

st.markdown(
    """
    실제 데이터의 처음 다섯 줄을 확인해 봅시다.

    각 행은 한 사람의 정보를 나타내고,
    각 열은 해당 사람의 특성을 나타냅니다.
    """
)

st.dataframe(
    df.head(5),
    use_container_width=True,
    hide_index=True
)


# ==========================================
# 9. 데이터 출처 입력 및 저장
# ==========================================

st.divider()

st.header("📚 4. 데이터 출처")

st.markdown(
    """
    교재에 소개된 데이터 출처를 아래에 기록하세요.
    입력한 내용은 자동으로 JSON 파일에 저장됩니다.
    """
)

source_text = st.text_area(
    "데이터 출처",
    key="source_text",
    height=120,
    placeholder="교재를 참고하여 데이터 출처를 입력하세요."
)

# 출처 저장
save_notes(
    meanings,
    source_text
)

st.success("학습 기록이 저장되었습니다.")

st.caption(
    f"저장 위치: {SAVE_FILE}"
)

st.divider()

st.markdown(
    """
    <div style="text-align:center; color:gray;">
        🧠 뇌졸중 예측 실습실
        <br>
        데이터 분석과 머신러닝 학습을 위한 공간
    </div>
    """,
    unsafe_allow_html=True
)
