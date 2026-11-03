import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


# --------------------------------------------------
# 기본 설정
# --------------------------------------------------
APP_TITLE = "뇌졸중 예측 실습실"
DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/stroke.csv"
RECORD_DIR = Path("내_기록")
RECORD_FILE = RECORD_DIR / "stroke_column_notes.json"

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🧠",
    layout="wide",
)


# --------------------------------------------------
# 기록 파일 불러오기 / 저장하기
# --------------------------------------------------
def load_notes():
    """저장된 우리말 뜻과 데이터 출처를 불러온다."""
    if RECORD_FILE.exists():
        try:
            with RECORD_FILE.open("r", encoding="utf-8") as file:
                saved = json.load(file)
            return saved
        except (json.JSONDecodeError, OSError):
            st.warning("기록 파일을 읽지 못해 빈 기록으로 시작합니다.")
    return {"meanings": {}, "source": ""}


def save_notes(meanings, source):
    """우리말 뜻과 데이터 출처를 JSON 파일에 저장한다."""
    RECORD_DIR.mkdir(parents=True, exist_ok=True)
    saved = {
        "meanings": meanings,
        "source": source,
    }
    with RECORD_FILE.open("w", encoding="utf-8") as file:
        json.dump(saved, file, ensure_ascii=False, indent=2)


# --------------------------------------------------
# 데이터 불러오기
# --------------------------------------------------
@st.cache_data
def load_data():
    data = pd.read_csv(DATA_URL, encoding="utf-8")
    return data


st.title("🧠 뇌졸중 예측 실습실")
st.caption("데이터를 살펴보고 각 열의 의미를 직접 정리해 보는 실습 공간입니다.")

try:
    df = load_data()
except Exception as error:
    st.error("데이터를 불러오지 못했습니다. 인터넷 연결과 데이터 주소를 확인해 주세요.")
    st.exception(error)
    st.stop()

notes = load_notes()

# 데이터 요약
total_people = len(df)
column_count = len(df.columns)
stroke_count = int((df["stroke"] == 1).sum()) if "stroke" in df.columns else 0
stroke_ratio = (stroke_count / total_people * 100) if total_people else 0.0

st.subheader("데이터 한눈에 보기")
metric_cols = st.columns(4)
metric_cols[0].metric("전체 사람 수", f"{total_people:,}명")
metric_cols[1].metric("열 개수", f"{column_count:,}개")
metric_cols[2].metric("뇌졸중(stroke=1)인 사람 수", f"{stroke_count:,}명")
metric_cols[3].metric("뇌졸중 비율", f"{stroke_ratio:.2f}%")

st.divider()

# 열 정보 표
st.subheader("열 이름과 데이터 정보")
st.write("각 열의 우리말 뜻을 교재에서 찾아 입력해 보세요. 입력한 내용은 자동으로 저장됩니다.")

def describe_values(series):
    if pd.api.types.is_numeric_dtype(series):
        return "숫자형"
    return "문자형"

column_info = pd.DataFrame({
    "열 이름": df.columns,
    "우리말 뜻": [
        notes.get("meanings", {}).get(column, "")
        for column in df.columns
    ],
    "값의 종류": [describe_values(df[column]) for column in df.columns],
    "빈 값 개수": [int(df[column].isna().sum()) for column in df.columns],
})

edited_info = st.data_editor(
    column_info,
    hide_index=True,
    use_container_width=True,
    column_config={
        "열 이름": st.column_config.TextColumn("열 이름", disabled=True),
        "우리말 뜻": st.column_config.TextColumn(
            "우리말 뜻",
            help="교재를 참고해 각 열의 뜻을 입력하세요.",
        ),
        "값의 종류": st.column_config.TextColumn("값의 종류", disabled=True),
        "빈 값 개수": st.column_config.NumberColumn("빈 값 개수", disabled=True),
    },
    disabled=["열 이름", "값의 종류", "빈 값 개수"],
    key="column_info_editor",
)

# 원본 열 이름에 대응해 입력 내용을 저장
meanings = {
    row["열 이름"]: row["우리말 뜻"]
    for _, row in edited_info.iterrows()
}

st.divider()

# 처음 다섯 줄
st.subheader("데이터 처음 다섯 줄")
st.dataframe(df.head(5), use_container_width=True)

st.divider()

# 출처 기록
st.subheader("데이터 출처")
source_text = st.text_area(
    "교재에 제시된 데이터 출처를 입력하세요.",
    value=notes.get("source", ""),
    placeholder="교재에 적힌 데이터 출처를 여기에 입력하세요.",
    height=100,
    key="source_editor",
)

if st.button("기록 저장하기", type="primary"):
    save_notes(meanings, source_text)
    st.success(f"기록을 저장했습니다. 저장 위치: {RECORD_FILE}")

st.caption("입력한 우리말 뜻과 데이터 출처는 '내_기록/stroke_column_notes.json' 파일에 저장됩니다.")
