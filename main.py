
import json
from pathlib import Path

import pandas as pd
import streamlit as st


# --------------------------------------------------
# 1. 기본 설정
# --------------------------------------------------

st.set_page_config(
    page_title="뇌졸중 예측 실습실",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 뇌졸중 예측 실습실")
st.caption("데이터를 살펴보고 뇌졸중 예측을 위한 기초를 학습하는 공간입니다.")

DATA_URL = (
    "https://raw.githubusercontent.com/"
    "greatsong/modudata/main/data/stroke.csv"
)

RECORD_DIR = Path("내_기록")
RECORD_FILE = RECORD_DIR / "설명_기록.json"

COLUMNS = [
    "id",
    "gender",
    "age",
    "hypertension",
    "heart_disease",
    "ever_married",
    "work_type",
    "Residence_type",
    "avg_glucose_level",
    "bmi",
    "smoking_status",
    "stroke"
]


# --------------------------------------------------
# 2. 기록 파일 불러오기 및 저장하기
# --------------------------------------------------

def load_records():
    """기존 JSON 기록을 불러옵니다."""
    if RECORD_FILE.exists():
        try:
            with open(RECORD_FILE, "r", encoding="utf-8") as file:
                return json.load(file)
        except (json.JSONDecodeError, OSError):
            return {}

    return {}


def save_records(meanings, source):
    """우리말 뜻과 데이터 출처를 JSON으로 저장합니다."""
    RECORD_DIR.mkdir(parents=True, exist_ok=True)

    records = {
        "meanings": meanings,
        "source": source
    }

    with open(RECORD_FILE, "w", encoding="utf-8") as file:
        json.dump(
            records,
            file,
            ensure_ascii=False,
            indent=4
        )


# --------------------------------------------------
# 3. 데이터 불러오기
# --------------------------------------------------

@st.cache_data
def load_data():
    """CSV 파일을 다운로드하고 읽어옵니다."""
    df = pd.read_csv(
        DATA_URL,
        encoding="utf-8"
    )

    return df


try:
    df = load_data()

except Exception as e:
    st.error("데이터를 불러오는 중 문제가 발생했습니다.")
    st.error(f"오류 내용: {e}")
    st.info("인터넷 연결을 확인한 후 다시 실행해 주세요.")
    st.stop()


# --------------------------------------------------
# 4. 기존 기록 불러오기
# --------------------------------------------------

records = load_records()

saved_meanings = records.get("meanings", {})
saved_source = records.get("source", "")


# --------------------------------------------------
# 5. 데이터 소개
# --------------------------------------------------

st.header("📊 데이터 소개")

st.markdown(
    """
    이 데이터는 사람들의 건강 및 생활 습관 정보와
    뇌졸중 발생 여부를 담고 있습니다.

    각 행은 한 사람의 정보를 나타내며,
    `stroke` 열을 통해 뇌졸중 발생 여부를 확인할 수 있습니다.
    """
)

# 주요 통계 계산
total_people = len(df)
total_columns = len(df.columns)
stroke_count = int((df["stroke"] == 1).sum())
stroke_ratio = (
    stroke_count / total_people * 100
    if total_people > 0
    else 0
)


# --------------------------------------------------
# 6. 숫자 카드 4개
# --------------------------------------------------

st.subheader("전체 데이터 요약")

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


# --------------------------------------------------
# 7. 열 정보 표
# --------------------------------------------------

st.header("📝 열 이름과 우리말 뜻")

st.write(
    "각 열의 우리말 뜻을 교재에서 찾아 직접 입력해 보세요. "
    "입력한 내용은 자동으로 JSON 파일에 저장됩니다."
)

# 열별 정보 구성
column_info = []

for column in df.columns:

    # 값의 종류
    unique_count = df[column].nunique(dropna=True)

    if pd.api.types.is_numeric_dtype(df[column]):
        value_type = f"수치형 ({unique_count}종류)"
    else:
        value_type = f"범주형 ({unique_count}종류)"

    # 빈 값 개수
    missing_count = int(df[column].isna().sum())

    column_info.append({
        "열 이름": column,
        "우리말 뜻": saved_meanings.get(column, ""),
        "값의 종류": value_type,
        "빈 값 개수": missing_count
    })

info_df = pd.DataFrame(column_info)

# 우리말 뜻만 편집 가능
edited_df = st.data_editor(
    info_df,
    use_container_width=True,
    hide_index=True,
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
            "우리말 뜻 (직접 입력)",
            width="large"
        ),
        "값의 종류": st.column_config.TextColumn(
            "값의 종류"
        ),
        "빈 값 개수": st.column_config.NumberColumn(
            "빈 값 개수",
            format="%d"
        )
    },
    key="meaning_editor"
)

# 수정된 우리말 뜻 저장
current_meanings = {}

for _, row in edited_df.iterrows():
    current_meanings[row["열 이름"]] = row["우리말 뜻"]

try:
    save_records(
        current_meanings,
        saved_source
    )
    st.caption("💾 우리말 뜻이 자동 저장되었습니다.")

except OSError as e:
    st.error(f"기록 저장에 실패했습니다: {e}")


st.divider()


# --------------------------------------------------
# 8. 데이터 처음 다섯 줄
# --------------------------------------------------

st.header("🔎 데이터 미리 보기")

st.write("실제 CSV 데이터의 처음 다섯 줄입니다.")

st.dataframe(
    df.head(5),
    use_container_width=True,
    hide_index=True
)


st.divider()


# --------------------------------------------------
# 9. 데이터 출처 입력
# --------------------------------------------------

st.header("📚 데이터 출처")

st.write(
    "교재에 안내된 데이터 출처를 아래에 입력하세요. "
    "입력한 내용은 저장 버튼을 누르면 보관됩니다."
)

source_text = st.text_area(
    "데이터 출처",
    value=saved_source,
    placeholder="교재에 있는 데이터 출처를 입력하세요.",
    height=120,
    key="source_editor"
)

if st.button(
    "💾 데이터 출처 저장",
    type="primary"
):
    try:
        save_records(
            current_meanings,
            source_text
        )
        st.success("데이터 출처가 저장되었습니다!")

    except OSError as e:
        st.error(f"기록 저장에 실패했습니다: {e}")


st.caption(
    "저장 위치: 내_기록/설명_기록.json"
)

st.divider()

st.caption(
    "※ 이 실습실은 데이터 탐색과 머신러닝 학습을 위한 교육용 앱입니다. "
    "실제 의료 진단이나 뇌졸중 위험 판정에 사용하지 마세요."
)
