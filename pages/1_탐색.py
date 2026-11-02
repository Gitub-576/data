
import streamlit as st
import pandas as pd
import json
import os


# ==========================================
# 1. 페이지 설정
# ==========================================

st.set_page_config(
    page_title="뇌졸중 예측 실습실",
    page_icon="🧠",
    layout="wide"
)


# ==========================================
# 2. 기본 설정
# ==========================================

DATA_URL = (
    "https://raw.githubusercontent.com/"
    "greatsong/modudata/main/data/stroke.csv"
)

SAVE_FOLDER = "내_기록"
SAVE_FILE = os.path.join(SAVE_FOLDER, "기록.json")


# ==========================================
# 3. 데이터 불러오기
# ==========================================

@st.cache_data
def load_data():
    return pd.read_csv(
        DATA_URL,
        encoding="utf-8"
    )


try:
    df = load_data()

except Exception as e:
    st.error("데이터를 불러오는 중 오류가 발생했습니다.")
    st.error(str(e))
    st.stop()


# ==========================================
# 4. 기록 불러오기
# ==========================================

def load_records():

    if os.path.exists(SAVE_FILE):

        try:
            with open(
                SAVE_FILE,
                "r",
                encoding="utf-8"
            ) as file:

                return json.load(file)

        except (json.JSONDecodeError, OSError):
            return {
                "translations": {},
                "source": ""
            }

    return {
        "translations": {},
        "source": ""
    }


if "records" not in st.session_state:
    st.session_state.records = load_records()


# ==========================================
# 5. 기록 저장하기
# ==========================================

def save_records(translations, source):

    os.makedirs(
        SAVE_FOLDER,
        exist_ok=True
    )

    records = {
        "translations": translations,
        "source": source
    }

    with open(
        SAVE_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            records,
            file,
            ensure_ascii=False,
            indent=4
        )

    st.session_state.records = records


# ==========================================
# 6. 앱 제목
# ==========================================

st.title("🧠 뇌졸중 예측 실습실")

st.markdown(
    """
    뇌졸중 예측에 사용되는 데이터를 살펴보고,
    데이터의 구조와 각 변수의 의미를 학습하는 공간입니다.

    먼저 데이터를 탐색하고 각 열의 의미를 직접 정리해 봅시다.
    """
)

st.divider()


# ==========================================
# 7. 데이터 소개
# ==========================================

st.header("📊 데이터 소개")

st.markdown(
    """
    이 데이터는 사람들의 건강 상태와 생활 습관 정보를 담고 있으며,
    뇌졸중 발생 여부를 예측하는 데 활용할 수 있습니다.

    한 행은 한 사람의 정보를 나타내며,
    `stroke` 열은 뇌졸중 발생 여부를 나타냅니다.
    """
)


# ==========================================
# 8. 주요 통계 카드
# ==========================================

total_people = len(df)
total_columns = len(df.columns)

stroke_count = int(
    (df["stroke"] == 1).sum()
)

stroke_ratio = (
    stroke_count / total_people * 100
    if total_people > 0
    else 0
)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("👥 전체 사람 수", f"{total_people:,}명")

with col2:
    st.metric("📋 전체 열 개수", f"{total_columns}개")

with col3:
    st.metric("🧠 뇌졸중 발생 인원", f"{stroke_count:,}명")

with col4:
    st.metric("📈 뇌졸중 발생 비율", f"{stroke_ratio:.2f}%")


st.divider()


# ==========================================
# 9. 열 이름 및 변수 정보
# ==========================================

st.header("📝 데이터 열 살펴보기")

st.markdown(
    """
    각 열의 우리말 뜻을 교재를 참고하여 직접 작성해 보세요.

    작성한 내용은 저장 버튼을 누르면 JSON 파일에 저장됩니다.
    """
)

saved_translations = st.session_state.records.get(
    "translations", {}
)

column_info = []

for column in df.columns:

    unique_count = df[column].nunique(dropna=True)

    if pd.api.types.is_numeric_dtype(df[column]):

        if unique_count <= 10:
            value_type = "범주형"
        else:
            value_type = "수치형"

    else:
        value_type = "범주형"

    column_info.append({
        "열 이름": column,
        "우리말 뜻": saved_translations.get(column, ""),
        "값의 종류": value_type,
        "빈 값 개수": int(df[column].isna().sum())
    })


info_df = pd.DataFrame(column_info)

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
            "값의 종류",
            width="medium"
        ),
        "빈 값 개수": st.column_config.NumberColumn(
            "빈 값 개수",
            width="small"
        )
    },
    key="column_editor"
)


# ==========================================
# 10. 데이터 처음 다섯 줄
# ==========================================

st.divider()

st.header("🔍 데이터 미리보기")

st.markdown(
    "실제 CSV 파일에 저장된 데이터의 처음 다섯 행입니다."
)

st.dataframe(
    df.head(5),
    use_container_width=True,
    hide_index=True
)


# ==========================================
# 11. 데이터 출처 입력
# ==========================================

st.divider()

st.header("📚 데이터 출처")

st.markdown(
    "교재에 소개된 데이터 출처를 아래에 작성해 보세요."
)

saved_source = st.session_state.records.get(
    "source", ""
)

source_text = st.text_area(
    "데이터 출처 작성",
    value=saved_source,
    placeholder="교재를 참고하여 데이터 출처를 입력하세요.",
    height=120,
    key="source_editor"
)


# ==========================================
# 12. 저장 버튼
# ==========================================

st.divider()

col_save, col_info = st.columns([1, 3])

with col_save:

    if st.button(
        "💾 기록 저장하기",
        use_container_width=True,
        type="primary"
    ):

        translations = {}

        for _, row in edited_df.iterrows():

            translations[row["열 이름"]] = (
                row["우리말 뜻"]
                if pd.notna(row["우리말 뜻"])
                else ""
            )

        try:

            save_records(
                translations,
                source_text
            )

            st.success(
                "우리말 뜻과 데이터 출처가 저장되었습니다!"
            )

        except OSError as e:

            st.error(
                f"파일 저장 중 오류가 발생했습니다: {e}"
            )


with col_info:

    st.caption(
        "저장된 내용은 내_기록/기록.json 파일에 보관됩니다. "
        "앱을 다시 실행하거나 새로고침해도 저장된 내용을 불러옵니다."
    )


# ==========================================
# 13. 마무리
# ==========================================

st.divider()

st.markdown(
    """
    ### 🎯 학습 목표

    - 데이터의 전체적인 구조 이해하기
    - 각 열의 의미 파악하기
    - 결측치가 있는 열 확인하기
    - 뇌졸중 발생 여부(stroke) 이해하기

    **이제 데이터를 직접 살펴보며 뇌졸중 예측을 위한
    첫 번째 데이터 탐색을 시작해 봅시다!**
    """
)
