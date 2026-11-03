
import streamlit as st
import pandas as pd
import plotly.express as px


# --------------------------------------------------
# 1. 페이지 설정
# --------------------------------------------------

st.set_page_config(
    page_title="탐색 | 뇌졸중 예측 실습실",
    page_icon="🔎",
    layout="wide"
)

st.title("🔎 데이터 탐색")
st.caption("뇌졸중 데이터의 분포와 변수 사이의 관계를 살펴봅니다.")


# --------------------------------------------------
# 2. 데이터 불러오기
# --------------------------------------------------

DATA_URL = (
    "https://raw.githubusercontent.com/"
    "greatsong/modudata/main/data/stroke.csv"
)


@st.cache_data
def load_data():
    df = pd.read_csv(
        DATA_URL,
        encoding="utf-8"
    )

    # bmi의 빈 값은 결측치로 처리
    df["bmi"] = pd.to_numeric(
        df["bmi"],
        errors="coerce"
    )

    return df


try:
    df = load_data()

except Exception as e:
    st.error("데이터를 불러오지 못했습니다.")
    st.error(f"오류 내용: {e}")
    st.stop()


# --------------------------------------------------
# 3. 데이터 기본 확인
# --------------------------------------------------

total_people = len(df)

stroke_count = int(
    (df["stroke"] == 1).sum()
)

stroke_ratio = (
    stroke_count / total_people * 100
    if total_people > 0
    else 0
)

st.subheader("📊 데이터 기본 정보")

col1, col2, col3 = st.columns(3)

col1.metric("전체 사람 수", f"{total_people:,}명")
col2.metric("뇌졸중 발생 인원", f"{stroke_count:,}명")
col3.metric("전체 뇌졸중 비율", f"{stroke_ratio:.2f}%")


st.divider()


# --------------------------------------------------
# 4. 나이와 평균 혈당 분포
# --------------------------------------------------

st.header("1. 나이와 평균 혈당의 분포")

st.write(
    "히스토그램을 통해 사람들의 나이와 평균 혈당이 "
    "어떤 범위에 많이 분포하는지 확인합니다."
)

fig_age = px.histogram(
    df,
    x="age",
    nbins=30,
    title="나이 분포",
    labels={
        "age": "나이",
        "count": "사람 수"
    },
    color_discrete_sequence=["#4C78A8"]
)

fig_age.update_layout(
    xaxis_title="나이",
    yaxis_title="사람 수",
    bargap=0.05
)

fig_glucose = px.histogram(
    df,
    x="avg_glucose_level",
    nbins=30,
    title="평균 혈당 분포",
    labels={
        "avg_glucose_level": "평균 혈당",
        "count": "사람 수"
    },
    color_discrete_sequence=["#F58518"]
)

fig_glucose.update_layout(
    xaxis_title="평균 혈당",
    yaxis_title="사람 수",
    bargap=0.05
)

col1, col2 = st.columns(2)

with col1:
    st.plotly_chart(
        fig_age,
        use_container_width=True
    )

with col2:
    st.plotly_chart(
        fig_glucose,
        use_container_width=True
    )


st.divider()


# --------------------------------------------------
# 5. 뇌졸중 발생 여부에 따른 상자그림
# --------------------------------------------------

st.header("2. 뇌졸중 발생 여부에 따른 비교")

st.write(
    "뇌졸중을 겪은 사람과 겪지 않은 사람의 "
    "나이 및 평균 혈당 분포를 비교합니다."
)

# 그룹 이름 생성
df_box = df.copy()

df_box["뇌졸중 여부"] = df_box["stroke"].map({
    0: "뇌졸중 없음",
    1: "뇌졸중 있음"
})

# 나이 상자그림
fig_age_box = px.box(
    df_box,
    x="뇌졸중 여부",
    y="age",
    color="뇌졸중 여부",
    title="뇌졸중 여부에 따른 나이 비교",
    labels={
        "age": "나이",
        "뇌졸중 여부": "뇌졸중 여부"
    },
    color_discrete_map={
        "뇌졸중 없음": "#4C78A8",
        "뇌졸중 있음": "#E45756"
    }
)

fig_age_box.update_layout(
    showlegend=False
)

# 평균 혈당 상자그림
fig_glucose_box = px.box(
    df_box,
    x="뇌졸중 여부",
    y="avg_glucose_level",
    color="뇌졸중 여부",
    title="뇌졸중 여부에 따른 평균 혈당 비교",
    labels={
        "avg_glucose_level": "평균 혈당",
        "뇌졸중 여부": "뇌졸중 여부"
    },
    color_discrete_map={
        "뇌졸중 없음": "#4C78A8",
        "뇌졸중 있음": "#E45756"
    }
)

fig_glucose_box.update_layout(
    showlegend=False
)

col1, col2 = st.columns(2)

with col1:
    st.plotly_chart(
        fig_age_box,
        use_container_width=True
    )

with col2:
    st.plotly_chart(
        fig_glucose_box,
        use_container_width=True
    )


# 그룹별 평균값 계산
mean_table = (
    df_box
    .groupby("뇌졸중 여부")
    .agg(
        사람 수=("stroke", "count"),
        평균 나이=("age", "mean"),
        평균 혈당=("avg_glucose_level", "mean")
    )
    .reindex(["뇌졸중 없음", "뇌졸중 있음"])
    .reset_index()
)

mean_table["평균 나이"] = mean_table["평균 나이"].round(2)
mean_table["평균 혈당"] = mean_table["평균 혈당"].round(2)

st.subheader("그룹별 평균값 비교")

st.dataframe(
    mean_table,
    use_container_width=True,
    hide_index=True
)


st.divider()


# --------------------------------------------------
# 6. 고혈압과 뇌졸중 비율
# --------------------------------------------------

st.header("3. 고혈압 여부에 따른 뇌졸중 비율")

st.write(
    "고혈압이 있는 사람과 없는 사람의 "
    "뇌졸중 발생 비율을 비교합니다."
)


def make_rate_table(data, column, labels):
    """특정 변수의 값에 따른 뇌졸중 발생 비율 계산"""

    result = (
        data.groupby(column, dropna=False)
        .agg(
            전체 인원=("stroke", "size"),
            뇌졸중 인원=("stroke", "sum")
        )
        .reset_index()
    )

    result["뇌졸중 비율(%)"] = (
        result["뇌졸중 인원"]
        / result["전체 인원"]
        * 100
    ).round(2)

    result[column] = result[column].map(labels)

    return result


hypertension_labels = {
    0: "고혈압 없음",
    1: "고혈압 있음"
}

hypertension_table = make_rate_table(
    df,
    "hypertension",
    hypertension_labels
)

fig_hypertension = px.bar(
    hypertension_table,
    x="hypertension",
    y="뇌졸중 비율(%)",
    text="뇌졸중 비율(%)",
    color="hypertension",
    title="고혈압 여부에 따른 뇌졸중 비율",
    labels={
        "hypertension": "고혈압 여부",
        "뇌졸중 비율(%)": "뇌졸중 비율 (%)"
    },
    color_discrete_sequence=["#4C78A8", "#E45756"]
)

fig_hypertension.update_traces(
    texttemplate="%{text:.2f}%",
    textposition="outside"
)

fig_hypertension.update_layout(
    showlegend=False,
    yaxis_title="뇌졸중 비율 (%)",
    xaxis_title="고혈압 여부"
)

st.plotly_chart(
    fig_hypertension,
    use_container_width=True
)


st.subheader("고혈압 여부별 통계")

st.dataframe(
    hypertension_table,
    use_container_width=True,
    hide_index=True
)


st.divider()


# --------------------------------------------------
# 7. 심장병과 뇌졸중 비율
# --------------------------------------------------

st.header("4. 심장병 여부에 따른 뇌졸중 비율")

st.write(
    "심장병이 있는 사람과 없는 사람의 "
    "뇌졸중 발생 비율을 비교합니다."
)

heart_labels = {
    0: "심장병 없음",
    1: "심장병 있음"
}

heart_table = make_rate_table(
    df,
    "heart_disease",
    heart_labels
)

fig_heart = px.bar(
    heart_table,
    x="heart_disease",
    y="뇌졸중 비율(%)",
    text="뇌졸중 비율(%)",
    color="heart_disease",
    title="심장병 여부에 따른 뇌졸중 비율",
    labels={
        "heart_disease": "심장병 여부",
        "뇌졸중 비율(%)": "뇌졸중 비율 (%)"
    },
    color_discrete_sequence=["#72B7B2", "#E45756"]
)

fig_heart.update_traces(
    texttemplate="%{text:.2f}%",
    textposition="outside"
)

fig_heart.update_layout(
    showlegend=False,
    yaxis_title="뇌졸중 비율 (%)",
    xaxis_title="심장병 여부"
)

st.plotly_chart(
    fig_heart,
    use_container_width=True
)


st.subheader("심장병 여부별 통계")

st.dataframe(
    heart_table,
    use_container_width=True,
    hide_index=True
)


st.divider()


# --------------------------------------------------
# 8. BMI 결측치와 뇌졸중 비율
# --------------------------------------------------

st.header("5. BMI가 비어 있는 사람들의 뇌졸중 비율")

st.write(
    "체질량지수(BMI)가 기록되지 않은 사람들을 따로 살펴보고, "
    "전체 뇌졸중 비율과 비교합니다."
)

# BMI 결측치만 추출
missing_bmi_df = df[df["bmi"].isna()]

missing_bmi_count = len(missing_bmi_df)

missing_bmi_stroke_count = int(
    (missing_bmi_df["stroke"] == 1).sum()
)

missing_bmi_ratio = (
    missing_bmi_stroke_count / missing_bmi_count * 100
    if missing_bmi_count > 0
    else 0
)

# 비교 표 작성
bmi_comparison = pd.DataFrame({
    "구분": [
        "BMI가 비어 있는 사람",
        "전체 사람"
    ],
    "전체 인원": [
        missing_bmi_count,
        total_people
    ],
    "뇌졸중 인원": [
        missing_bmi_stroke_count,
        stroke_count
    ],
    "뇌졸중 비율(%)": [
        round(missing_bmi_ratio, 2),
        round(stroke_ratio, 2)
    ]
})

st.dataframe(
    bmi_comparison,
    use_container_width=True,
    hide_index=True
)

st.info(
    f"BMI 결측 인원은 전체 {total_people:,}명 중 "
    f"{missing_bmi_count:,}명입니다."
)


st.divider()


# --------------------------------------------------
# 9. 흡연 상태별 사람 수
# --------------------------------------------------

st.header("6. 흡연 상태별 사람 수")

st.write(
    "흡연 상태에 따라 사람 수를 세어 봅니다. "
    "결측값이 있다면 별도 항목으로 표시합니다."
)

smoking_table = (
    df["smoking_status"]
    .fillna("결측값")
    .value_counts()
    .rename_axis("흡연 상태")
    .reset_index(name="사람 수")
)

st.dataframe(
    smoking_table,
    use_container_width=True,
    hide_index=True
)


st.divider()


# --------------------------------------------------
# 10. 마무리
# --------------------------------------------------

st.success(
    "데이터 탐색이 끝났습니다! "
    "각 변수와 뇌졸중 발생 사이의 차이를 살펴보세요."
)

st.caption(
    "※ 그래프와 통계는 데이터의 특성을 탐색하기 위한 교육용 자료이며, "
    "인과관계나 개인의 의료적 위험을 의미하지 않습니다."
)
