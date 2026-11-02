
import streamlit as st
import pandas as pd
import plotly.express as px


# ==========================================
# 1. 페이지 설정
# ==========================================

st.set_page_config(
    page_title="탐색 | 뇌졸중 예측 실습실",
    page_icon="🔎",
    layout="wide"
)


# ==========================================
# 2. 데이터 불러오기
# ==========================================

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

    return df


try:

    df = load_data()

except Exception as e:

    st.error("데이터를 불러오는 중 오류가 발생했습니다.")
    st.error(str(e))
    st.stop()


# ==========================================
# 3. 제목
# ==========================================

st.title("🔎 데이터 탐색")

st.markdown(
    """
    데이터를 다양한 그래프와 표로 살펴보면서
    사람들의 건강 상태와 뇌졸중 발생 사이의 관계를 탐색합니다.
    """
)

st.divider()


# ==========================================
# 4. 나이와 평균 혈당 분포
# ==========================================

st.header("1. 나이와 평균 혈당의 분포")

st.markdown(
    """
    히스토그램을 이용하여 사람들의 나이와 평균 혈당이
    어떤 범위에 많이 분포하는지 살펴봅니다.
    """
)

col1, col2 = st.columns(2)

with col1:

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

    st.plotly_chart(
        fig_age,
        use_container_width=True
    )


with col2:

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

    st.plotly_chart(
        fig_glucose,
        use_container_width=True
    )


st.divider()


# ==========================================
# 5. 뇌졸중 여부에 따른 나이와 평균 혈당 비교
# ==========================================

st.header("2. 뇌졸중 여부에 따른 비교")

st.markdown(
    """
    뇌졸중을 겪은 사람과 겪지 않은 사람의
    나이와 평균 혈당 분포를 상자그림으로 비교합니다.

    상자 안의 가로선은 중앙값을 나타내며,
    표에는 두 집단의 평균값을 함께 표시합니다.
    """
)


# 그룹 이름 만들기
df["stroke_group"] = df["stroke"].map({
    0: "뇌졸중 없음",
    1: "뇌졸중 있음"
})


col1, col2 = st.columns(2)

with col1:

    fig_box_age = px.box(
        df,
        x="stroke_group",
        y="age",
        color="stroke_group",
        title="뇌졸중 여부에 따른 나이 비교",
        labels={
            "stroke_group": "뇌졸중 여부",
            "age": "나이"
        },
        category_orders={
            "stroke_group": [
                "뇌졸중 없음",
                "뇌졸중 있음"
            ]
        },
        color_discrete_map={
            "뇌졸중 없음": "#4C78A8",
            "뇌졸중 있음": "#E45756"
        }
    )

    fig_box_age.update_layout(
        showlegend=False
    )

    st.plotly_chart(
        fig_box_age,
        use_container_width=True
    )


with col2:

    fig_box_glucose = px.box(
        df,
        x="stroke_group",
        y="avg_glucose_level",
        color="stroke_group",
        title="뇌졸중 여부에 따른 평균 혈당 비교",
        labels={
            "stroke_group": "뇌졸중 여부",
            "avg_glucose_level": "평균 혈당"
        },
        category_orders={
            "stroke_group": [
                "뇌졸중 없음",
                "뇌졸중 있음"
            ]
        },
        color_discrete_map={
            "뇌졸중 없음": "#4C78A8",
            "뇌졸중 있음": "#E45756"
        }
    )

    fig_box_glucose.update_layout(
        showlegend=False
    )

    st.plotly_chart(
        fig_box_glucose,
        use_container_width=True
    )


# 평균값 계산
mean_df = (
    df.groupby("stroke_group")[
        ["age", "avg_glucose_level"]
    ]
    .mean()
    .reindex([
        "뇌졸중 없음",
        "뇌졸중 있음"
    ])
    .reset_index()
)

mean_df.columns = [
    "뇌졸중 여부",
    "평균 나이",
    "평균 혈당"
]

mean_df["평균 나이"] = mean_df["평균 나이"].round(2)
mean_df["평균 혈당"] = mean_df["평균 혈당"].round(2)

st.subheader("두 집단의 평균값 비교")

st.dataframe(
    mean_df,
    use_container_width=True,
    hide_index=True
)


st.divider()


# ==========================================
# 6. 고혈압과 심장병에 따른 뇌졸중 비율
# ==========================================

st.header("3. 질병 여부에 따른 뇌졸중 비율")

st.markdown(
    """
    고혈압과 심장병이 있는 집단과 없는 집단을 나누어
    각각의 뇌졸중 발생 비율을 비교합니다.
    """
)


# 비율 계산 함수
def calculate_stroke_rate(data, column):

    result = (
        data.groupby(column)
        .agg(
            전체_사람수=("stroke", "count"),
            뇌졸중_인원=("stroke", "sum")
        )
        .reset_index()
    )

    result["뇌졸중_비율"] = (
        result["뇌졸중_인원"]
        / result["전체_사람수"]
        * 100
    )

    return result


# 고혈압
hypertension_df = calculate_stroke_rate(
    df,
    "hypertension"
)

hypertension_df["구분"] = hypertension_df[
    "hypertension"
].map({
    0: "고혈압 없음",
    1: "고혈압 있음"
})


# 심장병
heart_df = calculate_stroke_rate(
    df,
    "heart_disease"
)

heart_df["구분"] = heart_df[
    "heart_disease"
].map({
    0: "심장병 없음",
    1: "심장병 있음"
})


col1, col2 = st.columns(2)

with col1:

    fig_hypertension = px.bar(
        hypertension_df,
        x="구분",
        y="뇌졸중_비율",
        text="뇌졸중_비율",
        title="고혈압 여부에 따른 뇌졸중 비율",
        labels={
            "구분": "고혈압 여부",
            "뇌졸중_비율": "뇌졸중 비율 (%)"
        },
        color="구분",
        color_discrete_map={
            "고혈압 없음": "#4C78A8",
            "고혈압 있음": "#E45756"
        }
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


with col2:

    fig_heart = px.bar(
        heart_df,
        x="구분",
        y="뇌졸중_비율",
        text="뇌졸중_비율",
        title="심장병 여부에 따른 뇌졸중 비율",
        labels={
            "구분": "심장병 여부",
            "뇌졸중_비율": "뇌졸중 비율 (%)"
        },
        color="구분",
        color_discrete_map={
            "심장병 없음": "#4C78A8",
            "심장병 있음": "#E45756"
        }
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


# 비율 표
st.subheader("질병 여부별 상세 수치")

rate_table = pd.DataFrame({
    "구분": [
        "고혈압 없음",
        "고혈압 있음",
        "심장병 없음",
        "심장병 있음"
    ],
    "전체 사람 수": [
        hypertension_df.iloc[0]["전체_사람수"],
        hypertension_df.iloc[1]["전체_사람수"],
        heart_df.iloc[0]["전체_사람수"],
        heart_df.iloc[1]["전체_사람수"]
    ],
    "뇌졸중 인원": [
        hypertension_df.iloc[0]["뇌졸중_인원"],
        hypertension_df.iloc[1]["뇌졸중_인원"],
        heart_df.iloc[0]["뇌졸중_인원"],
        heart_df.iloc[1]["뇌졸중_인원"]
    ],
    "뇌졸중 비율 (%)": [
        hypertension_df.iloc[0]["뇌졸중_비율"],
        hypertension_df.iloc[1]["뇌졸중_비율"],
        heart_df.iloc[0]["뇌졸중_비율"],
        heart_df.iloc[1]["뇌졸중_비율"]
    ]
})

rate_table["뇌졸중 비율 (%)"] = (
    rate_table["뇌졸중 비율 (%)"].round(2)
)

st.dataframe(
    rate_table,
    use_container_width=True,
    hide_index=True
)


st.divider()


# ==========================================
# 7. BMI 결측치 분석
# ==========================================

st.header("4. BMI가 비어 있는 사람들의 뇌졸중 비율")

st.markdown(
    """
    체질량지수(bmi)가 비어 있는 사람들만 따로 살펴봅니다.

    BMI가 비어 있는 사람들의 뇌졸중 비율과
    전체 사람들의 뇌졸중 비율을 비교합니다.
    """
)


# BMI 결측치 데이터
missing_bmi_df = df[df["bmi"].isna()]


# BMI 결측치 집단
missing_bmi_count = len(missing_bmi_df)

missing_bmi_stroke_count = int(
    (missing_bmi_df["stroke"] == 1).sum()
)

missing_bmi_rate = (
    missing_bmi_stroke_count / missing_bmi_count * 100
    if missing_bmi_count > 0
    else 0
)


# 전체 데이터
total_count = len(df)

total_stroke_count = int(
    (df["stroke"] == 1).sum()
)

total_stroke_rate = (
    total_stroke_count / total_count * 100
    if total_count > 0
    else 0
)


bmi_comparison = pd.DataFrame({

    "비교 대상": [
        "BMI가 비어 있는 사람",
        "전체 사람"
    ],

    "전체 사람 수": [
        missing_bmi_count,
        total_count
    ],

    "뇌졸중 인원": [
        missing_bmi_stroke_count,
        total_stroke_count
    ],

    "뇌졸중 비율 (%)": [
        round(missing_bmi_rate, 2),
        round(total_stroke_rate, 2)
    ]

})


st.dataframe(
    bmi_comparison,
    use_container_width=True,
    hide_index=True
)


st.info(
    "BMI 결측 집단의 비율이 전체 비율과 다르더라도, "
    "그 차이만으로 BMI 결측이 뇌졸중의 원인이라고 판단할 수는 없습니다."
)


st.divider()


# ==========================================
# 8. 흡연 상태별 사람 수
# ==========================================

st.header("5. 흡연 상태별 사람 수")

st.markdown(
    """
    흡연 상태(smoking_status)에 어떤 값들이 있는지 확인하고
    각 상태에 해당하는 사람 수를 살펴봅니다.
    """
)


smoking_count = (
    df["smoking_status"]
    .fillna("결측")
    .value_counts()
    .rename_axis("흡연 상태")
    .reset_index(name="사람 수")
)


st.dataframe(
    smoking_count,
    use_container_width=True,
    hide_index=True
)


# 흡연 상태별 그래프
fig_smoking = px.bar(
    smoking_count,
    x="흡연 상태",
    y="사람 수",
    text="사람 수",
    title="흡연 상태별 사람 수",
    color="흡연 상태",
    color_discrete_sequence=px.colors.qualitative.Set2
)

fig_smoking.update_traces(
    textposition="outside"
)

fig_smoking.update_layout(
    showlegend=False,
    xaxis_title="흡연 상태",
    yaxis_title="사람 수"
)

st.plotly_chart(
    fig_smoking,
    use_container_width=True
)


# ==========================================
# 9. 마무리
# ==========================================

st.divider()

st.header("🎯 탐색 결과 정리")

st.markdown(
    """
    이번 탐색에서는 다음 내용을 살펴보았습니다.

    - 나이와 평균 혈당의 분포
    - 뇌졸중 여부에 따른 나이와 혈당의 차이
    - 고혈압 및 심장병 여부에 따른 뇌졸중 비율
    - BMI 결측치가 있는 사람들의 뇌졸중 비율
    - 흡연 상태별 사람 수

    **주의:** 그래프에서 확인한 차이는 데이터의 통계적 관계를 보여주는 것이며,
    특정 건강 상태가 뇌졸중을 직접 일으킨다는 인과관계를 증명하지는 않습니다.
    """
)
