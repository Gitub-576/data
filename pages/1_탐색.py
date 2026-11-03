import pandas as pd
import plotly.express as px
import streamlit as st


# --------------------------------------------------
# 페이지 설정 및 데이터 불러오기
# --------------------------------------------------
DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/stroke.csv"

st.title("🔎 데이터 탐색")
st.caption("나이, 혈당, 질병 여부, 흡연 상태 등에 따라 데이터를 탐색합니다.")


@st.cache_data
def load_data():
    """지정된 주소에서 UTF-8 인코딩으로 데이터를 읽는다."""
    return pd.read_csv(DATA_URL, encoding="utf-8")


try:
    df = load_data()
except Exception as error:
    st.error("데이터를 불러오지 못했습니다. 인터넷 연결과 데이터 주소를 확인해 주세요.")
    st.exception(error)
    st.stop()


# 분석에 필요한 열이 있는지 확인
required_columns = [
    "age",
    "avg_glucose_level",
    "stroke",
    "hypertension",
    "heart_disease",
    "bmi",
    "smoking_status",
]
missing_columns = [column for column in required_columns if column not in df.columns]

if missing_columns:
    st.error("데이터에 필요한 열이 없습니다: " + ", ".join(missing_columns))
    st.stop()


# 뇌졸중 여부를 읽기 쉬운 이름으로 표시
df["stroke_group"] = df["stroke"].map({
    0: "뇌졸중 없음",
    1: "뇌졸중 있음",
})


# --------------------------------------------------
# 1. 나이와 평균 혈당 분포
# --------------------------------------------------
st.header("1. 나이와 평균 혈당 분포")

hist_col1, hist_col2 = st.columns(2)

with hist_col1:
    age_fig = px.histogram(
        df,
        x="age",
        nbins=30,
        title="나이 분포",
        labels={"age": "나이", "count": "사람 수"},
    )
    age_fig.update_layout(
        xaxis_title="나이",
        yaxis_title="사람 수",
        bargap=0.08,
    )
    st.plotly_chart(age_fig, use_container_width=True)

with hist_col2:
    glucose_fig = px.histogram(
        df,
        x="avg_glucose_level",
        nbins=30,
        title="평균 혈당 분포",
        labels={"avg_glucose_level": "평균 혈당", "count": "사람 수"},
    )
    glucose_fig.update_layout(
        xaxis_title="평균 혈당",
        yaxis_title="사람 수",
        bargap=0.08,
    )
    st.plotly_chart(glucose_fig, use_container_width=True)


# --------------------------------------------------
# 2. 뇌졸중 여부에 따른 나이와 평균 혈당 비교
# --------------------------------------------------
st.header("2. 뇌졸중 여부에 따른 비교")

box_col1, box_col2 = st.columns(2)

with box_col1:
    age_box = px.box(
        df.dropna(subset=["age", "stroke_group"]),
        x="stroke_group",
        y="age",
        color="stroke_group",
        title="뇌졸중 여부에 따른 나이",
        labels={
            "stroke_group": "뇌졸중 여부",
            "age": "나이",
        },
        category_orders={"stroke_group": ["뇌졸중 없음", "뇌졸중 있음"]},
    )
    age_box.update_layout(showlegend=False, xaxis_title="뇌졸중 여부", yaxis_title="나이")
    st.plotly_chart(age_box, use_container_width=True)

with box_col2:
    glucose_box = px.box(
        df.dropna(subset=["avg_glucose_level", "stroke_group"]),
        x="stroke_group",
        y="avg_glucose_level",
        color="stroke_group",
        title="뇌졸중 여부에 따른 평균 혈당",
        labels={
            "stroke_group": "뇌졸중 여부",
            "avg_glucose_level": "평균 혈당",
        },
        category_orders={"stroke_group": ["뇌졸중 없음", "뇌졸중 있음"]},
    )
    glucose_box.update_layout(
        showlegend=False,
        xaxis_title="뇌졸중 여부",
        yaxis_title="평균 혈당",
    )
    st.plotly_chart(glucose_box, use_container_width=True)

st.subheader("뇌졸중 여부 그룹별 평균값")
mean_table = (
    df.groupby("stroke", dropna=False)
    .agg(
        사람_수=("stroke", "size"),
        평균_나이=("age", "mean"),
        평균_혈당=("avg_glucose_level", "mean"),
    )
    .reset_index()
)
mean_table["뇌졸중 여부"] = mean_table["stroke"].map({
    0: "뇌졸중 없음",
    1: "뇌졸중 있음",
}).fillna("기타/결측")
mean_table = mean_table[["뇌졸중 여부", "사람_수", "평균_나이", "평균_혈당"]]
mean_table = mean_table.rename(columns={
    "사람_수": "사람 수",
    "평균_나이": "평균 나이",
    "평균_혈당": "평균 혈당",
})
mean_table["평균 나이"] = mean_table["평균 나이"].round(2)
mean_table["평균 혈당"] = mean_table["평균 혈당"].round(2)
st.dataframe(mean_table, hide_index=True, use_container_width=True)


# --------------------------------------------------
# 3. 고혈압 및 심장병 여부에 따른 뇌졸중 비율
# --------------------------------------------------
def make_stroke_rate_table(data, group_column, label_map):
    """그룹별 전체 인원과 뇌졸중 인원, 비율을 계산한다."""
    summary = (
        data.groupby(group_column, dropna=False)
        .agg(
            전체_인원=("stroke", "size"),
            뇌졸중_인원=("stroke", lambda values: (values == 1).sum()),
        )
        .reset_index()
    )
    summary["뇌졸중 비율(%)"] = (
        summary["뇌졸중_인원"] / summary["전체_인원"] * 100
    )
    summary["구분"] = summary[group_column].map(label_map)
    summary["구분"] = summary["구분"].fillna("결측")
    return summary


st.header("3. 질병 여부에 따른 뇌졸중 비율")

hypertension_table = make_stroke_rate_table(
    df,
    "hypertension",
    {0: "고혈압 없음", 1: "고혈압 있음"},
)
heart_table = make_stroke_rate_table(
    df,
    "heart_disease",
    {0: "심장병 없음", 1: "심장병 있음"},
)

rate_col1, rate_col2 = st.columns(2)

with rate_col1:
    hypertension_fig = px.bar(
        hypertension_table,
        x="구분",
        y="뇌졸중 비율(%)",
        text=hypertension_table["뇌졸중 비율(%)"].map(lambda value: f"{value:.2f}%"),
        title="고혈압 여부에 따른 뇌졸중 비율",
        labels={"구분": "고혈압 여부"},
    )
    hypertension_fig.update_traces(textposition="outside")
    hypertension_fig.update_layout(
        xaxis_title="고혈압 여부",
        yaxis_title="뇌졸중 비율(%)",
        yaxis_rangemode="tozero",
    )
    st.plotly_chart(hypertension_fig, use_container_width=True)
    st.dataframe(
        hypertension_table[["구분", "전체_인원", "뇌졸중_인원", "뇌졸중 비율(%)"]]
        .rename(columns={
            "전체_인원": "전체 인원",
            "뇌졸중_인원": "뇌졸중 인원",
        }).round(2),
        hide_index=True,
        use_container_width=True,
    )

with rate_col2:
    heart_fig = px.bar(
        heart_table,
        x="구분",
        y="뇌졸중 비율(%)",
        text=heart_table["뇌졸중 비율(%)"].map(lambda value: f"{value:.2f}%"),
        title="심장병 여부에 따른 뇌졸중 비율",
        labels={"구분": "심장병 여부"},
    )
    heart_fig.update_traces(textposition="outside")
    heart_fig.update_layout(
        xaxis_title="심장병 여부",
        yaxis_title="뇌졸중 비율(%)",
        yaxis_rangemode="tozero",
    )
    st.plotly_chart(heart_fig, use_container_width=True)
    st.dataframe(
        heart_table[["구분", "전체_인원", "뇌졸중_인원", "뇌졸중 비율(%)"]]
        .rename(columns={
            "전체_인원": "전체 인원",
            "뇌졸중_인원": "뇌졸중 인원",
        }).round(2),
        hide_index=True,
        use_container_width=True,
    )


# --------------------------------------------------
# 4. BMI 결측자와 전체 데이터의 뇌졸중 비율 비교
# --------------------------------------------------
st.header("4. BMI 빈 값이 있는 사람들의 뇌졸중 비율")

missing_bmi = df[df["bmi"].isna()]
overall_stroke_rate = (df["stroke"] == 1).mean() * 100 if len(df) else 0
missing_bmi_stroke_rate = (
    (missing_bmi["stroke"] == 1).mean() * 100
    if len(missing_bmi)
    else 0
)

bmi_comparison = pd.DataFrame({
    "구분": ["BMI 빈 값이 있는 사람", "전체 사람"],
    "사람 수": [len(missing_bmi), len(df)],
    "뇌졸중 인원": [
        int((missing_bmi["stroke"] == 1).sum()),
        int((df["stroke"] == 1).sum()),
    ],
    "뇌졸중 비율(%)": [
        round(missing_bmi_stroke_rate, 2),
        round(overall_stroke_rate, 2),
    ],
})
st.dataframe(bmi_comparison, hide_index=True, use_container_width=True)

if len(missing_bmi) == 0:
    st.info("BMI 값이 비어 있는 사람이 없습니다.")


# --------------------------------------------------
# 5. 흡연 상태별 사람 수
# --------------------------------------------------
st.header("5. 흡연 상태별 사람 수")

smoking_counts = (
    df["smoking_status"]
    .fillna("결측")
    .value_counts()
    .rename_axis("흡연 상태")
    .reset_index(name="사람 수")
)
st.dataframe(smoking_counts, hide_index=True, use_container_width=True)

st.caption("참고: 비율은 각 그룹의 전체 인원 중 stroke 값이 1인 사람의 비율로 계산했습니다.")
