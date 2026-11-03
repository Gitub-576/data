
import streamlit as st
import pandas as pd
import plotly.express as px

# ==========================================
# 1. 페이지 기본 설정
# ==========================================

st.set_page_config(
    page_title="탐색 | 뇌졸중 예측 실습실",
    page_icon="🔎",
    layout="wide"
)

DATA_URL = (
    "https://raw.githubusercontent.com/"
    "greatsong/modudata/main/data/stroke.csv"
)


# ==========================================
# 2. 데이터 불러오기
# ==========================================

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
    st.error("데이터를 불러오지 못했습니다.")
    st.error(f"오류 내용: {e}")
    st.stop()


# ==========================================
# 3. 제목
# ==========================================

st.title("🔎 데이터 탐색")

st.markdown(
    """
    뇌졸중 데이터를 다양한 그래프와 표로 살펴봅시다.

    나이, 혈당, 고혈압, 심장병, BMI, 흡연 상태에 따라
    데이터에 어떤 특징이 나타나는지 확인해 봅니다.
    """
)

st.divider()


# ==========================================
# 4. 나이와 평균 혈당 분포
# ==========================================

st.header("📊 1. 나이와 평균 혈당의 분포")

st.markdown(
    """
    히스토그램은 데이터가 어떤 구간에 많이 분포하는지
    확인할 수 있는 그래프입니다.
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
        bargap=0.1
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
        bargap=0.1
    )

    st.plotly_chart(
        fig_glucose,
        use_container_width=True
    )


st.divider()


# ==========================================
# 5. 뇌졸중 여부에 따른 나이와 혈당 비교
# ==========================================

st.header("📦 2. 뇌졸중 여부에 따른 분포 비교")

st.markdown(
    """
    뇌졸중을 겪은 사람과 겪지 않은 사람의
    나이와 평균 혈당 분포를 상자그림으로 비교합니다.

    상자그림을 통해 중앙값, 데이터의 퍼짐 정도,
    이상치 등을 살펴볼 수 있습니다.
    """
)

df["stroke_label"] = df["stroke"].map({
    0: "뇌졸중 없음",
    1: "뇌졸중 있음"
})

col1, col2 = st.columns(2)

with col1:

    fig_age_box = px.box(
        df,
        x="stroke_label",
        y="age",
        color="stroke_label",
        title="뇌졸중 여부에 따른 나이 비교",
        labels={
            "stroke_label": "뇌졸중 여부",
            "age": "나이"
        },
        color_discrete_map={
            "뇌졸중 없음": "#4C78A8",
            "뇌졸중 있음": "#E45756"
        }
    )

    fig_age_box.update_layout(
        showlegend=False
    )

    st.plotly_chart(
        fig_age_box,
        use_container_width=True
    )


with col2:

    fig_glucose_box = px.box(
        df,
        x="stroke_label",
        y="avg_glucose_level",
        color="stroke_label",
        title="뇌졸중 여부에 따른 평균 혈당 비교",
        labels={
            "stroke_label": "뇌졸중 여부",
            "avg_glucose_level": "평균 혈당"
        },
        color_discrete_map={
            "뇌졸중 없음": "#4C78A8",
            "뇌졸중 있음": "#E45756"
        }
    )

    fig_glucose_box.update_layout(
        showlegend=False
    )

    st.plotly_chart(
        fig_glucose_box,
        use_container_width=True
    )


# 평균값 계산
mean_comparison = (
    df.groupby("stroke_label")[
        ["age", "avg_glucose_level"]
    ]
    .mean()
    .reset_index()
)

mean_comparison = mean_comparison.rename(
    columns={
        "stroke_label": "뇌졸중 여부",
        "age": "평균 나이",
        "avg_glucose_level": "평균 혈당"
    }
)

mean_comparison["평균 나이"] = (
    mean_comparison["평균 나이"].round(2)
)

mean_comparison["평균 혈당"] = (
    mean_comparison["평균 혈당"].round(2)
)

st.subheader("그룹별 평균값 비교")

st.dataframe(
    mean_comparison,
    use_container_width=True,
    hide_index=True
)


st.divider()


# ==========================================
# 6. 고혈압 여부에 따른 뇌졸중 비율
# ==========================================

st.header("❤️ 3. 고혈압 여부에 따른 뇌졸중 비율")

st.markdown(
    """
    고혈압이 있는 사람과 없는 사람을 구분하여
    각 그룹에서 뇌졸중이 발생한 비율을 비교합니다.
    """
)

hypertension_data = (
    df.groupby("hypertension")
    .agg(
        전체_인원=("stroke", "count"),
        뇌졸중_인원=("stroke", "sum")
    )
    .reset_index()
)

hypertension_data["뇌졸중_비율"] = (
    hypertension_data["뇌졸중_인원"]
    / hypertension_data["전체_인원"] * 100
)

hypertension_data["고혈압 여부"] = (
    hypertension_data["hypertension"].map({
        0: "고혈압 없음",
        1: "고혈압 있음"
    })
)

fig_hypertension = px.bar(
    hypertension_data,
    x="고혈압 여부",
    y="뇌졸중_비율",
    text="뇌졸중_비율",
    title="고혈압 여부에 따른 뇌졸중 발생 비율",
    labels={
        "고혈압 여부": "고혈압 여부",
        "뇌졸중_비율": "뇌졸중 비율(%)"
    },
    color="고혈압 여부",
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
    yaxis_range=[
        0,
        max(hypertension_data["뇌졸중_비율"].max() * 1.2, 1)
    ]
)

st.plotly_chart(
    fig_hypertension,
    use_container_width=True
)


st.divider()


# ==========================================
# 7. 심장병 여부에 따른 뇌졸중 비율
# ==========================================

st.header("💓 4. 심장병 여부에 따른 뇌졸중 비율")

st.markdown(
    """
    심장병이 있는 사람과 없는 사람을 구분하여
    각 그룹의 뇌졸중 발생 비율을 비교합니다.
    """
)

heart_data = (
    df.groupby("heart_disease")
    .agg(
        전체_인원=("stroke", "count"),
        뇌졸중_인원=("stroke", "sum")
    )
    .reset_index()
)

heart_data["뇌졸중_비율"] = (
    heart_data["뇌졸중_인원"]
    / heart_data["전체_인원"] * 100
)

heart_data["심장병 여부"] = (
    heart_data["heart_disease"].map({
        0: "심장병 없음",
        1: "심장병 있음"
    })
)

fig_heart = px.bar(
    heart_data,
    x="심장병 여부",
    y="뇌졸중_비율",
    text="뇌졸중_비율",
    title="심장병 여부에 따른 뇌졸중 발생 비율",
    labels={
        "심장병 여부": "심장병 여부",
        "뇌졸중_비율": "뇌졸중 비율(%)"
    },
    color="심장병 여부",
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
    yaxis_range=[
        0,
        max(heart_data["뇌졸중_비율"].max() * 1.2, 1)
    ]
)

st.plotly_chart(
    fig_heart,
    use_container_width=True
)


st.divider()


# ==========================================
# 8. BMI 결측치와 뇌졸중 비율
# ==========================================

st.header("🧮 5. BMI 빈 값과 뇌졸중 비율")

st.markdown(
    """
    BMI(체질량지수)가 비어 있는 사람들을 따로 살펴봅니다.

    BMI가 비어 있는 사람들의 뇌졸중 비율과
    전체 사람들의 뇌졸중 비율을 비교합니다.
    """
)

# BMI가 비어 있는 사람들
missing_bmi = df[df["bmi"].isna()]

missing_bmi_count = len(missing_bmi)

missing_bmi_stroke = int(
    (missing_bmi["stroke"] == 1).sum()
)

missing_bmi_ratio = (
    missing_bmi_stroke / missing_bmi_count * 100
    if missing_bmi_count > 0 else 0
)

# 전체 사람들
total_count = len(df)

total_stroke_count = int(
    (df["stroke"] == 1).sum()
)

total_stroke_ratio = (
    total_stroke_count / total_count * 100
    if total_count > 0 else 0
)

bmi_comparison = pd.DataFrame({
    "구분": [
        "BMI 빈 값인 사람",
        "전체 사람"
    ],
    "전체 인원": [
        missing_bmi_count,
        total_count
    ],
    "뇌졸중 인원": [
        missing_bmi_stroke,
        total_stroke_count
    ],
    "뇌졸중 비율(%)": [
        round(missing_bmi_ratio, 2),
        round(total_stroke_ratio, 2)
    ]
})

st.dataframe(
    bmi_comparison,
    use_container_width=True,
    hide_index=True
)

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "BMI 빈 값 인원",
        f"{missing_bmi_count:,}명"
    )

with col2:
    st.metric(
        "BMI 빈 값인 사람의 뇌졸중 비율",
        f"{missing_bmi_ratio:.2f}%"
    )

st.info(
    "BMI가 비어 있다는 사실만으로 뇌졸중 발생의 원인이나 "
    "위험성을 판단할 수는 없습니다."
)


st.divider()


# ==========================================
# 9. 흡연 상태별 사람 수
# ==========================================

st.header("🚬 6. 흡연 상태별 사람 수")

st.markdown(
    """
    흡연 상태별로 몇 명의 사람이 데이터에 포함되어 있는지
    확인합니다.

    빈 값이 있다면 별도의 항목으로 표시합니다.
    """
)

smoking_data = (
    df["smoking_status"]
    .fillna("빈 값")
    .value_counts(dropna=False)
    .rename_axis("흡연 상태")
    .reset_index(name="사람 수")
)

st.dataframe(
    smoking_data,
    use_container_width=True,
    hide_index=True
)

fig_smoking = px.bar(
    smoking_data,
    x="흡연 상태",
    y="사람 수",
    text="사람 수",
    title="흡연 상태별 사람 수",
    color="흡연 상태",
    labels={
        "흡연 상태": "흡연 상태",
        "사람 수": "사람 수"
    }
)

fig_smoking.update_traces(
    textposition="outside"
)

fig_smoking.update_layout(
    showlegend=False
)

st.plotly_chart(
    fig_smoking,
    use_container_width=True
)


# ==========================================
# 10. 마무리
# ==========================================

st.divider()

st.subheader("💡 탐색 결과 생각해 보기")

st.markdown(
    """
    다음 질문에 대해 생각해 봅시다.

    1. 나이가 많아질수록 뇌졸중 발생 비율은 어떻게 달라질까요?
    2. 뇌졸중을 겪은 사람과 겪지 않은 사람의 평균 혈당에는 차이가 있을까요?
    3. 고혈압이나 심장병이 있는 그룹의 뇌졸중 비율은 어떨까요?
    4. BMI가 비어 있는 데이터는 분석할 때 어떻게 처리하면 좋을까요?
    5. 흡연 상태별 사람 수는 어떻게 다를까요?

    위 결과는 데이터에서 관찰한 통계적 특징이며,
    개별 사람의 뇌졸중 발생 여부를 진단하거나 예측하는
    의학적 근거로 사용할 수는 없습니다.
    """
)

st.caption("🧠 뇌졸중 예측 실습실 | 데이터 탐색")
