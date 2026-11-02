
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score


# ==========================================
# 1. 페이지 설정
# ==========================================

st.set_page_config(
    page_title="분류 모델 | 뇌졸중 예측 실습실",
    page_icon="🤖",
    layout="wide"
)


# ==========================================
# 2. 기본 설정
# ==========================================

DATA_URL = (
    "https://raw.githubusercontent.com/"
    "greatsong/modudata/main/data/stroke.csv"
)

RANDOM_STATE = 42

FEATURE_NAMES = {
    "age": "나이",
    "avg_glucose_level": "평균 혈당",
    "bmi": "체질량지수",
    "hypertension": "고혈압",
    "heart_disease": "심장병"
}


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
# 4. 제목
# ==========================================

st.title("🤖 분류 모델")

st.markdown(
    """
    건강 정보를 입력으로 사용하여 뇌졸중 발생 여부를 예측하는
    분류 모델을 만들어 봅니다.

    이번 실습에서는 로지스틱 회귀와 의사결정트리를 비교합니다.

    - 0: 뇌졸중 아님 (음성)
    - 1: 뇌졸중 (양성)
    """
)

st.divider()


# ==========================================
# 5. 입력 속성 선택
# ==========================================

st.header("1. 입력 속성 선택")

st.markdown(
    """
    뇌졸중을 예측할 때 사용할 속성을 선택하세요.

    처음에는 체질량지수를 제외한 네 가지 속성이 선택되어 있습니다.
    """
)

feature_options = list(FEATURE_NAMES.keys())

selected_features = st.multiselect(
    "예측에 사용할 속성",
    options=feature_options,
    default=[
        "age",
        "avg_glucose_level",
        "hypertension",
        "heart_disease"
    ],
    format_func=lambda x: FEATURE_NAMES[x],
    key="selected_features"
)


# 두 개 미만 선택 시 중단
if len(selected_features) < 2:

    st.warning(
        "입력 속성을 두 개 이상 선택해 주세요."
    )

    st.stop()


st.write(
    "선택한 속성: "
    + ", ".join(
        FEATURE_NAMES[x]
        for x in selected_features
    )
)

st.divider()


# ==========================================
# 6. 데이터 정렬 및 훈련 / 테스트 분리
# ==========================================

st.header("2. 훈련 데이터와 테스트 데이터 나누기")

st.markdown(
    """
    사람의 번호(id)를 기준으로 오름차순 정렬합니다.

    정렬된 데이터를 10명씩 묶고,
    각 묶음의 앞 세 명을 테스트 데이터로 사용합니다.

    나머지 일곱 명만 훈련 데이터로 사용합니다.
    """
)


# 번호순 정렬
sorted_df = df.sort_values(
    by="id",
    ascending=True
).reset_index(drop=True)


# 10명 단위로 묶었을 때 앞의 3명 선택
test_mask = (
    sorted_df.index % 10 < 3
)

train_df = sorted_df.loc[
    ~test_mask
].copy()

test_df = sorted_df.loc[
    test_mask
].copy()


# 데이터 개수
train_count = len(train_df)
test_count = len(test_df)


# 요청한 데이터 개수 확인
if test_count != 1533:

    st.warning(
        f"현재 테스트 데이터는 {test_count:,}명입니다. "
        "요청한 1,533명과 다른 데이터 개수입니다."
    )


col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "전체 데이터",
        f"{len(sorted_df):,}명"
    )

with col2:
    st.metric(
        "훈련 데이터",
        f"{train_count:,}명"
    )

with col3:
    st.metric(
        "테스트 데이터",
        f"{test_count:,}명"
    )


st.info(
    "테스트 데이터는 10명씩 묶은 각 그룹의 앞 세 명입니다. "
    "훈련 데이터와 테스트 데이터는 서로 겹치지 않습니다."
)


# ==========================================
# 7. 입력 데이터와 정답 분리
# ==========================================

X_train = train_df[selected_features].copy()

X_test = test_df[selected_features].copy()

y_train = train_df["stroke"].astype(int)

y_test = test_df["stroke"].astype(int)


# ==========================================
# 8. BMI 결측치 처리
# ==========================================

st.header("3. 결측치 처리")

st.markdown(
    """
    체질량지수(bmi)를 입력 속성으로 선택한 경우에만
    비어 있는 값을 채웁니다.

    중앙값은 훈련 데이터에서만 계산합니다.
    테스트 데이터에는 훈련 데이터에서 계산한 동일한 중앙값을 사용합니다.
    """
)


imputation_medians = {}

if "bmi" in selected_features:

    bmi_median = X_train["bmi"].median()

    imputation_medians["bmi"] = bmi_median

    train_missing = int(
        X_train["bmi"].isna().sum()
    )

    test_missing = int(
        X_test["bmi"].isna().sum()
    )

    X_train["bmi"] = X_train["bmi"].fillna(
        bmi_median
    )

    X_test["bmi"] = X_test["bmi"].fillna(
        bmi_median
    )

    st.success(
        f"체질량지수의 훈련 데이터 중앙값 "
        f"{bmi_median:.2f}으로 결측치를 채웠습니다."
    )

    bmi_info = pd.DataFrame({
        "구분": [
            "훈련 데이터",
            "테스트 데이터"
        ],
        "결측치 처리 전 개수": [
            train_missing,
            test_missing
        ],
        "대체에 사용한 중앙값": [
            round(bmi_median, 2),
            round(bmi_median, 2)
        ]
    })

    st.dataframe(
        bmi_info,
        use_container_width=True,
        hide_index=True
    )

else:

    st.info(
        "체질량지수를 선택하지 않았으므로 "
        "BMI 결측치 처리를 하지 않습니다."
    )


# ==========================================
# 9. 입력 데이터 표준화
# ==========================================

st.header("4. 입력 데이터 크기 맞추기")

st.markdown(
    """
    로지스틱 회귀 모델은 입력 속성의 크기에 영향을 받을 수 있습니다.

    따라서 표준화 도구를 사용합니다.

    평균과 표준편차는 훈련 데이터만으로 계산하며,
    테스트 데이터에는 같은 기준을 적용합니다.
    """
)


# ==========================================
# 10. 모델 학습
# ==========================================

st.header("5. 분류 모델 만들기")


# 로지스틱 회귀
logistic_model = Pipeline([
    (
        "scaler",
        StandardScaler()
    ),
    (
        "model",
        LogisticRegression(
            random_state=RANDOM_STATE,
            max_iter=2000
        )
    )
])


# 의사결정트리
tree_model = DecisionTreeClassifier(
    max_depth=3,
    min_samples_leaf=5,
    random_state=RANDOM_STATE
)


# 두 모델 훈련
logistic_model.fit(
    X_train,
    y_train
)

tree_model.fit(
    X_train,
    y_train
)


# 예측
logistic_train_pred = logistic_model.predict(
    X_train
)

logistic_test_pred = logistic_model.predict(
    X_test
)

tree_train_pred = tree_model.predict(
    X_train
)

tree_test_pred = tree_model.predict(
    X_test
)


# ==========================================
# 11. 입력 없이 예측하는 기준 모델
# ==========================================

# 훈련 데이터에서 가장 많은 정답 선택
majority_class = int(
    y_train.mode().iloc[0]
)


baseline_train_pred = [
    majority_class
] * len(y_train)

baseline_test_pred = [
    majority_class
] * len(y_test)


# ==========================================
# 12. 정확도 계산
# ==========================================

logistic_train_accuracy = accuracy_score(
    y_train,
    logistic_train_pred
)

logistic_test_accuracy = accuracy_score(
    y_test,
    logistic_test_pred
)

tree_train_accuracy = accuracy_score(
    y_train,
    tree_train_pred
)

tree_test_accuracy = accuracy_score(
    y_test,
    tree_test_pred
)

baseline_train_accuracy = accuracy_score(
    y_train,
    baseline_train_pred
)

baseline_test_accuracy = accuracy_score(
    y_test,
    baseline_test_pred
)


# ==========================================
# 13. 정확도 카드
# ==========================================

st.subheader("모델 정확도 비교")

st.markdown(
    """
    큰 숫자는 테스트 데이터의 정확도입니다.

    아래에는 훈련 데이터와 테스트 데이터의 정확도를 함께 표시합니다.
    """
)


def show_accuracy_card(
    title,
    train_accuracy,
    test_accuracy
):

    with st.container(border=True):

        st.markdown(f"**{title}**")

        st.metric(
            label="테스트 데이터 정확도",
            value=f"{test_accuracy * 100:.2f}%"
        )

        st.caption(
            f"훈련 데이터 정확도: "
            f"{train_accuracy * 100:.2f}%"
        )

        st.caption(
            f"테스트 데이터 정확도: "
            f"{test_accuracy * 100:.2f}%"
        )


col1, col2, col3 = st.columns(3)

with col1:

    show_accuracy_card(
        "로지스틱 회귀 (확률로 답하는 모델)",
        logistic_train_accuracy,
        logistic_test_accuracy
    )


with col2:

    show_accuracy_card(
        "의사결정트리 (질문으로 답하는 모델)",
        tree_train_accuracy,
        tree_test_accuracy
    )


with col3:

    show_accuracy_card(
        "입력을 보지 않는 기준 모델",
        baseline_train_accuracy,
        baseline_test_accuracy
    )


st.caption(
    f"기준 모델은 훈련 데이터에서 더 많은 정답인 "
    f"{'뇌졸중' if majority_class == 1 else '뇌졸중 아님'}"
    "만 답합니다."
)


st.divider()


# ==========================================
# 14. 두 속성 선택
# ==========================================

st.header("6. 두 속성으로 분류 결과 살펴보기")

st.markdown(
    """
    선택한 입력 속성 가운데 두 가지를 골라
    가로축과 세로축으로 사용합니다.

    나머지 속성은 테스트 데이터의 중앙값으로 고정하여
    분류 경계와 분류 영역을 그립니다.
    """
)


axis_col1, axis_col2 = st.columns(2)

with axis_col1:

    x_feature = st.selectbox(
        "가로축 속성",
        options=selected_features,
        format_func=lambda x: FEATURE_NAMES[x],
        index=0
    )


y_options = [
    feature
    for feature in selected_features
    if feature != x_feature
]

with axis_col2:

    y_feature = st.selectbox(
        "세로축 속성",
        options=y_options,
        format_func=lambda x: FEATURE_NAMES[x],
        index=0
    )


# ==========================================
# 15. 테스트 데이터 중앙값 계산
# ==========================================

fixed_values = {}

for feature in selected_features:

    if feature not in [x_feature, y_feature]:

        fixed_values[feature] = float(
            X_test[feature].median()
        )


st.subheader("그림을 그리기 위해 고정한 속성")

if fixed_values:

    fixed_table = pd.DataFrame({
        "속성": [
            FEATURE_NAMES[x]
            for x in fixed_values.keys()
        ],
        "테스트 데이터 중앙값": [
            round(value, 3)
            for value in fixed_values.values()
        ]
    })

    st.dataframe(
        fixed_table,
        use_container_width=True,
        hide_index=True
    )

else:

    st.info(
        "선택한 두 속성 모두 그림의 축으로 사용하므로 "
        "고정하는 나머지 속성이 없습니다."
    )


# ==========================================
# 16. 그래프 범위와 격자 만들기
# ==========================================

x_min = float(X_test[x_feature].min())
x_max = float(X_test[x_feature].max())

y_min = float(X_test[y_feature].min())
y_max = float(X_test[y_feature].max())


# 그래프 여백
x_padding = (x_max - x_min) * 0.05
y_padding = (y_max - y_min) * 0.05

if x_padding == 0:
    x_padding = 0.5

if y_padding == 0
