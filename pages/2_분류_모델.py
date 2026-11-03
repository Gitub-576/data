
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score


# ==========================================
# 1. 페이지 기본 설정
# ==========================================

st.set_page_config(
    page_title="분류 모델 | 뇌졸중 예측 실습실",
    page_icon="🤖",
    layout="wide"
)

DATA_URL = (
    "https://raw.githubusercontent.com/"
    "greatsong/modudata/main/data/stroke.csv"
)

RANDOM_STATE = 42

FEATURE_LABELS = {
    "age": "나이",
    "avg_glucose_level": "평균 혈당",
    "bmi": "체질량지수",
    "hypertension": "고혈압",
    "heart_disease": "심장병"
}

DEFAULT_FEATURES = [
    "age",
    "avg_glucose_level",
    "hypertension",
    "heart_disease"
]


# ==========================================
# 2. 데이터 불러오기
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
    st.error("데이터를 불러오지 못했습니다.")
    st.error(str(e))
    st.stop()


# 필요한 열 확인
required_columns = [
    "id",
    "age",
    "avg_glucose_level",
    "bmi",
    "hypertension",
    "heart_disease",
    "stroke"
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    st.error(f"필요한 열이 없습니다: {missing_columns}")
    st.stop()


# ==========================================
# 3. 화면 제목
# ==========================================

st.title("🤖 분류 모델")

st.markdown("""
건강 정보를 이용하여 뇌졸중 발생 여부를 분류하는
머신러닝 모델을 만들어 봅시다.

이번 실습에서는 두 가지 분류 모델을 학습하고
테스트 데이터를 이용하여 성능을 비교합니다.
""")

st.info(
    "stroke가 1이면 뇌졸중(양성), "
    "0이면 뇌졸중 아님(음성)입니다."
)

st.divider()


# ==========================================
# 4. 입력 속성 선택
# ==========================================

st.header("📋 1. 입력 속성 선택")

st.markdown("""
예측에 사용할 속성을 선택하세요.

처음에는 체질량지수(BMI)를 제외한
네 가지 속성이 선택되어 있습니다.
""")

selected_features = st.multiselect(
    "예측에 사용할 속성",
    options=list(FEATURE_LABELS.keys()),
    default=DEFAULT_FEATURES,
    format_func=lambda x: FEATURE_LABELS[x]
)

if len(selected_features) < 2:
    st.warning(
        "입력 속성을 두 개 이상 선택해 주세요. "
        "두 개보다 적으면 모델을 학습할 수 없습니다."
    )
    st.stop()

st.write(
    "선택한 속성:",
    ", ".join(
        FEATURE_LABELS[x]
        for x in selected_features
    )
)

st.divider()


# ==========================================
# 5. 번호 순 정렬 및 데이터 분리
# ==========================================

st.header("✂️ 2. 학습 데이터와 테스트 데이터")

st.markdown("""
사람의 번호(id)를 오름차순으로 정렬합니다.

정렬된 데이터를 10명씩 묶고,
각 묶음의 앞 3명은 테스트용으로,
나머지 7명은 학습용으로 사용합니다.
""")

df = df.sort_values(
    by="id",
    ascending=True
).reset_index(drop=True)

# 10명씩 묶어 앞의 3명을 테스트용으로 지정
test_positions = [
    i for i in range(len(df))
    if i % 10 < 3
]

train_positions = [
    i for i in range(len(df))
    if i % 10 >= 3
]

train_df = df.iloc[train_positions].copy()
test_df = df.iloc[test_positions].copy()

# 입력 데이터와 정답 데이터 분리
X_train = train_df[selected_features].copy()
X_test = test_df[selected_features].copy()

y_train = train_df["stroke"].astype(int).copy()
y_test = test_df["stroke"].astype(int).copy()


# 데이터 수 표시
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("전체 데이터", f"{len(df):,}명")

with col2:
    st.metric("학습 데이터", f"{len(train_df):,}명")

with col3:
    st.metric("테스트 데이터", f"{len(test_df):,}명")


# 학습 및 테스트 데이터 구성표
split_table = pd.DataFrame({
    "구분": ["학습 데이터", "테스트 데이터"],
    "전체 인원": [
        len(train_df),
        len(test_df)
    ],
    "뇌졸중 인원(양성)": [
        int((y_train == 1).sum()),
        int((y_test == 1).sum())
    ],
    "뇌졸중 아님(음성)": [
        int((y_train == 0).sum()),
        int((y_test == 0).sum())
    ]
})

st.dataframe(
    split_table,
    use_container_width=True,
    hide_index=True
)

if len(test_df) != 1533:
    st.error(
        f"테스트 데이터가 1,533명이 아닙니다. "
        f"현재 {len(test_df):,}명입니다. "
        "원본 데이터의 행 수를 확인하세요."
    )
    st.stop()

st.divider()


# ==========================================
# 6. BMI 결측치 처리
# ==========================================

st.header("🧹 3. 빈 값 처리")

bmi_median = None

if "bmi" in selected_features:

    # 학습 데이터에서만 중앙값 계산
    bmi_median = X_train["bmi"].median()

    if pd.isna(bmi_median):
        st.error("학습 데이터에서 BMI 중앙값을 계산할 수 없습니다.")
        st.stop()

    train_missing = int(
        X_train["bmi"].isna().sum()
    )

    test_missing = int(
        X_test["bmi"].isna().sum()
    )

    # 학습 데이터 중앙값으로 결측치 채우기
    X_train["bmi"] = X_train["bmi"].fillna(
        bmi_median
    )

    # 테스트 데이터에도 학습 데이터 중앙값 적용
    X_test["bmi"] = X_test["bmi"].fillna(
        bmi_median
    )

    st.success(
        f"BMI 결측치를 학습 데이터 중앙값 "
        f"{bmi_median:.2f}으로 채웠습니다."
    )

    st.write(f"- 학습 데이터 빈 값: {train_missing:,}명")
    st.write(f"- 테스트 데이터 빈 값: {test_missing:,}명")

else:

    st.info(
        "체질량지수(BMI)를 선택하지 않았으므로 "
        "결측치 처리를 하지 않았습니다."
    )

st.divider()


# ==========================================
# 7. 입력 데이터 크기 맞추기
# ==========================================

st.header("📐 4. 입력 데이터 크기 맞추기")

st.markdown("""
StandardScaler를 이용하여 입력 속성의 크기를 맞춥니다.

크기 맞추기 도구는 학습 데이터로만 학습하고,
테스트 데이터에는 같은 기준을 적용합니다.
""")

scaler = StandardScaler()

# 학습 데이터로만 크기 맞추기 기준 계산
X_train_scaled = scaler.fit_transform(X_train)

# 테스트 데이터는 학습 데이터 기준으로 변환
X_test_scaled = scaler.transform(X_test)

st.success(
    "학습 데이터로 크기 맞추기 기준을 만들고 "
    "테스트 데이터에 동일한 기준을 적용했습니다."
)

st.divider()


# ==========================================
# 8. 로지스틱 회귀 모델
# ==========================================

st.header("🤖 5. 분류 모델 학습")

st.subheader(
    "로지스틱 회귀(확률로 답하는 모델)"
)

logistic_model = LogisticRegression(
    max_iter=2000,
    random_state=RANDOM_STATE
)

logistic_model.fit(
    X_train_scaled,
    y_train
)

logistic_train_pred = logistic_model.predict(
    X_train_scaled
)

logistic_test_pred = logistic_model.predict(
    X_test_scaled
)

logistic_train_accuracy = accuracy_score(
    y_train,
    logistic_train_pred
)

logistic_test_accuracy = accuracy_score(
    y_test,
    logistic_test_pred
)


# ==========================================
# 9. 의사결정트리 모델
# ==========================================

st.subheader(
    "의사결정트리(질문으로 답하는 모델)"
)

tree_model = DecisionTreeClassifier(
    max_depth=3,
    min_samples_leaf=5,
    random_state=RANDOM_STATE
)

tree_model.fit(
    X_train,
    y_train
)

tree_train_pred = tree_model.predict(
    X_train
)

tree_test_pred = tree_model.predict(
    X_test
)

tree_train_accuracy = accuracy_score(
    y_train,
    tree_train_pred
)

tree_test_accuracy = accuracy_score(
    y_test,
    tree_test_pred
)


# ==========================================
# 10. 기준 모델
# ==========================================

st.subheader(
    "기준 모델(입력을 보지 않고 많은 쪽으로 답하는 모델)"
)

baseline_model = DummyClassifier(
    strategy="most_frequent"
)

baseline_model.fit(
    X_train,
    y_train
)

baseline_train_pred = baseline_model.predict(
    X_train
)

baseline_test_pred = baseline_model.predict(
    X_test
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
# 1부 끝
# ==========================================

# ==========================================
# 11. 모델별 정확도 카드
# ==========================================

st.divider()

st.header("📊 6. 모델별 정확도 비교")

st.markdown("""
큰 숫자는 테스트 데이터의 정확도입니다.

작은 글씨에서는 훈련 데이터와 테스트 데이터의
정확도를 함께 비교할 수 있습니다.
""")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 로지스틱 회귀")

    st.metric(
        "테스트 데이터 정확도",
        f"{logistic_test_accuracy * 100:.2f}%"
    )

    st.caption(
        f"훈련 데이터: {logistic_train_accuracy * 100:.2f}%"
    )

    st.caption(
        f"테스트 데이터: {logistic_test_accuracy * 100:.2f}%"
    )

with col2:
    st.markdown("### 의사결정트리")

    st.metric(
        "테스트 데이터 정확도",
        f"{tree_test_accuracy * 100:.2f}%"
    )

    st.caption(
        f"훈련 데이터: {tree_train_accuracy * 100:.2f}%"
    )

    st.caption(
        f"테스트 데이터: {tree_test_accuracy * 100:.2f}%"
    )

with col3:
    st.markdown("### 기준 모델")

    st.metric(
        "테스트 데이터 정확도",
        f"{baseline_test_accuracy * 100:.2f}%"
    )

    st.caption(
        f"훈련 데이터: {baseline_train_accuracy * 100:.2f}%"
    )

    st.caption(
        f"테스트 데이터: {baseline_test_accuracy * 100:.2f}%"
    )


# 정확도 비교 표
accuracy_table = pd.DataFrame({
    "모델": [
        "로지스틱 회귀(확률로 답하는 모델)",
        "의사결정트리(질문으로 답하는 모델)",
        "기준 모델(다수 클래스 예측)"
    ],
    "훈련 데이터 정확도(%)": [
        round(logistic_train_accuracy * 100, 2),
        round(tree_train_accuracy * 100, 2),
        round(baseline_train_accuracy * 100, 2)
    ],
    "테스트 데이터 정확도(%)": [
        round(logistic_test_accuracy * 100, 2),
        round(tree_test_accuracy * 100, 2),
        round(baseline_test_accuracy * 100, 2)
    ]
})

st.dataframe(
    accuracy_table,
    use_container_width=True,
    hide_index=True
)

st.caption(
    "정확도는 전체 데이터 중 정답을 맞힌 비율입니다. "
    "뇌졸중 환자처럼 양성 데이터가 적은 경우에는 "
    "정확도만으로 모델의 성능을 판단하기 어렵습니다."
)


# ==========================================
# 12. 두 속성으로 분류 결과 살펴보기
# ==========================================

st.divider()

st.header("🎨 7. 두 속성으로 분류 결과 살펴보기")

st.markdown("""
선택한 입력 속성 중 두 가지를 골라
가로축과 세로축으로 사용합니다.

테스트 데이터를 점으로 표시하고,
실제 뇌졸중 여부에 따라 색을 구분합니다.

배경의 옅은 색은 의사결정트리의 분류 영역이며,
검은 선은 로지스틱 회귀의 확률 0.5 경계입니다.
""")

axis_col1, axis_col2 = st.columns(2)

with axis_col1:
    x_feature = st.selectbox(
        "가로축 속성",
        options=selected_features,
        format_func=lambda x: FEATURE_LABELS[x],
        key="x_feature"
    )

with axis_col2:
    y_options = [
        feature
        for feature in selected_features
        if feature != x_feature
    ]

    y_feature = st.selectbox(
        "세로축 속성",
        options=y_options,
        format_func=lambda x: FEATURE_LABELS[x],
        key="y_feature"
    )


# ==========================================
# 13. 나머지 속성의 중앙값 고정
# ==========================================

st.subheader("그림을 그릴 때 고정한 속성")

# 테스트 데이터의 중앙값 계산
test_medians = X_test.median()

fixed_values = {}

for feature in selected_features:

    if feature not in [x_feature, y_feature]:

        fixed_values[feature] = float(
            test_medians[feature]
        )


if fixed_values:

    fixed_table = pd.DataFrame({
        "속성": [
            FEATURE_LABELS[feature]
            for feature in fixed_values
        ],
        "고정값": [
            round(value, 3)
            for value in fixed_values.values()
        ]
    })

    st.dataframe(
        fixed_table,
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "가로축과 세로축이 아닌 속성은 "
        "테스트 데이터의 중앙값에 고정했습니다."
    )

else:

    st.info(
        "선택한 속성이 두 개뿐이므로 "
        "고정할 나머지 속성이 없습니다."
    )


# ==========================================
# 14. 그림의 범위 설정
# ==========================================

x_min = float(X_test[x_feature].min())
x_max = float(X_test[x_feature].max())

y_min = float(X_test[y_feature].min())
y_max = float(X_test[y_feature].max())

x_padding = (x_max - x_min) * 0.05
y_padding = (y_max - y_min) * 0.05

if x_padding == 0:
    x_padding = max(abs(x_min) * 0.05, 0.5)

if y_padding == 0:
    y_padding = max(abs(y_min) * 0.05, 0.5)

x_start = x_min - x_padding
x_end = x_max + x_padding

y_start = y_min - y_padding
y_end = y_max + y_padding

# 그림의 해상도
grid_size = 70

x_values = [
    x_start + (x_end - x_start) * i / (grid_size - 1)
    for i in range(grid_size)
]

y_values = [
    y_start + (y_end - y_start) * i / (grid_size - 1)
    for i in range(grid_size)
]


# ==========================================
# 15. 격자 데이터 만들기
# ==========================================

grid_rows = []

for y_value in y_values:

    for x_value in x_values:

        row = {}

        for feature in selected_features:

            if feature == x_feature:

                row[feature] = x_value

            elif feature == y_feature:

                row[feature] = y_value

            else:

                row[feature] = fixed_values[feature]

        grid_rows.append(row)


grid_df = pd.DataFrame(
    grid_rows,
    columns=selected_features
)


# ==========================================
# 16. BMI 결측치 처리
# ==========================================

if "bmi" in selected_features:

    grid_df["bmi"] = grid_df["bmi"].fillna(
        bmi_median
    )


# ==========================================
# 17. 로지스틱 회귀 확률 계산
# ==========================================

# 학습 데이터로 만든 크기 맞추기 기준 적용
grid_scaled = scaler.transform(grid_df)

# 뇌졸중(1)일 확률 계산
grid_probabilities = logistic_model.predict_proba(
    grid_scaled
)[:, 1]

# 2차원 배열로 변환
probability_grid = [
    grid_probabilities[
        i * grid_size:(i + 1) * grid_size
    ].tolist()
    for i in range(grid_size)
]


# ==========================================
# 18. 의사결정트리 분류 영역 계산
# ==========================================

grid_tree_predictions = tree_model.predict(
    grid_df
)

tree_grid = [
    grid_tree_predictions[
        i * grid_size:(i + 1) * grid_size
    ].tolist()
    for i in range(grid_size)
]


# ==========================================
# 19. Plotly 그림 만들기
# ==========================================

fig = go.Figure()


# ------------------------------------------
# 19-1. 의사결정트리 분류 영역
# ------------------------------------------

fig.add_trace(
    go.Heatmap(
        x=x_values,
        y=y_values,
        z=tree_grid,
        zmin=0,
        zmax=1,
        colorscale=[
            [0, "rgba(76,120,168,0.24)"],
            [0.5, "rgba(76,120,168,0.24)"],
            [0.5, "rgba(228,87,86,0.24)"],
            [1, "rgba(228,87,86,0.24)"]
        ],
        showscale=False,
        hoverinfo="skip",
        name="의사결정트리 분류 영역"
    )
)


# ------------------------------------------
# 19-2. 로지스틱 회귀 확률 0.5 경계
# ------------------------------------------

fig.add_trace(
    go.Contour(
        x=x_values,
        y=y_values,
        z=probability_grid,
        contours=dict(
            start=0.5,
            end=0.5,
            size=1,
            coloring="none"
        ),
        line=dict(
            color="black",
            width=3
        ),
        showscale=False,
        hoverinfo="skip",
        name="로지스틱 회귀 0.5 경계"
    )
)


# ------------------------------------------
# 19-3. 실제 뇌졸중이 아닌 사람
# ------------------------------------------

negative_mask = y_test == 0

fig.add_trace(
    go.Scatter(
        x=X_test.loc[
            negative_mask,
            x_feature
        ],
        y=X_test.loc[
            negative_mask,
            y_feature
        ],
        mode="markers",
        name="실제 뇌졸중 아님(0)",
        marker=dict(
            color="#4C78A8",
            size=7,
            opacity=0.7,
            line=dict(
                color="white",
                width=0.5
            )
        )
    )
)


# ------------------------------------------
# 19-4. 실제 뇌졸중인 사람
# ------------------------------------------

positive_mask = y_test == 1

fig.add_trace(
    go.Scatter(
        x=X_test.loc[
            positive_mask,
            x_feature
        ],
        y=X_test.loc[
            positive_mask,
            y_feature
        ],
        mode="markers",
        name="실제 뇌졸중(1)",
        marker=dict(
            color="#E45756",
            size=8,
            opacity=0.8,
            line=dict(
                color="white",
                width=0.5
            )
        )
    )
)


# ==========================================
# 20. 그림 설정 및 출력
# ==========================================

fig.update_layout(
    title="테스트 데이터와 분류 모델의 분류 영역",
    xaxis_title=FEATURE_LABELS[x_feature],
    yaxis_title=FEATURE_LABELS[y_feature],
    legend_title="실제 뇌졸중 여부",
    template="plotly_white",
    height=650
)

fig.update_xaxes(
    range=[x_start, x_end]
)

fig.update_yaxes(
    range=[y_start, y_end]
)

st.plotly_chart(
    fig,
    use_container_width=True
)


# ==========================================
# 21. 확률 0.5 경계 확인
# ==========================================

min_probability = float(
    grid_probabilities.min()
)

max_probability = float(
    grid_probabilities.max()
)

if min_probability <= 0.5 <= max_probability:

    st.success(
        "로지스틱 회귀의 확률 0.5 경계가 "
        "현재 그림의 범위 안에 있습니다."
    )

else:

    st.warning(
        "로지스틱 회귀의 확률 0.5 경계가 "
        "현재 그림 범위 안에 나타나지 않습니다. "
        "경계가 그림 밖에 있거나 현재 범위에서 "
        "0.5를 통과하지 않는 경우입니다."
    )


st.caption("""
파란색 배경은 의사결정트리가 뇌졸중 아님(0)으로,
붉은색 배경은 뇌졸중(1)으로 분류하는 영역입니다.

검은 선은 로지스틱 회귀의 뇌졸중 확률이 0.5인 경계입니다.
""")

# ==========================================
# 2부 끝
# ==========================================

# ==========================================
# 22. 의사결정트리 질문 구조
# ==========================================

st.divider()

st.header("🌳 8. 의사결정트리가 던지는 질문")

st.markdown("""
의사결정트리는 질문을 통해 데이터를 나눕니다.

이번 실습에서는 다음 조건을 적용합니다.

- 질문은 최대 세 번까지만 허용합니다.
- 한 마디에 사람이 다섯 명 미만이면 더 나누지 않습니다.
- 마지막 마디는 예측 결과에 따라 색을 다르게 표시합니다.
""")


# ==========================================
# 23. 의사결정트리의 마디 정보 계산
# ==========================================

tree_structure = tree_model.tree_

# 훈련 데이터가 각 마디에 도달했는지 확인
decision_path = tree_model.decision_path(X_train)

node_information = {}

leaf_count = 0
negative_leaf_count = 0
used_features = set()


for node_id in range(tree_structure.node_count):

    # 해당 마디에 도달한 훈련 데이터
    node_mask = (
        decision_path[:, node_id]
        .toarray()
        .ravel()
    )

    node_indices = [
        i for i, reached in enumerate(node_mask)
        if reached
    ]

    # 전체 인원
    people = len(node_indices)

    # 실제 뇌졸중 인원
    positive = sum(
        int(y_train.iloc[i])
        for i in node_indices
    )

    # 실제 뇌졸중이 아닌 인원
    negative = people - positive

    # 뇌졸중 비율
    ratio = (
        positive / people * 100
        if people > 0
        else 0
    )

    # 마지막 마디인지 확인
    is_leaf = (
        tree_structure.children_left[node_id]
        == tree_structure.children_right[node_id]
    )

    if is_leaf:

        leaf_count += 1

        predicted_class = int(
            tree_structure.value[node_id][0].argmax()
        )

        if predicted_class == 0:
            negative_leaf_count += 1

    else:

        feature_index = tree_structure.feature[node_id]

        feature_name = selected_features[
            feature_index
        ]

        used_features.add(feature_name)

    node_information[node_id] = {
        "people": people,
        "positive": positive,
        "negative": negative,
        "ratio": ratio,
        "is_leaf": is_leaf
    }


# ==========================================
# 24. DOT 문자열 만들기
# ==========================================

dot_lines = [
    "digraph DecisionTree {",

    'graph [rankdir=TB, bgcolor="white", nodesep=0.35, ranksep=0.55];',

    'node [shape=box, style="rounded,filled", fontname="Malgun Gothic", fontsize=11, margin=0.15];',

    'edge [fontname="Malgun Gothic", fontsize=10];'
]


# ------------------------------------------
# 각 마디 만들기
# ------------------------------------------

for node_id in range(tree_structure.node_count):

    info = node_information[node_id]

    people = info["people"]
    positive = info["positive"]
    ratio = info["ratio"]

    # 마디에 표시할 통계
    stats_text = (
        f"전체 {people}명\\n"
        f"뇌졸중 {positive}명\\n"
        f"뇌졸중 비율 {ratio:.1f}%"
    )

    if info["is_leaf"]:

        # 마지막 마디의 예측 결과
        predicted_class = int(
            tree_structure.value[node_id][0].argmax()
        )

        if predicted_class == 1:

            answer_text = "답: 뇌졸중 (1)"
            fill_color = "#F4CCCC"

        else:

            answer_text = "답: 뇌졸중 아님 (0)"
            fill_color = "#CFE2F3"

        label = (
            f"{stats_text}\\n"
            f"{answer_text}"
        )

    else:

        # 질문에 사용한 속성
        feature_index = tree_structure.feature[node_id]

        feature_name = selected_features[
            feature_index
        ]

        # 질문의 기준값
        threshold = float(
            tree_structure.threshold[node_id]
        )

        label = (
            f"{FEATURE_LABELS[feature_name]} ≤ "
            f"{threshold:.2f}?\\n"
            f"{stats_text}"
        )

        fill_color = "#FFF2CC"

    # DOT 노드 추가
    dot_lines.append(
        f'{node_id} [label="{label}", fillcolor="{fill_color}"];'
    )


# ==========================================
# 25. 가지 연결하기
# ==========================================

for node_id in range(tree_structure.node_count):

    # 마지막 마디는 가지가 없음
    if node_information[node_id]["is_leaf"]:
        continue

    left_child = int(
        tree_structure.children_left[node_id]
    )

    right_child = int(
        tree_structure.children_right[node_id]
    )

    # 왼쪽 가지: 질문에 예라고 답한 경우
    dot_lines.append(
        f'{node_id} -> {left_child} [label="예 (≤)"];'
    )

    # 오른쪽 가지: 질문에 아니요라고 답한 경우
    dot_lines.append(
        f'{node_id} -> {right_child} [label="아니요 (>)"];'
    )


# DOT 문자열 끝내기
dot_lines.append("}")

dot = "\n".join(dot_lines)


# ==========================================
# 26. 의사결정트리 그림 출력
# ==========================================

st.subheader("의사결정트리 질문 가지 그림")

st.markdown("""
각 마디에는 다음 정보가 표시됩니다.

- 전체: 해당 마디에 도달한 훈련 데이터 인원
- 뇌졸중: 실제 뇌졸중 환자 인원
- 뇌졸중 비율: 해당 마디의 뇌졸중 환자 비율
""")

st.graphviz_chart(
    dot,
    use_container_width=True
)


# ==========================================
# 27. 의사결정트리 요약
# ==========================================

st.subheader("의사결정트리 요약")

col1, col2 = st.columns(2)

with col1:

    st.metric(
        "답을 내는 마지막 마디 수",
        f"{leaf_count}칸"
    )

with col2:

    st.metric(
        "뇌졸중 아님(0)으로 답하는 마디 수",
        f"{negative_leaf_count}칸"
    )


# 실제 질문에 사용한 속성 정리
used_feature_text = ", ".join(
    FEATURE_LABELS[feature]
    for feature in selected_features
    if feature in used_features
)

if not used_feature_text:

    used_feature_text = "사용한 질문 속성이 없습니다."


st.markdown(
    f"**이 나무가 실제로 질문한 속성:** {used_feature_text}"
)

st.markdown(
    f"**마지막 마디:** 전체 {leaf_count}칸 중 "
    f"{negative_leaf_count}칸이 "
    "뇌졸중 아님(0)으로 답합니다."
)

st.caption(
    "마디의 인원 수와 뇌졸중 인원 수는 훈련 데이터 기준입니다. "
    "마지막 마디의 예측 결과는 해당 마디의 다수 클래스입니다."
)


# ==========================================
# 28. 실습 정리
# ==========================================

st.divider()

st.header("💡 생각해 보기")

st.markdown("""
다음 질문을 생각해 봅시다.

1. 로지스틱 회귀와 의사결정트리 중 테스트 정확도가 높은 모델은 무엇인가요?

2. 기준 모델과 비교했을 때 두 모델의 정확도는 어떠한가요?

3. 훈련 데이터와 테스트 데이터의 정확도 차이가 큰 모델은 무엇인가요?

4. 의사결정트리가 가장 먼저 던진 질문은 무엇인가요?

5. 입력 속성을 바꾸면 모델의 정확도와 분류 그림은 어떻게 달라지나요?

6. 정확도가 높다는 사실만으로 뇌졸중 환자를 잘 찾아낸다고 할 수 있을까요?
""")


# ==========================================
# 29. 주의사항
# ==========================================

st.warning("""
이 페이지는 교육용 분류 모델 실습입니다.

실제 의료 진단이나 개인의 뇌졸중 위험 평가에
사용해서는 안 됩니다.
""")


# ==========================================
# 30. 페이지 하단
# ==========================================

st.divider()

st.caption(
    "🧠 뇌졸중 예측 실습실 | 분류 모델"
)

# ==========================================
# 3부 끝
# ==========================================
