import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import openai
from io import StringIO

# 페이지 설정
st.set_page_config(
    page_title="보스턴 집 값 분석 AI 챗봇",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 사이드바 - API 키 입력
with st.sidebar:
    st.title("⚙️ 설정")
    api_key = st.text_input("OpenAI API Key", type="password", key="api_key")
    
    if api_key:
        openai.api_key = api_key
        st.success("✅ API 키가 설정되었습니다.")
    else:
        st.warning("⚠️ API 키를 입력해주세요.")
    
    st.divider()
    st.markdown("### 📊 앱 정보")
    st.info("""
    이 앱은 보스턴 주택 가격 데이터를 분석하고, 
    회귀 분석 결과를 AI 챗봇과 대화하며 해석할 수 있습니다.
    """)

# 데이터 로드 및 캐싱
@st.cache_data
def load_data():
    """보스턴 주택 데이터 로드"""
    try:
        # OpenML에서 보스턴 주택 데이터 로드
        boston = fetch_openml(name='boston', version=1, parser='auto')
        X = pd.DataFrame(boston.data, columns=boston.feature_names)
        # 데이터 타입을 숫자형으로 변환 (OpenML에서 문자열로 반환될 수 있음)
        X = X.astype(float)
        y = pd.Series(boston.target, name='MEDV').astype(float)
        
        # 전체 데이터프레임 생성
        df = pd.concat([X, y], axis=1)
        
        return df, X, y, boston.feature_names
    except:
        # 대체 방법: 직접 데이터 생성
        st.warning("OpenML에서 데이터를 가져올 수 없어 샘플 데이터를 생성합니다.")
        np.random.seed(42)
        n_samples = 506
        
        data = {
            'CRIM': np.random.exponential(3.5, n_samples),
            'ZN': np.random.choice([0, 12.5, 25, 50], n_samples),
            'INDUS': np.random.uniform(0.5, 27, n_samples),
            'CHAS': np.random.choice([0, 1], n_samples, p=[0.93, 0.07]),
            'NOX': np.random.uniform(0.3, 0.9, n_samples),
            'RM': np.random.normal(6.3, 0.7, n_samples),
            'AGE': np.random.uniform(2, 100, n_samples),
            'DIS': np.random.uniform(1, 12, n_samples),
            'RAD': np.random.choice(range(1, 25), n_samples),
            'TAX': np.random.uniform(180, 720, n_samples),
            'PTRATIO': np.random.uniform(12, 22, n_samples),
            'B': np.random.uniform(0.3, 400, n_samples),
            'LSTAT': np.random.uniform(2, 38, n_samples)
        }
        
        X = pd.DataFrame(data)
        # 간단한 선형 관계로 타겟 생성
        y = pd.Series(
            5 * X['RM'] - 0.5 * X['LSTAT'] + 0.1 * X['DIS'] + np.random.normal(0, 3, n_samples),
            name='MEDV'
        )
        y = y.clip(5, 50)  # 현실적인 범위로 제한
        
        df = pd.concat([X, y], axis=1)
        feature_names = list(data.keys())
        
        return df, X, y, feature_names

# 회귀 모델 학습 및 캐싱
@st.cache_data
def train_model(X, y):
    """회귀 모델 학습 및 평가"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # 예측
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # 평가 지표
    metrics = {
        'train': {
            'R2': r2_score(y_train, y_pred_train),
            'RMSE': np.sqrt(mean_squared_error(y_train, y_pred_train)),
            'MAE': mean_absolute_error(y_train, y_pred_train)
        },
        'test': {
            'R2': r2_score(y_test, y_pred_test),
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_test)),
            'MAE': mean_absolute_error(y_test, y_pred_test)
        }
    }
    
    # 계수 정보
    coefficients = pd.DataFrame({
        'Feature': X.columns,
        'Coefficient': model.coef_
    }).sort_values('Coefficient', key=abs, ascending=False)
    
    return model, metrics, coefficients, X_train, X_test, y_train, y_test, y_pred_test

# 데이터 및 모델 로드
df, X, y, feature_names = load_data()
model, metrics, coefficients, X_train, X_test, y_train, y_test, y_pred_test = train_model(X, y)

# 메인 타이틀
st.title("🏠 보스턴 주택 가격 분석 AI 챗봇")
st.markdown("---")

# 탭 생성
tab1, tab2, tab3 = st.tabs(["📈 데이터 분석", "📊 회귀 분석 결과", "💬 AI 챗봇"])

# 탭 1: 데이터 분석
with tab1:
    st.header("📈 데이터 탐색")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("전체 데이터 수", len(df))
    with col2:
        st.metric("특성 개수", len(feature_names))
    with col3:
        st.metric("평균 주택 가격", f"${y.mean():.2f}K")
    with col4:
        st.metric("가격 표준편차", f"${y.std():.2f}K")
    
    st.subheader("📋 데이터 미리보기")
    st.dataframe(df.head(10), use_container_width=True)
    
    st.subheader("📊 기술 통계")
    st.dataframe(df.describe(), use_container_width=True)
    
    st.subheader("📉 주요 변수 시각화")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(y, bins=30, edgecolor='black', alpha=0.7, color='skyblue')
        ax.set_xlabel('주택 가격 (MEDV) - $1000 단위', fontsize=12)
        ax.set_ylabel('빈도', fontsize=12)
        ax.set_title('주택 가격 분포', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        st.pyplot(fig)
    
    with col2:
        fig, ax = plt.subplots(figsize=(10, 6))
        correlation = df.corr()['MEDV'].sort_values(ascending=False)[1:11]
        colors = ['green' if x > 0 else 'red' for x in correlation]
        ax.barh(correlation.index, correlation.values, color=colors, alpha=0.7)
        ax.set_xlabel('상관계수', fontsize=12)
        ax.set_title('주택 가격과의 상관관계 (Top 10)', fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        st.pyplot(fig)
    
    st.subheader("🔥 상관관계 히트맵")
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, ax=ax, cbar_kws={'label': '상관계수'})
    ax.set_title('특성 간 상관관계 히트맵', fontsize=14, fontweight='bold')
    st.pyplot(fig)

# 탭 2: 회귀 분석 결과
with tab2:
    st.header("📊 선형 회귀 분석 결과")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 학습 데이터 성능")
        st.metric("R² Score", f"{metrics['train']['R2']:.4f}")
        st.metric("RMSE", f"${metrics['train']['RMSE']:.4f}K")
        st.metric("MAE", f"${metrics['train']['MAE']:.4f}K")
    
    with col2:
        st.subheader("🎯 테스트 데이터 성능")
        st.metric("R² Score", f"{metrics['test']['R2']:.4f}")
        st.metric("RMSE", f"${metrics['test']['RMSE']:.4f}K")
        st.metric("MAE", f"${metrics['test']['MAE']:.4f}K")
    
    st.subheader("📉 회귀 계수 분석")
    st.dataframe(coefficients, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 회귀 계수 시각화")
        fig, ax = plt.subplots(figsize=(10, 8))
        colors = ['green' if x > 0 else 'red' for x in coefficients['Coefficient']]
        ax.barh(coefficients['Feature'], coefficients['Coefficient'], 
                color=colors, alpha=0.7)
        ax.set_xlabel('계수', fontsize=12)
        ax.set_title('특성별 회귀 계수', fontsize=14, fontweight='bold')
        ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
        ax.grid(axis='x', alpha=0.3)
        st.pyplot(fig)
    
    with col2:
        st.subheader("🎯 예측 vs 실제")
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.scatter(y_test, y_pred_test, alpha=0.6, edgecolors='k', s=50)
        ax.plot([y_test.min(), y_test.max()], 
                [y_test.min(), y_test.max()], 
                'r--', lw=2, label='완벽한 예측')
        ax.set_xlabel('실제 가격 ($1000)', fontsize=12)
        ax.set_ylabel('예측 가격 ($1000)', fontsize=12)
        ax.set_title('예측 정확도', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        st.pyplot(fig)
    
    st.subheader("📉 잔차 분석")
    residuals = y_test - y_pred_test
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(y_pred_test, residuals, alpha=0.6, edgecolors='k', s=50)
        ax.axhline(y=0, color='r', linestyle='--', linewidth=2)
        ax.set_xlabel('예측 가격 ($1000)', fontsize=12)
        ax.set_ylabel('잔차', fontsize=12)
        ax.set_title('잔차 플롯', fontsize=14, fontweight='bold')
        ax.grid(alpha=0.3)
        st.pyplot(fig)
    
    with col2:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(residuals, bins=30, edgecolor='black', alpha=0.7, color='lightcoral')
        ax.set_xlabel('잔차', fontsize=12)
        ax.set_ylabel('빈도', fontsize=12)
        ax.set_title('잔차 분포', fontsize=14, fontweight='bold')
        ax.axvline(x=0, color='r', linestyle='--', linewidth=2)
        ax.grid(axis='y', alpha=0.3)
        st.pyplot(fig)

# 탭 3: AI 챗봇
with tab3:
    st.header("💬 AI 분석 챗봇")
    
    # 세션 상태 초기화
    if "messages" not in st.session_state:
        st.session_state.messages = []
        # 초기 컨텍스트 메시지
        context_message = f"""
당신은 데이터 분석 전문가입니다. 보스턴 주택 가격 데이터에 대한 회귀 분석 결과를 해석하는 것을 돕고 있습니다.

**데이터셋 정보:**
- 전체 샘플 수: {len(df)}
- 특성 개수: {len(feature_names)}
- 타겟 변수: MEDV (주택 가격, $1000 단위)

**회귀 모델 성능:**
- 학습 R² Score: {metrics['train']['R2']:.4f}
- 테스트 R² Score: {metrics['test']['R2']:.4f}
- 테스트 RMSE: ${metrics['test']['RMSE']:.4f}K
- 테스트 MAE: ${metrics['test']['MAE']:.4f}K

**주요 회귀 계수 (상위 5개):**
{coefficients.head(5).to_string(index=False)}

**상관관계 (MEDV와 상위 5개 특성):**
{df.corr()['MEDV'].sort_values(ascending=False)[1:6].to_string()}

사용자의 질문에 대해 이 정보를 바탕으로 친절하고 명확하게 답변해주세요.
"""
        st.session_state.messages.append({"role": "system", "content": context_message})
    
    # 대화 기록 표시
    for message in st.session_state.messages:
        if message["role"] != "system":  # 시스템 메시지는 표시하지 않음
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    
    # 사용자 입력
    if prompt := st.chat_input("분석 결과에 대해 질문해보세요..."):
        if not api_key:
            st.error("⚠️ 먼저 사이드바에서 OpenAI API 키를 입력해주세요.")
        else:
            # 사용자 메시지 추가
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # AI 응답 (스트리밍)
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""
                
                try:
                    # OpenAI API 호출 (스트리밍)
                    stream = openai.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": m["role"], "content": m["content"]}
                            for m in st.session_state.messages
                        ],
                        stream=True,
                        temperature=0.7,
                        max_tokens=1000
                    )
                    
                    # 스트림 응답 처리
                    for chunk in stream:
                        if chunk.choices[0].delta.content is not None:
                            full_response += chunk.choices[0].delta.content
                            message_placeholder.markdown(full_response + "▌")
                    
                    message_placeholder.markdown(full_response)
                    
                except Exception as e:
                    error_message = f"❌ 오류가 발생했습니다: {str(e)}"
                    message_placeholder.markdown(error_message)
                    full_response = error_message
                
                # 응답 저장
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": full_response
                })
    
    # 대화 초기화 버튼
    if st.button("🔄 대화 초기화"):
        st.session_state.messages = []
        st.rerun()
    
    # 추천 질문
    st.divider()
    st.subheader("💡 추천 질문")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("가장 중요한 특성은 무엇인가요?"):
            st.session_state.messages.append({
                "role": "user", 
                "content": "이 회귀 모델에서 가장 중요한 특성은 무엇이고, 그 이유는 무엇인가요?"
            })
            st.rerun()
        
        if st.button("모델 성능을 어떻게 해석하나요?"):
            st.session_state.messages.append({
                "role": "user", 
                "content": "R² Score와 RMSE를 바탕으로 이 모델의 성능을 어떻게 해석해야 하나요?"
            })
            st.rerun()
    
    with col2:
        if st.button("음수 계수의 의미는?"):
            st.session_state.messages.append({
                "role": "user", 
                "content": "음수 회귀 계수를 가진 특성들은 무엇을 의미하나요?"
            })
            st.rerun()
        
        if st.button("모델 개선 방법은?"):
            st.session_state.messages.append({
                "role": "user", 
                "content": "이 모델의 성능을 개선하기 위한 방법을 제안해주세요."
            })
            st.rerun()

# 푸터
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
        🏠 보스턴 주택 가격 분석 AI 챗봇 | Powered by Streamlit & OpenAI
    </div>
    """,
    unsafe_allow_html=True
)
