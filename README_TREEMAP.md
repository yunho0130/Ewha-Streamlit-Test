# 주식 트리맵 기능 추가 완료

## 작업 요약

`boston_housing_app.py`에 주식 시장 Position Map 스타일의 인터랙티브 트리맵 기능을 성공적으로 추가했습니다.

## 추가된 기능

### 1. 새로운 탭: "📊 주식 트리맵"
- 기존 3개 탭에 4번째 탭으로 추가
- Position Map 스타일의 인터랙티브 시각화

### 2. 주요 기능

#### 필터 옵션
- **Industry**: 산업별 필터링 (36개 산업)
- **Sector**: 섹터별 필터링 (9개 섹터)
- **Mnemonic (Ticker)**: 특정 종목 선택

#### 시각화 옵션
- **Size (크기)**:
  - MktValue (시가총액) - 기본값
  - Volume (거래량)
  - CurrentPrice (현재가)

- **Color (색상)**:
  - PriceChangePct (가격 변동률) - 기본값
    - 빨강: 하락 (-10% ~ 0%)
    - 파랑: 상승 (0% ~ +10%)
  - PE_Ratio (주가수익비율)
  - DividendYield (배당수익률)

#### 샘플 데이터
- 9개 주요 섹터의 주식 데이터 생성
  - Technology
  - Financials
  - Health Care
  - Consumer Goods
  - Consumer Services
  - Industrials
  - Basic Materials
  - Utilities
  - Telecommunications

#### 추가 정보
- 요약 통계 (총 기업 수, 총 시가총액, 평균 변동률, 상승 종목 비율)
- 섹터별 상세 통계
- 상위/하위 변동 종목 (각 10개)
- 호버 시 상세 정보 표시

## 파일 구조

```
C:\_dev\ewha-antigravity-test\
├── boston_housing_app.py          # 메인 애플리케이션 (업데이트됨)
├── requirements.txt               # 패키지 의존성 (업데이트됨)
└── README_TREEMAP.md             # 이 파일
```

## 실행 방법

### 중요: Python 환경 요구사항

**현재 시스템에 32비트 Python 3.12.7이 설치되어 있어 최신 패키지 설치에 제한이 있습니다.**

### 권장 사항: 64비트 Python 사용

1. **64비트 Python 3.9 이상 설치** (권장)
   - https://www.python.org/downloads/ 에서 64비트 버전 다운로드
   - 설치 시 "Add Python to PATH" 옵션 체크

2. **가상환경 생성 및 활성화**
   ```bash
   cd C:\_dev\ewha-antigravity-test
   python -m venv venv
   venv\Scripts\activate
   ```

3. **패키지 설치**
   ```bash
   pip install -r requirements.txt
   ```

4. **애플리케이션 실행**
   ```bash
   streamlit run boston_housing_app.py
   ```

5. **웹 브라우저에서 확인**
   - 자동으로 브라우저가 열립니다 (기본: http://localhost:8501)
   - "📊 주식 트리맵" 탭 클릭

## 현재 32비트 Python에서 실행하는 경우

32비트 Python 환경에서는 최신 pandas, matplotlib, scikit-learn 등의 패키지가 설치되지 않을 수 있습니다.

### 대안 1: Anaconda 사용 (권장)
```bash
conda create -n stock_treemap python=3.10
conda activate stock_treemap
conda install streamlit pandas numpy matplotlib seaborn scikit-learn plotly openai
cd C:\_dev\ewha-antigravity-test
streamlit run boston_housing_app.py
```

### 대안 2: Docker 사용
```bash
# Dockerfile 생성 후
docker build -t stock-treemap .
docker run -p 8501:8501 stock-treemap
```

## 코드 변경 사항

### 1. 새로운 import 추가
```python
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
```

### 2. 주식 데이터 생성 함수 추가
```python
@st.cache_data
def generate_stock_data():
    """주식 시장 샘플 데이터 생성"""
    # 9개 섹터, 각 섹터당 5-15개 기업
    # 시가총액, 가격변동률, 거래량 등 생성
```

### 3. 트리맵 시각화
- Plotly treemap 사용
- 계층 구조: Sector → Industry → Ticker
- 인터랙티브 호버 정보
- 색상 스케일: RdBu (빨강-파랑)

## 주요 라이브러리

- **streamlit**: 웹 애플리케이션 프레임워크
- **plotly**: 인터랙티브 시각화 (트리맵)
- **pandas**: 데이터 처리
- **numpy**: 수치 계산 및 랜덤 데이터 생성
- **sklearn**: 회귀 분석 (기존 기능)
- **matplotlib/seaborn**: 정적 시각화 (기존 기능)

## 트러블슈팅

### 문제: 패키지 설치 실패
**해결**: 64비트 Python 또는 Anaconda 사용 (위 참조)

### 문제: "No module named 'plotly'" 에러
```bash
pip install plotly
```

### 문제: 트리맵이 표시되지 않음
- 필터에서 최소 1개 이상의 Sector/Industry 선택 확인
- 브라우저 콘솔에서 에러 확인

### 문제: OpenAI API 관련 에러 (기존 기능)
- 사이드바에서 OpenAI API 키 입력 필요
- AI 챗봇 탭에서만 필요하며, 트리맵 기능에는 영향 없음

## 기능 데모

1. **전체 시장 보기**: 모든 필터를 기본값으로 유지
2. **Technology 섹터만**: Sector에서 "Technology"만 선택
3. **가격 급등/급락 찾기**: Color를 "PriceChangePct"로 설정하고 빨간색/파란색 박스 확인
4. **대형주 중심 보기**: Size를 "MktValue"로 설정
5. **배당주 찾기**: Color를 "DividendYield"로 변경

## 추가 개선 아이디어

1. 실시간 데이터 연동 (Yahoo Finance API, Alpha Vantage 등)
2. 히스토리컬 데이터 및 시계열 분석
3. 포트폴리오 시뮬레이션
4. 섹터 회전 분석
5. CSV 데이터 업로드 기능
6. 데이터 다운로드 버튼

## 라이선스 및 참고사항

- 샘플 데이터는 랜덤 생성된 데모용 데이터입니다.
- 실제 투자 결정에 사용하지 마세요.
- 실제 데이터 사용 시 해당 데이터 제공업체의 라이선스를 확인하세요.

---

**개발 완료**: 2025-12-12
**테스트 환경**: Windows, Python 3.12 (32비트 - 제약 있음)
**권장 환경**: Windows/Mac/Linux, Python 3.9-3.11 (64비트)
