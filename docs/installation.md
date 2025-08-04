# 설치 가이드

## 📦 PyPI에서 설치

```bash
pip install ranx-k
```

## 🔧 개발 버전 설치

GitHub에서 최신 개발 버전을 설치하려면:

```bash
pip install git+https://github.com/tsdata/rank-k.git
```

## 🐍 개발 환경 설정

프로젝트에 기여하거나 개발하려면:

```bash
# 저장소 클론
git clone https://github.com/tsdata/rank-k.git
cd rank-k

# 가상환경 생성 (uv 사용)
uv venv ranx-k-dev
source ranx-k-dev/bin/activate  # Linux/Mac
# ranx-k-dev\Scripts\activate  # Windows

# 개발 모드로 설치
uv pip install -e ".[dev]"
```

## ✅ 설치 확인

설치가 올바르게 되었는지 확인:

```python
from ranx_k.tokenizers import KiwiTokenizer

tokenizer = KiwiTokenizer()
print("✅ ranx-k 설치 완료!")
```

## 📋 의존성

### 필수 의존성
- Python ≥ 3.8
- kiwipiepy ≥ 0.15.0
- rouge-score ≥ 0.1.2
- sentence-transformers ≥ 2.2.0
- scikit-learn ≥ 1.0.0
- numpy ≥ 1.21.0
- tqdm ≥ 4.62.0
- ranx ≥ 0.3.0

### 개발 의존성
- pytest ≥ 7.0.0
- pytest-cov ≥ 4.0.0
- black ≥ 22.0.0
- isort ≥ 5.0.0
- flake8 ≥ 5.0.0
- mypy ≥ 1.0.0

## 🚨 문제 해결

### Kiwi 설치 오류
```bash
# macOS에서 Kiwi 설치 오류 시
brew install cmake
pip install kiwipiepy
```

### M1 Mac 호환성
Apple Silicon(M1/M2) Mac에서는 다음과 같이 설치:

```bash
# Rosetta 없이 네이티브 설치
arch -arm64 pip install ranx-k
```

### 메모리 부족 오류
대용량 임베딩 모델 사용 시 메모리 부족이 발생할 수 있습니다:

```python
# 경량 모델 사용
from ranx_k.evaluation import evaluate_with_ranx_similarity

results = evaluate_with_ranx_similarity(
    # ... 기타 매개변수
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"  # 경량 모델
)
```