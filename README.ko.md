# ranx-k: 한국어 최적화 ranx IR 평가 도구 🇰🇷

[![PyPI version](https://badge.fury.io/py/ranx-k.svg)](https://badge.fury.io/py/ranx-k)
[![Python version](https://img.shields.io/pypi/pyversions/ranx-k.svg)](https://pypi.org/project/ranx-k/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**[English](README.md) | [한국어](README.ko.md)**

**ranx-k**는 한국어에 최적화된 정보 검색(IR) 평가 도구로, 기존 ranx 라이브러리를 확장하여 Kiwi 토크나이저와 한국어 임베딩을 지원합니다. RAG(Retrieval-Augmented Generation) 시스템의 성능을 정확하게 평가할 수 있습니다.

## 🚀 주요 특징

- **한국어 특화**: Kiwi 형태소 분석기를 활용한 정확한 토큰화
- **ranx 기반**: 검증된 IR 평가 메트릭 (Hit@K, NDCG@K, MRR, MAP@K 등) 지원
- **LangChain 호환**: LangChain 검색기 인터페이스 표준 지원
- **다양한 평가 방법**: ROUGE, 임베딩 유사도, 의미적 유사도 기반 평가
- **등급별 관련성 지원**: NDCG 계산을 위해 유사도 점수를 관련성 등급으로 사용
- **구성 가능한 ROUGE 타입**: ROUGE-1, ROUGE-2, ROUGE-L 선택 가능
- **엄격한 임계값 적용**: 유사도 임계값 미만 문서는 검색 실패로 올바르게 처리
- **실용적 설계**: 프로토타입부터 프로덕션까지 단계별 평가 지원
- **높은 성능**: 기존 방법 대비 30~80% 한국어 평가 정확도 향상
- **이중언어 출력**: 국제적 접근성을 위한 영어-한국어 병기 출력 지원

## 📦 설치

```bash
pip install ranx-k
```

또는 개발 버전 설치:

```bash
pip install "ranx-k[dev]"
```

## 🔗 검색기 호환성

ranx-k는 **LangChain 검색기 인터페이스**를 지원합니다:

```python
# 검색기는 invoke() 메서드를 구현해야 합니다
class YourRetriever:
    def invoke(self, query: str) -> List[Document]:
        # Document 객체 리스트 반환 (page_content 속성 필요)
        pass

# LangChain Document 사용 예시
from langchain.schema import Document
doc = Document(page_content="텍스트 내용")
```

> **참고**: LangChain은 MIT 라이선스 하에 배포됩니다. 자세한 내용은 [문서](docs/ko/quickstart.md#langchain-license)를 참조하세요.

## 🔧 빠른 시작

### 기본 사용법

```python
from ranx_k.evaluation import simple_kiwi_rouge_evaluation

# 간단한 Kiwi ROUGE 평가
results = simple_kiwi_rouge_evaluation(
    retriever=your_retriever,
    questions=your_questions,
    reference_contexts=your_reference_contexts,
    k=5
)

print(f"ROUGE-1: {results['kiwi_rouge1@5']:.3f}")
print(f"ROUGE-2: {results['kiwi_rouge2@5']:.3f}")
print(f"ROUGE-L: {results['kiwi_rougeL@5']:.3f}")
```

### 향상된 평가 (Rouge Score + Kiwi)

```python
from ranx_k.evaluation import rouge_kiwi_enhanced_evaluation

# 검증된 rouge_score 라이브러리 + Kiwi 토크나이저
results = rouge_kiwi_enhanced_evaluation(
    retriever=your_retriever,
    questions=your_questions,
    reference_contexts=your_reference_contexts,
    k=5,
    tokenize_method='morphs',  # 'morphs' 또는 'nouns'
    use_stopwords=True
)
```

### 의미적 유사도 기반 ranx 평가

```python
from ranx_k.evaluation import evaluate_with_ranx_similarity

# 참조 기반 평가 (정확한 재현율을 위해 권장)
results = evaluate_with_ranx_similarity(
    retriever=your_retriever,
    questions=your_questions,
    reference_contexts=your_reference_contexts,
    k=5,
    method='embedding',
    similarity_threshold=0.6,
    use_graded_relevance=False,        # 이진 관련성 (기본값)
    evaluation_mode='reference_based'  # 모든 참조 문서 대상 평가
)

print(f"Hit@5: {results['hit_rate@5']:.3f}")
print(f"NDCG@5: {results['ndcg@5']:.3f}")
print(f"MRR: {results['mrr']:.3f}")
print(f"MAP@5: {results['map@5']:.3f}")
```

#### 다른 임베딩 모델 사용

```python
# OpenAI 임베딩 모델 (API 키 필요)
results = evaluate_with_ranx_similarity(
    retriever=your_retriever,
    questions=your_questions,
    reference_contexts=your_reference_contexts,
    k=5,
    method='openai',
    similarity_threshold=0.7,
    embedding_model="text-embedding-3-small"
)

# 최신 BGE-M3 모델 (한국어 우수)
results = evaluate_with_ranx_similarity(
    retriever=your_retriever,
    questions=your_questions,
    reference_contexts=your_reference_contexts,
    k=5,
    method='embedding',
    similarity_threshold=0.6,
    embedding_model="BAAI/bge-m3"
)

# 한국어 특화 Kiwi ROUGE 방법 - 구성 가능한 ROUGE 타입
results = evaluate_with_ranx_similarity(
    retriever=your_retriever,
    questions=your_questions,
    reference_contexts=your_reference_contexts,
    k=5,
    method='kiwi_rouge',
    similarity_threshold=0.3,  # Kiwi ROUGE는 낮은 임계값 권장
    rouge_type='rougeL',      # 'rouge1', 'rouge2', 'rougeL' 선택
    tokenize_method='morphs', # 'morphs' 또는 'nouns' 선택
    use_stopwords=True        # 불용어 필터링 설정
)
```

### 종합 평가

```python
from ranx_k.evaluation import comprehensive_evaluation_comparison

# 모든 평가 방법 비교
comparison = comprehensive_evaluation_comparison(
    retriever=your_retriever,
    questions=your_questions,
    reference_contexts=your_reference_contexts,
    k=5
)
```

## 📊 평가 방법

### 1. Kiwi ROUGE 평가
- **장점**: 빠른 속도, 직관적 해석
- **용도**: 프로토타이핑, 빠른 피드백

### 2. Enhanced ROUGE (Rouge Score + Kiwi)
- **장점**: 검증된 라이브러리, 안정성
- **용도**: 프로덕션 환경, 신뢰성 중요한 평가

### 3. 의미적 유사도 기반 ranx
- **장점**: 전통적 IR 메트릭, 의미적 유사도
- **용도**: 연구, 벤치마킹, 상세 분석

## 🎯 성능 개선 사례

```python
# 기존 방법 (영어 토크나이저)
basic_rouge1 = 0.234

# ranx-k (Kiwi 토크나이저)
ranxk_rouge1 = 0.421  # +79.9% 향상!
```

## 📊 추천 임베딩 모델

| 모델 | 용도 | 임계값 | 특징 |
|------|------|--------|------|
| `paraphrase-multilingual-MiniLM-L12-v2` | 기본 | 0.6 | 빠름, 가벼움 |
| `text-embedding-3-small` (OpenAI) | 정확도 | 0.7 | 높은 정확도, 비용 효율적 |
| `BAAI/bge-m3` | 한국어 | 0.6 | 최신, 다국어 우수 |
| `text-embedding-3-large` (OpenAI) | 프리미엄 | 0.8 | 최고 성능 |

## 📈 점수 해석 가이드

| 점수 범위 | 평가 | 권장 조치 |
|-----------|------|-----------|
| 0.7 이상 | 🟢 매우 좋음 | 현재 설정 유지 |
| 0.5~0.7 | 🟡 양호 | 미세 조정 고려 |
| 0.3~0.5 | 🟠 보통 | 개선 필요 |
| 0.3 미만 | 🔴 부족 | 대폭 수정 필요 |

## 🔍 고급 사용법

### 등급별 관련성 모드

```python
# 등급별 관련성 모드 - 유사도 점수를 관련성 등급으로 사용
results = evaluate_with_ranx_similarity(
    retriever=your_retriever,
    questions=questions,
    reference_contexts=references,
    method='embedding',
    similarity_threshold=0.6,
    use_graded_relevance=True   # 유사도 점수를 관련성 등급으로 사용
)

print(f"NDCG@5: {results['ndcg@5']:.3f}")
```

> **등급별 관련성 참고사항**: `use_graded_relevance` 매개변수는 주로 NDCG (Normalized Discounted Cumulative Gain) 계산에 영향을 미칩니다. Hit@K, MRR, MAP 같은 다른 메트릭들은 ranx 라이브러리에서 관련성을 이진으로 처리합니다. 문서 관련성의 품질 차이를 구분해야 할 때 등급별 관련성을 사용하세요.

### 커스텀 임베딩 모델

```python
# 커스텀 임베딩 모델 사용
results = evaluate_with_ranx_similarity(
    retriever=your_retriever,
    questions=questions,
    reference_contexts=references,
    method='embedding',
    embedding_model="your-custom-model-name",
    similarity_threshold=0.6
)
```

### 구성 가능한 ROUGE 타입

```python
# 다양한 ROUGE 메트릭 비교
for rouge_type in ['rouge1', 'rouge2', 'rougeL']:
    results = evaluate_with_ranx_similarity(
        retriever=your_retriever,
        questions=questions,
        reference_contexts=references,
        method='kiwi_rouge',
        rouge_type=rouge_type,
        tokenize_method='morphs',
        similarity_threshold=0.3
    )
    print(f"{rouge_type.upper()}: Hit@5 = {results['hit_rate@5']:.3f}")
```

### 임계값 민감도 분석

```python
# 다양한 임계값이 평가에 미치는 영향 분석
thresholds = [0.3, 0.5, 0.7]
for threshold in thresholds:
    results = evaluate_with_ranx_similarity(
        retriever=your_retriever,
        questions=questions,
        reference_contexts=references,
        similarity_threshold=threshold
    )
    print(f"임계값 {threshold}: Hit@5={results['hit_rate@5']:.3f}, NDCG@5={results['ndcg@5']:.3f}")
```

## 📚 예제

- [기본 토크나이저 예제](examples/basic_tokenizer.py)
- [BGE-M3 평가 예제](examples/bge_m3_evaluation.py)
- [임베딩 모델 비교](examples/embedding_models_comparison.py)
- [종합 비교](examples/comprehensive_comparison.py)

## 🤝 기여하기

기여를 환영합니다! 이슈와 풀 리퀘스트를 자유롭게 제출해 주세요.

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.

## 🙏 감사의 말

- Elias Bassani의 [ranx](https://github.com/AmenRa/ranx)를 기반으로 구축
- [Kiwi](https://github.com/bab2min/kiwipiepy)를 통한 한국어 형태소 분석
- [sentence-transformers](https://github.com/UKPLab/sentence-transformers)를 통한 임베딩 지원

## 📞 지원

- 🐛 이슈 트래커: GitHub에서 이슈를 제출해 주세요
- 📧 이메일: ontofinance@gmail.com

---

**ranx-k** - 정확하고 쉬운 한국어 RAG 평가를 위한 도구!