# 빠른 시작

이 가이드는 ranx-k의 기본 사용법을 소개합니다.

## 🚀 5분 만에 시작하기

### 1. 설치

```bash
pip install ranx-k
```

### 2. 기본 토크나이저 사용

```python
from ranx_k.tokenizers import KiwiTokenizer

# 형태소 분석 기반 토크나이저
tokenizer = KiwiTokenizer(method='morphs', use_stopwords=True)

text = "자연어처리는 인공지능의 핵심 기술입니다."
tokens = tokenizer.tokenize(text)
print(f"토큰: {tokens}")
# 출력: ['자연어처리', '인공지능', '핵심', '기술']
```

### 3. 명사 추출

```python
# 명사만 추출하는 토크나이저
noun_tokenizer = KiwiTokenizer(method='nouns')

text = "RAG 시스템은 검색과 생성을 결합합니다."
nouns = noun_tokenizer.tokenize(text)
print(f"명사: {nouns}")
# 출력: ['시스템', '검색', '생성', '결합']
```

### 4. 불용어 커스터마이징

```python
tokenizer = KiwiTokenizer(use_stopwords=True)

# 커스텀 불용어 추가
tokenizer.add_stopwords(['시스템', '방법'])

# 불용어 제거
tokenizer.remove_stopwords(['기술'])

# 현재 불용어 확인
stopwords = tokenizer.get_stopwords()
print(f"불용어 개수: {len(stopwords)}")
```

## 📊 평가 함수 사용

### 1. 간단한 ROUGE 평가

```python
from ranx_k.evaluation import simple_kiwi_rouge_evaluation

# 가상의 검색기와 데이터 (실제 사용 시 교체)
# retriever = your_retriever
# questions = ["질문1", "질문2", ...]
# reference_contexts = [["정답문서1", "정답문서2"], ...]

# 평가 실행
results = simple_kiwi_rouge_evaluation(
    retriever=retriever,
    questions=questions,
    reference_contexts=reference_contexts,
    k=5
)

print(f"ROUGE-1: {results['kiwi_rouge1@5']:.3f}")
print(f"ROUGE-2: {results['kiwi_rouge2@5']:.3f}")
print(f"ROUGE-L: {results['kiwi_rougeL@5']:.3f}")
```

### 2. 향상된 ROUGE 평가

```python
from ranx_k.evaluation import rouge_kiwi_enhanced_evaluation

results = rouge_kiwi_enhanced_evaluation(
    retriever=retriever,
    questions=questions,
    reference_contexts=reference_contexts,
    k=5,
    tokenize_method='morphs',
    use_stopwords=True
)
```

### 3. 의미적 유사도 기반 ranx 평가

```python
from ranx_k.evaluation import evaluate_with_ranx_similarity

results = evaluate_with_ranx_similarity(
    retriever=retriever,
    questions=questions,
    reference_contexts=reference_contexts,
    k=5,
    method='kiwi_rouge',
    similarity_threshold=0.6
)

print(f"Hit@5: {results['hit_rate@5']:.3f}")
print(f"NDCG@5: {results['ndcg@5']:.3f}")
print(f"MRR: {results['mrr']:.3f}")
```

## 🔧 실전 예제

### RAG 시스템 평가

```python
import ranx_k

# 1. 토크나이저 초기화
tokenizer = ranx_k.KiwiTokenizer(method='morphs')

# 2. 테스트 데이터 준비
questions = [
    "자연어처리란 무엇인가요?",
    "RAG 시스템의 장점은?",
    "한국어 토큰화의 어려움은?"
]

reference_contexts = [
    ["자연어처리는 컴퓨터가 인간의 언어를 이해하고 처리하는 기술입니다."],
    ["RAG는 검색과 생성을 결합하여 더 정확한 답변을 제공합니다."],
    ["한국어는 교착어적 특성으로 인해 토큰화가 복잡합니다."]
]

# 3. 종합 평가 실행
from ranx_k.evaluation import comprehensive_evaluation_comparison

results = comprehensive_evaluation_comparison(
    retriever=your_retriever,
    questions=questions,
    reference_contexts=reference_contexts,
    k=5
)
```

## 📈 성능 최적화 팁

### 1. 배치 처리
```python
# 대량 데이터 처리 시 배치 단위로 나누어 처리
batch_size = 100
for i in range(0, len(questions), batch_size):
    batch_questions = questions[i:i+batch_size]
    batch_contexts = reference_contexts[i:i+batch_size]
    # 평가 실행
```

### 2. 경량 모델 사용
```python
# 메모리 절약을 위한 경량 임베딩 모델
results = evaluate_with_ranx_similarity(
    # ... 기타 매개변수
    method='kiwi_rouge'  # 임베딩 대신 ROUGE 사용
)
```

### 3. 캐싱 활용
```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def cached_tokenize(text):
    return tokenizer.tokenize(text)
```

## 🎯 다음 단계

- [평가 방법론](evaluation-methods.md)에서 더 자세한 평가 방법 학습
- [API 레퍼런스](api-reference.md)에서 전체 함수 목록 확인
- [예제](examples.md)에서 실제 사용 사례 학습