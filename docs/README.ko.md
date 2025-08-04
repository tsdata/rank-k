# ranx-k 문서

## Language / 언어
[English](index.md) | [한국어](README.ko.md)

한국어 최적화된 ranx IR 평가 도구킷인 ranx-k에 오신 것을 환영합니다!

## 빠른 네비게이션

### 한국어 문서
- [설치 가이드](ko/installation.md)
- [빠른 시작](ko/quickstart.md)
- [API 참조](ko/api-reference.md)

### English Documentation
- [Installation Guide](en/installation.md)
- [Quick Start](en/quickstart.md) 
- [API Reference](en/api-reference.md)

## ranx-k 소개

ranx-k는 한국어 RAG (Retrieval-Augmented Generation) 시스템 평가를 위한 특화된 도구킷입니다:

- **한국어 최적화 토크나이저** - Kiwi 형태소 분석기 사용
- **다양한 평가 방법** - ROUGE, 임베딩 유사도, ranx 메트릭
- **종합적인 예제**와 문서
- **프로덕션 준비** 완료된 테스트 커버리지

## 주요 기능

### 🔤 한국어 토큰화
```python
from ranx_k.tokenizers import KiwiTokenizer

tokenizer = KiwiTokenizer(method='morphs')
tokens = tokenizer.tokenize('한국어 자연어처리 도구입니다.')
print(tokens)  # ['한국어', '자연어', '처리', '도구']
```

### 📊 평가 방법
```python
from ranx_k.evaluation import simple_kiwi_rouge_evaluation

results = simple_kiwi_rouge_evaluation(
    retriever=your_retriever,
    questions=questions,
    reference_contexts=references,
    k=5
)
```

## 시작하기

1. **설치**: `pip install ranx-k`
2. **따라하기**: [빠른 시작 가이드](ko/quickstart.md)
3. **탐색하기**: [예제](https://github.com/tsdata/rank-k/tree/main/examples)

## 링크

- **GitHub**: https://github.com/tsdata/rank-k
- **PyPI**: https://pypi.org/project/ranx-k/
- **이슈**: https://github.com/tsdata/rank-k/issues