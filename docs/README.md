# ranx-k 문서

ranx-k는 한국어에 최적화된 정보 검색(IR) 평가 도구입니다.

## 📚 문서 목록

- [설치 가이드](installation.md) - ranx-k 설치 방법
- [빠른 시작](quickstart.md) - 기본 사용법
- [API 레퍼런스](api-reference.md) - 상세 API 문서
- [평가 방법론](evaluation-methods.md) - 지원하는 평가 방법들
- [한국어 토크나이저](korean-tokenizer.md) - Kiwi 토크나이저 사용법
- [예제 및 튜토리얼](examples.md) - 실제 사용 예제들

## 🎯 주요 특징

1. **한국어 특화**: Kiwi 형태소 분석기를 활용한 정확한 토큰화
2. **ranx 기반**: 검증된 IR 평가 메트릭 지원
3. **다양한 평가**: ROUGE, 임베딩 유사도, 의미적 유사도 평가
4. **실용적 설계**: 프로토타입부터 프로덕션까지

## 🚀 시작하기

```bash
pip install ranx-k
```

```python
from ranx_k.tokenizers import KiwiTokenizer

tokenizer = KiwiTokenizer(method='morphs')
tokens = tokenizer.tokenize('한국어 자연어처리 도구입니다.')
print(tokens)  # ['한국어', '자연어', '처리', '도구']
```

## 📞 지원

- GitHub Issues: https://github.com/tsdata/rank-k/issues
- 이메일: ontofinance@gmail.com