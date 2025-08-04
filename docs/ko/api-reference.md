# API 레퍼런스

## 📋 Navigation
- [← 빠른 시작](quickstart.md) | [메인](index.md) | [설치](installation.md)

ranx-k의 모든 클래스와 함수에 대한 상세한 문서입니다.

## 🔤 토크나이저 (ranx_k.tokenizers)

### KiwiTokenizer

한국어 특화 토크나이저 클래스입니다.

```python
class KiwiTokenizer:
    def __init__(self, use_stemmer=False, method='morphs', use_stopwords=True):
        """
        Parameters:
            use_stemmer (bool): rouge_score 호환성을 위한 매개변수 (사용되지 않음)
            method (str): 토큰화 방법 ('morphs' 또는 'nouns')
            use_stopwords (bool): 한국어 불용어 필터링 여부
        """
```

#### 메서드

**tokenize(text: str) -> List[str]**
- 텍스트를 토큰화합니다.
- Parameters: `text` - 토큰화할 텍스트
- Returns: 토큰 리스트

**add_stopwords(stopwords: List[str]) -> None**
- 커스텀 불용어를 추가합니다.
- Parameters: `stopwords` - 추가할 불용어 리스트

**remove_stopwords(stopwords: List[str]) -> None**
- 불용어를 제거합니다.
- Parameters: `stopwords` - 제거할 불용어 리스트

**get_stopwords() -> Set[str]**
- 현재 불용어 집합을 반환합니다.
- Returns: 불용어 집합

## 📊 평가 함수 (ranx_k.evaluation)

### simple_kiwi_rouge_evaluation

Kiwi 토크나이저 기반 간단한 ROUGE 평가를 수행합니다.

```python
def simple_kiwi_rouge_evaluation(
    retriever, 
    questions: List[str], 
    reference_contexts: List[List[str]], 
    k: int = 5
) -> Dict[str, float]:
    """
    Parameters:
        retriever: 문서 검색기 객체 (invoke 메서드 필요)
        questions: 질문 리스트
        reference_contexts: 정답 문서 리스트의 리스트
        k: 상위 k개 문서 평가
        
    Returns:
        평가 결과 딕셔너리 {
            'kiwi_rouge1@k': float,
            'kiwi_rouge2@k': float, 
            'kiwi_rougeL@k': float
        }
    """
```

### rouge_kiwi_enhanced_evaluation

검증된 rouge_score 라이브러리와 Kiwi 토크나이저를 결합한 평가입니다.

```python
def rouge_kiwi_enhanced_evaluation(
    retriever,
    questions: List[str],
    reference_contexts: List[List[str]],
    k: int = 5,
    tokenize_method: str = 'morphs',
    use_stopwords: bool = True
) -> Dict[str, float]:
    """
    Parameters:
        retriever: 문서 검색기 객체
        questions: 질문 리스트
        reference_contexts: 정답 문서 리스트의 리스트
        k: 상위 k개 문서 평가
        tokenize_method: 토큰화 방법 ('morphs' 또는 'nouns')
        use_stopwords: 불용어 필터링 여부
        
    Returns:
        평가 결과 딕셔너리 {
            'enhanced_rouge1@k': float,
            'enhanced_rouge2@k': float,
            'enhanced_rougeL@k': float
        }
    """
```

### evaluate_with_ranx_similarity

의미적 유사도를 ranx 메트릭으로 변환하여 평가합니다.

```python
def evaluate_with_ranx_similarity(
    retriever,
    questions: List[str],
    reference_contexts: List[List[str]],
    k: int = 5,
    method: str = 'embedding',
    similarity_threshold: float = 0.7
) -> Dict[str, float]:
    """
    Parameters:
        retriever: 문서 검색기 객체
        questions: 질문 리스트
        reference_contexts: 정답 문서 리스트의 리스트
        k: 상위 k개 문서 평가
        method: 유사도 계산 방법 ('embedding' 또는 'kiwi_rouge')
        similarity_threshold: 관련 문서 판정 임계값
        
    Returns:
        ranx 평가 결과 딕셔너리 {
            'hit_rate@k': float,
            'ndcg@k': float,
            'map@k': float,
            'mrr': float
        }
    """
```

### comprehensive_evaluation_comparison

모든 평가 방법을 종합적으로 비교합니다.

```python
def comprehensive_evaluation_comparison(
    retriever,
    questions: List[str],
    reference_contexts: List[List[str]],
    k: int = 5
) -> Dict[str, Dict[str, float]]:
    """
    Parameters:
        retriever: 문서 검색기 객체
        questions: 질문 리스트
        reference_contexts: 정답 문서 리스트의 리스트
        k: 상위 k개 문서 평가
        
    Returns:
        방법별 평가 결과 딕셔너리 {
            'Kiwi ROUGE': {...},
            'Enhanced ROUGE': {...},
            'Similarity ranx': {...}
        }
    """
```

## 🛠️ 유틸리티 클래스

### EmbeddingSimilarityCalculator

임베딩 기반 유사도 계산기입니다.

```python
class EmbeddingSimilarityCalculator:
    def __init__(self, model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        """
        Parameters:
            model_name: 사용할 sentence-transformers 모델명
        """
    
    def calculate_similarity_matrix(self, ref_texts: List[str], ret_texts: List[str]) -> np.ndarray:
        """
        임베딩 유사도 매트릭스를 계산합니다.
        
        Parameters:
            ref_texts: 참조 텍스트 리스트
            ret_texts: 검색된 텍스트 리스트
            
        Returns:
            코사인 유사도 매트릭스
        """
```

### KiwiRougeSimilarityCalculator

Kiwi + ROUGE 기반 유사도 계산기입니다.

```python
class KiwiRougeSimilarityCalculator:
    def __init__(self):
        """Kiwi 토크나이저와 ROUGE 메트릭을 초기화합니다."""
    
    def calculate_similarity_matrix(self, ref_texts: List[str], ret_texts: List[str]) -> np.ndarray:
        """
        Kiwi ROUGE 유사도 매트릭스를 계산합니다.
        
        Parameters:
            ref_texts: 참조 텍스트 리스트
            ret_texts: 검색된 텍스트 리스트
            
        Returns:
            ROUGE-L F1 점수 매트릭스
        """
```

## 📊 평가 메트릭 해석

### ROUGE 점수
- **ROUGE-1**: 단어 수준 겹침 (0.0-1.0)
- **ROUGE-2**: 바이그램 수준 겹침 (0.0-1.0)  
- **ROUGE-L**: 최장 공통 부분 수열 기반 (0.0-1.0)

### ranx 메트릭
- **Hit@K**: 상위 K개 중 관련 문서 발견 비율 (0.0-1.0)
- **NDCG@K**: 정규화된 할인 누적 이득 (0.0-1.0)
- **MAP@K**: 평균 정밀도 (0.0-1.0)
- **MRR**: 평균 역순위 (0.0-1.0)

## 🎯 성능 기준

| 점수 범위 | 평가 | 의미 |
|-----------|------|------|
| 0.8-1.0 | 🟢 우수 | 매우 높은 정확도 |
| 0.6-0.8 | 🟡 좋음 | 실용적 수준 |
| 0.4-0.6 | 🟠 보통 | 개선 필요 |
| 0.0-0.4 | 🔴 낮음 | 시스템 재검토 필요 |