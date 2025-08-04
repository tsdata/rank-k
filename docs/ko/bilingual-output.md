# 이중언어 출력 시스템

ranx-k는 국제적 접근성을 보장하면서 한국 사용자를 위한 모국어 지원을 유지하기 위해 이중언어 출력 메시지를 제공합니다.

## 개요

ranx-k의 모든 사용자 대면 출력 메시지는 일관된 이중언어 형식을 따릅니다:
```
English Text | Korean Text
```

이 접근 방식은 다음을 보장합니다:
- 국제 사용자들이 영어로 출력을 이해할 수 있음
- 한국 사용자들이 모국어 지원을 받을 수 있음
- 모든 모듈에서 일관된 인터페이스 유지

## 구현된 구성 요소

### 핵심 평가 모듈

#### 향상된 ROUGE 평가 (`enhanced_rouge.py`)
- ✅ 모든 print 문이 이중언어 형식으로 변환됨
- 예시: `"Enhanced ROUGE Evaluation Results | 향상된 ROUGE 평가 결과"`

#### 예제들

##### 기본 토크나이저 (`basic_tokenizer.py`)
- ✅ 모든 사용자 출력이 이중언어 형식으로 변환됨
- 예시:
  - `"Morphological Analysis Tokenizer | 형태소 분석 토크나이저"`
  - `"Token Count | 토큰 개수: {count}"`

##### ROUGE 평가 (`rouge_evaluation.py`)
- ✅ 모든 print 문이 이중언어 형식으로 변환됨
- 예시:
  - `"ROUGE Evaluation Example | ROUGE 평가 예제"`
  - `"Processing Time | 처리 시간: {time:.2f}초"`

##### 종합 비교 (`comprehensive_comparison.py`)
- ✅ 모든 출력 메시지가 이중언어 형식으로 변환됨
- 예시:
  - `"Performance Comparison Table | 성능 비교 테이블"`
  - `"Completed | 완료 ({time:.2f}초)"`

## 형식 가이드라인

### 표준 패턴
```python
print("English Message | Korean Message")
```

### 변수 포함
```python
print(f"English Message | Korean Message: {variable}")
```

### 진행 메시지
```python
print("✅ Completed | 완료")
print("❌ Error | 오류: {error_message}")
```

### 헤더 및 섹션
```python
print("🔤 Section Title | 섹션 제목")
print("📊 Results | 결과")
```

## 장점

1. **국제적 접근성**: 영어 우선 접근 방식으로 전 세계 사용자가 도구에 접근 가능
2. **로컬 지원**: 한국어 번역으로 모국어 지원 유지
3. **일관성**: 모든 모듈에서 균일한 형식
4. **전문적 표준**: 국제 소프트웨어 개발 관행 준수

## 구현 참고사항

- 모든 기술적 주석과 docstring은 영어로 유지 (CLAUDE.md 가이드라인 준수)
- 사용자 대면 출력 메시지만 이중언어 형식 사용
- 오류 메시지는 더 나은 디버깅을 위해 양 언어 포함
- 성능 메트릭과 타임스탬프는 이중언어 레이블 유지

## 코드 스타일 준수

이 이중언어 출력 시스템은 프로젝트의 코드 스타일 가이드라인을 준수합니다:
- 코드 내 영어 전용 주석 및 docstring
- 코드 로직에 이모지 없음 (사용자 출력에만)
- 전문적 국제 표준
- 일관된 형식 패턴

## 사용 예시

### 평가 결과
```python
print("📊 Enhanced ROUGE Evaluation Results | 향상된 ROUGE 평가 결과:")
for metric, score in results.items():
    print(f"  {metric}: {score:.3f}")
```

### 진행 추적
```python
print("🚀 Starting Evaluation | 평가 시작")
# ... 처리 ...
print("✅ Evaluation Completed | 평가 완료")
```

### 오류 처리
```python
except Exception as e:
    print(f"❌ Error | 오류: {str(e)}")
```

## 향후 개선사항

- 출력 언어 선호도를 위한 구성 옵션 고려
- 추가 언어 지원 가능성
- 이중언어 로그 메시지를 위한 로깅 시스템과의 통합