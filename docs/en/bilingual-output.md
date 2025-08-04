# Bilingual Output System

ranx-k provides bilingual output messages to ensure international accessibility while maintaining Korean language support for local users.

## Overview

All user-facing output messages in ranx-k follow a consistent bilingual format:
```
English Text | Korean Text
```

This approach ensures that:
- International users can understand the output in English
- Korean users have native language support
- The interface remains consistent across all modules

## Implemented Components

### Core Evaluation Modules

#### Enhanced ROUGE Evaluation (`enhanced_rouge.py`)
- ✅ All print statements converted to bilingual format
- Example: `"Enhanced ROUGE Evaluation Results | 향상된 ROUGE 평가 결과"`

#### Examples

##### Basic Tokenizer (`basic_tokenizer.py`)
- ✅ All user output converted to bilingual format
- Examples:
  - `"Morphological Analysis Tokenizer | 형태소 분석 토크나이저"`
  - `"Token Count | 토큰 개수: {count}"`

##### ROUGE Evaluation (`rouge_evaluation.py`)
- ✅ All print statements converted to bilingual format
- Examples:
  - `"ROUGE Evaluation Example | ROUGE 평가 예제"`
  - `"Processing Time | 처리 시간: {time:.2f}초"`

##### Comprehensive Comparison (`comprehensive_comparison.py`)
- ✅ All output messages converted to bilingual format
- Examples:
  - `"Performance Comparison Table | 성능 비교 테이블"`
  - `"Completed | 완료 ({time:.2f}초)"`

## Format Guidelines

### Standard Pattern
```python
print("English Message | Korean Message")
```

### With Variables
```python
print(f"English Message | Korean Message: {variable}")
```

### Progress Messages
```python
print("✅ Completed | 완료")
print("❌ Error | 오류: {error_message}")
```

### Headers and Sections
```python
print("🔤 Section Title | 섹션 제목")
print("📊 Results | 결과")
```

## Benefits

1. **International Accessibility**: English-first approach makes the tool accessible to global users
2. **Local Support**: Korean translations maintain native language support
3. **Consistency**: Uniform format across all modules
4. **Professional Standards**: Follows international software development practices

## Implementation Notes

- All technical comments and docstrings remain in English (following CLAUDE.md guidelines)
- Only user-facing output messages use bilingual format
- Error messages include both languages for better debugging
- Performance metrics and timestamps maintain bilingual labels

## Code Style Compliance

This bilingual output system complies with the project's code style guidelines:
- English-only comments and docstrings in code
- No emojis in code logic (only in user output)
- Professional international standards
- Consistent formatting patterns

## Usage Examples

### Evaluation Results
```python
print("📊 Enhanced ROUGE Evaluation Results | 향상된 ROUGE 평가 결과:")
for metric, score in results.items():
    print(f"  {metric}: {score:.3f}")
```

### Progress Tracking
```python
print("🚀 Starting Evaluation | 평가 시작")
# ... processing ...
print("✅ Evaluation Completed | 평가 완료")
```

### Error Handling
```python
except Exception as e:
    print(f"❌ Error | 오류: {str(e)}")
```

## Future Enhancements

- Consider adding configuration option for output language preference
- Potential support for additional languages
- Integration with logging systems for bilingual log messages