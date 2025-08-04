# Claude Code Guidelines for ranx-k

This file contains development guidelines and conventions for the ranx-k project when working with Claude Code.

## Code Style Guidelines

### Comments and Documentation
- **All code comments MUST be written in English**
- **All docstrings MUST be written in English** 
- **All inline comments MUST be written in English**
- **No emojis in code comments, docstrings, or variable names**
- User-facing documentation can be multilingual (English primary, Korean secondary)
- Example scripts and demo content can contain Korean text and emojis in print statements

### Variable Names and Functions
- Use English names for all variables, functions, classes
- Use descriptive names that clearly indicate purpose
- Follow Python PEP 8 naming conventions
- No emojis in variable names or function names

### Examples

✅ **Correct:**
```python
class KiwiTokenizer:
    """Korean tokenizer using Kiwi morphological analyzer."""
    
    def __init__(self, method='morphs'):
        # Initialize Kiwi tokenizer with specified method
        self.kiwi = Kiwi()
        self.method = method  # 'morphs' or 'nouns'
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize Korean text using Kiwi analyzer.
        
        Args:
            text: Input Korean text to tokenize
            
        Returns:
            List of tokens
        """
        # Extract morphemes or nouns based on method
        if self.method == 'morphs':
            return self._extract_morphs(text)
        else:
            return self._extract_nouns(text)
```

❌ **Incorrect:**
```python
class KiwiTokenizer:
    """🔤 Kiwi 형태소 분석기를 사용한 한국어 토크나이저."""
    
    def __init__(self, method='morphs'):
        # 🚀 Kiwi 토크나이저를 지정된 방법으로 초기화
        self.kiwi = Kiwi()
        self.method = method  # 'morphs' 또는 'nouns'
```

## Documentation Standards

### Multilingual Documentation Structure
```
docs/
├── index.md (English default with language selector)
├── README.md (English)
├── README.ko.md (Korean)
├── en/ (English documentation - PRIMARY)
│   ├── installation.md
│   ├── quickstart.md
│   └── api-reference.md
└── ko/ (Korean documentation - SECONDARY)
    ├── installation.md
    ├── quickstart.md
    └── api-reference.md
```

### Language Priority
1. **English**: Primary language for all technical documentation
2. **Korean**: Secondary language for user accessibility
3. All technical docs must exist in English first
4. Korean translations are optional but recommended for user-facing content

### API Documentation
- All public APIs must have English docstrings
- Follow Google or NumPy docstring format
- Include parameter types and return types
- Provide usage examples
- No emojis in docstrings

### Documentation Links
- Each document should include language selector at the top
- Cross-references should maintain language consistency
- Example: `[English](file.md) | [한국어](../ko/file.md)`

## Version Control Guidelines

### Commit Messages
- Write commit messages in English only
- Use conventional commit format when possible
- Be descriptive about changes made
- **NO emojis in commit messages**
- **DO NOT include Claude Code attribution in commit messages**

✅ **Good commit messages:**
```
feat: add BGE-M3 embedding model support
fix: resolve Korean tokenization encoding issue  
docs: add English documentation structure
test: add comprehensive evaluation tests
refactor: improve error handling in similarity calculator
```

❌ **Avoid:**
```
feat: ✨ add BGE-M3 support 🚀
fix: 🐛 Korean tokenization issue
docs: 📚 add English docs

🤖 Generated with [Claude Code](https://claude.ai/code)
Co-Authored-By: Claude <noreply@anthropic.com>
```

### Git Push Policy
- **Never include Claude Code attribution in commits**
- Remove any auto-generated Claude attribution before pushing
- Keep commit history clean and professional
- Focus on technical changes rather than authorship attribution

## Testing Guidelines

### Test Code Standards
- Test function names in English
- Test comments and assertions in English
- Test data can contain Korean strings for realistic testing
- No emojis in test code
- Use descriptive test names that explain what is being tested

```python
def test_kiwi_tokenizer_morphs_method():
    """Test KiwiTokenizer with morphs tokenization method."""
    tokenizer = KiwiTokenizer(method='morphs')
    
    # Test with Korean text containing various morphemes
    korean_text = "자연어처리는 인공지능의 핵심 기술입니다."
    tokens = tokenizer.tokenize(korean_text)
    
    # Verify tokenization results
    assert isinstance(tokens, list)
    assert len(tokens) > 0
    assert '자연어처리' in tokens
```

## Error Handling Standards

### Exception Messages
- Use English for all exception messages
- Provide helpful context and solutions
- Use appropriate exception types
- No emojis in error messages

```python
if not OPENAI_AVAILABLE:
    raise ImportError(
        "OpenAI library required for OpenAI embeddings. "
        "Install with: pip install openai"
    )
```

### Logging
- Log messages in English
- Use appropriate log levels
- Include relevant context for debugging

## Performance Guidelines

### Memory Management
- Use generators for large datasets when possible
- Implement batch processing for API calls
- Cache expensive computations when appropriate
- Document memory requirements for large models

### API Integration
- Implement retry logic for external API calls
- Provide fallback methods when possible
- Handle rate limiting gracefully
- Estimate costs for paid APIs (OpenAI, etc.)

## Security Guidelines

### API Keys and Secrets
- Never hardcode API keys in source code
- Use environment variables or secure configuration files
- Provide clear instructions for secure key management
- Add sensitive files to .gitignore

```python
# Correct approach
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    raise ValueError(
        "OpenAI API key required. Set OPENAI_API_KEY environment variable"
    )
```

## File Organization

### Import Order
1. Standard library imports
2. Third-party imports  
3. Local application imports

### Optional Dependencies
- Handle optional dependencies gracefully
- Provide clear error messages for missing packages
- Use try/except blocks for optional imports

```python
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
```

## Example Scripts Standards

### User-Facing Output Standards

#### Bilingual Output System
- **All user-facing print statements MUST use bilingual format: `"English Text | Korean Text"`**
- This ensures international accessibility while maintaining Korean language support
- Format applies to all example scripts, evaluation outputs, and user messages
- Technical code comments and docstrings remain English-only

#### Implementation Guidelines
```python
def main():
    """Main evaluation function."""
    # Use bilingual format for all user output
    print("🚀 BGE-M3 Korean RAG Evaluation Example | BGE-M3 모델을 사용한 한국어 RAG 평가 예제")
    
    # Initialize tokenizer with proper English comments
    tokenizer = KiwiTokenizer(method='morphs')
    
    try:
        results = evaluate_model(tokenizer)
        print("✅ Evaluation Completed | 평가 완료!")
        print(f"📊 Results | 결과: {results}")
    except Exception as e:
        print(f"❌ Evaluation Failed | 평가 실패: {e}")
```

#### Bilingual Format Examples
```python
# Progress messages
print("🚀 Starting Evaluation | 평가 시작")
print("✅ Completed | 완료")
print("❌ Error | 오류: {error_message}")

# Results and metrics
print("📊 Performance Results | 성능 결과:")
print(f"Processing Time | 처리 시간: {time:.2f}초")
print(f"Token Count | 토큰 개수: {count}")

# Section headers
print("1️⃣ Basic Tokenizer Example | 기본 토크나이저 예제")
print("📈 Performance Comparison | 성능 비교")
```

#### Benefits of Bilingual Output
- **International Accessibility**: English-first approach for global users
- **Local Support**: Korean translations for native speakers  
- **Consistency**: Uniform format across all modules
- **Professional Standards**: Follows international development practices

## Summary

**Core Rules:**
1. All code comments and docstrings in English
2. No emojis in code, comments, docstrings, or variable names
3. English variable and function names following PEP 8
4. English primary documentation with Korean secondary
5. **All user-facing print statements MUST use bilingual format: `"English | Korean"`**
6. No Claude Code attribution in commits, code, or documentation
7. Keep Git history clean and professional
8. Handle optional dependencies gracefully
9. Provide clear error messages and troubleshooting info

**Documentation Structure:**
- English documentation is primary and required
- Korean documentation is secondary and optional
- All technical APIs documented in English first
- User guides can be bilingual for accessibility

**Output Standards:**
- All user-facing messages use bilingual format for international accessibility
- Technical code remains English-only for maintainability
- Consistent `"English | Korean"` pattern across all modules
- Error messages, progress indicators, and results all follow bilingual format

**Remember**: The goal is to maintain professional, readable, and maintainable code that follows international development standards while serving Korean language processing needs and ensuring global accessibility through bilingual output.