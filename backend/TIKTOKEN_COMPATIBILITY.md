# Tiktoken Compatibility with GPT-4.1 and Newer Models

## Overview

This document explains the compatibility of tiktoken with GPT-4.1 and other
newer OpenAI models, and how our token counting implementation handles them.

## Current Status (February 2025)

✅ **tiktoken 0.9.0** (latest) supports newer models including:

- o3-mini
- o1, o1-mini, o1-preview
- GPT-4o variants

❓ **GPT-4.1** is not yet officially supported by tiktoken, but our
implementation handles this gracefully with fallback mechanisms.

## Encoding Differences

### Model Encoding Mapping

| Model Family | Encoding | Notes |
|--------------|----------|-------|
| GPT-4, GPT-3.5-turbo | `cl100k_base` | Standard encoding |
| o1, o1-mini, o3-mini | `o200k_base` | Newer encoding with larger vocabulary |
| GPT-4.1 (when available) | TBD | Will likely use `o200k_base` or newer |

### Token Count Differences

For the same text, different encodings may produce slightly different token
counts:

```python
text = "Hello, this is a test for GPT-4.1 compatibility."

# GPT-4 (cl100k_base): 15 tokens
# o3-mini (o200k_base): 14 tokens
```

## Implementation Features

### Automatic Model Detection

Our `TokenCounter` class automatically detects the correct encoding for
supported models:

```python
from app.utils.token_counter import TokenCounter

counter = TokenCounter()

# Automatically uses correct encoding for each model
gpt4_tokens = counter.count_tokens_for_model(text, "gpt-4")        # cl100k_base
o3_tokens = counter.count_tokens_for_model(text, "o3-mini")        # o200k_base
```

### Graceful Fallback

For unsupported models (like GPT-4.1), the implementation falls back to the
default encoding:

```python
# GPT-4.1 not yet supported, falls back to cl100k_base
gpt41_tokens = counter.count_tokens_for_model(text, "gpt-4.1")
```

### Model Compatibility Checking

New helper methods to check model support:

```python
# Check if a model is supported
is_supported = TokenCounter.is_model_supported("gpt-4.1")  # False

# Get detailed encoding information
info = TokenCounter.get_model_encoding_info("o3-mini")
# Returns: {
#     "model_name": "o3-mini",
#     "encoding_name": "o200k_base",
#     "supported": True,
#     "max_token_value": 200018,
#     "vocab_size": 200019
# }
```

## Usage Recommendations

### For Current Implementation

1. **Use model-specific counting** when possible:

   ```python
   tokens = counter.count_tokens_for_model(text, model_name)
   ```

2. **Check model support** before processing:

   ```python
   if TokenCounter.is_model_supported(model_name):
       tokens = counter.count_tokens_for_model(text, model_name)
   else:
       # Use fallback or default counting
       tokens = counter.count_tokens(text)
   ```

### For GPT-4.1 Preparation

1. **Keep tiktoken updated**:

   ```bash
   pip install --upgrade tiktoken
   ```

2. **Monitor tiktoken releases** for GPT-4.1 support

3. **Use fallback gracefully** - our implementation already handles
   unsupported models

## Document Processing Integration

The token counting is integrated into our document processing pipeline:

- **Document parsing**: Automatically counts tokens for parsed content
- **Metadata storage**: Token counts are stored in `DocumentMetadata.total_tokens`
- **API responses**: Token information is included in processing stats
- **Cost estimation**: Can estimate API costs based on token counts

## Testing

Comprehensive tests ensure compatibility:

```bash
# Run all token counter tests
python -m pytest tests/test_token_counter.py -v

# Test specific newer model compatibility
python -m pytest tests/test_token_counter.py::TestTokenCounter::test_newer_models_compatibility -v
```

## Future Considerations

When GPT-4.1 becomes available:

1. **Automatic support**: Our implementation will automatically work once
   tiktoken adds support
2. **No code changes needed**: The fallback mechanism ensures continuity
3. **Encoding updates**: May use `o200k_base` or a newer encoding
4. **Token count accuracy**: Will become more accurate once native support
   is added

## Dependencies

- `tiktoken>=0.9.0` - Latest version with newer model support
- Automatic fallback for unsupported models
- Comprehensive error handling and logging
