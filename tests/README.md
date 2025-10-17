# Tests Directory

This directory contains all test files for the project.

## Test Files

### `test_azure_connection.py`
Tests Azure OpenAI API connections for both chat and embedding models.

**Usage:**
```bash
python tests/test_azure_connection.py
```

**What it tests:**
- GPT-4o-mini chat model connectivity
- Text embedding model connectivity
- API key validity
- Endpoint configuration

### `test_cluster.py`
Tests clustering algorithms used for innovation deduplication.

**Usage:**
```bash
python tests/test_cluster.py
```

## Running All Tests

```bash
# Run individual test
python tests/test_azure_connection.py

# Or run from project root
python -m tests.test_azure_connection
```

## Test Requirements

All tests require:
- Valid `.env` configuration file
- Active Azure OpenAI API keys
- Internet connection for API calls
