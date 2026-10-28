# KB Service

This service handles document ingestion for the QuantumHound RAG application.

## Testing

The service includes comprehensive unit tests for the document ingestion workflow. To run the tests:

1. Make sure you have pytest installed:
   ```
   pip install pytest
   ```

2. Run the tests from the kb-service directory:
   ```
   cd kb-service
   pytest test_ingest_documents.py -v
   ```

The tests cover the following scenarios:

### Document Embedding Tests
- Successful document embedding creation
- Error handling for embedding failures
- Proper metadata handling for both success and error cases

### Document Storage Tests
- Successful document storage
- Error handling for storage failures
- Proper response message formatting

### Complete Workflow Tests
- End-to-end document ingestion process
- Error handling at different stages of the workflow
- Proper handling of multiple documents
- Integration with mock ChromaDB for vector storage

The test suite uses pytest fixtures to:
- Set up mock environment variables
- Provide test document data
- Mock external dependencies (ChromaDB, embedding service)

## Components

- `ingest_documents.py`
