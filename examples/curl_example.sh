#!/bin/bash

# Example curl command to send a chat request to the Flask API
curl -X POST http://localhost:5000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "test-session-1",
    "message": "What was Broadcom'\''s revenue in Q4?",
    "use_rag": true,
    "namespace": "sp500-qa-demo",
    "top_k": 10,
    "model": "gpt-4",
    "galileo_project": "custom-project-name",
    "galileo_log_stream": "custom-log-stream"
  }'

# Example curl command to send a chat request with default Galileo settings
curl -X POST http://localhost:5000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "test-session-2",
    "message": "What was Broadcom'\''s revenue in Q4 and how does it compare to Q3?"
  }'

# Example curl command to check active sessions
curl -X GET http://localhost:5000/api/sessions

# Example curl command to get a specific session's conversation history
curl -X GET http://localhost:5000/api/sessions/test-session-1

# Example curl command to delete a session
curl -X DELETE http://localhost:5000/api/sessions/test-session-1 