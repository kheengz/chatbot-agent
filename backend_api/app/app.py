from fastapi import FastAPI, Response
from fastapi.responses import JSONResponse
from utils.llm import ask_the_llm
# add ddtrace import here
# enable Agent Observability
from ddtrace.llmobs import LLMObs

# add ddtrace method to enable Agent Observability here
import os

LLMObs.enable(
  ml_app=os.getenv("DD_LLMOBS_ML_APP"),
  api_key=os.getenv("DD_API_KEY"),
  site=os.getenv("DD_SITE"),
)

app = FastAPI()

@app.get("/ask")
def ask(prompt: str | None):
    """
    Endpoint to send a prompt to the LLM and return a generated response.
    """
    if not prompt:
        return Response(
            content="Did you have a question?", media_type="application/json"
        )
    response = ask_the_llm(prompt)
    return Response(content=response, media_type="application/json")


@app.get("/health")
def health():
    """
    Simple health check endpoint that returns a 200 status code.
    """
    return JSONResponse(content={"status": "ok"}, status_code=200)
