"""
AI Use-Case Discovery Copilot (FastAPI + Modern Tailwind Frontend)
==================================================================
Production-grade enterprise AI strategy engine built on FastAPI, LangChain LCEL,
and Pydantic v2, serving a modern, high-conversion Vercel/Linear-style web app.

RUN:
    uv run app.py
Then open http://127.0.0.1:7860
"""

import os
import time
from datetime import datetime
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import uvicorn

from langchain_deepseek import ChatDeepSeek
from langchain_core.prompts import ChatPromptTemplate

# --- 1. Credential & Environment Verification ------------------------------
load_dotenv()
api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()

# --- 2. Production Data Contracts (Pydantic Schemas) -----------------------
class UseCaseAssessment(BaseModel):
    """Production schema for structured AI feasibility evaluation."""
    domain: str = Field(description="The enterprise industry or domain")
    problem_summary: str = Field(description="Executive 1-line summary of the core operational pain point")
    recommended_use_case: str = Field(description="The single most practical, high-ROI AI use case to build first")
    quadrant: str = Field(description="One of: 'QUICK WIN' (High Value/Low Complexity), 'STRATEGIC INVESTMENT' (High/High), 'LOW HANGING FRUIT' (Low/Low), or 'RECONSIDER' (Low/High)")
    business_value_score: int = Field(ge=1, le=10, description="Business impact & ROI from 1 to 10")
    complexity_score: int = Field(ge=1, le=10, description="Technical and integration difficulty from 1 to 10")
    alternative_ideas: list[str] = Field(description="4 other potential AI use cases for this problem")
    required_data_sources: list[str] = Field(description="Specific internal databases, APIs, or documents needed")
    prototype_roadmap: list[str] = Field(description="4 sequential, actionable steps to build the first 2-week MVP")
    risks_and_guardrails: list[str] = Field(description="2 key hallucination, security, or compliance guardrails required")


class DiscoveryRequest(BaseModel):
    domain: str
    role: str = "Department Lead"
    problem: str
    task: str = ""
    outcome: str = ""
    creativity: str = "Balanced (0.3 - Feasibility Analysis)"


# --- 3. Prompt Architecture ------------------------------------------------
system_prompt = (
    "You are a Principal AI Solutions Architect and Enterprise Transformation Consultant.\n"
    "Your mission: Evaluate operational business bottlenecks with extreme rigor, honesty, and pragmatic engineering realism.\n"
    "Avoid vague AI buzzwords. Provide concrete architectures, specific data requirements, and honest 1-10 scores.\n"
    "Categorize the initiative into the classic 2x2 portfolio matrix: QUICK WIN, STRATEGIC INVESTMENT, LOW HANGING FRUIT, or RECONSIDER."
)

human_template = (
    "Enterprise Domain: {domain}\n"
    "Target Role: {role}\n"
    "Operational Bottleneck: {problem}\n"
    "Repetitive Manual Tasks: {task}\n"
    "Desired Business Outcome: {outcome}\n\n"
    "Provide a complete, structured AI Feasibility Assessment matching the requested schema."
)

prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", human_template),
])


# --- 4. FastAPI Application Initialization ---------------------------------
app = FastAPI(
    title="AI Use-Case Discovery Copilot",
    description="Enterprise AI Feasibility Engine powered by LangChain LCEL and Pydantic v2",
    version="2.0.0"
)

# Ensure static directory exists
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def serve_index():
    index_file = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {"message": "AI Use-Case Discovery Copilot API is running. index.html not found."}


@app.post("/api/discover")
def discover_use_case(req: DiscoveryRequest):
    if not req.domain.strip() or not req.problem.strip():
        raise HTTPException(status_code=400, detail="Domain and Problem are required fields.")

    temp_map = {
        "Conservative (0.1 - Audit & Code)": 0.1,
        "Balanced (0.3 - Feasibility Analysis)": 0.3,
        "Creative (0.7 - Idea Generation)": 0.7
    }
    temperature = temp_map.get(req.creativity, 0.3)
    start_time = time.time()

    try:
        current_api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
        if not current_api_key or current_api_key == "your_deepseek_api_key_here":
            raise HTTPException(status_code=500, detail="DEEPSEEK_API_KEY is not configured in .env or cloud environment.")

        llm = ChatDeepSeek(model="deepseek-chat", temperature=temperature, api_key=current_api_key)
        structured_llm = llm.with_structured_output(UseCaseAssessment)
        chain = prompt_template | structured_llm

        input_payload = {
            "domain": req.domain,
            "role": req.role or "Operations Director",
            "problem": req.problem,
            "task": req.task or "Manual spreadsheet entry and portal lookups",
            "outcome": req.outcome or "80% reduction in turnaround time",
        }

        assessment: UseCaseAssessment = chain.invoke(input_payload)
        elapsed = round(time.time() - start_time, 2)

        # Build clean markdown brief
        report_md = f"""# AI Feasibility Brief: {assessment.recommended_use_case}
**Industry:** {assessment.domain} | **Portfolio Classification:** {assessment.quadrant} | **Latency:** {elapsed}s

## 1. Feasibility Scores
- **Business Value / ROI:** {assessment.business_value_score} / 10
- **Technical Complexity:** {assessment.complexity_score} / 10

## 2. Core Operational Problem
{assessment.problem_summary}

## 3. Two-Week Prototype MVP Roadmap
"""
        for idx, step in enumerate(assessment.prototype_roadmap, 1):
            report_md += f"{idx}. **Phase {idx}**: {step}\n"

        report_md += "\n## 4. Required Internal Data Assets & APIs\n"
        for item in assessment.required_data_sources:
            report_md += f"- {item}\n"

        report_md += "\n## 5. Production Guardrails & Mitigations\n"
        for risk in assessment.risks_and_guardrails:
            report_md += f"- {risk}\n"

        report_md += "\n## 6. Alternative Ideas\n"
        for alt in assessment.alternative_ideas:
            report_md += f"- {alt}\n"

        return {
            "status": "success",
            "elapsed": elapsed,
            "assessment": assessment.model_dump(),
            "report_markdown": report_md
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "ai-discovery-copilot", "engine": "LangChain + Pydantic v2"}


# --- 5. Application Launch -------------------------------------------------
if __name__ == "__main__":
    port = int(os.getenv("PORT", 7860))
    server_name = os.getenv("GRADIO_SERVER_NAME", "0.0.0.0")
    print(f"Starting Production AI Discovery Copilot Web App on http://{server_name}:{port} ...")
    uvicorn.run(app, host=server_name, port=port)
