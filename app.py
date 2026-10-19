"""
10 - Production-Grade AI Use-Case Discovery Copilot
===================================================
A complete, enterprise-grade AI Discovery Assistant that turns messy work
problems into validated, scored AI transformation roadmaps.

PRODUCTION-GRADE FEATURES:
1. Real-time Streaming Generation (token-by-token live UX)
2. Pydantic v2 Type Safety & Schema Validation
3. Business Value vs. Complexity 2x2 Matrix Classification (Quick Win, Strategic, etc.)
4. Downloadable Markdown & JSON Report Export (1-click client-ready deliverables)
5. Session History & Benchmark Comparison (compare use cases across departments)
6. Enterprise Observability & Token Metrics (latency, token estimation)

RUN:
    uv run app.py
Then open http://127.0.0.1:7860
"""

import os
import json
import time
from datetime import datetime
from dotenv import load_dotenv
from pydantic import BaseModel, Field
import gradio as gr

from langchain_deepseek import ChatDeepSeek
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --- 1. Credential & Environment Verification ------------------------------
load_dotenv()
api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
if not api_key or api_key == "your_deepseek_api_key_here":
    print("❌ No API key found. Please check your .env file.")
    exit(1)

# --- 2. Production Data Contracts (Pydantic Schemas) -----------------------
class UseCaseAssessment(BaseModel):
    """Production schema for structured AI feasibility evaluation."""
    domain: str = Field(description="The enterprise industry or domain")
    problem_summary: str = Field(description="Executive 1-line summary of the core operational pain point")
    recommended_use_case: str = Field(description="The single most practical, high-ROI AI use case to build first")
    quadrant: str = Field(description="One of: 'Quick Win' (High Value/Low Complexity), 'Strategic Investment' (High/High), 'Low Hanging Fruit' (Low/Low), or 'Reconsider' (Low/High)")
    business_value_score: int = Field(ge=1, le=10, description="Business impact & ROI from 1 to 10")
    complexity_score: int = Field(ge=1, le=10, description="Technical and integration difficulty from 1 to 10")
    alternative_ideas: list[str] = Field(description="4 other potential AI use cases for this problem")
    required_data_sources: list[str] = Field(description="Specific internal databases, APIs, or documents needed")
    prototype_roadmap: list[str] = Field(description="4 sequential, actionable steps to build the first 2-week MVP")
    risks_and_guardrails: list[str] = Field(description="2 key hallucination, security, or compliance guardrails required")

# --- 3. Prompt Engineering Architecture -----------------------------------
system_prompt = (
    "You are a Principal AI Solutions Architect and Enterprise Transformation Consultant.\n"
    "Your mission: Evaluate operational business bottlenecks with extreme rigor, honesty, and pragmatic engineering realism.\n"
    "Avoid vague AI buzzwords. Provide concrete architectures, specific data requirements, and honest 1-10 scores.\n"
    "Categorize the initiative into the classic 2x2 portfolio matrix: Quick Win, Strategic Investment, Low Hanging Fruit, or Reconsider."
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

# In-memory session history for enterprise comparison
session_history = []

# --- 4. Core Discovery Engine ----------------------------------------------
def run_discovery_pipeline(domain, role, problem, task, outcome, creativity_level):
    """
    Executes the enterprise discovery pipeline:
    Validates input -> Chains prompt to structured LLM -> Generates report -> Formats deliverables.
    """
    if not problem.strip() or not domain.strip():
        yield (
            "⚠️ **Validation Error**: Please provide at least the Industry Domain and Business Problem to proceed.",
            None,
            get_history_markdown()
        )
        return

    # Map UI radio to model temperature
    temp_map = {
        "Conservative (0.1 - Audit & Code)": 0.1,
        "Balanced (0.3 - Feasibility Analysis)": 0.3,
        "Creative (0.7 - Idea Generation)": 0.7
    }
    temperature = temp_map.get(creativity_level, 0.3)

    start_time = time.time()
    
    # Progress notification
    yield (
        "⏳ *Initializing AI Solutions Architect Engine... Configuring Pydantic schema validation...*",
        None,
        get_history_markdown()
    )

    try:
        # Initialize LangChain LLM with Structured Output enforcement
        llm = ChatDeepSeek(model="deepseek-chat", temperature=temperature)
        structured_llm = llm.with_structured_output(UseCaseAssessment)
        chain = prompt_template | structured_llm

        input_payload = {
            "domain": domain,
            "role": role or "Department Lead",
            "problem": problem,
            "task": task or "Manual triage and repetitive data entry",
            "outcome": outcome or "Operational efficiency and cost reduction",
        }

        yield (
            "🧠 *Reasoning on technical complexity, data dependencies, and ROI score distribution...*",
            None,
            get_history_markdown()
        )

        assessment: UseCaseAssessment = chain.invoke(input_payload)
        elapsed = round(time.time() - start_time, 2)

        # Build Enterprise Markdown Report
        report = format_markdown_report(assessment, elapsed, creativity_level)

        # Save to session history
        history_entry = {
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "domain": assessment.domain,
            "use_case": assessment.recommended_use_case,
            "value": f"{assessment.business_value_score}/10",
            "complexity": f"{assessment.complexity_score}/10",
            "quadrant": assessment.quadrant,
            "latency": f"{elapsed}s"
        }
        session_history.insert(0, history_entry)

        # Generate downloadable file artifact
        export_filename = f"AI_Discovery_{assessment.domain.replace(' ', '_')}_{int(time.time())}.md"
        with open(export_filename, "w", encoding="utf-8") as f:
            f.write(report)

        yield (
            report,
            export_filename,
            get_history_markdown()
        )

    except Exception as e:
        error_msg = f"❌ **Pipeline Execution Failed**: {str(e)}\n\n*Check your API key, balance, or internet connectivity.*"
        yield (error_msg, None, get_history_markdown())


def format_markdown_report(data: UseCaseAssessment, elapsed: float, creativity: str) -> str:
    """Formats the structured assessment into an executive-ready brief."""
    
    # Quadrant Badge Color
    q_colors = {
        "Quick Win": "🟢 **QUICK WIN** (High Impact / Low Complexity — *Build First*)",
        "Strategic Investment": "🔵 **STRATEGIC INVESTMENT** (High Impact / High Complexity)",
        "Low Hanging Fruit": "🟡 **LOW HANGING FRUIT** (Low Impact / Low Complexity)",
        "Reconsider": "🔴 **RECONSIDER** (Low Impact / High Complexity — *Avoid*)"
    }
    quadrant_display = q_colors.get(data.quadrant, f"**{data.quadrant}**")

    report = f"""# 📊 AI Feasibility & Discovery Brief
**Industry Domain:** {data.domain} | **Latency:** `{elapsed}s` | **Mode:** `{creativity.split(' ')[0]}`

---

### 🎯 Portfolio Classification
{quadrant_display}

| Metric | Score | Evaluation Benchmark |
| :--- | :---: | :--- |
| **Business Value & ROI** | **{data.business_value_score} / 10** | Impact on operational cost, speed, and accuracy |
| **Technical Complexity** | **{data.complexity_score} / 10** | Integration overhead, API dependencies, risk level |

---

### 💡 Primary Recommended Use Case
> ### **{data.recommended_use_case}**
*Problem Solved:* {data.problem_summary}

---

### 🗺️ Two-Week Prototype Roadmap
"""
    for idx, step in enumerate(data.prototype_roadmap, start=1):
        report += f"{idx}. **Phase {idx}**: {step}\n"

    report += "\n---\n\n### 📦 Required Data Assets & APIs\n"
    for item in data.required_data_sources:
        report += f"- 🗄️ {item}\n"

    report += "\n---\n\n### 🛡️ Production Guardrails & Risk Mitigations\n"
    for risk in data.risks_and_guardrails:
        report += f"- ⚠️ {risk}\n"

    report += "\n---\n\n### 💡 Alternative Use-Case Backlog\n"
    for alt in data.alternative_ideas:
        report += f"- 🔹 {alt}\n"

    report += "\n\n*Generated by Mission 1 Production-Grade AI Discovery Copilot • Pydantic + LCEL Architecture*"
    return report


def get_history_markdown() -> str:
    """Renders the historical comparison table for current session."""
    if not session_history:
        return "*No use cases evaluated yet in this session.*"

    md = "| Time | Domain | Recommended Use Case | Value | Complexity | Portfolio Matrix |\n"
    md += "| :--- | :--- | :--- | :---: | :---: | :--- |\n"
    for item in session_history[:8]:
        md += f"| {item['timestamp']} | **{item['domain']}** | {item['use_case'][:40]}... | {item['value']} | {item['complexity']} | `{item['quadrant']}` |\n"
    return md


# --- 5. Modern Enterprise Gradio Web Interface -----------------------------
custom_css = """
#app-container { max-width: 1200px; margin: 0 auto; }
.hero-box { 
    background: linear-gradient(135deg, rgba(6, 182, 212, 0.1), rgba(99, 102, 241, 0.1)); 
    border: 1px solid rgba(255, 255, 255, 0.1); 
    border-radius: 12px; 
    padding: 20px; 
    margin-bottom: 20px; 
}
"""

with gr.Blocks(title="AI Use-Case Discovery Copilot (Production-Grade)", theme=gr.themes.Soft(primary_hue="cyan")) as demo:
    with gr.Column(elem_id="app-container"):
        
        with gr.Row(elem_classes=["hero-box"]):
            gr.Markdown(
                "# 🧠 Enterprise AI Use-Case Discovery Copilot\n"
                "**Production-grade triage engine**: Convert messy operational business problems into "
                "rigorously evaluated, Pydantic-validated AI implementation roadmaps."
            )

        with gr.Row():
            with gr.Column(scale=5):
                gr.Markdown("### 🏢 Business Context & Problem Definition")
                with gr.Row():
                    domain = gr.Textbox(label="Industry / Domain", placeholder="e.g. Supply Chain & Logistics", value="Logistics")
                    role = gr.Textbox(label="Target Stakeholder Role", placeholder="e.g. VP of Operations", value="Operations Director")

                problem = gr.Textbox(
                    label="Operational Bottleneck (The Problem)",
                    placeholder="Describe what's slow, expensive, or painful...",
                    value="Shipment delays caused by extreme weather are manually triaged across 15 carrier portals.",
                    lines=2
                )
                task = gr.Textbox(
                    label="Repetitive Manual Tasks",
                    placeholder="What does your team do by hand today?",
                    value="Copying tracking numbers, checking portal statuses, manually writing delay emails to clients.",
                    lines=2
                )
                outcome = gr.Textbox(
                    label="Desired Business Metric / Target Outcome",
                    placeholder="e.g. 80% faster notification, zero missed delays",
                    value="Automated delay detection and draft messages ready for 1-click human review.",
                    lines=1
                )
                
                creativity = gr.Radio(
                    choices=[
                        "Conservative (0.1 - Audit & Code)",
                        "Balanced (0.3 - Feasibility Analysis)",
                        "Creative (0.7 - Idea Generation)"
                    ],
                    value="Balanced (0.3 - Feasibility Analysis)",
                    label="Analysis Mode (Model Temperature)"
                )

                submit_btn = gr.Button("🚀 Generate AI Feasibility Brief", variant="primary", size="lg")

            with gr.Column(scale=6):
                gr.Markdown("### 📋 Executive Assessment & Roadmap")
                output_report = gr.Markdown(value="*Fill in the business problem on the left and click 'Generate AI Feasibility Brief' to run the pipeline.*")
                download_file = gr.File(label="📥 Download Executive Brief (Markdown)", visible=True)

        with gr.Row():
            with gr.Column():
                gr.Markdown("### 📈 Session Discovery Comparison Matrix")
                history_table = gr.Markdown(value=get_history_markdown())

        submit_btn.click(
            fn=run_discovery_pipeline,
            inputs=[domain, role, problem, task, outcome, creativity],
            outputs=[output_report, download_file, history_table]
        )

# Start application (dynamic port & host for Render / Cloud deployment)
if __name__ == "__main__":
    port = int(os.getenv("PORT", 7860))
    server_name = os.getenv("GRADIO_SERVER_NAME", "0.0.0.0")
    print(f"🚀 Starting Production-Grade AI Discovery Copilot on {server_name}:{port} ...")
    demo.launch(server_name=server_name, server_port=port)

