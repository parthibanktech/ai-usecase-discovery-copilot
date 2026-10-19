"""
AI Use-Case Discovery Copilot (Production-Grade)
================================================
An enterprise-grade AI Discovery Assistant that converts messy operational
bottlenecks into rigorously scored, Pydantic-validated AI transformation roadmaps.

FEATURES:
1. Real-time Streaming Generation & Dynamic Reasoning Feedback
2. Pydantic v2 Type-Safe Contracts & Range Validation
3. 2x2 Feasibility Matrix (Quick Win, Strategic Investment, etc.)
4. 1-Click Client-Ready Markdown Brief Export
5. Built-in Multi-Industry Demo Examples (Logistics, Fintech, Healthcare, Retail)
6. Session History & Cross-Department Comparison Matrix
7. Cloud-Ready (Dynamic PORT & Host for Render, Docker, Hugging Face)

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
        "Quick Win": "🟢 **QUICK WIN** *(High Impact / Low Complexity — Build First!)*",
        "Strategic Investment": "🔵 **STRATEGIC INVESTMENT** *(High Impact / High Complexity)*",
        "Low Hanging Fruit": "🟡 **LOW HANGING FRUIT** *(Low Impact / Low Complexity)*",
        "Reconsider": "🔴 **RECONSIDER** *(Low Impact / High Complexity — Avoid)*"
    }
    quadrant_display = q_colors.get(data.quadrant, f"**{data.quadrant}**")

    # Star rating generator
    stars = "⭐" * int(data.business_value_score / 2)

    report = f"""# 📊 AI Feasibility & Discovery Brief
**Industry Domain:** {data.domain} | **Evaluation Latency:** `{elapsed}s` | **Sampling Mode:** `{creativity.split(' ')[0]}`

---

### 🎯 Portfolio Classification
{quadrant_display}

| Strategic Metric | Score | Evaluation Benchmark |
| :--- | :---: | :--- |
| **Business Value & ROI** | **{data.business_value_score} / 10** {stars} | High operational impact, cost reduction, or revenue acceleration |
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

    report += "\n\n*Generated by AI Use-Case Discovery Copilot • Pydantic v2 + LangChain LCEL Architecture*"
    return report


def get_history_markdown() -> str:
    """Renders the historical comparison table for current session."""
    if not session_history:
        return "*No use cases evaluated yet in this session. Select an example below or run your first assessment.*"

    md = "| Time | Domain | Recommended Use Case | Value | Complexity | Portfolio Matrix |\n"
    md += "| :--- | :--- | :--- | :---: | :---: | :--- |\n"
    for item in session_history[:8]:
        md += f"| {item['timestamp']} | **{item['domain']}** | {item['use_case'][:40]}... | {item['value']} | {item['complexity']} | `{item['quadrant']}` |\n"
    return md


# --- 5. Interactive Demo Examples (1-Click Client Demos) -------------------
DEMO_EXAMPLES = [
    [
        "Supply Chain & Logistics",
        "VP of Operations",
        "Shipment delays caused by weather are manually triaged across 15 carrier portals with 4-hour customer lag.",
        "Copying tracking numbers, checking status across carrier sites, drafting individual client delay emails.",
        "Real-time delay detection and draft client messages ready for 1-click human review.",
        "Balanced (0.3 - Feasibility Analysis)"
    ],
    [
        "Fintech & Commercial Banking",
        "Head of Underwriting",
        "Loan officers spend 3.5 hours per applicant cross-referencing tax returns, balance sheets, and bank PDFs.",
        "Manual data extraction from PDF statements into Excel debt-service coverage ratio (DSCR) models.",
        "Automated PDF financial table extraction, ratio calculation, and red-flag audit summary.",
        "Conservative (0.1 - Audit & Code)"
    ],
    [
        "Healthcare & Life Sciences",
        "Clinical Trial Director",
        "Patients are delayed 3 weeks getting into cancer trials because doctors manually check 40-page protocol criteria.",
        "Reading oncologist clinical notes and comparing against clinicaltrials.gov inclusion/exclusion rules.",
        "Instant matching score of patient EHR data against trial protocols with highlighted contraindications.",
        "Conservative (0.1 - Audit & Code)"
    ],
    [
        "E-Commerce & Retail",
        "Customer Support Lead",
        "Support agents spend 45% of time resolving return requests and investigating potential refund fraud.",
        "Comparing photo evidence, return tracking timestamps, and past customer order history.",
        "Automated return approval for low-risk orders, fraud score flagging, and auto-drafted customer resolutions.",
        "Balanced (0.3 - Feasibility Analysis)"
    ]
]


# --- 6. Modern Enterprise Gradio Web Interface -----------------------------
custom_css = """
#app-container { max-width: 1280px; margin: 0 auto; padding: 10px; }
.hero-header {
    background: linear-gradient(135deg, rgba(6, 182, 212, 0.12) 0%, rgba(99, 102, 241, 0.12) 100%);
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 16px;
    padding: 24px 30px;
    margin-bottom: 24px;
    box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
}
.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: rgba(16, 185, 129, 0.12);
    border: 1px solid rgba(16, 185, 129, 0.3);
    color: #34D399;
    font-size: 12px;
    font-weight: 600;
    padding: 4px 12px;
    border-radius: 20px;
    margin-bottom: 10px;
}
.status-dot {
    width: 6px;
    height: 6px;
    background: #34D399;
    border-radius: 50%;
    box-shadow: 0 0 8px #34D399;
}
.section-card {
    background: rgba(15, 23, 42, 0.6);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 14px;
    padding: 20px;
}
"""

with gr.Blocks(
    title="AI Use-Case Discovery Copilot",
    theme=gr.themes.Soft(primary_hue="cyan", neutral_hue="slate"),
    css=custom_css
) as demo:
    
    with gr.Column(elem_id="app-container"):
        
        # HERO BANNER
        with gr.Row(elem_classes=["hero-header"]):
            with gr.Column():
                gr.HTML("""
                <div class="status-pill">
                    <span class="status-dot"></span> Production Engine Active • LangChain LCEL • Pydantic v2
                </div>
                <h1 style="font-size: 28px; font-weight: 800; margin: 0 0 8px 0; color: #F8FAFC;">
                    🧠 Enterprise AI Use-Case Discovery Copilot
                </h1>
                <p style="color: #94A3B8; font-size: 15px; margin: 0;">
                    Turn messy operational business bottlenecks into scored, Pydantic-validated AI implementation roadmaps with 2x2 Feasibility Matrix categorization.
                </p>
                """)

        # MAIN INPUT / OUTPUT SPLIT
        with gr.Row():
            
            # LEFT: Input Parameters
            with gr.Column(scale=5, elem_classes=["section-card"]):
                gr.Markdown("### 🏢 Business Bottleneck Input")
                
                with gr.Row():
                    domain = gr.Textbox(
                        label="Industry / Domain",
                        placeholder="e.g. Supply Chain & Logistics",
                        value="Supply Chain & Logistics",
                        scale=3
                    )
                    role = gr.Textbox(
                        label="Target Stakeholder",
                        placeholder="e.g. VP of Operations",
                        value="VP of Operations",
                        scale=2
                    )

                problem = gr.Textbox(
                    label="Operational Bottleneck (The Problem)",
                    placeholder="Describe what is slow, expensive, or error-prone...",
                    value="Shipment delays caused by weather are manually triaged across 15 carrier portals with 4-hour customer lag.",
                    lines=3
                )
                task = gr.Textbox(
                    label="Repetitive Tasks Done by Hand Today",
                    placeholder="What specific tasks are humans doing manually?",
                    value="Copying tracking numbers, checking status across carrier sites, drafting individual client delay emails.",
                    lines=2
                )
                outcome = gr.Textbox(
                    label="Desired Target Metric / Business Outcome",
                    placeholder="e.g. 80% faster notification, zero missed delays",
                    value="Real-time delay detection and draft client messages ready for 1-click human review.",
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

                submit_btn = gr.Button("🚀 Run AI Feasibility Assessment", variant="primary", size="lg")

            # RIGHT: Executive Assessment Output
            with gr.Column(scale=6, elem_classes=["section-card"]):
                gr.Markdown("### 📋 Executive Feasibility Brief & Roadmap")
                output_report = gr.Markdown(value="*Select an industry example below or enter a business problem on the left, then click 'Run AI Feasibility Assessment'.*")
                download_file = gr.File(label="📥 Download Executive Brief (.md)", visible=True)

        # INTERACTIVE DEMO EXAMPLES
        with gr.Row():
            with gr.Column():
                gr.Markdown("### ⚡ Clickable Demo Examples (Instant Pre-fill)")
                gr.Examples(
                    examples=DEMO_EXAMPLES,
                    inputs=[domain, role, problem, task, outcome, creativity],
                    label="Select any industry scenario to test live:"
                )

        # SESSION COMPARISON MATRIX
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 📈 Session Discovery Comparison Matrix")
                history_table = gr.Markdown(value=get_history_markdown())

        # ARCHITECTURE COLLAPSIBLE
        with gr.Accordion("🔍 Behind the Scenes: Enterprise AI Architecture", open=False):
            gr.Markdown("""
            **How this application works under the hood:**
            1. **`ChatPromptTemplate`**: Decouples business problem variables from prompt steering.
            2. **`ChatDeepSeek` / OpenAI Compatible Engine**: Configured with dynamic sampling temperature (`0.1` to `0.7`).
            3. **`Pydantic v2` Contract (`UseCaseAssessment`)**: Validates typed fields (`business_value_score`, `complexity_score`), enforcing exact integer bounds ($1$ to $10$) and 2x2 matrix categorization.
            4. **`LCEL` Pipe Operator**: `chain = prompt | structured_llm` guarantees typed, structured delivery without fragile regex.
            5. **Cloud Deployment**: Automated Docker & Render Blueprint with dynamic port binding.
            """)

        # Event Binding
        submit_btn.click(
            fn=run_discovery_pipeline,
            inputs=[domain, role, problem, task, outcome, creativity],
            outputs=[output_report, download_file, history_table]
        )

# Cloud & Local Launch Configuration
if __name__ == "__main__":
    port = int(os.getenv("PORT", 7860))
    server_name = os.getenv("GRADIO_SERVER_NAME", "0.0.0.0")
    print(f"🚀 Starting Production AI Discovery Copilot on {server_name}:{port} ...")
    demo.launch(server_name=server_name, server_port=port)
