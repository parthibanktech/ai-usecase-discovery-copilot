"""
AI Use-Case Discovery Copilot (Production-Grade)
================================================
An enterprise AI Strategy & Feasibility Copilot that converts manual business
bottlenecks into rigorously evaluated, scored AI implementation roadmaps.

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

# --- 1. Credential & Environment Verification ------------------------------
load_dotenv()
api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
if not api_key or api_key == "your_deepseek_api_key_here":
    print("WARNING: No DEEPSEEK_API_KEY found in .env file.")

# --- 2. Production Data Contracts (Pydantic Schema) ------------------------
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

session_history = []

# --- 4. Preset Industry Scenarios (Instant Demo Data) ----------------------
SCENARIOS = {
    "logistics": (
        "Supply Chain & Logistics",
        "VP of Global Operations",
        "Shipment delays caused by extreme weather are manually triaged across 15 carrier portals with 4-hour customer lag.",
        "Copying tracking numbers, checking portal statuses, manually writing delay emails to clients.",
        "Real-time delay detection and draft messages ready for 1-click human review.",
        "Balanced (0.3 - Feasibility Analysis)"
    ),
    "fintech": (
        "Commercial Banking & FinTech",
        "Head of Credit Underwriting",
        "Underwriters spend 3.5 hours per applicant manually extracting and reconciling data from unstructured tax returns, balance sheets, and bank PDFs.",
        "Manual data extraction from PDF statements into Excel debt-service coverage ratio (DSCR) models.",
        "Automated PDF financial table extraction, ratio calculation, and red-flag audit summary.",
        "Conservative (0.1 - Audit & Code)"
    ),
    "healthcare": (
        "Clinical Research & Healthcare",
        "Clinical Trial Director",
        "Eligible oncology patients face 3-week delays entering trials because oncologists must cross-reference patient histories against 40-page inclusion criteria.",
        "Reviewing oncologist clinical notes, lab records, and comparing against inclusion/exclusion rules.",
        "Instant matching score of patient EHR data against trial protocols with highlighted contraindications.",
        "Conservative (0.1 - Audit & Code)"
    ),
    "retail": (
        "E-Commerce & Retail",
        "Customer Experience Lead",
        "Customer support agents spend 45% of their workday reviewing high-volume return claims and investigating potential refund fraud.",
        "Cross-checking photo evidence, return tracking timestamps, and past customer order history.",
        "Instant return approval for low-risk orders, fraud score flagging, and auto-drafted customer resolutions.",
        "Balanced (0.3 - Feasibility Analysis)"
    )
}

def load_scenario(key: str):
    return SCENARIOS.get(key, SCENARIOS["logistics"])

# --- 5. Discovery Engine ---------------------------------------------------
def run_discovery_pipeline(domain, role, problem, task, outcome, creativity_level):
    if not problem.strip() or not domain.strip():
        yield (
            "### ⚠️ Validation Error\nPlease provide at least the **Industry Domain** and **Business Problem** to proceed.",
            None,
            get_history_markdown()
        )
        return

    temp_map = {
        "Conservative (0.1 - Audit & Code)": 0.1,
        "Balanced (0.3 - Feasibility Analysis)": 0.3,
        "Creative (0.7 - Idea Generation)": 0.7
    }
    temperature = temp_map.get(creativity_level, 0.3)
    start_time = time.time()

    yield (
        "### ⏳ Evaluating Feasibility...\n*Running LangChain LCEL chain with DeepSeek and Pydantic v2 validation...*",
        None,
        get_history_markdown()
    )

    try:
        current_api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
        llm = ChatDeepSeek(model="deepseek-chat", temperature=temperature, api_key=current_api_key)
        structured_llm = llm.with_structured_output(UseCaseAssessment)
        chain = prompt_template | structured_llm

        input_payload = {
            "domain": domain,
            "role": role or "Department Lead",
            "problem": problem,
            "task": task or "Manual data extraction and repetitive triage",
            "outcome": outcome or "Operational efficiency and cost reduction",
        }

        assessment: UseCaseAssessment = chain.invoke(input_payload)
        elapsed = round(time.time() - start_time, 2)

        # Build clean markdown brief
        report = format_markdown_report(assessment, elapsed, creativity_level)

        # Save session history
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

        export_filename = f"AI_Discovery_{assessment.domain.replace(' ', '_')}_{int(time.time())}.md"
        with open(export_filename, "w", encoding="utf-8") as f:
            f.write(report)

        yield (
            report,
            export_filename,
            get_history_markdown()
        )

    except Exception as e:
        error_msg = f"### ❌ Pipeline Execution Failed\n\n**Error Details:** `{str(e)}`\n\n*Please verify your `DEEPSEEK_API_KEY` in `.env` or Render environment settings.*"
        yield (error_msg, None, get_history_markdown())


def format_markdown_report(data: UseCaseAssessment, elapsed: float, creativity: str) -> str:
    # Quadrant indicators
    q_badge = {
        "QUICK WIN": "🟢 **QUICK WIN** (High Value / Low Complexity — *Build First!*)",
        "STRATEGIC INVESTMENT": "🔵 **STRATEGIC INVESTMENT** (High Value / High Complexity)",
        "LOW HANGING FRUIT": "🟡 **LOW HANGING FRUIT** (Low Value / Low Complexity)",
        "RECONSIDER": "🔴 **RECONSIDER** (Low Value / High Complexity — *Avoid*)"
    }.get(data.quadrant.upper(), f"**{data.quadrant}**")

    # Star rating
    stars = "⭐" * int(round(data.business_value_score / 2))

    report = f"""# 📊 AI Feasibility & Executive Brief

### 🎯 Portfolio Classification: {q_badge}

| Strategic Metric | Score | Evaluation Benchmark |
| :--- | :---: | :--- |
| **Business Value & ROI** | **{data.business_value_score} / 10** {stars} | High operational impact, cost reduction, or revenue acceleration |
| **Technical Complexity** | **{data.complexity_score} / 10** | Integration overhead, API dependencies, risk level |
| **Evaluation Speed** | **{elapsed}s** | Validated via Pydantic v2 structured output |

---

### 💡 Primary Recommended AI Use Case
> ## **{data.recommended_use_case}**
*Problem Solved:* {data.problem_summary}

---

### 🗺️ 2-Week Prototype MVP Roadmap
"""
    for idx, step in enumerate(data.prototype_roadmap, start=1):
        report += f"{idx}. **Phase {idx}**: {step}\n"

    report += "\n---\n\n### 📦 Required Internal Data Assets & APIs\n"
    for item in data.required_data_sources:
        report += f"- 🗄️ {item}\n"

    report += "\n---\n\n### 🛡️ Production Guardrails & Risk Mitigations\n"
    for risk in data.risks_and_guardrails:
        report += f"- ⚠️ {risk}\n"

    report += "\n---\n\n### 💡 Alternative Backlog Ideas\n"
    for alt in data.alternative_ideas:
        report += f"- 🔹 {alt}\n"

    report += f"\n\n*Generated by AI Use-Case Discovery Copilot • Domain: {data.domain} • Mode: {creativity.split(' ')[0]}*"
    return report


def get_history_markdown() -> str:
    if not session_history:
        return "*No use cases evaluated yet in this session. Click any scenario button above to run a demo.*"

    md = "| Time | Industry Domain | Recommended AI Use Case | Business Value | Complexity | Portfolio Quadrant |\n"
    md += "| :--- | :--- | :--- | :---: | :---: | :--- |\n"
    for item in session_history[:8]:
        md += f"| {item['timestamp']} | **{item['domain']}** | {item['use_case'][:42]}... | `{item['value']}` | `{item['complexity']}` | **{item['quadrant']}** |\n"
    return md


# --- 6. Gradio Interface Construction --------------------------------------
app_theme = gr.themes.Soft(
    primary_hue="blue",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"]
)

with gr.Blocks(title="AI Use-Case Discovery Copilot") as demo:
    
    # 1. CLEAR HEADER: WHAT THIS APP IS & EXACT INPUT ➔ OUTPUT USAGE
    with gr.Row():
        gr.Markdown("""
        # 🧠 Enterprise AI Use-Case Discovery Copilot
        
        ### 🎯 What is this app?
        An automated **AI Feasibility & Strategy Engine** that evaluates whether an operational problem is worth solving with Generative AI before wasting engineering budget.
        
        ---
        
        ### 📥 What You Provide ➔ 📤 What The Copilot Gives You:
        
        | 📥 What You Enter (Input) | 📤 What You Get Back in 10s (Deliverable) |
        | :--- | :--- |
        | 🏢 **Industry Domain & Role** *(e.g. Logistics, VP of Ops)* | 🎯 **2x2 Portfolio Classification**: Is it a **🟢 QUICK WIN** (Build first!), **🔵 STRATEGIC**, or **🔴 RECONSIDER** (Avoid)? |
        | ⚠️ **Operational Bottleneck** *(What is slow, expensive, or painful)* | 📊 **Feasibility Scores**: **Business Value / ROI (1–10)** vs. **Technical Complexity (1–10)** |
        | ✍️ **Repetitive Tasks Done by Hand** *(What humans do manually)* | 💡 **Recommended AI Use Case**: The single best AI architecture to build first |
        | 🎯 **Desired Target Outcome** *(e.g. 80% faster turnaround)* | 🗺️ **2-Week Prototype MVP Roadmap**: Step 1 ➔ Step 2 ➔ Step 3 ➔ Step 4 |
        | 🎛️ **Analysis Mode** *(Conservative, Balanced, or Creative)* | 🗄️ **Data & APIs Needed**: Exactly what internal databases, PDFs, or APIs are required |
        | | 🛡️ **Risk Guardrails**: Hallucination mitigations, security, and compliance checks |
        | | 📥 **1-Click Executive Brief (.md)**: Download a client-ready proposal to share with executives |
        """)

    gr.Markdown("---")

    # 2. INSTANT DEMO BUTTONS

    gr.Markdown("### ⚡ Click any industry scenario below to test live in 1-click:")
    with gr.Row():
        btn_logistics = gr.Button("🚚 Logistics (Shipment Delays)", variant="secondary")
        btn_fintech = gr.Button("💳 Banking (Loan Doc Extraction)", variant="secondary")
        btn_healthcare = gr.Button("🏥 Healthcare (Trial Matching)", variant="secondary")
        btn_retail = gr.Button("🛍️ Retail (Return Fraud Triage)", variant="secondary")

    gr.Markdown("---")

    # 3. MAIN WORKSPACE (2 COLUMNS)
    with gr.Row():
        
        # LEFT: Inputs
        with gr.Column(scale=5):
            gr.Markdown("### 🏢 1. Business Problem Input")
            
            with gr.Row():
                domain = gr.Textbox(
                    label="Industry Domain",
                    value="Supply Chain & Logistics",
                    placeholder="e.g. Logistics, FinTech, Healthcare",
                    scale=3
                )
                role = gr.Textbox(
                    label="Stakeholder Role",
                    value="VP of Global Operations",
                    placeholder="e.g. VP of Operations",
                    scale=2
                )

            problem = gr.Textbox(
                label="Operational Bottleneck (The Core Problem)",
                value="Shipment delays caused by extreme weather are manually triaged across 15 carrier portals with 4-hour customer lag.",
                placeholder="Describe what is slow, expensive, or painful...",
                lines=3
            )
            
            task = gr.Textbox(
                label="Repetitive Tasks Done by Hand Today",
                value="Copying tracking numbers, checking portal statuses, manually writing delay emails to clients.",
                placeholder="What specific tasks are humans doing manually?",
                lines=2
            )
            
            outcome = gr.Textbox(
                label="Desired Business Metric / Target Outcome",
                value="Real-time delay detection and draft messages ready for 1-click human review.",
                placeholder="e.g. 80% faster notification, zero missed delays",
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

        # RIGHT: Output Report
        with gr.Column(scale=7):
            gr.Markdown("### 📋 2. Executive Feasibility Brief & Roadmap")
            output_report = gr.Markdown(
                value="*Click any demo scenario button above, then click **'Run AI Feasibility Assessment'** to generate a complete roadmap.*"
            )
            download_file = gr.File(label="📥 Download Executive Brief (.md)", visible=True)

    gr.Markdown("---")

    # 4. SESSION HISTORY BENCHMARK
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 📈 Cross-Department Discovery Matrix (Session History)")
            history_table = gr.Markdown(value=get_history_markdown())

    # 5. ARCHITECTURE DETAILS
    with gr.Accordion("🔍 Under the Hood: Production Architecture & Tech Stack", open=False):
        gr.Markdown("""
        - **LangChain LCEL**: Declarative execution chain (`prompt | structured_llm`)
        - **Pydantic v2**: Strict schema validation enforcing integer bounds ($1$ to $10$) and 2x2 matrix classification
        - **DeepSeek / OpenAI Engine**: OpenAI-compatible LLM endpoint with dynamic temperature switching
        - **Cloud Deployment**: Automated Dockerfile and Render Blueprint (`render.yaml`)
        """)

    # --- Button Wiring ---
    btn_logistics.click(
        fn=lambda: load_scenario("logistics"),
        outputs=[domain, role, problem, task, outcome, creativity]
    )
    btn_fintech.click(
        fn=lambda: load_scenario("fintech"),
        outputs=[domain, role, problem, task, outcome, creativity]
    )
    btn_healthcare.click(
        fn=lambda: load_scenario("healthcare"),
        outputs=[domain, role, problem, task, outcome, creativity]
    )
    btn_retail.click(
        fn=lambda: load_scenario("retail"),
        outputs=[domain, role, problem, task, outcome, creativity]
    )

    submit_btn.click(
        fn=run_discovery_pipeline,
        inputs=[domain, role, problem, task, outcome, creativity],
        outputs=[output_report, download_file, history_table]
    )

# Launch
if __name__ == "__main__":
    port = int(os.getenv("PORT", 7860))
    server_name = os.getenv("GRADIO_SERVER_NAME", "0.0.0.0")
    print(f"Starting Production AI Discovery Copilot on {server_name}:{port} ...")
    demo.launch(server_name=server_name, server_port=port, theme=app_theme)
