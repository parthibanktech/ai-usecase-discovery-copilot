# 🚀 AI Use-Case Discovery Copilot (LLM Foundations)

[![CI & Code Quality](https://github.com/parthibanktech/ai-usecase-discovery-copilot/actions/workflows/ci.yml/badge.svg)](https://github.com/parthibanktech/ai-usecase-discovery-copilot/actions/workflows/ci.yml)
[![Render Ready](https://img.shields.io/badge/Render-Deploy%20Ready-46E3B7?logo=render&logoColor=white)](render.yaml)

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-1.3%2B-1C3C3C.svg?logo=langchain&logoColor=white)](https://www.langchain.com/)
[![Pydantic v2](https://img.shields.io/badge/Pydantic-v2.12-E92063.svg?logo=pydantic&logoColor=white)](https://docs.pydantic.dev/)

A production-grade, progressive curriculum and enterprise application suite for mastering LLM engineering from scratch—from raw API tokens to Pydantic structured output, LCEL pipelines, and interactive web apps.

---

## 📂 Project Structure & Learning Progression

| File | Module | Key Architectural Concepts |
| :--- | :--- | :--- |
| [`00_setup_check.py`](00_setup_check.py) | **Environment Setup** | Python 3.11+, `uv` package manager, `.env` secret hygiene |
| [`01_first_llm_call.py`](01_first_llm_call.py) | **Raw API Protocol** | OpenAI-compatible DeepSeek client, request payload, tokens vs words |
| [`02_why_a_framework.py`](02_why_a_framework.py) | **Framework Abstraction** | Raw HTTP boilerplate vs LangChain unified `.invoke()` |
| [`03_temperature.py`](03_temperature.py) | **Temperature Tuning** | Deterministic (0.0) for code/audit vs Creative (0.9) for ideation |
| [`04_system_and_user_roles.py`](04_system_and_user_roles.py) | **System vs User Roles** | Persona steering: SRE incident analysis vs Business executive summary |
| [`05_streaming.py`](05_streaming.py) | **Token Streaming** | `.stream()`, Time-To-First-Token (TTFT), real-time terminal UX |
| [`06_prompt_templates.py`](06_prompt_templates.py) | **Prompt Templates** | `ChatPromptTemplate` with `{placeholders}` for code/data decoupling |
| [`07_structured_output.py`](07_structured_output.py) | **Structured Outputs** | Pydantic `BaseModel` & `Field` for strongly-typed data contracts |
| [`08_chain_lcel.py`](08_chain_lcel.py) | **Chains (LCEL)** | Declarative pipe operator `prompt \| llm \| parser` |
| [`09_capstone_use_case_app.py`](09_capstone_use_case_app.py) | **Capstone UI App** | Interactive Gradio web interface, dynamic temperature selector |
| [`10_production_grade_app.py`](10_production_grade_app.py) | **Enterprise Discovery Copilot** | **Production-grade**: 2x2 Feasibility Matrix, 1-click report export, history tracking |
| [`llm_foundations_infographic.html`](llm_foundations_infographic.html) | **Developer Infographic** | Pixel-perfect 9-step roadmap generator with 1-click 4K PNG export |

---

## 🌐 Cloud Deployment (Render & Docker)

This repository includes native configuration for zero-downtime cloud hosting on **Render**:

### Option A: 1-Click Render Blueprint Deployment
1. Push this repository to your GitHub account.
2. In the [Render Dashboard](https://dashboard.render.com/), select **New +** ➔ **Blueprint**.
3. Connect your repository. Render automatically reads [`render.yaml`](render.yaml) and configures:
   - Python 3.11 environment
   - Dependency installation via `requirements.txt`
   - Port binding on `0.0.0.0`
4. Add your `DEEPSEEK_API_KEY` under the service's **Environment** tab.
5. Click **Apply**—your app is live on a public URL!

### Option B: Containerized Deployment (Docker)
Build and run anywhere with Docker:
```bash
docker build -t llm-discovery-copilot .
docker run -p 7860:7860 -e DEEPSEEK_API_KEY="your_api_key_here" llm-discovery-copilot
```

---

## 🤖 GitHub Actions Automation

This repository includes automated CI/CD workflows under [`.github/workflows/`](.github/workflows/):

- **[`ci.yml`](.github/workflows/ci.yml)**: Automatically runs on every push and pull request. Validates Python syntax, checks dependencies, and enforces Pydantic data contracts.
- **[`deploy.yml`](.github/workflows/deploy.yml)**: Automatically triggers Render zero-downtime deployment on push to `main` when `RENDER_DEPLOY_HOOK_URL` secret is configured.

---

## 🛠️ Local Development & Quickstart

1. **Install dependencies** (using `uv` or `pip`):
   ```bash
   uv sync
   # or: pip install -r requirements.txt
   ```

2. **Configure your API key** in `.env`:
   ```env
   DEEPSEEK_API_KEY="your_real_deepseek_api_key"
   ```

3. **Verify setup**:
   ```bash
   uv run 00_setup_check.py
   ```

4. **Run the Production-Grade Application**:
   ```bash
   uv run 10_production_grade_app.py
   ```
   Open [http://127.0.0.1:7860](http://127.0.0.1:7860) in your browser.

---

## 📖 Complete Learning Roadmap & LinkedIn Strategy

Read the complete study guide, technical architecture explanations, and copy-paste ready LinkedIn posts in:
👉 [**`LLM_FOUNDATIONS_ROADMAP_AND_LINKEDIN.md`**](LLM_FOUNDATIONS_ROADMAP_AND_LINKEDIN.md)
