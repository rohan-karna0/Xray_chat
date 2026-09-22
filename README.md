# MedRAX: Chest X-ray Reasoning Agent

**Author:** Rohan Karna
**Repository:** [github.com/rohan-karna0/MedRAX](https://github.com/rohan-karna0/MedRAX)

MedRAX is a medical AI application for analyzing chest X-ray images and DICOM studies through a multimodal reasoning agent. It combines a Gradio interface, configurable medical imaging tools, and an OpenAI-compatible vision-language model endpoint.

> This project is for research and demonstration only. It is not a medical diagnostic system.

## My Implementation

I reworked the application around an open-source Qwen vision-language model and built a configurable workflow for image-based medical reasoning.

Key work in this version:

- Qwen2.5-VL tools model as the default reasoning model
- OpenAI-compatible support for local and hosted model endpoints
- Gradio chat interface for image and DICOM uploads
- DICOM conversion and browser-friendly image previews
- Selective initialization of medical imaging tools
- ChestAgentBench evaluation script with JSONL result output
- Automatic port selection for the Gradio server
- Environment-based configuration for models, API endpoints, and runtime options

## Features

- Chest X-ray classification
- Chest X-ray segmentation
- Visual question answering
- Medical report generation
- Phrase grounding and finding localization
- DICOM processing and visualization
- Optional image generation tools
- Benchmark evaluation for vision-language models

## Architecture

```text
User image or DICOM study
            |
            v
      Gradio interface
            |
            v
       MedRAX agent
       /          \
Medical tools    Qwen2.5-VL
       \          /
        Reasoned response
```

The application uses LangChain and LangGraph for agent orchestration. Model requests are sent through the OpenAI-compatible client, so the same code can work with a local server or a hosted provider.

## Requirements

- Python 3.10 or newer
- CUDA-enabled GPU recommended for local medical imaging tools
- Access to an OpenAI-compatible vision-language model endpoint
- Model weights for the tools enabled in `main.py`

## Installation

```bash
git clone https://github.com/rohan-karna0/MedRAX.git
cd MedRAX
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

For development dependencies:

```bash
pip install -e ".[dev]"
```

## Configuration

Create a `.env` file in the project root. Do not commit this file.

```dotenv
OPENAI_BASE_URL=<openai-compatible-endpoint>
OPENAI_API_KEY=<your-api-key>
OPENAI_MODEL=rfsousa/qwen2.5vl:tools
GRADIO_SERVER_PORT=8585
```

For a local OpenAI-compatible server, use its base URL and a placeholder API key if the server does not require authentication. The default model can be changed without editing Python code.

## Run the Application

```bash
python main.py
```

The server selects an available port starting at `8585` and launches the Gradio interface. Upload a chest X-ray or DICOM file, choose a supported workflow, and submit a clinical question.

The default tools are configured in `main.py`. Tools that require additional model weights can be enabled or disabled in the `selected_tools` list.

## Evaluate With ChestAgentBench

Download the benchmark data and place it in the directory used by the evaluator. Then run:

```bash
python evaluate_chestagentbench.py \
  --benchmark-dir /path/to/chestagentbench \
  --max-questions 10
```

Results are written as JSONL files in `results/`. This directory is intentionally excluded from Git because evaluation output is generated locally.

## Project Structure

```text
.
├── main.py                         # Application entry point
├── interface.py                    # Gradio chat and upload workflow
├── evaluate_chestagentbench.py     # Benchmark evaluation script
├── medrax/
│   ├── agent/                      # Agent orchestration
│   ├── tools/                      # Medical imaging tools
│   └── utils/                      # Shared utilities
├── benchmark/                      # Benchmark helpers
├── data/                           # Metadata and data utilities
├── demo/                           # Sample chest X-ray studies
└── pyproject.toml                  # Package and dependency configuration
```

## Responsible Use

This software is an engineering and research project. Outputs may be incomplete or incorrect and must not be used as a substitute for evaluation by qualified healthcare professionals.

## Author

**Rohan Karna**
GitHub: [@rohan-karna0](https://github.com/rohan-karna0)
