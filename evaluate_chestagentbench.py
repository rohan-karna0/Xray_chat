#!/usr/bin/env python3
"""Evaluate an OpenAI-compatible vision model on ChestAgentBench."""

import argparse
import base64
import json
import mimetypes
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import openai
from dotenv import load_dotenv


load_dotenv()


ANSWER_RE = re.compile(r"\b([A-F])\b", re.IGNORECASE)
SYSTEM_PROMPT = (
    "You are a medical imaging expert. "
    "Provide only the letter corresponding to your answer choice (A/B/C/D/E/F)."
)


def default_benchmark_dir() -> Path:
    script_dir = Path(__file__).resolve().parent
    candidates = [
        script_dir / "chestagentbench",
        script_dir.parent.parent / "MedRAX" / "chestagentbench",
    ]
    for candidate in candidates:
        if (candidate / "metadata.jsonl").is_file():
            return candidate
    return candidates[0]


def load_examples(metadata_path: Path) -> list[dict[str, Any]]:
    examples = []
    with metadata_path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def encode_image(path: Path) -> str:
    with path.open("rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def data_url_for_image(path: Path) -> str:
    mime_type = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    return f"data:{mime_type};base64,{encode_image(path)}"


def normalize_image_paths(example: dict[str, Any], benchmark_dir: Path) -> list[Path]:
    paths = []
    for image_name in example.get("images", []):
        if not isinstance(image_name, str):
            continue
        relative_path = image_name.replace("figures/", "", 1)
        paths.append(benchmark_dir / "figures" / relative_path)
    return paths


def extract_answer(text: str | None) -> str | None:
    if not text:
        return None
    match = ANSWER_RE.search(text.strip().upper())
    return match.group(1) if match else None


def build_messages(example: dict[str, Any], image_paths: list[Path]) -> list[dict[str, Any]]:
    prompt = f"""Given the following medical case:
Please answer this multiple choice question:
{example["question"]}
Base your answer only on the provided images and case information.

Return only one uppercase letter: A, B, C, D, E, or F."""

    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for image_path in image_paths:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": data_url_for_image(image_path)},
            }
        )

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]


def evaluate_example(
    example: dict[str, Any],
    benchmark_dir: Path,
    client: openai.OpenAI,
    model: str,
    temperature: float,
) -> dict[str, Any]:
    image_paths = normalize_image_paths(example, benchmark_dir)
    missing_images = [str(path) for path in image_paths if not path.is_file()]
    if missing_images:
        return {
            "question_id": example.get("question_id", "unknown"),
            "case_id": example.get("case_id"),
            "status": "skipped",
            "reason": "missing_images",
            "missing_images": missing_images,
            "correct_answer": example.get("answer"),
            "question": example.get("question"),
        }

    messages = build_messages(example, image_paths)
    started_at = time.time()
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=50,
        temperature=temperature,
    )
    duration = time.time() - started_at

    model_answer = response.choices[0].message.content
    extracted_answer = extract_answer(model_answer)
    correct_answer = str(example.get("answer", "")).strip().upper()
    usage = getattr(response, "usage", None)

    return {
        "question_id": example.get("question_id", "unknown"),
        "full_question_id": example.get("full_question_id"),
        "case_id": example.get("case_id"),
        "status": "completed",
        "model": model,
        "temperature": temperature,
        "duration": round(duration, 2),
        "model_answer": model_answer,
        "extracted_answer": extracted_answer,
        "correct_answer": correct_answer,
        "correct": extracted_answer == correct_answer if extracted_answer else False,
        "question": example.get("question"),
        "explanation": example.get("explanation"),
        "categories": example.get("categories"),
        "images": [str(path) for path in image_paths],
        "usage": usage.model_dump() if usage is not None else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a vision model on ChestAgentBench")
    parser.add_argument("--benchmark-dir", type=Path, default=default_benchmark_dir())
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", "rfsousa/qwen2.5vl:tools"))
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-questions", type=int, default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    metadata_path = args.metadata or args.benchmark_dir / "metadata.jsonl"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"ChestAgentBench metadata not found: {metadata_path}")

    base_url = os.getenv("OPENAI_BASE_URL")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key and not base_url:
        raise ValueError(
            "OPENAI_API_KEY environment variable is not set. "
            "Set it in your shell or in .env. For a local OpenAI-compatible server, "
            "set OPENAI_BASE_URL and the evaluator will use a dummy API key."
        )

    client_kwargs: dict[str, str] = {"api_key": api_key or "dummy"}
    if base_url:
        client_kwargs["base_url"] = base_url
    client = openai.OpenAI(**client_kwargs)

    examples = load_examples(metadata_path)
    examples = examples[args.start :]
    if args.max_questions is not None:
        examples = examples[: args.max_questions]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    safe_model_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", args.model)
    output_path = args.output_dir / (
        f"chestagentbench_{safe_model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    )

    print(f"Benchmark dir: {args.benchmark_dir}")
    print(f"Metadata: {metadata_path}")
    print(f"Model: {args.model}")
    print(f"Questions: {len(examples)}")
    print(f"Writing results to: {output_path}")

    completed = 0
    skipped = 0
    correct = 0

    with output_path.open("w", encoding="utf-8") as output_file:
        for index, example in enumerate(examples, start=args.start + 1):
            try:
                result = evaluate_example(
                    example,
                    args.benchmark_dir,
                    client,
                    args.model,
                    args.temperature,
                )
            except Exception as exc:
                result = {
                    "question_id": example.get("question_id", "unknown"),
                    "case_id": example.get("case_id"),
                    "status": "error",
                    "error": str(exc),
                    "correct_answer": example.get("answer"),
                    "question": example.get("question"),
                }

            output_file.write(json.dumps(result, ensure_ascii=False) + "\n")
            output_file.flush()

            if result["status"] == "completed":
                completed += 1
                correct += int(result["correct"])
                print(
                    f"[{index}] {result['question_id']}: "
                    f"{result['extracted_answer']} / {result['correct_answer']} "
                    f"{'OK' if result['correct'] else 'WRONG'}"
                )
            else:
                skipped += 1
                print(f"[{index}] {result['question_id']}: {result['status']} ({result.get('reason') or result.get('error')})")

    accuracy = (correct / completed * 100) if completed else 0.0
    print("\nEvaluation complete")
    print(f"Completed: {completed}")
    print(f"Skipped/errors: {skipped}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
