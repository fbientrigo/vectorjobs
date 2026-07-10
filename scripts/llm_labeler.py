#!/usr/bin/env python
"""Deterministic LLM labeling tool using Gemini API for apolo-slm dataset preparation.

Supports checkpointing, JSON schema validation, evidence validation, and automatic retries.
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
logger = logging.getLogger("llm_labeler")

GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"

DEFAULT_MODEL = "gemini-1.5-flash"

SCHEMA_RESPONSE_CONFIG = {
    "type": "OBJECT",
    "properties": {
        "label": {
            "type": "STRING",
            "enum": [
                "HARD_SKILL",
                "DOMAIN_SKILL",
                "SOFT_SKILL",
                "EDUCATION",
                "EXPERIENCE",
                "RESPONSIBILITY",
                "BENEFIT",
                "LOCATION",
                "SCHEDULE",
                "CONTRACT",
                "IGNORE",
                "UNCERTAIN"
            ],
            "description": "The classified category for the candidate text."
        },
        "normalized_value": {
            "type": "STRING",
            "description": "Standardized canonical value (e.g., lowercase tech names, standard hours, modality). Keep empty if not applicable."
        },
        "evidence": {
            "type": "STRING",
            "description": "Literal, verbatim substring copied from candidate_text that justifies the label selection."
        },
        "confidence": {
            "type": "NUMBER",
            "description": "Confidence score between 0.0 and 1.0."
        },
        "notes": {
            "type": "STRING",
            "description": "Short reasoning explanation."
        }
    },
    "required": ["label", "normalized_value", "evidence", "confidence", "notes"]
}


def call_gemini_api(api_key: str, model: str, prompt: str, system_instruction: str) -> dict:
    """Call Gemini API with system instructions and JSON output schema constraints."""
    url = GEMINI_API_URL.format(model=model, api_key=api_key)
    headers = {"Content-Type": "application/json"}
    
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": prompt}]
            }
        ],
        "systemInstruction": {
            "parts": [{"text": system_instruction}]
        },
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": SCHEMA_RESPONSE_CONFIG,
            "temperature": 0.0,  # Ensure maximum determinism
        }
    }
    
    # Simple retry mechanism
    max_retries = 3
    backoff = 2
    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            if response.status_code == 200:
                data = response.json()
                text_response = data["candidates"][0]["content"]["parts"][0]["text"]
                return json.loads(text_response)
            elif response.status_code == 429:
                logger.warning(f"Rate limited (429). Retrying in {backoff} seconds...")
                time.sleep(backoff)
                backoff *= 2
            else:
                logger.error(f"API Error {response.status_code}: {response.text}")
                time.sleep(backoff)
                backoff *= 2
        except Exception as e:
            logger.error(f"Request failed: {e}")
            time.sleep(backoff)
            backoff *= 2
            
    raise RuntimeError("Failed to get response from Gemini API after retries.")


def main():
    parser = argparse.ArgumentParser(description="Deterministic LLM labeling using Gemini.")
    parser.add_argument("--input", type=str, default="data/silver/agent_labeling/agent_labeling_pack.jsonl", help="Path to agent_labeling_pack.jsonl")
    parser.add_argument("--output", type=str, default="data/silver/agent_labeling/model_labeled_candidates.jsonl", help="Destination path for labeled outputs")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of lines to process")
    parser.add_argument("--api-key", type=str, default=None, help="Gemini API Key (can also set via GEMINI_API_KEY environment variable)")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Gemini model name")
    parser.add_argument("--delay", type=float, default=1.0, help="Delay in seconds between API requests (to avoid rate limits)")
    
    args = parser.parse_args()
    
    api_key = args.api_key or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        logger.error("Gemini API key is required. Pass --api-key or set the GEMINI_API_KEY environment variable.")
        sys.exit(1)
        
    input_path = Path(args.input)
    output_path = Path(args.output)
    prompt_path = input_path.parent / "agent_labeling_prompt.md"
    
    if not input_path.exists():
        logger.error(f"Input file not found at {input_path}")
        sys.exit(1)
        
    if not prompt_path.exists():
        logger.error(f"Prompt guidelines file not found at {prompt_path}")
        sys.exit(1)
        
    system_instruction = prompt_path.read_text(encoding="utf-8")
    
    # Load already processed task_ids to support resume/checkpointing
    completed_tasks = {}
    if output_path.exists():
        logger.info(f"Output file {output_path} exists. Reading completed tasks for checkpointing...")
        with output_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        row = json.loads(line)
                        completed_tasks[row["task_id"]] = row
                    except Exception:
                        pass
        logger.info(f"Found {len(completed_tasks)} already labeled tasks.")
        
    # Read input tasks
    tasks = []
    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                tasks.append(json.loads(line))
                
    if args.limit:
        tasks = tasks[:args.limit]
        
    logger.info(f"Total tasks in queue: {len(tasks)}. Remaining to label: {len([t for t in tasks if t['task_id'] not in completed_tasks])}")
    
    # Open output file in append mode
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    evidence_failures = 0
    label_counts = {}
    
    # Process tasks
    with output_path.open("a", encoding="utf-8") as f_out:
        for i, task in enumerate(tasks):
            task_id = task["task_id"]
            if task_id in completed_tasks:
                # Add to statistics
                label = completed_tasks[task_id].get("label", "UNCERTAIN")
                label_counts[label] = label_counts.get(label, 0) + 1
                continue
                
            candidate_text = task["candidate_text"]
            logger.info(f"[{i+1}/{len(tasks)}] Labeling task {task_id}: '{candidate_text[:60]}...'")
            
            # Format task prompt
            task_prompt = (
                f"Candidate Text: \"{candidate_text}\"\n"
                f"Candidate Source: {task['candidate_source']}\n"
                f"Section Name: {task['section_name']}\n"
                f"Title Clean: {task['title_clean']}\n"
                f"Company Name: {task['company_name']}\n"
                f"Company Industry: {task['company_industry']}\n"
                f"Skills Norm (Regex baseline): {task['skills_normalized']}\n\n"
                f"Context Card: {task['context']['job_card_text']}\n"
            )
            
            try:
                response_json = call_gemini_api(
                    api_key=api_key,
                    model=args.model,
                    prompt=task_prompt,
                    system_instruction=system_instruction
                )
                
                # Validate evidence substring
                evidence = response_json.get("evidence", "")
                if evidence and evidence not in candidate_text:
                    logger.warning(f"Evidence '{evidence}' is not a literal substring of '{candidate_text}'!")
                    evidence_failures += 1
                    
                # Standardize record
                labeled_row = {
                    "task_id": task_id,
                    "job_id": task["job_id"],
                    "candidate_index": task["candidate_index"],
                    "candidate_text": candidate_text,
                    "candidate_source": task["candidate_source"],
                    "section_name": task["section_name"],
                    "label": response_json.get("label", "UNCERTAIN"),
                    "normalized_value": response_json.get("normalized_value", ""),
                    "evidence": evidence,
                    "confidence": response_json.get("confidence", 0.0),
                    "notes": response_json.get("notes", "")
                }
                
                # Write immediately
                f_out.write(json.dumps(labeled_row, ensure_ascii=False) + "\n")
                f_out.flush()
                
                # Update stats
                label = labeled_row["label"]
                label_counts[label] = label_counts.get(label, 0) + 1
                
                # Delay to handle rate limits
                if args.delay > 0:
                    time.sleep(args.delay)
                    
            except Exception as e:
                logger.error(f"Failed to label task {task_id}: {e}")
                # We do not crash so we can save other tasks, but print error
                continue
                
    # Re-calculate manifest
    all_labeled_rows = []
    if output_path.exists():
        with output_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    all_labeled_rows.append(json.loads(line))
                    
    final_label_counts = {}
    final_evidence_failures = 0
    for row in all_labeled_rows:
        lbl = row.get("label", "UNCERTAIN")
        final_label_counts[lbl] = final_label_counts.get(lbl, 0) + 1
        ev = row.get("evidence", "")
        txt = row.get("candidate_text", "")
        if ev and ev not in txt:
            final_evidence_failures += 1
            
    manifest = {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "n_input_rows_read": len(tasks),
        "n_labeled_rows": len(all_labeled_rows),
        "label_counts": final_label_counts,
        "evidence_substring_failures": final_evidence_failures,
        "created_at": datetime.now(tz=timezone.utc).isoformat(),
        "limitations": [
            f"Deterministic LLM-labeled sample using {args.model}.",
            "Labels based on candidate fragments and context prompts.",
            "JSON output format validated against schema."
        ]
    }
    
    manifest_path = output_path.parent / f"{output_path.stem}_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
        
    logger.info(f"Labeling complete. Labeled dataset output written to {output_path}")
    logger.info(f"Manifest written to {manifest_path}")


if __name__ == "__main__":
    main()
