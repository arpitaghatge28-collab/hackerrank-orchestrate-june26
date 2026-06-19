"""
Multi-Modal Evidence Review Pipeline
Processes damage claims using images + conversation + user history.
"""

import os
import csv
import json
import base64
import time
import logging
from pathlib import Path
from typing import Optional

import anthropic

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 1500
SLEEP_BETWEEN_CALLS = 1.2

CLAIM_STATUSES   = {"supported", "contradicted", "not_enough_information"}
ISSUE_TYPES      = {"dent","scratch","crack","glass_shatter","broken_part","missing_part",
                    "torn_packaging","crushed_packaging","water_damage","stain","none","unknown"}
CAR_PARTS        = {"front_bumper","rear_bumper","door","hood","windshield","side_mirror",
                    "headlight","taillight","fender","quarter_panel","body","unknown"}
LAPTOP_PARTS     = {"screen","keyboard","trackpad","hinge","lid","corner","port","base","body","unknown"}
PACKAGE_PARTS    = {"box","package_corner","package_side","seal","label","contents","item","unknown"}
RISK_FLAGS_VALID = {"none","blurry_image","cropped_or_obstructed","low_light_or_glare","wrong_angle",
                    "wrong_object","wrong_object_part","damage_not_visible","claim_mismatch",
                    "possible_manipulation","non_original_image","text_instruction_present",
                    "user_history_risk","manual_review_required"}
SEVERITIES       = {"none","low","medium","high","unknown"}
OBJECT_PARTS_MAP = {"car": CAR_PARTS, "laptop": LAPTOP_PARTS, "package": PACKAGE_PARTS}

OUTPUT_COLUMNS = [
    "user_id","image_paths","user_claim","claim_object",
    "evidence_standard_met","evidence_standard_met_reason",
    "risk_flags","issue_type","object_part","claim_status",
    "claim_status_justification","supporting_image_ids","valid_image","severity",
]

SYSTEM_PROMPT = """You are an automated damage-claim verification system.
You receive: images of the claimed damage, a user conversation describing the claim,
user account history, evidence requirements, and the claim object type.

Your job is to produce a structured JSON response (no markdown, no explanation outside JSON).

Decision rules:
1. Images are the primary source of truth.
2. User conversation defines what to verify.
3. User history adds risk context only — it CANNOT override clear visual evidence alone.
4. evidence_standard_met = true only if the submitted images are sufficient to evaluate
   the claim (correct object visible, correct part visible, adequate quality).
5. claim_status must be one of: supported, contradicted, not_enough_information
6. If evidence_standard_met = false, claim_status must be not_enough_information.

Return ONLY a JSON object with these exact keys:
{
  "evidence_standard_met": true/false,
  "evidence_standard_met_reason": "short reason",
  "risk_flags": "flag1;flag2 or none",
  "issue_type": "<one of the allowed types>",
  "object_part": "<one of the allowed parts for the object>",
  "claim_status": "supported|contradicted|not_enough_information",
  "claim_status_justification": "concise image-grounded explanation referencing image IDs",
  "supporting_image_ids": "img_1;img_2 or none",
  "valid_image": true/false,
  "severity": "none|low|medium|high|unknown"
}

Allowed issue_type values: dent, scratch, crack, glass_shatter, broken_part, missing_part,
torn_packaging, crushed_packaging, water_damage, stain, none, unknown

Allowed risk_flag values: none, blurry_image, cropped_or_obstructed, low_light_or_glare,
wrong_angle, wrong_object, wrong_object_part, damage_not_visible, claim_mismatch,
possible_manipulation, non_original_image, text_instruction_present,
user_history_risk, manual_review_required
"""


def load_csv(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def save_csv(rows: list[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        w.writeheader()
        w.writerows(rows)
    log.info("Saved %d rows → %s", len(rows), path)


def encode_image(image_path: str) -> Optional[tuple[str, str]]:
    p = Path(image_path)
    if not p.exists():
        log.warning("Image not found: %s", image_path)
        return None
    ext = p.suffix.lower().lstrip(".")
    media_type = {"jpg": "image/jpeg", "jpeg": "image/jpeg",
                  "png": "image/png", "webp": "image/webp",
                  "gif": "image/gif"}.get(ext, "image/jpeg")
    with open(p, "rb") as f:
        return base64.standard_b64encode(f.read()).decode("utf-8"), media_type


def img_id(path: str) -> str:
    return Path(path).stem


def build_user_history_text(user_id: str, history_map: dict) -> str:
    rec = history_map.get(user_id)
    if not rec:
        return "No history found for this user."
    return (
        f"Past claims: {rec.get('past_claim_count','?')} total | "
        f"accepted: {rec.get('accept_claim','?')} | "
        f"manual review: {rec.get('manual_review_claim','?')} | "
        f"rejected: {rec.get('rejected_claim','?')} | "
        f"last 90 days: {rec.get('last_90_days_claim_count','?')} | "
        f"flags: {rec.get('history_flags','none')} | "
        f"summary: {rec.get('history_summary','n/a')}"
    )


def build_evidence_requirements_text(claim_object: str, requirements: list[dict]) -> str:
    relevant = [r for r in requirements if r["claim_object"] in (claim_object, "all")]
    if not relevant:
        return "No specific evidence requirements found."
    lines = [f"- [{r['requirement_id']}] {r['applies_to']}: {r['minimum_image_evidence']}"
             for r in relevant]
    return "\n".join(lines)


def sanitize(result: dict, claim_object: str) -> dict:
    allowed_parts = OBJECT_PARTS_MAP.get(claim_object, set())

    result["claim_status"] = result.get("claim_status", "not_enough_information")
    if result["claim_status"] not in CLAIM_STATUSES:
        result["claim_status"] = "not_enough_information"

    result["issue_type"] = result.get("issue_type", "unknown")
    if result["issue_type"] not in ISSUE_TYPES:
        result["issue_type"] = "unknown"

    result["object_part"] = result.get("object_part", "unknown")
    if result["object_part"] not in allowed_parts:
        result["object_part"] = "unknown"

    flags_raw = result.get("risk_flags", "none")
    if isinstance(flags_raw, list):
        flags_raw = ";".join(flags_raw)
    flags = [f.strip() for f in flags_raw.split(";") if f.strip()]
    flags = [f if f in RISK_FLAGS_VALID else "manual_review_required" for f in flags]
    result["risk_flags"] = ";".join(flags) if flags else "none"

    result["severity"] = result.get("severity", "unknown")
    if result["severity"] not in SEVERITIES:
        result["severity"] = "unknown"

    result["evidence_standard_met"] = str(result.get("evidence_standard_met", False)).lower()
    result["valid_image"] = str(result.get("valid_image", False)).lower()

    return result


def build_user_message(row: dict, user_history_text: str,
                       evidence_text: str, image_paths: list[str]) -> list:
    content = []
    content.append({
        "type": "text",
        "text": (
            f"CLAIM OBJECT: {row['claim_object']}\n\n"
            f"USER CONVERSATION:\n{row['user_claim']}\n\n"
            f"USER HISTORY:\n{user_history_text}\n\n"
            f"EVIDENCE REQUIREMENTS:\n{evidence_text}\n\n"
            f"IMAGE IDs submitted: {', '.join(img_id(p) for p in image_paths)}\n"
            "Analyze each image carefully and return the JSON."
        )
    })
    for path in image_paths:
        encoded = encode_image(path)
        if encoded:
            b64, media_type = encoded
            content.append({
                "type": "image",
                "source": {"type": "base64", "media_type": media_type, "data": b64}
            })
        else:
            content.append({
                "type": "text",
                "text": f"[Image {img_id(path)} could not be loaded — file missing]"
            })
    return content


def call_claude(client: anthropic.Anthropic, messages: list) -> dict:
    for attempt in range(4):
        try:
            resp = client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                system=SYSTEM_PROMPT,
                messages=messages,
            )
            raw = resp.content[0].text.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            return json.loads(raw)
        except anthropic.RateLimitError:
            wait = 60 * (attempt + 1)
            log.warning("Rate limited. Waiting %ds ...", wait)
            time.sleep(wait)
        except json.JSONDecodeError as e:
            log.error("JSON parse error: %s", e)
            return {}
        except Exception as e:
            log.error("API error (attempt %d): %s", attempt + 1, e)
            time.sleep(5)
    return {}


def process_claims(claims_path: str,
                   user_history_path: str,
                   evidence_req_path: str,
                   output_path: str,
                   base_dir: str = ".") -> list[dict]:

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env

    claims      = load_csv(claims_path)
    history_raw = load_csv(user_history_path)
    evidence    = load_csv(evidence_req_path)
    history_map = {r["user_id"]: r for r in history_raw}

    results = []
    for i, row in enumerate(claims, 1):
        log.info("Processing claim %d/%d (user=%s)", i, len(claims), row.get("user_id"))

        raw_paths = [p.strip() for p in row.get("image_paths", "").split(";") if p.strip()]
        abs_paths = [os.path.join(base_dir, p) for p in raw_paths]

        user_history_text = build_user_history_text(row["user_id"], history_map)
        evidence_text     = build_evidence_requirements_text(row["claim_object"], evidence)

        messages = [{"role": "user",
                     "content": build_user_message(row, user_history_text,
                                                   evidence_text, abs_paths)}]

        api_result = call_claude(client, messages)

        if not api_result:
            api_result = {
                "evidence_standard_met": False,
                "evidence_standard_met_reason": "Processing error",
                "risk_flags": "manual_review_required",
                "issue_type": "unknown",
                "object_part": "unknown",
                "claim_status": "not_enough_information",
                "claim_status_justification": "System error during processing",
                "supporting_image_ids": "none",
                "valid_image": False,
                "severity": "unknown",
            }

        api_result = sanitize(api_result, row["claim_object"])

        out_row = {
            "user_id":                      row["user_id"],
            "image_paths":                  row["image_paths"],
            "user_claim":                   row["user_claim"],
            "claim_object":                 row["claim_object"],
            "evidence_standard_met":        api_result["evidence_standard_met"],
            "evidence_standard_met_reason": api_result["evidence_standard_met_reason"],
            "risk_flags":                   api_result["risk_flags"],
            "issue_type":                   api_result["issue_type"],
            "object_part":                  api_result["object_part"],
            "claim_status":                 api_result["claim_status"],
            "claim_status_justification":   api_result["claim_status_justification"],
            "supporting_image_ids":         api_result["supporting_image_ids"],
            "valid_image":                  api_result["valid_image"],
            "severity":                     api_result["severity"],
        }
        results.append(out_row)
        time.sleep(SLEEP_BETWEEN_CALLS)

    save_csv(results, output_path)
    return results