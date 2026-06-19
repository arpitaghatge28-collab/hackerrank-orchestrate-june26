import json


def extract_claim(text):

    text = text.lower()

    result = {
        "issue_type": "unknown",
        "object_part": "unknown"
    }

    issues = [
        "dent",
        "scratch",
        "crack",
        "broken",
        "water",
        "stain",
        "torn",
        "missing"
    ]

    for i in issues:
        if i in text:
            result["issue_type"] = i
            break

    parts = [
        "screen",
        "keyboard",
        "door",
        "bumper",
        "box",
        "package"
    ]

    for p in parts:
        if p in text:
            result["object_part"] = p
            break

    return result


# -------- RUN THE FUNCTION --------

claim_text = "My laptop screen has a crack"

output = extract_claim(claim_text)

print(json.dumps(output, indent=4))