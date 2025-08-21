
import os, hashlib, json, base64, requests
from flask import Flask, request, jsonify

OPENAI_API_KEY  = os.environ.get("OPENAI_API_KEY", "")
JOTFORM_API_KEY = os.environ.get("JOTFORM_API_KEY", "")
INTAKE_FORM_ID  = os.environ.get("INTAKE_FORM_ID", "")  # can be overridden by fields.json

app = Flask(__name__)

# ---------- load config ----------
def load_cfg():
    with open("fields.json", "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return cfg

CFG = load_cfg()
if CFG.get("intake_form_id"):
    INTAKE_FORM_ID = CFG["intake_form_id"]

# ---------- helpers ----------
def download_file(url):
    # For MVP: ensure Jotform "Require login to view uploads" is OFF
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    data = r.content
    return data, hashlib.sha256(data).hexdigest()

def openai_upload_file(pdf_bytes, filename="contract.pdf"):
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is missing.")
    url = "https://api.openai.com/v1/files"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
    files = {
        "file": (filename, pdf_bytes, "application/pdf")
    }
    data = {"purpose": "assistants"}  # works with the Responses API too
    r = requests.post(url, headers=headers, files=files, data=data, timeout=180)
    try:
        r.raise_for_status()
    except Exception:
        raise RuntimeError(f"OpenAI file upload error: {r.status_code} {r.text[:400]}")
    return r.json()["id"]

def make_json_schema(cfg_fields):
    """Build OpenAI JSON schema based on types in fields.json.
       Defaults to string if unknown."""
    props = {}
    for f in cfg_fields:
        key = f["key"]
        ftype = f.get("type","string")
        if ftype == "number":
            props[key] = {"type": "number", "nullable": True}
        elif ftype == "date":
            props[key] = {"type": "string", "nullable": True}
        elif ftype == "enum":
            props[key] = {"type": "string", "enum": f.get("enum", []), "nullable": True}
        elif ftype == "full_name":
            props[key] = {
                "type": "object", "nullable": True,
                "properties": {
                    "first": {"type":"string","nullable": True},
                    "last":  {"type":"string","nullable": True},
                    "middle":{"type":"string","nullable": True},
                    "suffix":{"type":"string","nullable": True}
                },
                "additionalProperties": False
            }
        elif ftype == "address":
            props[key] = {
                "type": "object", "nullable": True,
                "properties": {
                    "line1":   {"type":"string","nullable": True},
                    "line2":   {"type":"string","nullable": True},
                    "city":    {"type":"string","nullable": True},
                    "state":   {"type":"string","nullable": True},
                    "postal":  {"type":"string","nullable": True},
                    "country": {"type":"string","nullable": True}
                },
                "additionalProperties": False
            }
        elif ftype == "phone":
            props[key] = {
                "type": "object", "nullable": True,
                "properties": {
                    "area": {"type":"string","nullable": True},
                    "number": {"type":"string","nullable": True},
                    "full": {"type":"string","nullable": True}
                },
                "additionalProperties": False
            }
        else:
            # string, fixed_string, json default to string in extraction
            props[key] = {"type": "string", "nullable": True}

    # meta fields for review
    props["confidence"] = {"type":"number","minimum":0,"maximum":1,"nullable": True}
    props["page_refs"]  = {"type":"array","items":{"type":"integer"}, "nullable": True}

    return {
      "name": "intake_extract",
      "schema": {"type":"object", "properties": props, "required": [], "additionalProperties": False},
      "strict": True
    }

def extract_with_openai(pdf_bytes, schema):
    # 1) Upload file once
    file_id = openai_upload_file(pdf_bytes)

    # 2) Ask the model to read that file and return strict JSON
    url = "https://api.openai.com/v1/responses"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
    payload = {
      "model": "gpt-4o-mini",
      # The "input" format you used is fine
      "input": [{
        "role": "user",
        "content": [
          {
            "type": "input_text",
            "text": (
              "Read this real estate purchase contract. Extract the fields defined by the schema. "
              "Return null for anything not present. Keep dates as YYYY-MM-DD if possible. "
              "For names and addresses, fill sub-fields if available. "
              "Also include page_refs for where you found key values and an overall confidence (0-1)."
            )
          },
          { "type": "input_file", "file_id": file_id }
        ]
      }],
      # ✅ NEW: Structured output must be specified under the `text` key in the Responses API
      "modalities": ["text"],
      "text": {
        "format": "json_schema",
        "json_schema": schema
      },
      # Extraction: be deterministic
      "temperature": 0
    }

    r = requests.post(url, headers=headers, json=payload, timeout=180)
    try:
        r.raise_for_status()
    except Exception:
        # Bubble up the real server message so you see it in Render logs/Zapier
        raise RuntimeError(f"OpenAI API error: {r.status_code} {r.text[:400]}")
    data = r.json()

    # With structured outputs, most SDKs give you parsed JSON already.
    # Fallback logic below covers both shapes.
    parsed = data.get("output_parsed")
    if not parsed:
        # Some responses carry the JSON as text
        content = data.get("output", [{}])[0].get("content", [{}])[0].get("text", "{}")
        parsed = json.loads(content)
    return parsed

def jotform_create_submission(form_id, cfg_fields, extracted, file_hash, fub_id):
    """Post values to Jotform. Handles sub-fields for known types."""
    url = f"https://api.jotform.com/form/{form_id}/submissions?apiKey={JOTFORM_API_KEY}"
    payload = {}

    def put(qid, value, subkey=None):
        if subkey:
            payload[f"submission[{qid}][{subkey}]"] = value
        else:
            payload[f"submission[{qid}]"] = value

    # Optional: look for review_status/confidence_json in config
    has_review_status = next((f for f in cfg_fields if f.get("key")=="review_status"), None)
    has_conf_json     = next((f for f in cfg_fields if f.get("key")=="confidence_json"), None)

    for f in cfg_fields:
        key = f["key"]
        qid = f["qid"]
        ftype = f.get("type","string")
        val = extracted.get(key)

        if ftype == "fixed_string":
            put(qid, f.get("value",""))
            continue

        if ftype == "json":
            meta = {
              "overall": extracted.get("confidence"),
              "page_refs": extracted.get("page_refs"),
              "source_file_hash": file_hash,
              "fub_id": fub_id
            }
            put(qid, json.dumps(meta))
            continue

        if val is None or val == "":
            continue

        if ftype == "full_name" and isinstance(val, dict):
            if val.get("first"):  put(qid, val["first"], "first")
            if val.get("last"):   put(qid, val["last"], "last")
            if val.get("middle"): put(qid, val["middle"], "middle")
            if val.get("suffix"): put(qid, val["suffix"], "suffix")
        elif ftype == "address" and isinstance(val, dict):
            if val.get("line1"):   put(qid, val["line1"],   "addr_line1")
            if val.get("line2"):   put(qid, val["line2"],   "addr_line2")
            if val.get("city"):    put(qid, val["city"],    "city")
            if val.get("state"):   put(qid, val["state"],   "state")
            if val.get("postal"):  put(qid, val["postal"],  "postal")
            if val.get("country"): put(qid, val["country"], "country")
        elif ftype == "phone" and isinstance(val, dict):
            if val.get("full"):
                put(qid, val["full"], "full")
            else:
                if val.get("area"):   put(qid, val["area"],   "area")
                if val.get("number"): put(qid, val["number"], "phone")
        else:
            # string, number, enum, date
            put(qid, val)

    r = requests.post(url, data=payload, timeout=60)
    r.raise_for_status()
    return r.json()["content"]["submissionID"]

def jotform_edit_link(submission_id):
    return f"https://www.jotform.com/edit/{submission_id}"

# ---------- routes ----------
@app.route("/ingest", methods=["POST"])
def ingest():
    body = request.get_json(force=True)
    fub_id      = body.get("fub_id", "")
    agent_email = body.get("agent_email", "")
    uploaded_by = body.get("uploaded_by", "")
    file_url    = body.get("file_url", "")

    pdf_bytes, file_hash = download_file(file_url)
    schema = make_json_schema(CFG["fields"])
    extracted = extract_with_openai(pdf_bytes, schema)

    submission_id = jotform_create_submission(
        INTAKE_FORM_ID, CFG["fields"], extracted, file_hash, fub_id
    )
    edit_url = jotform_edit_link(submission_id)

    return jsonify({
      "ok": True,
      "submission_id": submission_id,
      "edit_url": edit_url,
      "file_hash": file_hash,
      "agent_email": agent_email,
      "uploaded_by": uploaded_by
    })

@app.route("/", methods=["GET"])
def health():
    return "Middleware is running.", 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)

@app.route("/ingest", methods=["POST"])
def ingest():
    try:
        body = request.get_json(force=True)
        fub_id      = body.get("fub_id", "")
        agent_email = body.get("agent_email", "")
        uploaded_by = body.get("uploaded_by", "")
        file_url    = body.get("file_url", "")

        pdf_bytes, file_hash = download_file(file_url)
        schema = make_json_schema(CFG["fields"])
        extracted = extract_with_openai(pdf_bytes, schema)

        submission_id = jotform_create_submission(
            INTAKE_FORM_ID, CFG["fields"], extracted, file_hash, fub_id
        )
        edit_url = jotform_edit_link(submission_id)

        return jsonify({
          "ok": True,
          "submission_id": submission_id,
          "edit_url": edit_url,
          "file_hash": file_hash,
          "agent_email": agent_email,
          "uploaded_by": uploaded_by
        })
    except Exception as e:
        # Send the message back so Zapier shows it in the task history
        return jsonify({"ok": False, "error": str(e)}), 400
