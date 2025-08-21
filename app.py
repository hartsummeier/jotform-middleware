# app.py (config-driven)
import os, hashlib, json, base64, requests
from flask import Flask, request, jsonify

OPENAI_API_KEY  = os.environ.get("OPENAI_API_KEY", "")
JOTFORM_API_KEY = os.environ.get("JOTFORM_API_KEY", "")
# We'll override intake_form_id from fields.json if present
INTAKE_FORM_ID  = os.environ.get("INTAKE_FORM_ID", "")

app = Flask(__name__)

# ---------- load config ----------
with open("fields.json", "r", encoding="utf-8") as f:
    CFG = json.load(f)

if CFG.get("intake_form_id"):
    INTAKE_FORM_ID = CFG["intake_form_id"]

# ---------- helpers ----------
def download_file(url):
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    data = r.content
    return data, hashlib.sha256(data).hexdigest()

def make_json_schema(cfg_fields):
    """Build OpenAI JSON schema from fields.json"""
    props = {}
    for f in cfg_fields:
        key = f["key"]
        ftype = f["type"]
        # Map config types to JSON schema
        if ftype == "number":
            props[key] = {"type": "number", "nullable": True}
        elif ftype == "date":
            props[key] = {"type": "string", "nullable": True}
        elif ftype == "enum":
            props[key] = {"type": "string", "enum": f["enum"], "nullable": True}
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
        elif ftype in ("json","fixed_string"):  # will be filled by us, not model
            # Still define them so the final JSON contains them if you want
            props[key] = {"type":"string","nullable": True}
        else:  # default to plain string
            props[key] = {"type": "string", "nullable": True}

    # meta: confidence + page_refs
    props["confidence"] = {"type":"number","minimum":0,"maximum":1,"nullable": True}
    props["page_refs"]  = {"type":"array","items":{"type":"integer"}, "nullable": True}

    schema = {
      "name": "intake_extract",
      "schema": {"type":"object", "properties": props, "required": [], "additionalProperties": False},
      "strict": True
    }
    return schema

def extract_with_openai(pdf_bytes, schema):
    b64 = base64.b64encode(pdf_bytes).decode("utf-8")
    url = "https://api.openai.com/v1/responses"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
    payload = {
      "model": "gpt-4o-mini",
      "input": [{
        "role":"user",
        "content":[
          {"type":"input_text","text":
           "Read this real estate contract. Extract the fields defined by the schema. "
           "Return null for anything not present. Keep dates as YYYY-MM-DD if possible. "
           "For names and addresses, fill the sub-fields if available. "
           "Also include page_refs for where you found key values and an overall confidence (0-1)."},
          {"type":"input_image","image_url": f"data:application/pdf;base64,{b64}"}
        ]
      }],
      "response_format": {"type":"json_schema","json_schema": schema}
    }
    r = requests.post(url, headers=headers, json=payload, timeout=180)
    r.raise_for_status()
    data = r.json()
    parsed = data.get("output_parsed")
    if not parsed:
        content = data.get("output", [{}])[0].get("content", [{}])[0].get("text", "{}")
        parsed = json.loads(content)
    return parsed

def jotform_create_submission(form_id, cfg_fields, extracted, file_hash, fub_id):
    """Build the payload for Jotform, including sub-fields where needed."""
    url = f"https://api.jotform.com/form/{form_id}/submissions?apiKey={JOTFORM_API_KEY}"
    payload = {}

    def put(qid, value, subkey=None):
        if subkey:
            payload[f"submission[{qid}][{subkey}]"] = value
        else:
            payload[f"submission[{qid}]"] = value

    # Fill fields from extraction
    for f in cfg_fields:
        key, ftype, qid = f["key"], f["type"], f["qid"]
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

        if val is None:
            continue

        if ftype == "full_name" and isinstance(val, dict):
            if val.get("first"):  put(qid, val["first"], "first")
            if val.get("last"):   put(qid, val["last"], "last")
            if val.get("middle"): put(qid, val["middle"], "middle")
            if val.get("suffix"): put(qid, val["suffix"], "suffix")
        elif ftype == "address" and isinstance(val, dict):
            # Jotform subkeys can vary; these common ones usually work
            if val.get("line1"):   put(qid, val["line1"],   "addr_line1")
            if val.get("line2"):   put(qid, val["line2"],   "addr_line2")
            if val.get("city"):    put(qid, val["city"],    "city")
            if val.get("state"):   put(qid, val["state"],   "state")
            if val.get("postal"):  put(qid, val["postal"],  "postal")
            if val.get("country"): put(qid, val["country"], "country")
        elif ftype == "phone" and isinstance(val, dict):
            # Prefer full if you have it; otherwise area/number
            if val.get("full"):
                put(qid, val["full"], "full")
            else:
                if val.get("area"):   put(qid, val["area"],   "area")
                if val.get("number"): put(qid, val["number"], "phone")
        else:
            # number, enum, string, date, etc.
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
