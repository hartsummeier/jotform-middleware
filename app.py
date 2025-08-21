# app.py
import os, json, base64, hashlib, logging, traceback, requests
from flask import Flask, request, jsonify

# -----------------------------------------------------------------------------
# Environment
# -----------------------------------------------------------------------------
OPENAI_API_KEY  = os.environ.get("OPENAI_API_KEY", "")
JOTFORM_API_KEY = os.environ.get("JOTFORM_API_KEY", "")
INTAKE_FORM_ID  = os.environ.get("INTAKE_FORM_ID", "")  # can be overridden by fields.json

# -----------------------------------------------------------------------------
# App & Logging
# -----------------------------------------------------------------------------
app = Flask(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# -----------------------------------------------------------------------------
# Load fields config (fields.json)
# -----------------------------------------------------------------------------
def load_cfg():
    try:
        with open("fields.json", "r", encoding="utf-8") as f:
            cfg = json.load(f)
        return cfg
    except FileNotFoundError:
        return None

CFG = load_cfg()
if CFG and CFG.get("intake_form_id"):
    INTAKE_FORM_ID = CFG["intake_form_id"]


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def err(message: str, status: int = 400, **extra):
    """Return a consistent JSON error to Zapier and log it."""
    logging.error(f"{message} | extra={extra}")
    payload = {"ok": False, "error": message}
    if extra:
        payload.update(extra)
    return jsonify(payload), status

def download_file(url: str):
    """
    Download the PDF. For MVP, ensure Jotform Setting 'Require log-in to view uploaded files' is OFF.
    Returns (bytes, sha256hex).
    """
    try:
        r = requests.get(url, timeout=90)
        r.raise_for_status()
        data = r.content
        return data, hashlib.sha256(data).hexdigest()
    except Exception as e:
        raise RuntimeError(f"Failed to download file from URL. "
                           f"Hint: turn OFF 'Require log-in to view uploaded files' for testing. Details: {e}")

def make_json_schema(cfg_fields):
    """
    Build an OpenAI JSON Schema from fields.json.
    Supported 'type's: string (default), number, date, enum, full_name, address, phone,
    fixed_string (we fill), json (we fill).
    """
    props = {}
    for f in cfg_fields:
        key = f["key"]
        ftype = f.get("type", "string")
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
                    "area":  {"type":"string","nullable": True},
                    "number":{"type":"string","nullable": True},
                    "full":  {"type":"string","nullable": True}
                },
                "additionalProperties": False
            }
        else:
            # string / fixed_string / json default to string in extraction
            props[key] = {"type": "string", "nullable": True}

    # meta fields from the model
    props["confidence"] = {"type": "number", "minimum": 0, "maximum": 1, "nullable": True}
    props["page_refs"]  = {"type": "array", "items": {"type":"integer"}, "nullable": True}

    return {
        "name": "intake_extract",
        "schema": {
            "type":"object",
            "properties": props,
            "required": [],
            "additionalProperties": False
        },
        "strict": True
    }

def openai_upload_file(pdf_bytes: bytes, filename="contract.pdf") -> str:
    """Upload the PDF to OpenAI Files API and return file_id."""
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is missing.")
    url = "https://api.openai.com/v1/files"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
    files = {"file": (filename, pdf_bytes, "application/pdf")}
    data  = {"purpose": "assistants"}
    r = requests.post(url, headers=headers, files=files, data=data, timeout=180)
    try:
        r.raise_for_status()
    except Exception:
        raise RuntimeError(f"OpenAI file upload error: {r.status_code} {r.text[:400]}")
    return r.json()["id"]

def extract_with_openai(pdf_bytes: bytes, schema: dict) -> dict:
    """
    Call the Responses API with structured outputs (schema under text.format).
    Returns a dict of extracted values keyed by the 'key' strings from fields.json.
    """
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is missing.")

    file_id = openai_upload_file(pdf_bytes)

    url = "https://api.openai.com/v1/responses"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
    payload = {
        "model": "gpt-4o-mini",
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
                {"type": "input_file", "file_id": file_id}
            ]
        }],
        # IMPORTANT: schema goes under text.format for the Responses API
        "text": {
            "format": {
                "type": "json_schema",
                "json_schema": schema
            }
        },
        "temperature": 0
    }

    r = requests.post(url, headers=headers, json=payload, timeout=180)
    try:
        r.raise_for_status()
    except Exception:
        raise RuntimeError(f"OpenAI API error: {r.status_code} {r.text[:400]}")

    data = r.json()
    # Prefer parsed when provided; otherwise fall back to text
    parsed = data.get("output_parsed")
    if not parsed:
        try:
            text_piece = (
                data.get("output", [{}])[0]
                    .get("content", [{}])[0]
                    .get("text", "{}")
            )
            parsed = json.loads(text_piece)
        except Exception:
            parsed = {}
    return parsed

def jotform_create_submission(form_id: str, cfg_fields: list, extracted: dict,
                              file_hash: str, fub_id: str) -> str:
    """
    Post values to Jotform. Handles subfields for full_name, address, phone.
    """
    if not JOTFORM_API_KEY:
        raise RuntimeError("JOTFORM_API_KEY is missing.")
    if not form_id:
        raise RuntimeError("INTAKE_FORM_ID is missing. Set env var or put intake_form_id in fields.json.")

    url = f"https://api.jotform.com/form/{form_id}/submissions?apiKey={JOTFORM_API_KEY}"
    payload = {}

    def put(qid, value, subkey=None):
        if subkey:
            payload[f"submission[{qid}][{subkey}]"] = value
        else:
            payload[f"submission[{qid}]"] = value

    for f in cfg_fields:
        key   = f["key"]
        qid   = f["qid"]
        ftype = f.get("type", "string")
        val   = extracted.get(key)

        if ftype == "fixed_string":
            put(qid, f.get("value", ""))
            continue

        if ftype == "json":
            meta = {
                "overall":   extracted.get("confidence"),
                "page_refs": extracted.get("page_refs"),
                "source_file_hash": file_hash,
                "fub_id": fub_id
            }
            put(qid, json.dumps(meta))
            continue

        if val in (None, ""):
            continue

        if ftype == "full_name" and isinstance(val, dict):
            if val.get("first"):  put(qid, val["first"],  "first")
            if val.get("last"):   put(qid, val["last"],   "last")
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

    r = requests.post(url, data=payload, timeout=90)
    try:
        r.raise_for_status()
    except Exception:
        raise RuntimeError(f"Jotform API error: {r.status_code} {r.text[:300]}")
    return r.json()["content"]["submissionID"]

def jotform_edit_link(submission_id: str) -> str:
    return f"https://www.jotform.com/edit/{submission_id}"


# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------
@app.route("/", methods=["GET"])
def health():
    return "Middleware is running.", 200

@app.route("/ingest", methods=["POST"])
def ingest():
    # Single, robust handler (no duplicate route names)
    try:
        if CFG is None:
            return err("fields.json not found in the working directory.", 500)
        if not OPENAI_API_KEY:
            return err("OPENAI_API_KEY env var not set.", 500)
        if not (JOTFORM_API_KEY and (INTAKE_FORM_ID or CFG.get("intake_form_id"))):
            return err("JOTFORM_API_KEY or INTAKE_FORM_ID missing.", 500)

        body = request.get_json(force=True) or {}
        fub_id      = body.get("fub_id", "")
        agent_email = body.get("agent_email", "")
        uploaded_by = body.get("uploaded_by", "")
        file_url    = body.get("file_url", "")

        if not file_url:
            return err("file_url is required in the payload.")

        # 1) Download PDF
        pdf_bytes, file_hash = download_file(file_url)

        # 2) Build schema & extract
        schema    = make_json_schema(CFG["fields"])
        extracted = extract_with_openai(pdf_bytes, schema)

        # 3) Create Jotform submission
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
        logging.error("ingest error: %s\n%s", e, traceback.format_exc())
        return err(str(e), 400)


# -----------------------------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
