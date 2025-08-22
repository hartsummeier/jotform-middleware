# app.py
import os, json, re, hashlib, logging, traceback, requests
from datetime import datetime
from flask import Flask, request, jsonify

# -----------------------------------------------------------------------------
# Environment
# -----------------------------------------------------------------------------
OPENAI_API_KEY  = os.environ.get("OPENAI_API_KEY", "")
JOTFORM_API_KEY = os.environ.get("JOTFORM_API_KEY", "")
# Optional fallback if you don't set intake_form_id in fields.json
INTAKE_FORM_ID  = os.environ.get("INTAKE_FORM_ID", "")

# -----------------------------------------------------------------------------
# App & logging
# -----------------------------------------------------------------------------
app = Flask(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
def load_cfg():
    with open("fields.json", "r", encoding="utf-8") as f:
        return json.load(f)

CFG = load_cfg()
if CFG.get("intake_form_id"):
    INTAKE_FORM_ID = CFG["intake_form_id"]

# -----------------------------------------------------------------------------
# Jotform helpers
# -----------------------------------------------------------------------------
def extract_form_id(value: str) -> str:
    """Accept either a numeric ID or a full URL; return numeric ID."""
    if not value:
        raise RuntimeError("INTAKE_FORM_ID is missing.")
    m = re.search(r"(\d{8,})", str(value))
    if not m:
        raise RuntimeError(f"Could not parse a numeric form id from '{value}'.")
    return m.group(1)

def jf_get_questions(form_id: str) -> dict:
    """Return dict of {qid: question_obj} from Jotform."""
    if not JOTFORM_API_KEY:
        raise RuntimeError("JOTFORM_API_KEY env var not set.")
    form_id = extract_form_id(form_id)
    url = f"https://api.jotform.com/form/{form_id}/questions?apiKey={JOTFORM_API_KEY}"
    r = requests.get(url, timeout=90)
    try:
        r.raise_for_status()
    except Exception:
        raise RuntimeError(f"Jotform /questions error: {r.status_code} {r.text[:400]}")
    return r.json()["content"]

def jf_type_to_model_type(jf_type: str, override: dict | None) -> str:
    """Map Jotform control types to our internal types for schema/submission."""
    if override and "type" in override:
        return override["type"]
    t = jf_type or ""
    # Common mappings
    if t in ("control_fullname",):               return "full_name"
    if t in ("control_address",):                return "address"
    if t in ("control_phone",):                  return "phone"
    if t in ("control_datetime",):               return "date"
    if t in ("control_number", "control_spinner"): return "number"
    if t in ("control_dropdown", "control_radio"): return "enum"
    # Everything else we treat as string (textbox, textarea, email, calc text, etc.)
    return "string"

def build_live_catalog():
    """Build a list of field descriptors from the live Jotform, honoring config."""
    qs = jf_get_questions(INTAKE_FORM_ID)  # dict keyed by qid
    exclude_types = set(CFG.get("exclude_types", []))
    overrides = CFG.get("overrides", {})

    fields = []
    for qid, q in qs.items():
        jf_type = q.get("type") or ""
        if jf_type in exclude_types:
            continue
        unique = q.get("name") or ""  # Jotform Unique Name (no spaces)
        label  = q.get("text") or unique
        if not unique:
            continue

        ov = overrides.get(unique, {})
        ftype = jf_type_to_model_type(jf_type, ov)

        # Build enum options for dropdown/radio if present
        enum = None
        if ftype == "enum":
            # Jotform stores options either as "options": "A|B|C" or list
            props = q.get("properties") or {}
            opts  = props.get("options")
            if isinstance(opts, str):
                enum = [o.strip() for o in opts.split("|") if o.strip()]
            elif isinstance(opts, list):
                enum = [str(o).strip() for o in opts if str(o).strip()]

        fields.append({
            "key": unique,        # used in extraction JSON
            "qid": str(qid),      # used when submitting back to Jotform
            "label": label,
            "type": ov.get("type", ftype),
            "enum": ov.get("enum", enum)
        })
    logging.info("Live catalog size: %d", len(fields))
    return fields

# -----------------------------------------------------------------------------
# OpenAI helpers
# -----------------------------------------------------------------------------
def download_file(url: str):
    """Download file; for Jotform uploads, ensure 'Require log-in to view uploaded files' is OFF."""
    try:
        r = requests.get(url, timeout=120)
        r.raise_for_status()
        data = r.content
        return data, hashlib.sha256(data).hexdigest()
    except Exception as e:
        raise RuntimeError(
            "Failed to download file from URL. "
            "If it's a Jotform file, make sure 'Require log-in to view uploaded files' is OFF. "
            f"Details: {e}"
        )

def openai_upload_file(pdf_bytes: bytes, filename="contract.pdf") -> str:
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

def make_json_schema_from_fields(fields: list[dict]) -> dict:
    props = {}
    for f in fields:
        key = f["key"]
        ftype = f.get("type", "string")
        if ftype == "number":
            props[key] = {"type": "number", "nullable": True}
        elif ftype == "date":
            props[key] = {"type": "string", "nullable": True}  # YYYY-MM-DD preferred
        elif ftype == "enum":
            enum = f.get("enum")
            if enum:
                props[key] = {"type": "string", "enum": enum, "nullable": True}
            else:
                props[key] = {"type": "string", "nullable": True}
        elif ftype == "full_name":
            props[key] = {
                "type": "object", "nullable": True,
                "properties": {
                    "first":  {"type": "string", "nullable": True},
                    "last":   {"type": "string", "nullable": True},
                    "middle": {"type": "string", "nullable": True},
                    "suffix": {"type": "string", "nullable": True}
                },
                "additionalProperties": False
            }
        elif ftype == "address":
            props[key] = {
                "type": "object", "nullable": True,
                "properties": {
                    "line1":   {"type": "string", "nullable": True},
                    "line2":   {"type": "string", "nullable": True},
                    "city":    {"type": "string", "nullable": True},
                    "state":   {"type": "string", "nullable": True},
                    "postal":  {"type": "string", "nullable": True},
                    "country": {"type": "string", "nullable": True}
                },
                "additionalProperties": False
            }
        elif ftype == "phone":
            props[key] = {
                "type": "object", "nullable": True,
                "properties": {
                    "full":   {"type": "string", "nullable": True},
                    "area":   {"type": "string", "nullable": True},
                    "number": {"type": "string", "nullable": True}
                },
                "additionalProperties": False
            }
        else:
            props[key] = {"type": "string", "nullable": True}

    # Global meta
    props["confidence"] = {"type": "number", "minimum": 0, "maximum": 1, "nullable": True}
    props["page_refs"]  = {"type": "array", "items": {"type": "integer"}, "nullable": True}

    required = list(props.keys())
    return {
        "name": "intake_extract",
        "schema": {
            "type": "object",
            "properties": props,
            "required": required if CFG.get("strict_schema", True) else [],
            "additionalProperties": False
        },
        "strict": CFG.get("strict_schema", True)
    }

def extract_with_openai(pdf_bytes: bytes, schema_obj: dict) -> dict:
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
                        "Read this real estate purchase contract (may include addenda). "
                        "Extract the fields in the schema. "
                        "Use null when information is not present. "
                        "Format dates as YYYY-MM-DD when possible. "
                        "For names/addresses/phones, fill sub-fields if available. "
                        "Also include 'page_refs' listing page numbers where key values were found "
                        "and an overall 'confidence' between 0 and 1."
                    )
                },
                {"type": "input_file", "file_id": file_id}
            ]
        }],
        # Structured output (Responses API)
        "text": { "format": schema_obj },
        "temperature": 0
    }
    r = requests.post(url, headers=headers, json=payload, timeout=240)
    try:
        r.raise_for_status()
    except Exception:
        raise RuntimeError(f"OpenAI API error: {r.status_code} {r.text[:500]}")
    data = r.json()
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

# -----------------------------------------------------------------------------
# Submission helpers
# -----------------------------------------------------------------------------
def put_field(payload: dict, qid: str, value, subkey: str | None = None):
    if subkey:
        payload[f"submission[{qid}][{subkey}]"] = value
    else:
        payload[f"submission[{qid}]"] = value

def parse_date_yyyy_mm_dd(s: str) -> tuple[int,int,int] | None:
    # Accept 'YYYY-MM-DD' or many common variants
    if not s or not isinstance(s, str):
        return None
    s = s.strip()
    # Try strict
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y", "%m-%d-%Y"):
        try:
            dt = datetime.strptime(s, fmt)
            return dt.month, dt.day, dt.year
        except Exception:
            pass
    # loose fallback: digits only
    m = re.findall(r"\d+", s)
    if len(m) == 3:
        a, b, c = [int(x) for x in m]
        # Heuristic: if 4-digit present first -> Y-M-D
        if len(m[0]) == 4:
            return b, c, a
        # If last has 4 digits -> M-D-Y
        if len(m[-1]) == 4:
            return a, b, c
    return None

def jotform_create_submission(form_id: str, fields: list[dict], extracted: dict,
                              file_hash: str, fub_id: str | None = None) -> str:
    form_id = extract_form_id(form_id)
    url = f"https://api.jotform.com/form/{form_id}/submissions?apiKey={JOTFORM_API_KEY}"
    payload = {}

    # Optional meta field with extraction info (skip if you don't have a target field)
    meta = {
        "overall": extracted.get("confidence"),
        "page_refs": extracted.get("page_refs"),
        "source_file_hash": file_hash,
        "fub_id": fub_id
    }
    # If you have a long text field to hold meta, you can set it here:
    # put_field(payload, "<QID_OF_META_FIELD>", json.dumps(meta))

    for f in fields:
        key  = f["key"]
        qid  = f["qid"]
        ftyp = f.get("type", "string")
        val  = extracted.get(key, None)

        if val in (None, ""):
            continue

        if ftyp == "full_name" and isinstance(val, dict):
            if val.get("first"):  put_field(payload, qid, val["first"],  "first")
            if val.get("last"):   put_field(payload, qid, val["last"],   "last")
            if val.get("middle"): put_field(payload, qid, val["middle"], "middle")
            if val.get("suffix"): put_field(payload, qid, val["suffix"], "suffix")
        elif ftyp == "address" and isinstance(val, dict):
            if val.get("line1"):   put_field(payload, qid, val["line1"],   "addr_line1")
            if val.get("line2"):   put_field(payload, qid, val["line2"],   "addr_line2")
            if val.get("city"):    put_field(payload, qid, val["city"],    "city")
            if val.get("state"):   put_field(payload, qid, val["state"],   "state")
            if val.get("postal"):  put_field(payload, qid, val["postal"],  "postal")
            if val.get("country"): put_field(payload, qid, val["country"], "country")
        elif ftyp == "phone" and isinstance(val, dict):
            if val.get("full"):
                put_field(payload, qid, val["full"], "full")
            else:
                if val.get("area"):   put_field(payload, qid, val["area"],   "area")
                if val.get("number"): put_field(payload, qid, val["number"], "phone")
        elif ftyp == "date":
            if isinstance(val, dict):
                # If the model returned components
                m = val.get("month"); d = val.get("day"); y = val.get("year")
                if m and d and y:
                    put_field(payload, qid, m, "month")
                    put_field(payload, qid, d, "day")
                    put_field(payload, qid, y, "year")
                elif val.get("raw"):
                    comp = parse_date_yyyy_mm_dd(val["raw"])
                    if comp:
                        m, d, y = comp
                        put_field(payload, qid, m, "month")
                        put_field(payload, qid, d, "day")
                        put_field(payload, qid, y, "year")
            elif isinstance(val, str):
                comp = parse_date_yyyy_mm_dd(val)
                if comp:
                    m, d, y = comp
                    put_field(payload, qid, m, "month")
                    put_field(payload, qid, d, "day")
                    put_field(payload, qid, y, "year")
        else:
            # string, number, enum -> single value
            put_field(payload, qid, val)

    r = requests.post(url, data=payload, timeout=120)
    try:
        r.raise_for_status()
    except Exception:
        raise RuntimeError(f"Jotform API error: {r.status_code} {r.text[:400]}")
    return r.json()["content"]["submissionID"]

def jotform_edit_link(submission_id: str) -> str:
    return f"https://www.jotform.com/edit/{submission_id}"

# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------
@app.route("/", methods=["GET", "HEAD"])
def health():
    return "Middleware is running.", 200

@app.route("/ingest", methods=["POST"])
def ingest():
    try:
        if not OPENAI_API_KEY:
            return jsonify({"ok": False, "error": "OPENAI_API_KEY not set"}), 500
        if not (JOTFORM_API_KEY and INTAKE_FORM_ID):
            return jsonify({"ok": False, "error": "JOTFORM_API_KEY or INTAKE_FORM_ID missing"}), 500

        body = request.get_json(force=True) or {}
        fub_id      = body.get("fub_id", "")
        agent_email = body.get("agent_email", "")
        uploaded_by = body.get("uploaded_by", "")
        file_url    = body.get("file_url", "")
        if not file_url:
            return jsonify({"ok": False, "error": "file_url is required"}), 400

        # 1) Download PDF
        pdf_bytes, file_hash = download_file(file_url)

        # 2) Build *live* catalog & schema from Jotform questions
        fields = build_live_catalog()
        schema_obj = make_json_schema_from_fields(fields)

        # 3) Extract with OpenAI
        extracted = extract_with_openai(pdf_bytes, schema_obj)

        # 4) Create Jotform submission
        submission_id = jotform_create_submission(
            INTAKE_FORM_ID, fields, extracted, file_hash, fub_id
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
        return jsonify({"ok": False, "error": str(e)}), 400

# -----------------------------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
