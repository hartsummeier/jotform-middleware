# app.py
import os, json, re, hashlib, logging, traceback, requests
from datetime import datetime
from flask import Flask, request, jsonify

OPENAI_API_KEY  = os.environ.get("OPENAI_API_KEY", "")
JOTFORM_API_KEY = os.environ.get("JOTFORM_API_KEY", "")
INTAKE_FORM_ID  = os.environ.get("INTAKE_FORM_ID", "")

app = Flask(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ---------- config ----------
def _strip_json_comments(text: str) -> str:
    # Remove // line comments and /* block */ comments (basic, good enough for our fields.json)
    text = re.sub(r"//.*?$", "", text, flags=re.MULTILINE)
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    return text

def load_cfg():
    with open("fields.json", "r", encoding="utf-8") as f:
        raw = f.read()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Try again after stripping comments if someone added them
        return json.loads(_strip_json_comments(raw))

CFG = load_cfg()
if CFG.get("intake_form_id"):
    INTAKE_FORM_ID = CFG["intake_form_id"]

# ---------- jotform helpers ----------
def extract_form_id(value: str) -> str:
    if not value:
        raise RuntimeError("INTAKE_FORM_ID is missing.")
    m = re.search(r"(\d{8,})", str(value))
    if not m:
        raise RuntimeError(f"Could not parse a numeric form id from '{value}'.")
    return m.group(1)

def jf_get_questions(form_id: str) -> dict:
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
    if override and "type" in override:
        return override["type"]
    t = jf_type or ""
    if t in ("control_fullname",):                  return "full_name"
    if t in ("control_address",):                   return "address"
    if t in ("control_phone",):                     return "phone"
    if t in ("control_datetime",):                  return "date"
    if t in ("control_number", "control_spinner"):  return "number"
    if t in ("control_dropdown", "control_radio"):  return "enum"
    return "string"

def build_live_catalog():
    qs = jf_get_questions(INTAKE_FORM_ID)
    exclude_types = set(CFG.get("exclude_types", []))
    overrides = CFG.get("overrides", {})

    fields = []
    for qid, q in qs.items():
        jf_type = q.get("type") or ""
        if jf_type in exclude_types:
            continue
        unique = q.get("name") or ""    # Unique Name
        label  = q.get("text") or unique
        if not unique:
            continue

        ov    = overrides.get(unique, {})
        ftype = jf_type_to_model_type(jf_type, ov)

        enum = None
        if ftype == "enum":
            props = q.get("properties") or {}
            opts  = props.get("options")
            if isinstance(opts, str):
                enum = [o.strip() for o in opts.split("|") if o.strip()]
            elif isinstance(opts, list):
                enum = [str(o).strip() for o in opts if str(o).strip()]

        fields.append({
            "key": unique,
            "qid": str(qid),
            "label": label,
            "type": ov.get("type", ftype),
            "enum": ov.get("enum", enum)
        })
    logging.info("Live catalog size: %d", len(fields))
    return fields

# ---------- openai helpers ----------
def _sanitize_jotform_file_url(u: str) -> str:
    u = u.strip().strip("<>").strip("\"'")
    u = re.sub(r'%3E(?=\?|$)', '', u, flags=re.IGNORECASE)
    u = re.sub(r'>(?=\?|$)', '', u)
    return u

def download_file(url: str):
    try:
        clean = _sanitize_jotform_file_url(url)
        r = requests.get(clean, timeout=120)
        r.raise_for_status()
        data = r.content
        return data, hashlib.sha256(data).hexdigest()
    except Exception as e:
        raise RuntimeError(
            "Failed to download file from URL. If it's a Jotform upload, turn OFF "
            "'Require log-in to view uploaded files' for testing. Details: %s" % e
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
        key = f["key"]; ftype = f.get("type", "string")
        if ftype == "number":
            props[key] = {"type": "number", "nullable": True}
        elif ftype == "date":
            props[key] = {"type": "string", "nullable": True}
        elif ftype == "enum":
            enum = f.get("enum")
            if enum:
                props[key] = {"type": "string", "enum": enum, "nullable": True}
            else:
                props[key] = {"type": "string", "nullable": True}
        elif ftype == "full_name":
            subprops = {
                "first":  {"type":"string","nullable": True},
                "last":   {"type":"string","nullable": True},
                "middle": {"type":"string","nullable": True},
                "suffix": {"type":"string","nullable": True}
            }
            props[key] = {
                "type": "object",
                "nullable": True,
                "properties": subprops,
                "required": list(subprops.keys()),           # <-- add this
                "additionalProperties": False
            }
        elif ftype == "address":
            subprops = {
                "line1":   {"type":"string","nullable": True},
                "line2":   {"type":"string","nullable": True},
                "city":    {"type":"string","nullable": True},
                "state":   {"type":"string","nullable": True},
                "postal":  {"type":"string","nullable": True},
                "country": {"type":"string","nullable": True}
            }
            props[key] = {
                "type": "object",
                "nullable": True,
                "properties": subprops,
                "required": list(subprops.keys()),           # <-- add this
                "additionalProperties": False
            }
        elif ftype == "phone":
            subprops = {
                "full":   {"type":"string","nullable": True},
                "area":   {"type":"string","nullable": True},
                "number": {"type":"string","nullable": True}
            }
            props[key] = {
                "type": "object",
                "nullable": True,
                "properties": subprops,
                "required": list(subprops.keys()),           # <-- add this
                "additionalProperties": False
            }
        else:
            props[key] = {"type": "string", "nullable": True}

    # meta from model
    props["confidence"] = {"type":"number","minimum":0,"maximum":1,"nullable": True}
    props["page_refs"]  = {"type":"array","items":{"type":"integer"}, "nullable": True}

    return {
        "name": "intake_extract",
        "schema": {
            "type":"object",
            "properties": props,
            "required": list(props.keys()),
            "additionalProperties": False
        },
        "strict": True
    }

def extract_with_openai(pdf_bytes: bytes, schema_obj: dict) -> dict:
    file_id = openai_upload_file(pdf_bytes)
    url = "https://api.openai.com/v1/responses"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}

    format_block = {"type": "json_schema", **schema_obj}  # << add type here

    payload = {
        "model": "gpt-4o-mini",
        "input": [{
            "role": "user",
            "content": [
                {"type":"input_text","text":
                 "Read this real estate purchase contract (and addenda). "
                 "Extract values that match the schema keys. "
                 "Use null when not present. Format dates as YYYY-MM-DD. "
                 "For names/addresses/phones, fill sub-fields if available. "
                 "Also return page_refs (page numbers) and an overall confidence 0-1."},
                {"type":"input_file","file_id": file_id}
            ]
        }],
        "text": { "format": format_block },  # <-- now includes 'type'
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

# ---------- submission helpers ----------
def put_field(payload: dict, qid: str, value, subkey: str | None = None):
    payload[f"submission[{qid}][{subkey}]" if subkey else f"submission[{qid}]"] = value

def parse_date_yyyy_mm_dd(s: str):
    if not s or not isinstance(s, str): return None
    s = s.strip()
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y", "%m-%d-%Y"):
        try:
            dt = datetime.strptime(s, fmt)
            return dt.month, dt.day, dt.year
        except Exception:
            pass
    m = re.findall(r"\d+", s)
    if len(m) == 3:
        a, b, c = [int(x) for x in m]
        if len(m[0]) == 4: return b, c, a
        if len(m[-1]) == 4: return a, b, c
    return None

def jotform_create_submission(form_id: str, fields: list[dict], extracted: dict,
                              file_hash: str, fub_id: str | None = None) -> str:
    form_id = extract_form_id(form_id)
    url = f"https://api.jotform.com/form/{form_id}/submissions?apiKey={JOTFORM_API_KEY}"
    payload = {}

    # Optional metadata sink — point this to a long text QID if you create one.
    # meta_qid = None
    meta = {
        "overall": extracted.get("confidence"),
        "page_refs": extracted.get("page_refs"),
        "source_file_hash": file_hash,
        "fub_id": fub_id
    }
    # if meta_qid: put_field(payload, meta_qid, json.dumps(meta))

    for f in fields:
        key, qid, ftyp = f["key"], f["qid"], f.get("type", "string")
        val = extracted.get(key, None)
        if val in (None, ""): continue

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
            if val.get("full"): put_field(payload, qid, val["full"], "full")
            else:
                if val.get("area"):   put_field(payload, qid, val["area"],   "area")
                if val.get("number"): put_field(payload, qid, val["number"], "phone")
        elif ftyp == "date":
            if isinstance(val, dict):
                m, d, y = val.get("month"), val.get("day"), val.get("year")
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
            put_field(payload, qid, val)

    r = requests.post(url, data=payload, timeout=120)
    try:
        r.raise_for_status()
    except Exception:
        raise RuntimeError(f"Jotform API error: {r.status_code} {r.text[:400]}")
    return r.json()["content"]["submissionID"]

def jotform_edit_link(submission_id: str) -> str:
    return f"https://www.jotform.com/edit/{submission_id}"

# ---------- routes ----------
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

        pdf_bytes, file_hash = download_file(file_url)
        fields     = build_live_catalog()
        schema_obj = make_json_schema_from_fields(fields)
        extracted  = extract_with_openai(pdf_bytes, schema_obj)

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

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
