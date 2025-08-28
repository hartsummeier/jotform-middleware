# app.py
import os, re, json, hashlib, logging, traceback, requests
from datetime import datetime
from io import BytesIO

from flask import Flask, request, jsonify
from pdfminer.high_level import extract_text
from rapidfuzz import fuzz, process as rf_process

# -----------------------------------------------------------------------------
# Environment
# -----------------------------------------------------------------------------
OPENAI_API_KEY  = os.environ.get("OPENAI_API_KEY", "")
JOTFORM_API_KEY = os.environ.get("JOTFORM_API_KEY", "")
INTAKE_FORM_ID  = os.environ.get("INTAKE_FORM_ID", "")  # can be overridden via fields.json

# -----------------------------------------------------------------------------
# App & logging
# -----------------------------------------------------------------------------
app = Flask(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# -----------------------------------------------------------------------------
# Config loader (fields.json is optional but supported)
# -----------------------------------------------------------------------------
def _strip_json_comments(text: str) -> str:
    text = re.sub(r"//.*?$", "", text, flags=re.MULTILINE)
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    return text

def load_cfg():
    try:
        with open("fields.json", "r", encoding="utf-8") as f:
            raw = f.read()
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return json.loads(_strip_json_comments(raw))
    except FileNotFoundError:
        return {
            "intake_form_id": INTAKE_FORM_ID or "",
            "include_all_questions": True,
            "exclude_types": [
                "control_head","control_pagebreak","control_button",
                "control_image","control_text","control_divider",
                "control_collapse","control_signature","control_matrix"
            ],
            "overrides": {},
            "strict_schema": True
        }

CFG = load_cfg()
if CFG.get("intake_form_id"):
    INTAKE_FORM_ID = CFG["intake_form_id"]

# -----------------------------------------------------------------------------
# Jotform helpers
# -----------------------------------------------------------------------------
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
        raise RuntimeError(f"Jotform /questions error: {r.status_code} {r.text[:600]}")
    return r.json().get("content", {})

def jf_type_to_model_type(jf_type: str, override: dict | None) -> str:
    if override and "type" in override:
        return override["type"]
    t = jf_type or ""
    if t in ("control_fullname",):                  return "full_name"
    if t in ("control_address",):                   return "address"
    if t in ("control_phone",):                     return "phone"
    if t in ("control_datetime", "control_birthdate", "control_time"): return "date"
    if t in ("control_number", "control_spinner", "control_currency"): return "number"
    if t in ("control_checkbox",):                  return "multi_enum"
    if t in ("control_dropdown", "control_radio"):  return "enum"
    return "string"

def build_live_catalog():
    """
    Returns a list of fields with:
      key  = Jotform unique name
      qid  = Jotform question id
      label= Jotform text/label
      type = normalized type (string, number, date, enum, multi_enum, full_name, address, phone)
      enum = list of options if enum/multi_enum
    """
    qs = jf_get_questions(INTAKE_FORM_ID)
    exclude_types = set(CFG.get("exclude_types", []))
    overrides = CFG.get("overrides", {})

    fields = []
    for qid, q in qs.items():
        jf_type = q.get("type") or ""
        if jf_type in exclude_types:
            continue
        unique = q.get("name") or ""   # Unique Name
        label  = q.get("text") or unique
        if not unique:
            continue

        ov    = overrides.get(unique, {})
        ftype = jf_type_to_model_type(jf_type, ov)

        enum = None
        if ftype in ("enum", "multi_enum"):
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

# -----------------------------------------------------------------------------
# File I/O helpers
# -----------------------------------------------------------------------------
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

# -----------------------------------------------------------------------------
# PDF text (heuristics pre-pass)
# -----------------------------------------------------------------------------
def pdf_to_text(pdf_bytes: bytes) -> str:
    try:
        return extract_text(BytesIO(pdf_bytes))
    except Exception as e:
        logging.warning("PDF text extraction failed: %s", e)
        return ""

_money_rx = re.compile(r"\$?\s*([0-9]{1,3}(?:[, ][0-9]{3})*(?:\.[0-9]{2})?)")
_zip_rx   = re.compile(r"\b\d{5}(?:-\d{4})?\b")

def find_money_after(label: str, text: str):
    """Find currency near a label (e.g., 'Purchase Price')."""
    block_rx = re.compile(rf"{re.escape(label)}[:\s]*([^\n]{{0,80}})", re.IGNORECASE)
    m = block_rx.search(text)
    if m:
        m2 = _money_rx.search(m.group(1))
        if m2:
            amt = m2.group(1).replace(" ", "").replace(",", "")
            return amt
    # global fallback (first money)
    m3 = _money_rx.search(text)
    if m3:
        return m3.group(1).replace(",", "").replace(" ", "")
    return None

def detect_payment_method(text: str, options: list[str] | None):
    """
    Tries to detect method of payment from text using fuzzy match over expected options.
    """
    if not options:
        # common defaults
        options = ["Cash", "Conventional", "Insured Conventional", "FHA", "VA"]
    best, score, _ = rf_process.extractOne(text, options, scorer=fuzz.WRatio) if text else (None, 0, 0)
    # naive: if the doc explicitly says CASH, force it
    if re.search(r"\bcash\b", text, re.IGNORECASE):
        return "Cash"
    return best if score >= 75 else None

def find_address_block(text: str):
    """
    Grab the first address-like pattern: line with number + street,
    followed by City, ST ZIP on next line.
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    for i in range(len(lines) - 1):
        ln = lines[i]
        if re.search(r"^\d{1,6}\s+\S+", ln):  # starts with house number
            nxt = lines[i+1]
            m = re.search(r"^(.+?),\s*([A-Z]{2})\s+(\d{5}(?:-\d{4})?)$", nxt)
            if m:
                line1 = ln
                city  = m.group(1).strip()
                state = m.group(2).strip()
                postal= m.group(3).strip()
                return {
                    "line1": line1,
                    "line2": "",
                    "city": city,
                    "state": state,
                    "postal": postal,
                    "country": "US"
                }
    return None

def detect_appraisal(text: str):
    """
    Look for appraisal Yes/No statements.
    """
    t = text.lower()
    if "appraisal" in t:
        if re.search(r"appraisal[^.\n]{0,30}\byes\b", t):
            return True
        if re.search(r"appraisal[^.\n]{0,30}\bno\b", t):
            return False
        # heuristic: if financing present, default to appraisal True
        if re.search(r"\bloan\b|\bmortgage\b|\bfinanc", t):
            return True
    return None

def heuristic_value_for(field: dict, text: str):
    """
    Try to compute a value for a single field based on its label / unique name and PDF text.
    Covers common cases: purchase price, earnest money, property address, method of payment, appraisal.
    Returns a value matching the field's expected type (string/number/object/list/etc) or None.
    """
    label = (field.get("label") or "").lower()
    uname = (field.get("key") or "").lower()
    ftype = field.get("type", "string")
    options = field.get("enum")

    # Purchase Price
    if "purchase price" in label or "purchase_price" in uname:
        amt = find_money_after("Purchase Price", text) or find_money_after("Price", text)
        return amt

    # Earnest Money Amount
    if ("earnest" in label and "amount" in label) or "earnest" in uname:
        amt = find_money_after("Earnest", text) or find_money_after("Earnest Money", text)
        return amt

    # Property Address (address widget)
    if ftype == "address" or "property address" in label:
        addr = find_address_block(text)
        return addr

    # Method of Payment / Financing Type
    if ("method of payment" in label) or ("financing" in label) or ("loan type" in label) or ("payment method" in label):
        m = detect_payment_method(text, options)
        return m

    # Appraisal (checkbox or yes/no)
    if "appraisal" in label:
        ap = detect_appraisal(text)
        if ap is None:
            return None
        # If this field is enum yes/no
        if ftype == "enum" and options:
            return "Yes" if ap else "No"
        # If checkbox exists, return a list containing "Appraisal"
        if ftype == "multi_enum" and options:
            if ap and any("appraisal" in o.lower() for o in options):
                # Select Appraisal in a multi-select contingencies control
                return [next(o for o in options if "appraisal" in o.lower())]
            else:
                return []
        # Otherwise return boolean-ish text
        return "Yes" if ap else "No"

    return None

def heuristic_extract(pdf_bytes: bytes, fields: list[dict]) -> dict:
    """
    Run heuristics across all fields and return a partial dict: {unique_name: value}
    Only fills what we can confidently find.
    """
    out = {}
    text = pdf_to_text(pdf_bytes)

    for f in fields:
        try:
            val = heuristic_value_for(f, text)
            if val not in (None, "", [], {}):
                out[f["key"]] = val
        except Exception as e:
            logging.debug("Heuristic failed for %s: %s", f.get("key"), e)

    return out

# -----------------------------------------------------------------------------
# OpenAI (Responses API) — Structured Outputs
# -----------------------------------------------------------------------------
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
        raise RuntimeError(f"OpenAI file upload error: {r.status_code} {r.text[:600]}")
    return r.json()["id"]

def make_json_schema_from_fields(fields: list[dict]) -> dict:
    """
    Build a permissive JSON Schema for the given fields.
    """
    props = {}
    for f in fields:
        key = f["key"]; ftype = f.get("type", "string")
        if ftype == "number":
            props[key] = {"type": "number"}
        elif ftype == "date":
            props[key] = {"type": "string"}
        elif ftype == "enum":
            enum = f.get("enum")
            props[key] = {"type": "string", "enum": enum} if enum else {"type": "string"}
        elif ftype == "multi_enum":
            enum = f.get("enum") or []
            if enum:
                props[key] = {"type": "array", "items": {"type": "string", "enum": enum}}
            else:
                props[key] = {"type": "array", "items": {"type": "string"}}
        elif ftype == "full_name":
            props[key] = {
                "type": "object",
                "properties": {
                    "first": {"type":"string"},
                    "last":  {"type":"string"},
                    "middle":{"type":"string"},
                    "suffix":{"type":"string"}
                },
                "additionalProperties": False
            }
        elif ftype == "address":
            props[key] = {
                "type": "object",
                "properties": {
                    "line1":   {"type":"string"},
                    "line2":   {"type":"string"},
                    "city":    {"type":"string"},
                    "state":   {"type":"string"},
                    "postal":  {"type":"string"},
                    "country": {"type":"string"}
                },
                "additionalProperties": False
            }
        elif ftype == "phone":
            props[key] = {
                "type": "object",
                "properties": {
                    "full":   {"type":"string"},
                    "area":   {"type":"string"},
                    "number": {"type":"string"}
                },
                "additionalProperties": False
            }
        else:
            props[key] = {"type": "string"}

    # add meta channels if we want to store them later via a long text
    props["confidence"] = {"type":"number"}
    props["page_refs"]  = {"type":"array","items":{"type":"integer"}}

    schema = {
        "type": "object",
        "properties": props,
        "required": [],
        "additionalProperties": False
    }
    return {
        "name": "intake_extract",
        "schema": schema,
        "strict": True
    }

def _build_json_schema_format(schema_obj: dict) -> dict:
    """
    Returns the correct 'text.format' block for the Responses API:
      "text": { "format": { "type":"json_schema", "name": "...", "schema": {...}, "strict": true } }
    """
    if not isinstance(schema_obj, dict):
        raise RuntimeError("schema_obj must be a dict.")

    name   = schema_obj.get("name") or "intake_extract"
    schema = schema_obj.get("schema") or schema_obj
    strict = schema_obj.get("strict", True)

    if not isinstance(schema, dict) or "type" not in schema or "properties" not in schema:
        if isinstance(schema_obj, dict) and "schema" in schema_obj and isinstance(schema_obj["schema"], dict):
            schema = schema_obj["schema"]

    if "type" not in schema or "properties" not in schema:
        raise RuntimeError("JSON schema missing 'type' or 'properties'.")

    return {
        "type":   "json_schema",
        "name":   name,
        "schema": schema,
        "strict": bool(strict),
    }

def extract_with_openai(pdf_bytes: bytes, schema_obj: dict) -> dict:
    file_id = openai_upload_file(pdf_bytes)
    format_block = _build_json_schema_format(schema_obj)

    url = "https://api.openai.com/v1/responses"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
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
                 "Also return page_refs (page numbers) and an overall confidence 0-1."
                },
                {"type":"input_file","file_id": file_id}
            ]
        }],
        "text": {"format": format_block},
        "temperature": 0
    }

    logging.info("OPENAI payload text.format keys: %s", list(payload["text"]["format"].keys()))

    r = requests.post(url, headers=headers, json=payload, timeout=240)
    try:
        r.raise_for_status()
    except Exception:
        raise RuntimeError(f"OpenAI API error: {r.status_code} {r.text[:800]}")

    data = r.json()
    parsed = data.get("output_parsed")
    if not parsed:
        try:
            text_piece = data.get("output", [{}])[0].get("content", [{}])[0].get("text", "{}")
            parsed = json.loads(text_piece)
        except Exception:
            parsed = {}
    return parsed

def llm_extract_missing(pdf_bytes: bytes, fields_subset: list[dict]) -> dict:
    """
    Fields subset: a small list of field dicts. We build a tiny schema & query once.
    """
    if not fields_subset:
        return {}
    subset_schema = make_json_schema_from_fields(fields_subset)
    file_id = openai_upload_file(pdf_bytes)
    format_block = _build_json_schema_format(subset_schema)

    url = "https://api.openai.com/v1/responses"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
    payload = {
        "model": "gpt-4o-mini",
        "input": [{
            "role": "user",
            "content": [
                {"type":"input_text","text":
                 "Fill ONLY the fields present in the schema. "
                 "Use null for anything you cannot find. Keep dates as YYYY-MM-DD."
                },
                {"type":"input_file","file_id": file_id}
            ]
        }],
        "text": {"format": format_block},
        "temperature": 0
    }

    logging.info("Missing-fields call — text.format keys: %s", list(payload["text"]["format"].keys()))

    r = requests.post(url, headers=headers, json=payload, timeout=180)
    try:
        r.raise_for_status()
    except Exception:
        raise RuntimeError(f"OpenAI API error: {r.status_code} {r.text[:800]}")

    data = r.json()
    parsed = data.get("output_parsed")
    if not parsed:
        try:
            text_piece = data.get("output", [{}])[0].get("content", [{}])[0].get("text", "{}")
            parsed = json.loads(text_piece)
        except Exception:
            parsed = {}
    return parsed

# -----------------------------------------------------------------------------
# Submission helpers
# -----------------------------------------------------------------------------
def put_field(payload: dict, qid: str, value, subkey: str | None = None):
    key = f"submission[{qid}]" if subkey is None else f"submission[{qid}][{subkey}]"
    payload[key] = value

def parse_date_yyyy_mm_dd(s: str):
    if not s or not isinstance(s, str):
        return None
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
    if not JOTFORM_API_KEY:
        raise RuntimeError("JOTFORM_API_KEY is missing.")
    form_id = extract_form_id(form_id)
    url = f"https://api.jotform.com/form/{form_id}/submissions?apiKey={JOTFORM_API_KEY}"
    payload = {}

    # Optional meta sink could be posted to a long text field if desired
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
        if key not in extracted:
            continue
        val = extracted.get(key)
        if val in (None, "", [], {}):
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

        elif ftyp == "multi_enum":
            # Jotform checkbox usually tolerates comma-separated string
            if isinstance(val, list):
                put_field(payload, qid, ", ".join([str(x) for x in val]))
            else:
                put_field(payload, qid, str(val))

        else:
            # string, number, enum, etc.
            put_field(payload, qid, val)

    r = requests.post(url, data=payload, timeout=120)
    try:
        r.raise_for_status()
    except Exception:
        raise RuntimeError(f"Jotform API error: {r.status_code} {r.text[:800]}")
    return r.json()["content"]["submissionID"]

def jotform_edit_link(submission_id: str) -> str:
    return f"https://www.jotform.com/edit/{submission_id}"

# -----------------------------------------------------------------------------
# Hybrid Extract (heuristics + LLM fallback)
# -----------------------------------------------------------------------------
def hybrid_extract(pdf_bytes: bytes, fields: list[dict]) -> dict:
    # Pass 1: heuristics
    partial = heuristic_extract(pdf_bytes, fields)

    # Determine still-missing subset (skip meta keys)
    missing = []
    for f in fields:
        k = f["key"]
        if k not in partial:
            missing.append(f)

    # If everything missing, or a lot missing, ask model for the whole set to start
    if len(partial) < 3:
        schema_full = make_json_schema_from_fields(fields)
        try:
            llm_full = extract_with_openai(pdf_bytes, schema_full)
            partial.update({k: v for k, v in llm_full.items() if k in {f["key"] for f in fields}})
        except Exception as e:
            logging.error("LLM full extract failed: %s", e)

    # Still missing? Ask for just the remainder
    still_missing = [f for f in fields if f["key"] not in partial]
    if still_missing:
        try:
            llm_extra = llm_extract_missing(pdf_bytes, still_missing)
            partial.update({k: v for k, v in llm_extra.items() if k in {f["key"] for f in fields}})
        except Exception as e:
            logging.error("LLM missing-fields extract failed: %s", e)

    return partial

# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------
@app.route("/", methods=["GET", "HEAD"])
def health():
    return "Middleware is running.", 200

@app.route("/debug/schema", methods=["POST"])
def debug_schema():
    try:
        body = request.get_json(force=True) or {}
        fields = body.get("fields") or []
        if not isinstance(fields, list) or not fields:
            return jsonify({"ok": False, "error": "fields (list) required"}), 400
        schema_obj = make_json_schema_from_fields(fields)
        fmt = _build_json_schema_format(schema_obj)
        return jsonify({"ok": True, "format": fmt})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400

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
        debug_only  = bool(body.get("debug", False))
        if not file_url:
            return jsonify({"ok": False, "error": "file_url is required"}), 400

        pdf_bytes, file_hash = download_file(file_url)
        fields = build_live_catalog()

        extracted = hybrid_extract(pdf_bytes, fields)

        if debug_only:
            return jsonify({
                "ok": True,
                "extracted_preview": {k: extracted.get(k) for k in sorted(extracted.keys())},
                "counts": {"fields_total": len(fields), "filled": len(extracted)},
                "file_hash": file_hash
            })

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
