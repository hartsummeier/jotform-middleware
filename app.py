# app.py
import os, io, re, json, hashlib, logging, traceback, requests
from datetime import datetime
from flask import Flask, request, jsonify

# PDF text extraction & fuzzy match
from pdfminer.high_level import extract_text
from rapidfuzz import process, fuzz

# -----------------------------------------------------------------------------
# Environment
# -----------------------------------------------------------------------------
OPENAI_API_KEY  = os.environ.get("OPENAI_API_KEY", "")
JOTFORM_API_KEY = os.environ.get("JOTFORM_API_KEY", "")
INTAKE_FORM_ID  = os.environ.get("INTAKE_FORM_ID", "")  # may be overridden by fields.json

# -----------------------------------------------------------------------------
# App & logging
# -----------------------------------------------------------------------------
app = Flask(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# -----------------------------------------------------------------------------
# Load fields.json (lightweight, comment-tolerant)
# -----------------------------------------------------------------------------
def _strip_json_comments(text: str) -> str:
    text = re.sub(r"//.*?$", "", text, flags=re.MULTILINE)
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    return text

def load_cfg():
    with open("fields.json", "r", encoding="utf-8") as f:
        raw = f.read()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return json.loads(_strip_json_comments(raw))

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
    """Return dict of question id -> question JSON."""
    if not JOTFORM_API_KEY:
        raise RuntimeError("JOTFORM_API_KEY is not set.")
    form_id = extract_form_id(form_id)
    url = f"https://api.jotform.com/form/{form_id}/questions?apiKey={JOTFORM_API_KEY}"
    r = requests.get(url, timeout=90)
    r.raise_for_status()
    return r.json()["content"]

def jf_type_to_model_type(jf_type: str, override: dict|None) -> str:
    if override and "type" in override:
        return override["type"]
    t = (jf_type or "").lower()
    if t == "control_fullname":  return "full_name"
    if t == "control_address":   return "address"
    if t == "control_phone":     return "phone"
    if t == "control_datetime":  return "date"
    if t in ("control_number", "control_spinner"): return "number"
    if t in ("control_dropdown", "control_radio"): return "enum"
    if t == "control_checkbox":  return "multi_enum"
    return "string"

def build_live_catalog():
    """Read Jotform questions and return a list of normalized fields with options."""
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

        # collect options for enums/checkbox
        enum = None
        if ftype in ("enum", "multi_enum"):
            props = q.get("properties") or {}
            opts  = props.get("options")
            if isinstance(opts, str):
                enum = [o.strip() for o in opts.split("|") if o.strip()]
            elif isinstance(opts, list):
                enum = [str(o).strip() for o in opts if str(o).strip()]

        fields.append({
            "key": unique,             # Unique Name
            "qid": str(qid),           # Question ID
            "label": label,            # Human label
            "jf_type": jf_type,        # Original Jotform type
            "type": ov.get("type", ftype),
            "enum": ov.get("enum", enum)
        })
    logging.info("Live catalog size: %d", len(fields))
    return fields

# Quick label finder for mapping
def key_by_label(fields, pattern):
    rx = re.compile(pattern, re.I)
    for f in fields:
        if rx.search(f["label"]):
            return f["key"]
    return None

# -----------------------------------------------------------------------------
# OpenAI helpers
# -----------------------------------------------------------------------------
def download_file(url: str):
    clean = url.strip().strip("<>").strip("\"'")
    clean = re.sub(r'%3E(?=\?|$)', '', clean, flags=re.IGNORECASE)
    clean = re.sub(r'>(?=\?|$)', '', clean)
    r = requests.get(clean, timeout=180)
    r.raise_for_status()
    data = r.content
    return data, hashlib.sha256(data).hexdigest()

def openai_upload_file(pdf_bytes: bytes, filename="contract.pdf") -> str:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is missing.")
    url = "https://api.openai.com/v1/files"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
    files = {"file": (filename, pdf_bytes, "application/pdf")}
    data  = {"purpose": "assistants"}
    r = requests.post(url, headers=headers, files=files, data=data, timeout=180)
    r.raise_for_status()
    return r.json()["id"]

def make_json_schema_from_subset(subset_fields):
    """
    Build a compact JSON Schema for the still-missing fields.
    Keep it simple to avoid API rejections.
    """
    props = {}
    for f in subset_fields:
        k, t = f["key"], f.get("type", "string")

        # Basic shapes only (avoid nullable for now)
        if t == "number":
            props[k] = {"type": "number"}
        elif t == "date":
            props[k] = {"type": "string"}
        elif t == "enum":
            enum = f.get("enum")
            if enum:
                props[k] = {"type": "string", "enum": enum}
            else:
                props[k] = {"type": "string"}
        elif t == "multi_enum":
            enum = f.get("enum") or []
            if enum:
                props[k] = {"type": "array", "items": {"type": "string", "enum": enum}}
            else:
                props[k] = {"type": "array", "items": {"type": "string"}}
        elif t == "full_name":
            props[k] = {
                "type": "object",
                "properties": {
                    "first":  {"type":"string"},
                    "last":   {"type":"string"},
                    "middle": {"type":"string"},
                    "suffix": {"type":"string"}
                },
                "additionalProperties": False
            }
        elif t == "address":
            props[k] = {
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
        elif t == "phone":
            props[k] = {
                "type": "object",
                "properties": {
                    "full":   {"type":"string"},
                    "area":   {"type":"string"},
                    "number": {"type":"string"}
                },
                "additionalProperties": False
            }
        else:
            props[k] = {"type": "string"}

    # Keep it permissive; we’ll post-process
    schema = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "properties": props,
        "required": [],                    # not forcing presence
        "additionalProperties": False
    }

    return {
        "name": "intake_extract_subset",
        "schema": schema,
        "strict": True
    }


def llm_extract_missing(pdf_bytes: bytes, missing_fields):
    """
    Ask the model ONLY for missing fields, in small batches (to keep schema small).
    Uses the Responses API with text.format.json_schema.
    """
    if not missing_fields:
        return {}

    # Upload the PDF once
    file_id = openai_upload_file(pdf_bytes)

    # Chunk missing fields to avoid oversized schemas
    BATCH = 24
    merged = {}

    for i in range(0, len(missing_fields), BATCH):
        batch = missing_fields[i:i+BATCH]
        schema_obj = make_json_schema_from_subset(batch)

        url = "https://api.openai.com/v1/responses"
        headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
        payload = {
            "model": "gpt-4o-mini",
            "input": [{
                "role":"user",
                "content":[
                    {"type":"input_text","text":
                     "Read this real estate purchase contract and extract ONLY the fields described by the JSON schema. "
                     "Use null when not present. Dates as YYYY-MM-DD if possible."
                    },
                    {"type":"input_file","file_id": file_id}
                ]
            }],
            "text": {
                "format": {
                    "type": "json_schema",
                    "json_schema": {                 # <<<<<< key difference
                        "name":   schema_obj["name"],
                        "schema": schema_obj["schema"],
                        "strict": True
                    }
                }
            },
            "temperature": 0
        }

        r = requests.post(url, headers=headers, json=payload, timeout=240)
        if not r.ok:
            # Log exact server message to Render logs AND surface it to caller
            try:
                detail = r.text[:800]
            except Exception:
                detail = "<no body>"
            raise RuntimeError(f"OpenAI API error: {r.status_code} {detail}")

        data = r.json()
        parsed = data.get("output_parsed")
        if parsed is None:
            try:
                txt = data.get("output",[{}])[0].get("content",[{}])[0].get("text","{}")
                parsed = json.loads(txt)
            except Exception:
                parsed = {}

        # merge per-batch
        if isinstance(parsed, dict):
            merged.update(parsed)

    return merged

# -----------------------------------------------------------------------------
# Rule-based extraction (template-driven)
# -----------------------------------------------------------------------------
def pdf_to_text(pdf_bytes: bytes) -> str:
    with io.BytesIO(pdf_bytes) as f:
        return extract_text(f) or ""

NUM_RE = r"(?:(?:\$?\s*)?([0-9][0-9,\.]*))"
DATE_RE = r"([A-Za-z]{3,9}\s+\d{1,2},\s*\d{4}|\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{2}-\d{2})"

def find_one(rx, text, flags=re.I):
    m = re.search(rx, text, flags)
    return (m.group(1).strip() if m else None)

def to_number(val):
    if val is None: return None
    if isinstance(val, (int, float)): return val
    if isinstance(val, str):
        s = re.sub(r"[^\d\.-]", "", val)
        try:
            n = float(s)
            return int(n) if n.is_integer() else n
        except Exception:
            return None
    return None

def parse_date_parts(s: str):
    if not s or not isinstance(s, str): return None
    s = s.strip()
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%m-%d-%Y", "%b %d, %Y", "%B %d, %Y", "%Y/%m/%d"):
        try:
            dt = datetime.strptime(s, fmt)
            return dt.month, dt.day, dt.year
        except: pass
    nums = re.findall(r"\d+", s)
    if len(nums) == 3:
        a,b,c = [int(x) for x in nums]
        if len(nums[0]) == 4: return b,c,a
        if len(nums[-1]) == 4: return a,b,c
    return None

def choose_option(value: str, options: list[str]) -> str|None:
    """Pick the closest label from options for a single-select enum."""
    if not value or not options: return None
    # normalize common synonyms
    v = value.strip().lower()
    synonyms = {
        "lb": "listing brokerage",
        "listing broker": "listing brokerage",
        "sb": "selling brokerage",
        "title": "title company",
        "cash purchase": "cash",
        "conv": "conventional",
        "insured conv": "insured conventional",
        "yes": "yes", "no": "no"
    }
    v = synonyms.get(v, v)
    best, score, _ = process.extractOne(v, options, scorer=fuzz.token_sort_ratio)
    return best if score >= 70 else None

def rule_extract(text: str) -> dict:
    """
    Pull the obvious fields by pattern.
    Tuned for your Purchase Agreement template family.
    """
    out = {}

    # Purchase Price
    # e.g., "Purchase Price: $330,000.00" or "PURCHASE PRICE $330,000"
    price = find_one(r"purchase\s+price[:\s]*" + NUM_RE, text)
    out["purchase_price"] = to_number(price)

    # Earnest Money amount
    em = find_one(r"earnest\s+money[^.\n]*" + NUM_RE, text)
    out["earnest_money"] = to_number(em)

    # Earnest Money due date (if present as a date)
    em_due = find_one(r"earnest\s+money[^.\n]*(?:due|delivered|deposit)[^.\n]*" + DATE_RE, text)
    out["earnest_due"] = em_due

    # Earnest Money holder
    em_holder = find_one(r"(?:held\s+by|deposited\s+with)\s+([A-Za-z ]+?)(?:\.|\n|,)", text)
    if em_holder:
        out["earnest_holder"] = em_holder

    # Property Address (simple pass: first address-like line)
    addr_line = find_one(r"(?i)(?:property\s+address|address)[:\s]*([^\n]+)\n", text)
    if addr_line:
        out["property_line1"] = addr_line

    # Method of Payment (Cash / Conventional / FHA / VA / etc.)
    mop = find_one(r"(?:method\s+of\s+payment|type\s+of\s+financing)[:\s]*([A-Za-z ]+)", text)
    if mop:
        out["method_of_payment"] = mop

    # Appraisal (common check on cash deals)
    app = find_one(r"(?:appraisal)[^.\n]*(yes|no)\b", text)
    if app:
        out["appraisal_cash"] = app

    return out

# -----------------------------------------------------------------------------
# Submission helpers (supports arrays for checkboxes)
# -----------------------------------------------------------------------------
def put_item(items, qid: str, value, subkey: str|None = None, is_array: bool = False):
    if value is None: return
    if is_array:
        items.append((f"submission[{qid}][]", value))
    else:
        if subkey:
            items.append((f"submission[{qid}][{subkey}]", value))
        else:
            items.append((f"submission[{qid}]", value))

def submit_to_jotform(form_id: str, fields: list[dict], extracted: dict,
                      file_hash: str, fub_id: str|None = None) -> str:
    form_id = extract_form_id(form_id)
    url = f"https://api.jotform.com/form/{form_id}/submissions?apiKey={JOTFORM_API_KEY}"

    items: list[tuple[str,str]] = []

    # (optional) metadata sink if you later add a Long Text field for debugging
    # meta_qid = None
    meta = {"source_file_hash": file_hash, "fub_id": fub_id}
    # if meta_qid: put_item(items, meta_qid, json.dumps(meta))

    # Build a quick lookup of key -> field spec
    field_by_key = {f["key"]: f for f in fields}

    for key, val in extracted.items():
        spec = field_by_key.get(key)
        if not spec or val in (None, ""):
            continue
        qid  = spec["qid"]
        ftyp = spec.get("type", "string")

        if ftyp == "full_name" and isinstance(val, dict):
            for sub in ("first","last","middle","suffix"):
                if val.get(sub): put_item(items, qid, val[sub], sub)
        elif ftyp == "address" and isinstance(val, dict):
            m = {
                "line1":"addr_line1","line2":"addr_line2",
                "city":"city","state":"state","postal":"postal","country":"country"
            }
            for sub, jf_sub in m.items():
                if val.get(sub): put_item(items, qid, val[sub], jf_sub)
        elif ftyp == "phone" and isinstance(val, dict):
            if val.get("full"): put_item(items, qid, val["full"], "full")
            else:
                if val.get("area"):   put_item(items, qid, val["area"], "area")
                if val.get("number"): put_item(items, qid, val["number"], "phone")
        elif ftyp == "date":
            if isinstance(val, dict):
                m,d,y = val.get("month"), val.get("day"), val.get("year")
                if m and d and y:
                    put_item(items, qid, m, "month")
                    put_item(items, qid, d, "day")
                    put_item(items, qid, y, "year")
            elif isinstance(val, str):
                comp = parse_date_parts(val)
                if comp:
                    m,d,y = comp
                    put_item(items, qid, m, "month")
                    put_item(items, qid, d, "day")
                    put_item(items, qid, y, "year")
        elif ftyp == "number":
            n = to_number(val)
            if n is not None:
                put_item(items, qid, n)
        elif ftyp == "multi_enum":
            if isinstance(val, list):
                for opt in val:
                    put_item(items, qid, opt, is_array=True)
        else:
            put_item(items, qid, val)

    r = requests.post(url, data=items, timeout=180)
    r.raise_for_status()
    return r.json()["content"]["submissionID"]

# -----------------------------------------------------------------------------
# Hybrid extraction orchestrator
# -----------------------------------------------------------------------------
def map_rule_results_to_keys(rule_out: dict, fields: list[dict]) -> dict:
    """
    Convert domain results (purchase_price, earnest_money, ...) into
    {<unique_name>: value} using label lookup so you don't have to maintain keys.
    """
    out = {}

    # Locate keys by label text
    k_price   = key_by_label(fields, r"what\s+is\s+the\s+purchase\s+price")
    k_em_amt  = key_by_label(fields, r"how\s+much\s+is\s+the\s+earnest\s+money")
    k_em_due  = key_by_label(fields, r"when\s+is\s+the\s+earnest\s+money\s+due")
    k_em_hold = key_by_label(fields, r"who\s+is\s+holding\s+earnest\s+money")
    k_addr    = key_by_label(fields, r"address$")
    k_mop     = key_by_label(fields, r"method\s+of\s+payment")
    k_app     = key_by_label(fields, r"will\s+buyer\s+have\s+an\s+appraisal")

    # Numbers & dates
    if rule_out.get("purchase_price") is not None and k_price:
        out[k_price] = rule_out["purchase_price"]
    if rule_out.get("earnest_money") is not None and k_em_amt:
        out[k_em_amt] = rule_out["earnest_money"]
    if rule_out.get("earnest_due") and k_em_due:
        out[k_em_due] = rule_out["earnest_due"]

    # Address (simple best-effort single-line -> line1)
    if rule_out.get("property_line1") and k_addr:
        out[k_addr] = {
            "line1": rule_out["property_line1"],
            "line2": None, "city": None, "state": None, "postal": None, "country": None
        }

    # Single-select enums: choose the nearest real option
    def choose_for_key(k, val):
        if not k or not val: return
        spec = next((f for f in fields if f["key"] == k), None)
        best = choose_option(str(val), spec.get("enum", [])) if spec else None
        if best: out[k] = best

    choose_for_key(k_em_hold, rule_out.get("earnest_holder"))
    choose_for_key(k_mop,     rule_out.get("method_of_payment"))
    choose_for_key(k_app,     rule_out.get("appraisal_cash"))

    return out

def hybrid_extract(pdf_bytes: bytes, fields: list[dict]) -> dict:
    """
    1) Parse PDF text with regex anchors (fast, deterministic).
    2) Ask the LLM only for fields still missing.
    """
    text = pdf_to_text(pdf_bytes)
    primary = map_rule_results_to_keys(rule_extract(text), fields)

    # figure out what's still missing (and worth asking the model)
    keyed = {f["key"] for f in fields}
    still_missing = []
    for f in fields:
        if f["key"] not in primary and f.get("type") in ("number","date","enum","multi_enum","address","full_name","phone","string"):
            still_missing.append(f)

    llm_extra = llm_extract_missing(pdf_bytes, still_missing)
    # merge (rule-based wins)
    merged = dict(llm_extra)
    merged.update(primary)
    return merged

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
        debug       = body.get("debug", False)

        if not file_url:
            return jsonify({"ok": False, "error": "file_url is required"}), 400

        pdf_bytes, file_hash = download_file(file_url)
        fields = build_live_catalog()

        extracted = hybrid_extract(pdf_bytes, fields)

        if debug:
            return jsonify({"ok": True, "mode": "debug", "extracted": extracted})

        submission_id = submit_to_jotform(
            INTAKE_FORM_ID, fields, extracted, file_hash, fub_id
        )
        edit_url = f"https://www.jotform.com/edit/{submission_id}"

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
