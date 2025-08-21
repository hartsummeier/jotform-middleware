# app.py
import os, hashlib, json, base64, requests
from flask import Flask, request, jsonify

# You'll add these on the hosting site (Render) as environment variables
OPENAI_API_KEY  = os.environ.get("OPENAI_API_KEY", "")
JOTFORM_API_KEY = os.environ.get("JOTFORM_API_KEY", "")
INTAKE_FORM_ID  = os.environ.get("INTAKE_FORM_ID", "")

app = Flask(__name__)

# ---------- helpers ----------

def download_file(url):
    # For MVP: make sure Jotform "Require login to view uploads" is OFF
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    data = r.content
    file_hash = hashlib.sha256(data).hexdigest()
    return data, file_hash

def extract_with_openai(pdf_bytes):
    """
    Sends the PDF to OpenAI and asks for clean JSON.
    We keep it simple and only pull a few common fields.
    You can add more later.
    """
    # Turn PDF into base64 so we can send it as data URL
    b64 = base64.b64encode(pdf_bytes).decode("utf-8")

    # This is the JSON "shape" we want back
    schema = {
      "name": "transaction_intake",
      "schema": {
        "type":"object",
        "properties":{
          "transaction_type":{"type":"string","enum":[
            "Traditional [private buyer, private seller]",
            "For sale by owner",
            "New construction [purchase agreement is on builder documents]",
            "New construction [purchase agreement is on MIBOR documents]",
            "Unimproved land [commercial/residential]"
          ], "nullable": True},
          "agent_role":{"type":"string","enum":[
            "Listing agent",
            "Selling agent",
            "Both the selling agent and the listing agent"
          ], "nullable": True},
          "purchase_price":{"type":"number","nullable": True},
          "commission_rate":{"type":"number","minimum":0,"maximum":1,"nullable": True},
          "buyer_name":{"type":"string","nullable": True},
          "seller_name":{"type":"string","nullable": True},
          "close_date":{"type":"string","nullable": True},   # e.g. 2025-08-20
          "page_refs":{"type":"array","items":{"type":"integer"}, "nullable": True},
          "confidence":{"type":"number","minimum":0,"maximum":1, "nullable": True}
        },
        "required": [],
        "additionalProperties": False
      },
      "strict": True
    }

    url = "https://api.openai.com/v1/responses"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
    payload = {
      "model": "gpt-4o-mini",
      "input": [{
        "role":"user",
        "content":[
          {"type":"input_text","text":
           "Read this real estate purchase contract. Fill the JSON schema exactly. "
           "Use null if unsure. Give page_refs for where you found the data and an overall confidence (0-1)."},
          {"type":"input_image","image_url": f"data:application/pdf;base64,{b64}"}
        ]
      }],
      "response_format": {"type":"json_schema","json_schema": schema}
    }
    r = requests.post(url, headers=headers, json=payload, timeout=180)
    r.raise_for_status()
    data = r.json()

    # Try to get parsed JSON cleanly
    parsed = data.get("output_parsed")
    if not parsed:
        # Some responses carry it as text—handle both
        content = data.get("output", [{}])[0].get("content", [{}])[0].get("text", "{}")
        parsed = json.loads(content)
    return parsed

def jotform_create_submission(form_id, field_map):
    """
    field_map is {"12": "For sale by owner", "15": 300000, ...}
    Keys are Jotform field IDs (strings).
    """
    url = f"https://api.jotform.com/form/{form_id}/submissions?apiKey={JOTFORM_API_KEY}"
    payload = {}
    for qid, value in field_map.items():
        # Jotform expects submission[FIELDID]=VALUE
        if isinstance(value, (dict, list)):
            value = json.dumps(value)
        payload[f"submission[{qid}]"] = value
    r = requests.post(url, data=payload, timeout=60)
    r.raise_for_status()
    resp = r.json()
    return resp["content"]["submissionID"]

def jotform_edit_link(submission_id):
    return f"https://www.jotform.com/edit/{submission_id}"

# ---------- main endpoint Zapier calls ----------

@app.route("/ingest", methods=["POST"])
def ingest():
    body = request.get_json(force=True)

    # from your Zap
    fub_id         = body.get("fub_id", "")
    agent_email    = body.get("agent_email", "")
    uploaded_by    = body.get("uploaded_by", "")
    file_url       = body.get("file_url", "")

    # 1) Download the PDF
    pdf_bytes, file_hash = download_file(file_url)

    # 2) Extract data with OpenAI
    extracted = extract_with_openai(pdf_bytes)

    # 3) Map to your actual Intake Form field IDs.
    #    TODO: Replace the right-side strings ("12","13",...) with YOUR real field IDs from Jotform.
    #    Finding field IDs: open your Intake Form → click a field → Properties (gear) → Advanced → Field Details → "Field ID".
    Q = {
      "transaction_type": "12",
      "agent_role":       "13",
      "purchase_price":   "15",
      "commission_rate":  "16",
      "buyer_name":       "17",
      "seller_name":      "18",
      "close_date":       "19",
      "review_status":    "99",   # hidden text field you added
      "confidence_json":  "100"   # hidden long text field you added
    }

    # 4) Build the values we’ll send to Jotform
    field_map = {
      Q["transaction_type"]: extracted.get("transaction_type") or "",
      Q["agent_role"]:       extracted.get("agent_role") or "",
      Q["purchase_price"]:   extracted.get("purchase_price") or "",
      Q["commission_rate"]:  extracted.get("commission_rate") or "",
      Q["buyer_name"]:       extracted.get("buyer_name") or "",
      Q["seller_name"]:      extracted.get("seller_name") or "",
      Q["close_date"]:       extracted.get("close_date") or "",
      Q["review_status"]:    "AUTO_FILLED_PENDING_REVIEW",
      Q["confidence_json"]:  {
          "overall": extracted.get("confidence"),
          "page_refs": extracted.get("page_refs"),
          "source_file_hash": file_hash,
          "fub_id": fub_id
      }
    }

    # 5) Create the Jotform submission
    submission_id = jotform_create_submission(INTAKE_FORM_ID, field_map)
    edit_url = jotform_edit_link(submission_id)

    # 6) Return info Zapier will use (to email the agent, log, etc.)
    return jsonify({
        "ok": True,
        "submission_id": submission_id,
        "edit_url": edit_url,
        "file_hash": file_hash,
        "agent_email": agent_email,
        "uploaded_by": uploaded_by
    })

@app.route("/", methods=["GET"])
def hello():
    return "Middleware is running.", 200

if __name__ == "__main__":
    # For local/dev; the host will set this differently
    app.run(host="0.0.0.0", port=8080)
