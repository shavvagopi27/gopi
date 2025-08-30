# app.py
import os
import base64
import sqlite3
from flask import Flask, request, jsonify, render_template, g
from google import genai
from google.genai.types import Part

# --- Configuration ---
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise RuntimeError("Set GOOGLE_API_KEY environment variable with your Gemini/Google API key.")
MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")  # change if needed

# Create Gemini client
client = genai.Client(api_key=API_KEY)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 80 * 1024 * 1024  # 80MB uploads

# --- Simple SQLite for rooms and uploaded doc references (MVP) ---
DB = "studybuddy.db"

def get_db():
    db = getattr(g, "_database", None)
    if db is None:
        db = g._database = sqlite3.connect(DB)
        db.row_factory = sqlite3.Row
    return db

def init_db():
    with app.app_context():
        db = get_db()
        db.execute("""
        CREATE TABLE IF NOT EXISTS rooms (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            room_name TEXT UNIQUE,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )""")
        db.execute("""
        CREATE TABLE IF NOT EXISTS docs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            room_id INTEGER,
            filename TEXT,
            data BLOB,
            mime TEXT,
            uploaded_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(room_id) REFERENCES rooms(id)
        )""")
        db.commit()

@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, "_database", None)
    if db is not None:
        db.close()

# --- Helper: call Gemini with parts (pdf, image, audio) and text prompts ---
def call_gemini(parts_or_texts, prefer_json=False, timeout_seconds=60):
    # parts_or_texts: list of either Part objects or dicts like {"text": "..."}
    # Returns the model response object
    response = client.models.generate_content(model=MODEL, contents=parts_or_texts)
    return response

# --- Routes ---
@app.route("/")
def index():
    return render_template("index.html")

# Create or get room
@app.route("/room", methods=["POST"])
def create_room():
    data = request.json or {}
    name = data.get("room_name") or "default"
    db = get_db()
    cur = db.execute("SELECT id FROM rooms WHERE room_name = ?", (name,))
    row = cur.fetchone()
    if row:
        room_id = row["id"]
    else:
        cur = db.execute("INSERT INTO rooms (room_name) VALUES (?)", (name,))
        db.commit()
        room_id = cur.lastrowid
    return jsonify({"room_id": room_id, "room_name": name})

# Upload PDF and associate to a room
@app.route("/upload_pdf", methods=["POST"])
def upload_pdf():
    room_id = request.form.get("room_id")
    pdf = request.files.get("pdf")
    if not pdf:
        return jsonify({"error": "No PDF uploaded"}), 400
    data = pdf.read()
    mime = pdf.mimetype or "application/pdf"
    filename = pdf.filename
    db = get_db()
    db.execute("INSERT INTO docs (room_id, filename, data, mime) VALUES (?, ?, ?, ?)",
               (room_id, filename, data, mime))
    db.commit()
    return jsonify({"status": "ok", "filename": filename})

# Internal helper: get all docs for a room as Parts
def get_room_parts(room_id):
    db = get_db()
    cur = db.execute("SELECT * FROM docs WHERE room_id = ?", (room_id,))
    rows = cur.fetchall()
    parts = []
    for r in rows:
        bytes_data = r["data"]
        mime = r["mime"]
        part = Part.from_bytes(bytes=bytes_data, mime_type=mime)
        parts.append(part)
    return parts

# Context-aware Q&A: ask question limited to uploaded PDFs in the room
@app.route("/ask_pdf", methods=["POST"])
def ask_pdf():
    payload = request.form or request.json or {}
    room_id = payload.get("room_id")
    question = payload.get("question") or payload.get("prompt") or ""
    level = payload.get("level") or "default"  # explain level: child/exam/professor

    if not question:
        return jsonify({"error": "No question provided"}), 400

    parts = []
    if room_id:
        parts = get_room_parts(room_id)

    # Build instruction to prefer only uploaded docs, request citations
    instruction = (
        "You are allowed to use ONLY the uploaded documents as source. "
        "Answer concisely. Include JSON 'citations' array where each item has fields: page, excerpt. "
        "If answer is not found inside uploaded documents, say so clearly. "
    )
    # Apply explanation-level hint
    if level == "child":
        instruction += "Explain like I'm 10. "
    elif level == "exam":
        instruction += "Explain concisely, exam-focused. "
    elif level == "professor":
        instruction += "Give an in-depth technical explanation. "

    user_prompt = instruction + "\nQuestion: " + question

    # Build contents: all parts first (PDFs), then text prompt
    contents = []
    contents.extend(parts)
    contents.append({"text": user_prompt})

    try:
        resp = call_gemini(contents)
        return jsonify({"answer": resp.text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Plain text Q&A (no uploaded PDF)
@app.route("/ask_text", methods=["POST"])
def ask_text():
    data = request.json or {}
    question = data.get("question", "")
    level = data.get("level", "default")
    if not question:
        return jsonify({"error": "No question"}), 400
    instr = ""
    if level == "child":
        instr = "Explain like I'm 10. "
    elif level == "exam":
        instr = "Explain concisely, exam-focused. "
    elif level == "professor":
        instr = "Explain in-depth like a professor. "
    prompt = instr + question
    resp = call_gemini([{"text": prompt}])
    return jsonify({"answer": resp.text})

# Image Q&A (camera)
@app.route("/ask_image", methods=["POST"])
def ask_image():
    data = request.json or {}
    image_b64 = data.get("image")
    question = data.get("question", "")
    if not image_b64:
        return jsonify({"error": "No image"}), 400
    header, b64 = image_b64.split(",", 1) if "," in image_b64 else (None, image_b64)
    img_bytes = base64.b64decode(b64)
    img_part = Part.from_bytes(bytes=img_bytes, mime_type="image/png")
    prompt = "Analyze the image and then answer: " + question
    resp = call_gemini([img_part, {"text": prompt}])
    return jsonify({"answer": resp.text})

# Voice Q&A (audio file as base64)
@app.route("/ask_voice", methods=["POST"])
def ask_voice():
    data = request.json or {}
    audio_b64 = data.get("audio")
    if not audio_b64:
        return jsonify({"error": "No audio"}), 400
    header, b64 = audio_b64.split(",", 1) if "," in audio_b64 else (None, audio_b64)
    audio_bytes = base64.b64decode(b64)
    audio_part = Part.from_bytes(bytes=audio_bytes, mime_type="audio/webm")
    prompt = "Transcribe and answer any question included, or summarize."
    resp = call_gemini([audio_part, {"text": prompt}])
    return jsonify({"answer": resp.text})

# Flashcards generation (from room docs)
@app.route("/flashcards", methods=["POST"])
def flashcards():
    data = request.json or {}
    room_id = data.get("room_id")
    num = int(data.get("num", 8))
    parts = get_room_parts(room_id) if room_id else []

    prompt = (
        f"From the uploaded documents generate up to {num} flashcards as JSON format: "
        '{"cards":[{"q":"...","a":"...","page":n}]} '
        "Use page citations when possible."
    )
    contents = []
    contents.extend(parts)
    contents.append({"text": prompt})
    resp = call_gemini(contents)
    return jsonify({"flashcards_raw": resp.text})

# Summarize a chapter or uploaded documents
@app.route("/summarize", methods=["POST"])
def summarize():
    data = request.json or {}
    room_id = data.get("room_id")
    parts = get_room_parts(room_id) if room_id else []
    prompt = (
        "Summarize the uploaded materials into key points, formulas, and a short study checklist. "
        "Output JSON: {summary: '...', key_points: [...], formulas: [...]}"
    )
    contents = []
    contents.extend(parts)
    contents.append({"text": prompt})
    resp = call_gemini(contents)
    return jsonify({"summary_raw": resp.text})

# Quiz generation: produce a short adaptive quiz
@app.route("/quiz", methods=["POST"])
def quiz():
    data = request.json or {}
    room_id = data.get("room_id")
    difficulty = data.get("difficulty", "mixed")
    num = int(data.get("num", 5))
    parts = get_room_parts(room_id) if room_id else []
    prompt = (
        f"From uploaded documents, generate {num} quiz questions with answers and hints as JSON: "
        '{"quiz":[{"q":"...","choices":["..."],"ans":"...","hint":"..."}]}'
    )
    contents = []
    contents.extend(parts)
    contents.append({"text": prompt})
    resp = call_gemini(contents)
    return jsonify({"quiz_raw": resp.text})

# Run
if __name__ == "__main__":
    init_db()
    app.run(debug=True, host="0.0.0.0", port=5000)
