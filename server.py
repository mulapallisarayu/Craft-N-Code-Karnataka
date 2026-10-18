import os
import re
from pathlib import Path
from io import BytesIO

from flask import Flask, jsonify, request, send_file
from google import genai
from google.genai import types
from pypdf import PdfReader

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 12 * 1024 * 1024

BASE_DIR = Path(__file__).resolve().parent
GAME_PAGE = BASE_DIR / "kart-study-garage_final.html"

# Models are tried in order. Override with GEMINI_MODELS="model-a,model-b" if Google renames them.
MODELS = [m.strip() for m in os.environ.get(
    "GEMINI_MODELS", "gemini-3.8-flash,gemini-2.5-flash,gemini-2.0-flash").split(",") if m.strip()]

TARIO_SYSTEM = (
    "You are Tario, the friendly pixel study buddy in Kart Kampus, a CBSE tutor for "
    "Indian school students in Classes 1 to 10. Explain school topics clearly, accurately "
    "and simply, step by step, with a small example where helpful. "
    "Write in plain text only. Do not use Markdown symbols such as asterisks, hash signs, "
    "backticks, horizontal rules made of dashes, or LaTeX. Write chemical equations with "
    "plain words or simple symbols, such as Carbon dioxide + Water -> Glucose + Oxygen. "
    "For lists, start each line with a dash. Keep a clear, friendly tone. If the student's "
    "message includes chapter context, use it when relevant and never mention that you were given context."
)

_client = None


def get_client():
    global _client
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        raise RuntimeError("GEMINI_API_KEY is not set. Set it in the same Terminal window and restart server.py.")
    if _client is None:
        _client = genai.Client(api_key=key)
    return _client


def generate(contents, config):
    """Try each model in turn. Returns (text, model_used). Raises with the last real error."""
    last_error = None
    for model in MODELS:
        try:
            response = get_client().models.generate_content(model=model, contents=contents, config=config)
            text = (response.text or "").strip()
            if text:
                return text, model
            last_error = f"{model} returned an empty answer"
        except RuntimeError:
            raise
        except Exception as e:
            app.logger.warning("Model %s failed: %s", model, e)
            last_error = f"{model}: {e}"
    raise RuntimeError(last_error or "No model available")


def clean_reply(text):
    """Turn any leftover Markdown or LaTeX into plain readable text."""
    t = str(text)
    t = re.sub(r"\$\$(.+?)\$\$", r"\1", t, flags=re.S)
    t = re.sub(r"\$([^$]+?)\$", r"\1", t, flags=re.S)
    t = t.replace("$", "")
    for _ in range(3):  # unwrap nested \text{...}
        t = re.sub(r"\\(?:text|mathrm|mathbf|textbf|textit|emph)\{([^{}]*)\}", r"\1", t)
    t = re.sub(r"\\xrightarrow\s*(\[[^\]]*\])?\s*(\{[^{}]*\})?", "->", t)
    t = re.sub(r"\\(rightarrow|to|longrightarrow)\b", "->", t)
    t = re.sub(r"\\[a-zA-Z]+\s*", "", t)
    t = re.sub(r"\*\*(.+?)\*\*", r"\1", t)
    t = re.sub(r"__(.+?)__", r"\1", t)
    t = re.sub(r"^\s{0,3}#{1,6}\s*", "", t, flags=re.M)
    t = re.sub(r"^\s*(-{3,}|\*{3,}|_{3,})\s*$", "", t, flags=re.M)
    t = re.sub(r"^\s*[*+]\s+", "- ", t, flags=re.M)
    t = t.replace("`", "").replace("*", "")
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def notebook_context(sources, limit=60000):
    parts, total = [], 0
    for s in sources[:8]:
        if not isinstance(s, dict):
            continue
        name = str(s.get("name", "Source"))[:60]
        text = str(s.get("text", ""))
        chunk = f"=== SOURCE: {name} ===\n{text}\n"
        if total + len(chunk) > limit:
            chunk = chunk[:max(0, limit - total)]
        parts.append(chunk)
        total += len(chunk)
        if total >= limit:
            break
    return "\n".join(parts).strip()


@app.route("/")
def home():
    page = GAME_PAGE if GAME_PAGE.exists() else BASE_DIR / "index.html"
    return send_file(page)


@app.route("/api/health")
def health():
    return jsonify({
        "ok": True,
        "key_set": bool(os.environ.get("GEMINI_API_KEY", "").strip()),
        "models_tried_in_order": MODELS,
    })


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    messages = data.get("messages", [])

    if not isinstance(messages, list) or not messages:
        return jsonify({"error": "Please enter a message."}), 400

    contents = []
    for message in messages[-12:]:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        content = message.get("content")
        if role in ("user", "assistant") and isinstance(content, str):
            content = content.strip()[:4000]
            if content:
                contents.append(types.Content(
                    role="user" if role == "user" else "model",
                    parts=[types.Part.from_text(text=content)],
                ))

    if not contents or contents[-1].role != "user":
        return jsonify({"error": "Please send a message first."}), 400

    try:
        reply, model = generate(
            contents,
            types.GenerateContentConfig(system_instruction=TARIO_SYSTEM, temperature=0.4, max_output_tokens=1800),
        )
        return jsonify({"reply": clean_reply(reply), "model": model})
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 502
    except Exception as e:
        app.logger.exception("Tario chat failed")
        return jsonify({"error": f"Tario couldn't reply: {e}"}), 502


@app.route("/api/extract", methods=["POST"])
def extract():
    f = request.files.get("file")
    if not f or not f.filename:
        return jsonify({"error": "Choose a PDF, TXT or MD file."}), 400
    name = f.filename.lower()
    raw = f.read()
    try:
        if name.endswith(".pdf"):
            reader = PdfReader(BytesIO(raw))
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
        elif name.endswith((".txt", ".md")):
            text = raw.decode("utf-8", errors="replace")
        else:
            return jsonify({"error": "Supported files: PDF, TXT and MD."}), 400
    except Exception:
        return jsonify({"error": "Could not read that file. Try a text-based PDF."}), 400
    text = text.strip()[:60000]
    if len(text) < 20:
        return jsonify({"error": "That file has very little readable text."}), 400
    return jsonify({"text": text})


@app.route("/api/notebook", methods=["POST"])
def notebook():
    data = request.get_json(silent=True) or {}
    task = str(data.get("task", "")).strip()[:500]
    sources = data.get("sources", [])
    if not isinstance(sources, list):
        sources = []
    if not task:
        return jsonify({"error": "Tell the notebook what to do with the sources."}), 400
    ctx = notebook_context(sources)
    if not ctx:
        return jsonify({"error": "Add at least one source first."}), 400

    prompt = (
        f"Using ONLY the sources below, {task}. If the answer is not in the sources, say so. "
        "Write in plain text: no Markdown symbols and no LaTeX.\n\n" + ctx
    )
    try:
        reply, model = generate(prompt, types.GenerateContentConfig(max_output_tokens=3000, temperature=0.3))
        return jsonify({"reply": clean_reply(reply), "model": model})
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 502
    except Exception as e:
        app.logger.exception("Notebook failed")
        return jsonify({"error": f"Notebook couldn't answer: {e}"}), 502


@app.errorhandler(413)
def file_too_large(error):
    return jsonify({"error": "File too large. Maximum upload size is 12 MB."}), 413


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
