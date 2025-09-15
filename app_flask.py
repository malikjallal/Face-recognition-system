# app_flask.py — FaceGate + Agent (Flask, single file) — ONNX/PyTorch with class_names.json

import io, time, json, hashlib, base64, webbrowser, re, difflib
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

from flask import Flask, request, jsonify, send_file, Response
import pandas as pd
from PIL import Image
import numpy as np
import httpx

# ----------------------------------------------------------------------------- #
# Paths / Folders
# ----------------------------------------------------------------------------- #
ROOT = Path(".").resolve()
LOGS = ROOT / "logs"; LOGS.mkdir(parents=True, exist_ok=True)
AUDIT = LOGS / "audit_log.csv"
TASKS = LOGS / "tasks.json"

ASSETS_DIR = ROOT / "assets"
MODELS_DIR = ROOT / "models"

CLASS_NAMES_PATH = ASSETS_DIR / "class_names.json"   # REQUIRED (list in training order)
THRESHOLD_PATH   = ASSETS_DIR / "threshold.json"     # optional

ONNX_PATH = MODELS_DIR / "best_efficientnet_b0.onnx"
PTH_PATH  = MODELS_DIR / "best_efficientnet_b0.pth"

# ----------------------------------------------------------------------------- #
# Config / Policy
# ----------------------------------------------------------------------------- #
LOCKOUT_AFTER = 3
LOCK_SECS = 60
_state: Dict[str, Dict[str, float]] = {}  # username -> {"fails": int, "lock_until": float}

# صلاحية استخدام أدوات الوكيل بعد نجاح التحقق (بالثواني)
AUTH_TTL = 5 * 60  # 5 دقائق
_verified_ok: Dict[str, float] = {}  # canon(username) -> last_ok_ts

# ----------------------------------------------------------------------------- #
# Read class names (EXACT training order) + threshold
# ----------------------------------------------------------------------------- #
try:
    CLASS_NAMES: List[str] = json.loads(CLASS_NAMES_PATH.read_text(encoding="utf-8"))
    if not isinstance(CLASS_NAMES, list) or not CLASS_NAMES:
        raise RuntimeError("assets/class_names.json must be a non-empty list")
    print(f"[FaceGate] Loaded class_names.json with {len(CLASS_NAMES)} classes")
except Exception as e:
    CLASS_NAMES = []
    raise RuntimeError(f"[FaceGate] Cannot load assets/class_names.json: {e}")

DEFAULT_THRESHOLD = 0.55
try:
    THRESHOLD_CONF = json.loads(THRESHOLD_PATH.read_text(encoding="utf-8"))
    CONF_THRESHOLD = float(THRESHOLD_CONF.get("max_softmax_threshold", DEFAULT_THRESHOLD))
except Exception:
    CONF_THRESHOLD = DEFAULT_THRESHOLD

# ----------------------------------------------------------------------------- #
# Canonicalization + Label resolving (flexible matching)
# ----------------------------------------------------------------------------- #
def canonize(s: str) -> str:
    """
    Canonicalize a string aggressively:
      - lowercase
      - remove leading 'pins' or 'pin' (optional)
      - strip non-alphanumerics (spaces, underscores, punctuation)
    Examples become 'alvaromorte': 'pins_Alvaro Morte' / 'alvaromorte' / 'Alvaro_Morte'
    """
    if s is None:
        return ""
    s = s.strip().lower()
    if s.startswith("pins"):
        s = s[4:]
    elif s.startswith("pin"):
        s = s[3:]
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s

def name_similarity(a: str, b: str) -> float:
    """Similarity in [0,1] using canonized strings."""
    ca, cb = canonize(a), canonize(b)
    if not ca or not cb:
        return 0.0
    return difflib.SequenceMatcher(None, ca, cb).ratio()

class LabelResolver:
    def __init__(self, original_labels: List[str]):
        self.original_labels = original_labels
        self.canon_to_original: Dict[str, str] = {}
        for name in original_labels:
            c = canonize(name)
            self.canon_to_original.setdefault(c, name)  # keep first if collision
        self.original_to_idx: Dict[str, int] = {}
        for i, name in enumerate(original_labels):
            self.original_to_idx.setdefault(name, i)

    def resolve_label(self, user_input: str) -> Optional[str]:
        cu = canonize(user_input)
        if not cu:
            return None
        if cu in self.canon_to_original:
            return self.canon_to_original[cu]
        for c_name, orig in self.canon_to_original.items():
            if cu in c_name or c_name in cu:
                return orig
        return None

    def get_index(self, original_label: str) -> Optional[int]:
        return self.original_to_idx.get(original_label)

RESOLVER = LabelResolver(CLASS_NAMES)

def refresh_labels_from_disk():
    """Reload labels at runtime if assets/class_names.json was modified."""
    global CLASS_NAMES, RESOLVER
    try:
        CLASS_NAMES = json.loads(CLASS_NAMES_PATH.read_text(encoding="utf-8"))
        if not isinstance(CLASS_NAMES, list) or not CLASS_NAMES:
            raise RuntimeError("assets/class_names.json must be a non-empty list")
        RESOLVER = LabelResolver(CLASS_NAMES)
        print(f"[FaceGate] Labels reloaded: {len(CLASS_NAMES)}")
    except Exception as e:
        print(f"[WARN] Reload labels failed: {e}")

# ----------------------------------------------------------------------------- #
# Model loading (prefer ONNX, fallback to Torch)
# ----------------------------------------------------------------------------- #
ONNX_SESSION = None
ONNX_INPUT_NAME = None
USE_ONNX = False

TORCH_MODEL = None
USE_TORCH = False

try:
    import onnxruntime as ort
    avail = ort.get_available_providers()
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if "CUDAExecutionProvider" in avail else ["CPUExecutionProvider"]
    if ONNX_PATH.exists():
        ONNX_SESSION = ort.InferenceSession(str(ONNX_PATH), providers=providers)
        ONNX_INPUT_NAME = ONNX_SESSION.get_inputs()[0].name
        USE_ONNX = True
        print(f"[FaceGate] ONNX loaded ({providers}) -> {ONNX_PATH.name}")
    else:
        print(f"[FaceGate] ONNX file not found at {ONNX_PATH}")
except Exception as e:
    print(f"[FaceGate] ONNX init failed: {e}")

if not USE_ONNX:
    try:
        import torch
        if PTH_PATH.exists():
            TORCH_MODEL = torch.load(str(PTH_PATH), map_location="cpu")
            if hasattr(TORCH_MODEL, "eval"):
                TORCH_MODEL.eval()
                USE_TORCH = True
                print(f"[FaceGate] PyTorch model loaded on CPU -> {PTH_PATH.name}")
            else:
                print("[FaceGate] Torch file loaded but not a model object.")
        else:
            print(f"[FaceGate] Torch file not found at {PTH_PATH}")
    except Exception as e:
        print(f"[FaceGate] Torch init failed: {e}")

# ----------------------------------------------------------------------------- #
# Pre/Post processing
# ----------------------------------------------------------------------------- #
IM_SIZE = 224
IM_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IM_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def preprocess_pil(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB").resize((IM_SIZE, IM_SIZE), Image.BILINEAR)
    arr = np.asarray(img).astype(np.float32) / 255.0
    arr = (arr - IM_MEAN) / IM_STD
    arr = np.transpose(arr, (2, 0, 1))  # CHW
    arr = np.expand_dims(arr, 0)        # NCHW
    return arr

def softmax(logits: np.ndarray) -> np.ndarray:
    x = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)

# ----------------------------------------------------------------------------- #
# Placeholder (if no model)
# ----------------------------------------------------------------------------- #
def _fake_score(img: Image.Image) -> float:
    img = img.convert("RGB").resize((64,64))
    b = io.BytesIO(); img.save(b, format="JPEG", quality=85)
    v = int(hashlib.md5(b.getvalue()).hexdigest()[:6], 16) / 0xFFFFFF
    return round(float(v), 3)

# ----------------------------------------------------------------------------- #
# Inference & decision (flexible)
# ----------------------------------------------------------------------------- #
def verify_face(
    username: str,
    image_bytes: bytes,
    threshold: float,
    accept: str = "claim",
    name_sim_thresh: float = 0.6
) -> Dict[str, Any]:
    # 0) image
    try:
        img = Image.open(io.BytesIO(image_bytes))
    except Exception:
        return {"ok": False, "score": 0.0, "reason": "invalid_image"}

    # no model fallback
    if not (USE_ONNX or USE_TORCH) or len(CLASS_NAMES) == 0:
        score = _fake_score(img)
        return {"ok": False, "score": score, "reason": "no_model"}

    # 1) preprocess
    inp = preprocess_pil(img)  # (1,3,224,224)

    # 2) logits
    if USE_ONNX:
        logits = ONNX_SESSION.run(None, {ONNX_INPUT_NAME: inp})[0]  # (1,C)
    else:
        import torch
        with torch.no_grad():
            t = torch.from_numpy(inp)
            out = TORCH_MODEL(t)                                     # (1,C)
            logits = out.detach().cpu().numpy()

    # 3) softmax + argmax
    probs = softmax(logits)
    pred_idx = int(np.argmax(probs[0]))
    score = float(probs[0, pred_idx])
    pred_label = CLASS_NAMES[pred_idx]
    print("[DEBUG] Predicted:", pred_label, "with score", score, "idx=", pred_idx)

    # 4) resolve claimed username -> original label -> index
    resolved_label = RESOLVER.resolve_label(username)
    if resolved_label is None:
        sim_pred = name_similarity(username, pred_label)
        return {
            "ok": False, "score": round(score,3), "reason": f"user_not_in_model:{username}",
            "pred_idx": pred_idx, "pred_label": pred_label, "resolved_label": None, "claim_idx": None,
            "claim_prob": None, "claim_rank": None,
            "sim_pred": round(sim_pred,3), "sim_claim": 0.0, "name_sim": round(sim_pred,3),
            "name_sim_thresh": name_sim_thresh
        }

    claim_idx = RESOLVER.get_index(resolved_label)
    if claim_idx is None:
        sim_pred = name_similarity(username, pred_label)
        sim_claim = name_similarity(username, resolved_label or "")
        return {
            "ok": False, "score": round(score,3), "reason": f"user_not_in_model:{resolved_label}",
            "pred_idx": pred_idx, "pred_label": pred_label, "resolved_label": resolved_label, "claim_idx": None,
            "claim_prob": None, "claim_rank": None,
            "sim_pred": round(sim_pred,3), "sim_claim": round(sim_claim,3), "name_sim": round(max(sim_pred, sim_claim),3),
            "name_sim_thresh": name_sim_thresh
        }

    # احتمال الكلاس المطلوب نفسه + رتبته
    claim_prob = float(probs[0, claim_idx])
    order = np.argsort(-probs[0]).tolist()
    claim_rank = int(order.index(claim_idx) + 1)

    # حساب تشابه الاسم
    sim_pred  = name_similarity(username, pred_label)
    sim_claim = name_similarity(username, resolved_label or "")
    max_sim   = max(sim_pred, sim_claim)

    # قرار القبول
    if accept == "strict":
        ok = (pred_idx == claim_idx) and (score >= threshold)
        reason = "ok" if ok else ("mismatch" if pred_idx != claim_idx else "low_conf")
    elif accept == "topk":
        TOPK, MARGIN = 3, 0.05
        ok = (claim_rank <= TOPK) and (claim_prob >= max(threshold, score - MARGIN))
        reason = "ok" if ok else "not_in_topk"
    elif accept == "name":
        ok = (max_sim >= name_sim_thresh) and ((score >= threshold) or (claim_prob >= threshold * 0.5))
        reason = "ok" if ok else f"name_sim<{name_sim_thresh:.2f}"
    else:  # "claim" (افتراضي)
        ok = (claim_prob >= threshold)
        reason = "ok" if ok else "low_conf_claim"

    return {
        "ok": bool(ok), "score": round(score,3), "reason": reason,
        "pred_idx": pred_idx, "pred_label": pred_label,
        "resolved_label": resolved_label, "claim_idx": claim_idx,
        "claim_prob": round(claim_prob,3), "claim_rank": claim_rank,
        "sim_pred": round(sim_pred,3), "sim_claim": round(sim_claim,3),
        "name_sim": round(max_sim,3), "name_sim_thresh": name_sim_thresh
    }

# ----------------------------------------------------------------------------- #
# Audit
# ----------------------------------------------------------------------------- #
def append_audit(username: str, decision: str, score: float, reason: str, client="web"):
    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "username": username, "decision": decision, "score": score,
        "reason": reason, "model_version": "onnx" if USE_ONNX else ("torch" if USE_TORCH else "stub"),
        "client": client
    }
    if AUDIT.exists():
        df = pd.read_csv(AUDIT)
        df.loc[len(df)] = row
    else:
        df = pd.DataFrame([row])
    df.to_csv(AUDIT, index=False)

def read_audit(n: int = 200) -> pd.DataFrame:
    if not AUDIT.exists():
        return pd.DataFrame(columns=["timestamp","username","decision","score","reason","model_version","client"])
    return pd.read_csv(AUDIT).tail(n)

# ----------------------------------------------------------------------------- #
# Agent tools
# ----------------------------------------------------------------------------- #
def tool_webhook(url: str, message: str) -> str:
    try:
        r = httpx.post(url, json={"message": message}, timeout=5.0)
        return f"Webhook: {r.status_code}"
    except Exception as e:
        return f"Webhook failed: {e}"

def tool_create_task(title: str, due: str) -> str:
    lst = json.loads(TASKS.read_text("utf-8")) if TASKS.exists() else []
    lst.append({"title": title, "due": due, "ts": time.time()})
    TASKS.write_text(json.dumps(lst, ensure_ascii=False, indent=2), encoding="utf-8")
    return f"✅ Task saved: {title} (due {due})"

def tool_summarize(text: str) -> str:
    s = " ".join(text.strip().split())
    words = s.split()
    return s if len(words) <= 60 else " ".join(words[:60]) + " …"

def tool_open(url: str) -> str:
    try:
        webbrowser.open(url); return f"Opened: {url}"
    except Exception as e:
        return f"Open failed: {e}"

def agent_router(text: str) -> str:
    t = (text or "").strip()
    low = t.lower()
    if low.startswith("open "):
        return tool_open(t.split(" ",1)[1])
    if low.startswith("webhook "):
        try:
            p = t.split(" ",1)[1]; url, msg = p.split("|",1)
            return tool_webhook(url.strip(), msg.strip())
        except Exception:
            return "Format: webhook <url> | <message>"
    if low.startswith("create task"):
        body = t[len("create task"):].strip()
        if "@" in body:
            title, due = body.split("@",1)
            return tool_create_task(title.strip(), due.strip())
        return "Format: create task <title> @<YYYY-MM-DD>"
    if low.startswith("summarize "):
        return tool_summarize(t.split(" ",1)[1])
    return "Commands: open <url> | webhook <url> | <msg> | create task <title> @<date> | summarize <text>"

# ----------------------------------------------------------------------------- #
# Flask app + endpoints
# ----------------------------------------------------------------------------- #
app = Flask(__name__)

def _is_agent_allowed(username: str) -> bool:
    """يُسمح باستخدام الـAgent إذا تحقق هذا المستخدم بنجاح خلال AUTH_TTL."""
    key = canonize(username)
    ts = _verified_ok.get(key, 0.0)
    return (time.time() - ts) <= AUTH_TTL

@app.get("/health")
def health():
    return jsonify({
        "status": "ok",
        "model": "onnx" if USE_ONNX else ("torch" if USE_TORCH else "stub"),
        "num_classes": len(CLASS_NAMES),
        "default_threshold": CONF_THRESHOLD
    })

@app.get("/audit")
def api_audit():
    return jsonify({"records": read_audit().to_dict(orient="records")})

@app.get("/audit.csv")
def download_audit():
    if not AUDIT.exists():
        return Response("no audit yet", status=404)
    return send_file(AUDIT, as_attachment=True, download_name="audit_log.csv")

@app.post("/verify")
def api_verify():
    username = request.form.get("username","").strip()
    threshold = float(request.form.get("threshold", CONF_THRESHOLD))
    accept = request.form.get("accept", "claim")
    try:
        name_sim = float(request.form.get("name_sim", 0.6))
    except:
        name_sim = 0.6

    file = request.files.get("image")
    data_url = request.form.get("image_data_url")  # base64 from webcam

    if not username:
        return jsonify({"ok": False, "error": "missing_username"}), 400
    if not file and not data_url:
        append_audit(username, "denied", 0.0, "no_image", "api")
        return jsonify({"ok": True, "decision": "denied", "score": 0.0, "reason": "no_image"})

    # lockout check
    now = time.time()
    st = _state.get(username, {"fails":0, "lock_until":0})
    if now < st["lock_until"]:
        left = int(st["lock_until"] - now)
        append_audit(username, "denied", 0.0, f"locked_{left}s", "api")
        return jsonify({"ok": True, "decision": "denied", "score": 0.0, "reason": f"locked_{left}s", "fails": st["fails"]})

    # read bytes
    try:
        if file:
            image_bytes = file.read()
        else:
            b64 = data_url.split(",",1)[1]
            image_bytes = base64.b64decode(b64)
    except Exception:
        append_audit(username, "denied", 0.0, "invalid_image", "api")
        return jsonify({"ok": True, "decision": "denied", "score": 0.0, "reason": "invalid_image"})

    # inference
    res = verify_face(username, image_bytes, threshold, accept=accept, name_sim_thresh=name_sim)
    decision = "granted" if res["ok"] else "denied"

    # state update
    if decision == "granted":
        st = {"fails":0, "lock_until":0}
        # سجّل هذا المستخدم على أنه متحقق واسمح له بالـAgent لفترة AUTH_TTL
        _verified_ok[canonize(username)] = time.time()
    else:
        st["fails"] += 1
        if st["fails"] >= LOCKOUT_AFTER:
            st["lock_until"] = now + LOCK_SECS
            st["fails"] = 0
    _state[username] = st

    append_audit(username, decision, res["score"], res["reason"], "api")

    # return extended debug fields too
    return jsonify({
        "ok": True,
        "decision": decision,
        "score": res.get("score"),
        "reason": res.get("reason"),
        "fails": st["fails"],
        "pred_idx": res.get("pred_idx"),
        "pred_label": res.get("pred_label"),
        "resolved_label": res.get("resolved_label"),
        "claim_idx": res.get("claim_idx"),
        "claim_prob": res.get("claim_prob"),
        "claim_rank": res.get("claim_rank"),
        "accept": accept,
        "sim_pred": res.get("sim_pred"),
        "sim_claim": res.get("sim_claim"),
        "name_sim": res.get("name_sim"),
        "name_sim_thresh": res.get("name_sim_thresh"),
        "agent_allowed": _is_agent_allowed(username)
    })

@app.get("/debug/labels")
def debug_labels():
    return jsonify({"count": len(CLASS_NAMES),
                    "labels": [{"idx": i, "name": n} for i, n in enumerate(CLASS_NAMES)]})

@app.get("/debug/resolve")
def debug_resolve():
    q = (request.args.get("q") or "").strip()
    resolved = RESOLVER.resolve_label(q)
    idx = RESOLVER.get_index(resolved) if resolved else None
    return jsonify({"q": q, "canon": canonize(q), "resolved": resolved, "idx": idx})

@app.post("/debug/topk")
def debug_topk():
    file = request.files.get("image")
    data_url = request.form.get("image_data_url")
    if not file and not data_url:
        return jsonify({"error":"no_image"}), 400
    if file:
        image_bytes = file.read()
    else:
        b64 = data_url.split(",",1)[1]
        image_bytes = base64.b64decode(b64)

    try:
        img = Image.open(io.BytesIO(image_bytes))
    except Exception:
        return jsonify({"error":"invalid_image"}), 400

    if not (USE_ONNX or USE_TORCH):
        return jsonify({"error":"no_model"}), 400

    inp = preprocess_pil(img)
    if USE_ONNX:
        logits = ONNX_SESSION.run(None, {ONNX_INPUT_NAME: inp})[0]
    else:
        import torch
        with torch.no_grad():
            t = torch.from_numpy(inp)
            logits = TORCH_MODEL(t).detach().cpu().numpy()

    probs = softmax(logits)[0]
    topk = int(min(5, len(probs)))
    idxs = np.argsort(-probs)[:topk].tolist()
    out = [{"idx": int(i), "label": CLASS_NAMES[int(i)], "p": float(probs[int(i)])} for i in idxs]
    return jsonify({"topk": out})

@app.post("/agent/chat")
def agent_chat():
    data = request.get_json(force=True, silent=True) or {}
    text = (data.get("text") or "").strip()
    username = (data.get("username") or "").strip()

    if not username:
        return jsonify({"reply": "Missing username"}), 400
    if not _is_agent_allowed(username):
        # رفض صريح مع حالة 403
        return jsonify({"reply": "Access denied: verify your face for this username first."}), 403

    reply = agent_router(text)
    return jsonify({"reply": reply})

@app.get("/tasks.json")
def get_tasks():
    if not TASKS.exists():
        return jsonify({"tasks": []})
    return jsonify({"tasks": json.loads(TASKS.read_text("utf-8"))})

# ----------------------------------------------------------------------------- #
# UI
# ----------------------------------------------------------------------------- #
HTML = r"""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width,initial-scale=1" />
<title>FaceGate — Flask + Agent</title>
<style>
:root{color-scheme: dark light}
body{font-family: ui-sans-serif,system-ui,Segoe UI,Roboto,Arial;background:#0b0f1a;color:#e6eaf2;margin:0}
.wrap{max-width:1150px;margin:28px auto;padding:0 16px}
.card{background:#0f1220;border:1px solid #232a4a;border-radius:16px;padding:16px;margin-bottom:16px}
h1{margin:0 0 8px}
.row{display:flex;gap:16px;flex-wrap:wrap}
.col{flex:1 1 0}
label{font-size:14px;opacity:.9}
input[type=text],input[type=number],input[type=url]{width:100%;padding:10px;border-radius:10px;border:1px solid #273056;background:#0b0f1a;color:#fff}
button{background:#6366f1;border:none;color:#fff;padding:10px 14px;border-radius:10px;cursor:pointer}
button.secondary{background:#1f2937}
.badge{padding:4px 10px;border-radius:999px;font-size:12px;border:1px solid}
.ok{background:#163d2a;border-color:#1f8a47;color:#22c55e}
.err{background:#401b1b;border-color:#b32626;color:#ef4444}
.gauge{background:#10142a;border:1px solid #1b2247;border-radius:12px;padding:8px}
.bar{height:12px;background:#1f2547;border-radius:6px;overflow:hidden}
.fill{height:100%;transition:width .35s}
table{width:100%;border-collapse:collapse}
th,td{padding:6px 8px;border-bottom:1px solid #1f2547;font-size:14px}
small{opacity:.8}
video,canvas{max-width:100%;border-radius:12px;border:1px solid #1f2547}
.chat{height:280px;overflow:auto;background:#0b0f1a;border:1px solid #273056;border-radius:12px;padding:10px}
.msg{margin:6px 0}
.msg.me{color:#a5b4fc}
.msg.bot{color:#93c5fd}
</style>
</head>
<body>
<div class="wrap">
  <h1>🔐 FaceGate <span class="badge ok">Flask</span> → 🧠 Agent</h1>
  <div class="row">
    <div class="col">
      <div class="card">
        <h3>Login / Verify</h3>
        <label>Username</label>
        <input id="username" type="text" placeholder="e.g. pins_Alvaro Morte"/>
        <div class="row">
          <div class="col">
            <label>Upload Face Image</label>
            <input id="file" type="file" accept="image/*"/>
          </div>
          <div class="col">
            <label>Or use Webcam</label><br/>
            <video id="vid" autoplay playsinline width="320" height="240"></video><br/>
            <button class="secondary" id="snap">Capture</button>
            <canvas id="canvas" width="320" height="240" style="display:none"></canvas>
          </div>
        </div>
        <div class="row">
          <div class="col">
            <label>Decision Threshold</label>
            <input id="thr" type="number" step="0.01" min="0.3" max="0.95" value="0.55"/>
          </div>
          <div class="col" style="display:flex;align-items:end;gap:8px">
            <button id="verify">Verify</button>
            <button class="secondary" id="clear">Clear</button>
            <a class="secondary" href="/audit.csv"><button class="secondary">Download Audit CSV</button></a>
          </div>
        </div>
        <p id="decision"></p>
        <div class="gauge"><div style="display:flex;justify-content:space-between;margin-bottom:6px">
          <div>Score: <b id="score">0.000</b></div>
          <div>Threshold: <b id="thrlbl">0.55</b></div>
        </div>
        <div class="bar"><div id="fill" class="fill" style="width:0%;background:#ef4444"></div></div></div>
        <p id="lock" class="badge err" style="display:none">Locked</p>
      </div>
    </div>
    <div class="col">
      <div class="card">
        <h3>Quick Stats</h3>
        <p>Health: <span class="badge ok" id="health">checking…</span></p>
        <table id="last"></table>
      </div>
      <div class="card">
        <h3>Agent Tools</h3>
        <div class="chat" id="chat"></div>
        <input id="agent_input" type="text" placeholder="Try: create task Fix login @2025-09-20, or webhook https://... | hi"/>
        <div style="margin-top:6px; display:flex; gap:8px">
          <button id="send">Send</button>
          <button class="secondary" id="clrchat">Clear</button>
        </div>
        <p id="agent_note" style="opacity:.8; font-size:13px">
          Agent is locked until you verify your face for the username above.
        </p>
      </div>
    </div>
  </div>

  <div class="card">
    <h3>Audit Trail (last 200)</h3>
    <table id="table"></table>
  </div>
</div>

<script>
async function health(){ try{
  const r = await fetch('/health'); const j = await r.json();
  document.getElementById('health').textContent = (j.status || 'ok').toUpperCase();
}catch(e){ document.getElementById('health').textContent='ERR'; } }
health();

async function loadAudit(){
  const r = await fetch('/audit'); const j = await r.json();
  const rows = j.records || [];
  const last = rows.slice(-5).reverse();
  const lastTbl = document.getElementById('last');
  lastTbl.innerHTML = "<tr><th>Time</th><th>User</th><th>Decision</th><th>Score</th><th>Reason</th></tr>" +
    last.map(x=>`<tr><td>${x.timestamp||""}</td><td>${x.username||""}</td><td>${x.decision||""}</td><td>${x.score||""}</td><td>${x.reason||""}</td></tr>`).join("");
  const tbl = document.getElementById('table');
  tbl.innerHTML = "<tr><th>Time</th><th>User</th><th>Decision</th><th>Score</th><th>Reason</th><th>Model</th><th>Client</th></tr>" +
    rows.slice().reverse().map(x=>`<tr><td>${x.timestamp||""}</td><td>${x.username||""}</td><td>${x.decision||""}</td><td>${x.score||""}</td><td>${x.reason||""}</td><td>${x.model_version||""}</td><td>${x.client||""}</td></tr>`).join("");
}
loadAudit();

let dataURL = null;
const vid = document.getElementById('vid');
const canvas = document.getElementById('canvas');
const snap = document.getElementById('snap');
if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
  navigator.mediaDevices.getUserMedia({video:true}).then(s=>{ vid.srcObject = s; }).catch(()=>{});
}
snap.onclick = () => {
  canvas.getContext('2d').drawImage(vid, 0, 0, canvas.width, canvas.height);
  dataURL = canvas.toDataURL('image/jpeg');
  canvas.style.display = 'block';
};

document.getElementById('thr').oninput = (e)=>{
  document.getElementById('thrlbl').textContent = parseFloat(e.target.value).toFixed(2);
}

function setGauge(score, thr){
  document.getElementById('score').textContent = score.toFixed(3);
  document.getElementById('thrlbl').textContent = thr.toFixed(2);
  const pct = Math.max(0, Math.min(100, Math.round(score*100)));
  const fill = document.getElementById('fill');
  fill.style.width = pct + "%";
  fill.style.background = (score >= thr) ? "#22c55e" : "#ef4444";
}

document.getElementById('clear').onclick = ()=>{
  document.getElementById('file').value = "";
  dataURL = null;
  setGauge(0, parseFloat(document.getElementById('thr').value));
  document.getElementById('decision').innerHTML = "";
  document.getElementById('lock').style.display = 'none';
}

document.getElementById('verify').onclick = async ()=>{
  const u = document.getElementById('username').value.trim();
  const thr = parseFloat(document.getElementById('thr').value);
  const f = document.getElementById('file').files[0];
  const fd = new FormData();
  fd.append('username', u);
  fd.append('threshold', thr.toString());
  // الوضع الذي تفضّله (مثال: "claim" أو "name")
  fd.append('accept', 'name');
  fd.append('name_sim', '0.60');

  if (f) fd.append('image', f);
  else if (dataURL) fd.append('image_data_url', dataURL);

  const r = await fetch('/verify', {method:'POST', body: fd});
  const j = await r.json();
  const decision = j.decision || 'denied';
  const score = j.score || 0.0;
  const reason = j.reason || '';
  setGauge(score, thr);

  const extra = [
    (j.claim_prob!=null?`claim_prob=${j.claim_prob}`:''),
    (j.claim_rank!=null?`rank=${j.claim_rank}`:''),
    (j.name_sim!=null?`name_sim=${j.name_sim} (≥ ${j.name_sim_thresh})`:'' )
  ].filter(Boolean).join(' • ');

  document.getElementById('decision').innerHTML =
    (decision==='granted')
      ? `<span class="badge ok">GRANTED</span> <small>${reason} • ${extra}</small>`
      : `<span class="badge err">DENIED</span> <small>${reason} • ${extra}</small>`;

  // تحديث ملاحظة الوكيل
  document.getElementById('agent_note').textContent =
    (j.agent_allowed ? "Agent unlocked for 5 minutes." : "Agent is locked until you verify your face.");

  if (String(reason).startsWith('locked_')) document.getElementById('lock').style.display='inline-block';
  else document.getElementById('lock').style.display='none';
  await loadAudit();
};

// Agent chat
const chat = document.getElementById('chat');
function push(role, text){
  const div = document.createElement('div');
  div.className = 'msg ' + role;
  div.textContent = (role==='me'?'You: ':'Agent: ') + text;
  chat.appendChild(div); chat.scrollTop = chat.scrollHeight;
}
document.getElementById('send').onclick = async ()=>{
  const inp = document.getElementById('agent_input');
  const text = inp.value.trim(); if(!text) return;
  const u = document.getElementById('username').value.trim();
  push('me', text); inp.value='';
  const r = await fetch('/agent/chat', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify({text, username: u})
  });
  if (r.status === 403){
    push('bot', 'Access denied: verify your face for this username first.');
    return;
  }
  const j = await r.json(); push('bot', j.reply || '...');
};
document.getElementById('clrchat').onclick = ()=>{ chat.innerHTML=''; };
</script>
</body>
</html>
"""

@app.get("/")
def ui():
    return Response(HTML, mimetype="text/html")

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
