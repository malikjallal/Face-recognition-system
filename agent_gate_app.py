# agent_gate_app.py  — FaceGate Pro UI + Agent
# ==========================================================
import io, time, json, hashlib, webbrowser, os
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
from PIL import Image
import httpx

from fastapi import FastAPI, UploadFile, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
import gradio as gr

# ---------- Paths ----------
ROOT = Path(".").resolve()
LOGS = ROOT / "logs"; LOGS.mkdir(parents=True, exist_ok=True)
AUDIT = LOGS / "audit_log.csv"
TASKS = LOGS / "tasks.json"

# ---------- TTS (safe) ----------
try:
    import pyttsx3
    _tts = pyttsx3.init()
except Exception:
    _tts = None
def say(text: str):
    try:
        if _tts:
            _tts.say(text); _tts.runAndWait()
    except Exception: pass

# ---------- Security/Business rules (editable) ----------
ALLOWLIST = {"demo","stark","admin"}  # مؤقتًا – احذفها لاحقًا
LOCKOUT_AFTER = 3                     # قفل بعد 3 فشل
LOCK_SECS = 60                        # مدة القفل
THRESHOLD = 0.55                      # عتبة القرار (قابلة للتعديل من واجهة الإعدادات)
_state = {}  # username -> {"fails":int,"lock_until":float}

# ---------- Placeholder scoring (بدّل بالموديل الحقيقي لاحقًا) ----------
def _fake_score(img: Image.Image) -> float:
    img = img.convert("RGB").resize((64,64))
    b = io.BytesIO(); img.save(b, format="JPEG", quality=85)
    return round(int(hashlib.md5(b.getvalue()).hexdigest()[:6], 16) / 0xFFFFFF, 3)

def verify_face(username: str, image_bytes: bytes, threshold: float = THRESHOLD) -> Dict[str, Any]:
    """
    بدّل محتوى هذه الدالة لاحقًا:
    - REST:
      r = httpx.post("http://127.0.0.1:9000/verify",
                     data={"username": username},
                     files={"image": ("img.jpg", image_bytes, "image/jpeg")}, timeout=10)
      rj = r.json()  # {"ok": bool, "score": float}
      return {"ok": rj["ok"], "score": round(rj["score"],3), "reason": "ok" if rj["ok"] else "mismatch"}
    - ONNX/torch: حمّل الموديل مرة واحدة أعلى الملف واستعمله هنا.
    """
    try:
        img = Image.open(io.BytesIO(image_bytes))
    except Exception:
        return {"ok": False, "score": 0.0, "reason": "invalid_image"}
    score = _fake_score(img)
    granted = (username.lower() in ALLOWLIST) and (score >= threshold)
    return {"ok": granted, "score": score, "reason": "ok" if granted else "mismatch"}

# ---------- Audit ----------
def _append_log(username, decision, score, reason, client="web"):
    row = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
           "username": username, "decision": decision, "score": score,
           "reason": reason, "model_version": "stub-0.2", "client": client}
    if AUDIT.exists(): df = pd.read_csv(AUDIT); df.loc[len(df)] = row
    else: df = pd.DataFrame([row])
    df.to_csv(AUDIT, index=False)

def _read_audit(n=200):
    if not AUDIT.exists(): return pd.DataFrame(columns=["timestamp","username","decision","score","reason","model_version","client"])
    return pd.read_csv(AUDIT).tail(n)

# ---------- FastAPI ----------
app = FastAPI(title="FaceGate Pro", version="0.2")

@app.get("/", response_class=HTMLResponse)
async def root():
    return '<h2>FaceGate Pro</h2><p>Open the UI: <a href="/ui">/ui</a> • Health: <a href="/health">/health</a></p>'

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/verify")
async def api_verify(username: str = Form(...), image: UploadFile = Form(...), threshold: float = Form(THRESHOLD), request: Request = None):
    if not username or not image:
        return JSONResponse({"ok": False, "error": "missing_fields"}, status_code=400)
    img_bytes = await image.read()
    if len(img_bytes) < 1024:
        _append_log(username, "denied", 0.0, "no_image", "api")
        return {"ok": True, "decision": "denied", "score": 0.0, "reason": "no_image"}

    # Lockout
    now = time.time()
    st = _state.get(username, {"fails":0,"lock_until":0})
    if now < st["lock_until"]:
        left = int(st["lock_until"]-now)
        _append_log(username, "denied", 0.0, f"locked_{left}s", "api")
        return {"ok": True, "decision": "denied", "score": 0.0, "reason": f"locked_{left}s"}

    res = verify_face(username, img_bytes, threshold)
    decision = "granted" if res["ok"] else "denied"
    if decision == "granted":
        st = {"fails":0, "lock_until":0}
    else:
        st["fails"] += 1
        if st["fails"] >= LOCKOUT_AFTER:
            st["lock_until"] = now + LOCK_SECS
            st["fails"] = 0
    _state[username] = st

    _append_log(username, decision, res["score"], res["reason"], "api")
    return {"ok": True, "decision": decision, "score": float(res["score"]), "reason": res["reason"], "fails": st["fails"], "lock_until": st["lock_until"]}

@app.get("/audit")
async def api_audit():
    return {"records": _read_audit().to_dict(orient="records")}

# ---------- UI helpers ----------
CSS = """
* {font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Arial}
#title {text-align:center}
.card {background:#0f1220; border:1px solid #232a4a; border-radius:16px; padding:14px}
.badge {padding:4px 10px; border-radius:999px; font-size:12px}
.badge.ok {background:#163d2a; color:#22c55e; border:1px solid #1f8a47}
.badge.err{background:#401b1b; color:#ef4444; border:1px solid #b32626}
.gaugewrap{width:100%;background:#0b0e1a;border:1px solid #1c2240;border-radius:10px;padding:8px}
.bar{height:12px;background:#1f2547;border-radius:6px;overflow:hidden}
.fill{height:100%;transition:width .35s}
"""

def gauge(score: float, thr: float) -> str:
    pct = int(max(0, min(100, round(score*100))))
    color = "#22c55e" if score >= thr else "#ef4444"
    return f"""
    <div class="gaugewrap">
      <div style="display:flex;justify-content:space-between;margin-bottom:6px">
        <div>Score: <b>{score:.3f}</b></div>
        <div>Threshold: <b>{thr:.2f}</b></div>
      </div>
      <div class="bar"><div class="fill" style="width:{pct}%; background:{color}"></div></div>
    </div>
    """

# ---------- Agent tools ----------
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

# ---------- Gradio UI ----------
with gr.Blocks(css=CSS, theme=gr.themes.Soft(primary_hue="indigo")) as gr_app:
    gr.Markdown('<h1 id="title">🔐 FaceGate <span class="badge ok">Pro</span> → 🧠 Agent</h1>')
    with gr.Row():
        with gr.Column(scale=2):
            with gr.Group():
                gr.Markdown("### Login / Verification", elem_classes=["card"])
                u = gr.Textbox(label="Username", placeholder="e.g. demo")
                im = gr.Image(sources=["upload","webcam"], type="filepath", label="Face (Upload/Webcam)")
                thr = gr.Slider(0.3, 0.9, value=THRESHOLD, step=0.01, label="Decision Threshold")
                with gr.Row():
                    verify_btn = gr.Button("Verify", variant="primary")
                    clear_btn = gr.Button("Clear")
                dec = gr.HTML()
                gauge_html = gr.HTML()
                attempts = gr.HTML()

        with gr.Column(scale=1):
            with gr.Group():
                gr.Markdown("### Quick Stats", elem_classes=["card"])
                health = gr.HTML()
                last5 = gr.Dataframe(headers=["timestamp","username","decision","score","reason"], wrap=True, row_count=5)

            with gr.Group():
                gr.Markdown("### Settings", elem_classes=["card"])
                lock_info = gr.Markdown(f"- Lock after **{LOCKOUT_AFTER}** failed attempts\n- Lock duration **{LOCK_SECS}s**\n- Allowed users (demo only): `{', '.join(sorted(ALLOWLIST))}`")
                dl = gr.File(label="Download Audit CSV", interactive=False)

    gr.Markdown("---")
    with gr.Tab("Agent"):
        gr.Markdown("#### Command Chat")
        bot = gr.Chatbot(height=280)
        msg = gr.Textbox(placeholder="Example: create task Fix login by tomorrow, or open https://…")
        with gr.Row():
            send = gr.Button("Send")
            clr = gr.Button("Clear Chat")

        with gr.Accordion("Tools", open=False):
            with gr.Row():
                w_url = gr.Textbox(label="Webhook URL")
                w_msg = gr.Textbox(label="Message")
                w_btn = gr.Button("Send Webhook")
            with gr.Row():
                t_title = gr.Textbox(label="Task Title")
                t_due = gr.Textbox(label="Due (YYYY-MM-DD)")
                t_btn = gr.Button("Create Task")
            with gr.Row():
                sum_in = gr.Textbox(label="Summarize text")
                sum_btn = gr.Button("Summarize")
                sum_out = gr.Textbox(label="Summary", interactive=False)
            with gr.Row():
                open_in = gr.Textbox(label="Open URL")
                open_btn = gr.Button("Open")
                open_out = gr.Textbox(label="Open status", interactive=False)

    gr.Markdown("### Audit Trail (last 200)", elem_classes=["card"])
    with gr.Row():
        refresh = gr.Button("Refresh Audit")
    audit_tbl = gr.Dataframe(headers=["timestamp","username","decision","score","reason","model_version","client"],
                             row_count=8, wrap=True)

    # ---- Handlers ----
    def do_health():
        return '<span class="badge ok">OK</span>'

    def do_verify(u_, img_path, thr_):
        if not u_: return ("<span class='badge err'>Missing username</span>", gauge(0.0, thr_), "")
        if not img_path: return ("<span class='badge err'>No image</span>", gauge(0.0, thr_), "")

        # simulate API call
        with open(img_path, "rb") as f:
            r = httpx.post("http://127.0.0.1:8000/verify", data={"username": u_, "threshold": thr_},
                           files={"image": ("img.jpg", f, "image/jpeg")}, timeout=10.0)
        rj = r.json()
        if not rj.get("ok", False):
            return ("<span class='badge err'>Denied</span>", gauge(float(rj.get("score",0.0)), thr_), "")
        decision = rj["decision"]; score = float(rj["score"]); reason = rj.get("reason","")
        color = "ok" if decision=="granted" else "err"
        if decision == "granted": say(f"Welcome {u_}. Access granted.")
        else: say("Access denied.")
        stat = _read_audit(5)[["timestamp","username","decision","score","reason"]]
        # file download
        file_path = str(AUDIT) if AUDIT.exists() else None
        left = ""
        if "lock" in reason:
            left = f"<span class='badge err'>Locked</span>"
        return (f"<span class='badge {color}'>{decision.upper()}</span> &nbsp;<small>{reason}</small>",
                gauge(score, thr_),
                left), stat, file_path

    def do_clear():
        return "", gauge(0.0, THRESHOLD), ""

    def load_audit():
        return _read_audit().reset_index(drop=True)

    def tool_router(history: List[List[str]], text: str):
        reply = ""
        t = text.strip()
        if t.lower().startswith("open "):
            url = t.split(" ",1)[1]; reply = tool_open(url)
        elif t.lower().startswith("webhook "):
            # webhook https://… | message
            try:
                p = t.split(" ",1)[1]
                url, msg = p.split("|",1); reply = tool_webhook(url.strip(), msg.strip())
            except Exception: reply = "Format: webhook <url> | <message>"
        elif t.lower().startswith("create task"):
            # create task Title @YYYY-MM-DD
            body = t[len("create task"):].strip()
            if "@" in body:
                title, due = body.split("@",1); reply = tool_create_task(title.strip(), due.strip())
            else:
                reply = "Format: create task <title> @<YYYY-MM-DD>"
        elif t.lower().startswith("summarize "):
            reply = tool_summarize(t.split(" ",1)[1])
        else:
            reply = "Commands: open <url> | webhook <url> | <msg> | create task <title> @<date> | summarize <text>"
        history = history + [[text, reply]]
        return history, ""

    # wire
    verify_btn.click(do_verify, [u, im, thr], [dec, gauge_html, attempts, last5, dl])
    clear_btn.click(do_clear, [], [dec, gauge_html, attempts])
    refresh.click(load_audit, [], [audit_tbl])
    send.click(tool_router, [bot, msg], [bot, msg])
    clr.click(lambda: [], [], [bot])
    w_btn.click(lambda url,msg: tool_webhook(url,msg), [w_url, w_msg], [open_out])
    t_btn.click(lambda title,due: tool_create_task(title,due), [t_title, t_due], [open_out])
    sum_btn.click(tool_summarize, [sum_in], [sum_out])
    open_btn.click(tool_open, [open_in], [open_out])
    health.value = do_health()
    last5.value = _read_audit(5)[["timestamp","username","decision","score","reason"]]

# Mount UI at /ui
gr.mount_gradio_app(app, gr_app, path="/ui")
