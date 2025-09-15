# 🔐 FaceGate + Agent (Flask + ONNX/PyTorch)

مشروع **FaceGate** هو نظام تحقق بالوجوه (Face Verification) مدمج مع **Agent Tools**.  
النظام يعتمد على **EfficientNet-B0** مدرّب مسبقًا (ONNX/PyTorch)، ويشتغل عبر **Flask API** مع واجهة ويب تفاعلية.

---

## 📂 هيكل المشروع
```
project-root/
│
├── app_flask.py              # الكود الرئيسي (Flask app + Agent tools)
├── requirements.txt           # المكتبات المطلوبة
├── README.md                  # هذا الملف
│
├── assets/                    # ملفات النظام
│   ├── class_names.json       # أسماء الكلاسات (بالترتيب الصحيح من التدريب)
│   ├── threshold.json         # قيمة العتبة الافتراضية (مثال: 0.55)
│
├── models/                    # النماذج
│   ├── best_efficientnet_b0.onnx   # الموديل ONNX
│   └── best_efficientnet_b0.pth    # نسخة Torch (اختياري)
│
├── logs/                      # ملفات التدقيق (Audit Logs)
│   └── audit_log.csv          # يتولّد أوتوماتيكياً
│
└── venv/                      # بيئة بايثون الافتراضية (اختياري)
```

---

## ⚙️ المتطلبات
- Python 3.9 – 3.11 (مُجرّب على 3.11.9)
- مكتبات Python (مذكورة في `requirements.txt`):
  - Flask
  - onnxruntime
  - torch (اختياري لو استعملت `.pth`)
  - numpy
  - pandas
  - pillow
  - httpx

---

## 🚀 خطوات التشغيل

### 1) تفعيل البيئة الافتراضية (venv)
على Windows:
```bash
python -m venv .venv
.venv\Scripts\activate
```

على Linux / macOS:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

### 2) تثبيت المكتبات
```bash
pip install -r requirements.txt
```

---

### 3) تشغيل السيرفر
```bash
python app_flask.py
```

رح يطلعلك سطر زي:
```
Running on http://127.0.0.1:5000/
```

افتح المتصفح وروح على الرابط:
👉 [http://127.0.0.1:5000/](http://127.0.0.1:5000/)

---

## 🖼️ استخدام الواجهة (UI)

1. اكتب **Username** (لازم يطابق اسم الكلاس من `class_names.json`).  
   ⚠️ تم تعديل الكود بحيث لازم الاسم والصورة يتطابقوا 100% عشان يسمح بالدخول.  
2. حمّل صورة الوجه أو استعمل الكاميرا.  
3. حدّد قيمة **Decision Threshold** (افتراضي 0.55).  
4. اضغط **Verify**.  
5. إذا الاسم والصورة متطابقين → النظام يعطيك **GRANTED ✅**  
   غير هيك → **DENIED ❌**.

---

## 📑 APIs المتوفرة

- **`GET /health`** → معلومات عن الموديل والعتبة.
- **`POST /verify`** → التحقق بالوجه + الاسم.
- **`GET /audit`** → آخر عمليات التحقق (JSON).
- **`GET /audit.csv`** → تحميل ملف الـ Audit بصيغة CSV.
- **`POST /agent/chat`** → تفاعل مع الـ Agent (مسموح فقط بعد تطابق الوجه والاسم).
- **`GET /tasks.json`** → استرجاع المهام المحفوظة.

---

## 🧠 ملاحظات مهمة
- النظام مشغّل بـ **ONNX Runtime** (يفضّل استخدام CUDA لو فيه GPU).  
- لازم ملف `class_names.json` يكون بنفس الترتيب اللي اتدرّب فيه الموديل.  
- تقدر تغيّر **Threshold** الافتراضية من `assets/threshold.json`.  
- الـ Agent ما رح يشتغل إلا إذا تم التحقق بالوجه والاسم بشكل صحيح.  

---

## 👨‍💻 خطوات إضافية
- كل عمليات الدخول والفشل بتتسجل في ملف: `logs/audit_log.csv`.  
- في حال فشل المستخدم أكثر من 3 مرات → يتم **Lockout** لمدة 60 ثانية.  
- النظام يدعم **Webcam Capture** مباشرة من المتصفح.  

---

## 👨‍🔬 المطور
- مشروع FaceGate مبني خصيصًا للاستخدام الأكاديمي والتجريبي.  
- غير مخصص للاستخدام الإنتاجي (Production) إلا بعد تقوية الحماية وإضافة HTTPS.  

---
