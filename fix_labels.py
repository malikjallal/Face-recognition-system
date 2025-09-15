import json, pathlib, sys

ASSETS = pathlib.Path("assets")
src_candidates = [
    ASSETS/"idx_to_class.json",
    ASSETS/"idx_to_class_dict.json",
    ASSETS/"class_map.json",
]
dst = ASSETS/"class_names.json"

# اختر المصدر تلقائياً
src = None
for c in src_candidates:
    if c.exists():
        src = c; break
if src is None:
    # لو class_names.json أصلاً موجود لكن مرتب غلط (list)، هنحاول استعماله كما هو
    src = dst
    if not src.exists():
        print("No mapping file found. Put your mapping in assets/ and re-run.")
        sys.exit(1)

obj = json.loads(src.read_text(encoding="utf-8"))

def from_name_to_idx(d):
    # {"Alvaro_Morte": 17, ...} -> list مرتبة
    items = sorted(d.items(), key=lambda kv: int(kv[1]))
    return [name for name, _ in items]

def from_idx_to_name(d):
    # {"0":"Alice", "1":"Bob"} -> list مرتبة
    items = sorted(d.items(), key=lambda kv: int(kv[0]))
    return [name for _k, name in items]

if isinstance(obj, dict):
    # حاول نميّز الشكل
    if all(isinstance(v, int) for v in obj.values()):
        class_list = from_name_to_idx(obj)
    elif all(str(k).isdigit() for k in obj.keys()):
        class_list = from_idx_to_name(obj)
    else:
        print("Unknown dict format. Please paste a snippet here so we can help.")
        sys.exit(2)
elif isinstance(obj, list):
    # هو أصلاً list  نكتب نسخة نظيفة فقط
    class_list = list(obj)
else:
    print("Unsupported JSON type.")
    sys.exit(3)

# اكتب نسخة احتياطية إن وجد ملف سابق
if dst.exists():
    bkp = dst.with_suffix(".json.bak")
    bkp.write_text(dst.read_text(encoding="utf-8"), encoding="utf-8")
    print("Backup:", bkp)

dst.write_text(json.dumps(class_list, ensure_ascii=False, indent=2), encoding="utf-8")
print("Wrote:", dst, "with", len(class_list), "classes")
