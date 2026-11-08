# remap.py — Remapea TrashNet a 4 clases de EcoVision
import os, shutil, random

SRC = "raw_trashnet"
DST = "dataset"
MAP = {
    "plastic":   "botella_plastica",
    "metal":     "lata",
    "paper":     "papel_carton",
    "cardboard": "papel_carton",
    "glass":     "no_identificado",
    "trash":     "no_identificado",
}
random.seed(42)

for split in ["train", "val", "test"]:
    for cls in set(MAP.values()):
        os.makedirs(f"{DST}/{split}/{cls}", exist_ok=True)

for cls_src, cls_dst in MAP.items():
    path = os.path.join(SRC, cls_src)
    if not os.path.isdir(path):
        print(f"AVISO: no existe {path}, lo salto")
        continue
    files = [f for f in os.listdir(path)
             if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    random.shuffle(files)
    n = len(files)
    for i, f in enumerate(files):
        split = "train" if i < 0.7*n else ("val" if i < 0.85*n else "test")
        shutil.copy(os.path.join(path, f),
                    f"{DST}/{split}/{cls_dst}/{cls_src}_{f}")

print("Dataset remapeado OK")
for split in ["train", "val", "test"]:
    print(f"\n[{split}]")
    for cls in sorted(os.listdir(f"{DST}/{split}")):
        n = len(os.listdir(f"{DST}/{split}/{cls}"))
        print(f"  {cls}: {n} imagenes")