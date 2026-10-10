# -*- coding: utf-8 -*-
"""Download the three UCI data sets used in the paper and verify their SHA-256 hashes."""
import hashlib
import os
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = "https://archive.ics.uci.edu/ml/machine-learning-databases/"
FILES = {
    "parkinsons.data": (BASE + "parkinsons/parkinsons.data",
                        "a36116c2deac07895b79ce97a2f99a1b3720ac4892995ff5c3b9ccb397ed2bc5"),
    "breast-cancer-wisconsin.data": (BASE + "breast-cancer-wisconsin/breast-cancer-wisconsin.data",
                                     "402c585309c399237740f635ef9919dc512cca12cbeb20de5e563a4593f22b64"),
    "processed.cleveland.data": (BASE + "heart-disease/processed.cleveland.data",
                                 "a74b7efa387bc9d108d7d0115d831fe9b414b29ae7124f331b622b4efa0427c8"),
}
out = os.path.join(ROOT, "data", "external", "p3")
os.makedirs(out, exist_ok=True)
for name, (url, sha) in FILES.items():
    fn = os.path.join(out, name)
    if not os.path.exists(fn):
        print("downloading", url)
        urllib.request.urlretrieve(url, fn)
    h = hashlib.sha256(open(fn, "rb").read()).hexdigest()
    print(f"{name}: {'OK' if h == sha else 'HASH MISMATCH ' + h}")
sav = os.path.join(ROOT, "data", "external", "qchat_poland_252.sav")
print("POL252:", "present" if os.path.exists(sav) else "missing — see data/README.md")
