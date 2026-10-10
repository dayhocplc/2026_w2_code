# Data (not redistributed)

| File | Source | SHA-256 |
|---|---|---|
| `external/qchat_poland_252.sav` | Niedźwiecka A. (2020) Q-CHAT scores of Polish toddlers with ASD and typically developing controls, Mendeley Data v2, https://doi.org/10.17632/tmpkt2mfkg.2 — download the SPSS file and save it under this name | 7fed516fc4e615f4750d0e44c1eea538bbdcdee989e8c2ff8de719976fa08a41 |
| `external/p3/parkinsons.data` | UCI Machine Learning Repository, Parkinsons | a36116c2deac07895b79ce97a2f99a1b3720ac4892995ff5c3b9ccb397ed2bc5 |
| `external/p3/breast-cancer-wisconsin.data` | UCI, Breast Cancer Wisconsin (Original) | 402c585309c399237740f635ef9919dc512cca12cbeb20de5e563a4593f22b64 |
| `external/p3/processed.cleveland.data` | UCI, Heart Disease (Cleveland) | a74b7efa387bc9d108d7d0115d831fe9b414b29ae7124f331b622b4efa0427c8 |

`python scripts/fetch_data.py` downloads the three UCI files and checks the hashes. The analysis of the
Mamdani layer (`run_w2_mamdani_theorem.py`) needs no data at all.
