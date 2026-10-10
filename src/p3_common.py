"""Hạ tầng dùng chung cho bộ thí nghiệm bài P3 (bản tiếng Anh, khuôn Elsevier).

Bài P3 hỏi: trên phiếu hỏi sàng lọc cỡ mẫu nhỏ, các kiến trúc linh hoạt hiện đại
(KAN / Transformer / Mamba) có thật sự hơn mô hình cộng tính không, và ta có đủ
dữ liệu để trả lời câu hỏi đó không?

Module này gom lại những thứ mọi giai đoạn đều cần:
  * nạp dữ liệu (D2 Ba Lan + các bộ bảng công khai dùng để lặp lại)
  * bộ mô hình sklearn (cộng tính -> linh hoạt), có đếm số tham số
  * hiệu chỉnh phương sai Nadeau-Bengio cho repeated CV
  * hiệu chỉnh so sánh bội Benjamini-Hochberg
  * sai số chuẩn AUC Hanley-McNeil và phân tích cỡ mẫu
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
from scipy import stats
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, SplineTransformer, StandardScaler

warnings.filterwarnings("ignore")

ROOT = __file__.rsplit("src", 1)[0]
SAV = ROOT + "data/external/qchat_poland_252.sav"
P3DIR = ROOT + "data/external/p3/"
OUT = ROOT + "out/reports/"
SEED = 0


# =========================================================================== DỮ LIỆU
@dataclass
class Dataset:
    key: str
    name: str
    X: np.ndarray
    y: np.ndarray
    label_source: str
    feature_kind: str
    note: str = ""
    item_scale: tuple | None = None
    meta: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.y)

    @property
    def p(self) -> int:
        return self.X.shape[1]

    @property
    def prevalence(self) -> float:
        return float(self.y.mean())


def load_poland() -> Dataset:
    """D2 - đoàn hệ Ba Lan, 25 mục Q-CHAT thang 0..4.

    Nhãn là chẩn đoán RLPTK do cơ sở lâm sàng cộng đồng đặt, NGOÀI nghiên cứu và
    độc lập với phiếu Q-CHAT (theo mô tả của bộ dữ liệu Mendeley 10.17632/tmpkt2mfkg.1).
    Bộ dữ liệu KHÔNG ghi quy trình chẩn đoán chi tiết; chỉ khẳng định được tính
    ĐỘC LẬP của nhãn với đặc trưng, và đó đúng là tính chất bài này cần.
    """
    import pyreadstat

    df, _ = pyreadstat.read_sav(SAV)
    items = [c for c in df.columns if c.startswith("qchat") and c.endswith("recode")]
    X = np.clip(df[items].to_numpy(float), 0, 4)
    y = (df["group"].to_numpy(float) == 1.0).astype(int)
    return Dataset("POL252", "Q-CHAT Poland", X, y,
                   label_source="community-based clinical ASD diagnosis, assigned outside the study",
                   feature_kind="25 ordinal questionnaire items, 0-4",
                   note="case-control, 53.6% positive", item_scale=(0.0, 4.0),
                   meta={"items": items})


def load_parkinsons() -> Dataset:
    raw = np.genfromtxt(P3DIR + "parkinsons.data", delimiter=",", skip_header=1,
                        dtype=str)
    vals = raw[:, 1:].astype(float)
    y = vals[:, 16].astype(int)
    X = np.delete(vals, 16, axis=1)
    return Dataset("PARK195", "Parkinson voice", X, y,
                   label_source="independent clinical diagnosis",
                   feature_kind="22 continuous acoustic measures",
                   note="UCI Parkinsons")


def load_cleveland() -> Dataset:
    raw = np.genfromtxt(P3DIR + "processed.cleveland.data", delimiter=",",
                        missing_values="?", filling_values=np.nan)
    y = (raw[:, 13] > 0).astype(int)
    X = SimpleImputer(strategy="median").fit_transform(raw[:, :13])
    return Dataset("HEART303", "Cleveland heart disease", X, y,
                   label_source="independent angiographic finding",
                   feature_kind="13 mixed clinical attributes",
                   note="UCI heart-disease, processed.cleveland")


def load_bcw_original() -> Dataset:
    """Ordinal 1-10, chín 'mục' - bộ công khai gần phiếu hỏi nhất về cấu trúc."""
    raw = np.genfromtxt(P3DIR + "breast-cancer-wisconsin.data", delimiter=",",
                        missing_values="?", filling_values=np.nan)
    ok = ~np.isnan(raw).any(axis=1)
    raw = raw[ok]
    y = (raw[:, 10] == 4).astype(int)
    X = raw[:, 1:10]
    return Dataset("BCW683", "Breast cancer Wisconsin (original)", X, y,
                   label_source="independent histopathology",
                   feature_kind="9 ordinal items, 1-10",
                   note="683 complete records of 699", item_scale=(1.0, 10.0))


def load_wdbc() -> Dataset:
    from sklearn.datasets import load_breast_cancer

    d = load_breast_cancer()
    return Dataset("WDBC569", "Breast cancer Wisconsin (diagnostic)",
                   d.data, (d.target == 0).astype(int),
                   label_source="independent histopathology",
                   feature_kind="30 continuous morphometric features",
                   note="scikit-learn bundled copy")


REPLICATION_LOADERS = {
    "POL252": load_poland,
    "PARK195": load_parkinsons,
    "HEART303": load_cleveland,
    "BCW683": load_bcw_original,
    "WDBC569": load_wdbc,
}


# ======================================================================== MÔ HÌNH
class SumScore(ClassifierMixin, BaseEstimator):
    """Tổng điểm thô - 0 tham số học. Chuẩn tham chiếu của mọi phiếu sàng lọc."""

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        s = X.sum(1).astype(float)
        self.sign_ = 1.0
        if len(self.classes_) == 2 and s.std() > 0:
            if np.corrcoef(s, np.asarray(y, float))[0, 1] < 0:
                self.sign_ = -1.0
        return self

    def predict_proba(self, X):
        s = self.sign_ * X.sum(1).astype(float)
        rng = s.max() - s.min()
        s = (s - s.min()) / (rng if rng else 1.0)
        return np.column_stack([1 - s, s])


def model_suite(p: int):
    """(mã, nhãn, #tham số, estimator) - xếp theo bậc linh hoạt tăng dần.

    Ba dòng đầu là 'cộng tính': không mô hình hoá tương tác giữa các mục.
    Dòng E là CẬN TRÊN của một tầng self-attention: attention trên các mục chỉ có
    thể khai thác quan hệ đôi một, và mô hình tương tác đôi một tường minh khai
    thác chúng TỐI ƯU, nên nó chặn trên mọi kiến trúc attention một tầng.
    """
    n_inter = p * (p - 1) // 2
    return [
        ("A", "Unweighted sum score", 0, SumScore()),
        ("B", "Logistic regression (L2)", p + 1,
         make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=5000))),
        ("C", "Additive splines, 5 knots (KAN-lite)", 7 * p + 1,
         make_pipeline(SplineTransformer(n_knots=5, degree=3),
                       LogisticRegression(C=0.05, max_iter=8000))),
        ("D", "Additive splines, 3 knots (KAN-lite)", 5 * p + 1,
         make_pipeline(SplineTransformer(n_knots=3, degree=3),
                       LogisticRegression(C=0.05, max_iter=8000))),
        ("E", "Pairwise interactions (attention upper bound)", p + n_inter + 1,
         make_pipeline(StandardScaler(),
                       PolynomialFeatures(2, interaction_only=True, include_bias=False),
                       LogisticRegression(C=0.02, max_iter=8000))),
        ("F", "MLP, one hidden layer of 32", 33 * p + 33,
         make_pipeline(StandardScaler(),
                       MLPClassifier((32,), alpha=1.0, max_iter=4000, random_state=SEED))),
        ("G", "MLP, 64-32", 65 * p + 2145,
         make_pipeline(StandardScaler(),
                       MLPClassifier((64, 32), alpha=1.0, max_iter=4000,
                                     random_state=SEED))),
        ("H", "Gradient boosting", None, GradientBoostingClassifier(random_state=SEED)),
        ("I", "Random forest", None,
         RandomForestClassifier(n_estimators=500, random_state=SEED, n_jobs=1)),
        ("J", "k-nearest neighbours, k = 7", 1,
         make_pipeline(StandardScaler(), KNeighborsClassifier(7, weights="distance"))),
    ]


def cv_scores(est, X, y, n_splits=5, n_repeats=10, seed=SEED, n_jobs=-1):
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,
                                 random_state=seed)
    return cross_val_score(est, X, y, cv=cv, scoring="roc_auc", n_jobs=n_jobs)


# ================================================================== THỐNG KÊ
def nadeau_bengio(a, b, n_train: int, n_test: int):
    """t ghép cặp có hiệu chỉnh phương sai cho repeated k-fold CV (Nadeau & Bengio 2003).

    Các lượt CV lặp dùng chung dữ liệu nên không độc lập; t-test thường coi chúng
    độc lập và cho p quá nhỏ. Hệ số 1/k + n_test/n_train sửa đúng chỗ đó.
    Trả về: (Delta, t, p_hiệu_chỉnh, p_ngây_thơ, nửa bề rộng KTC 95% hiệu chỉnh).
    """
    d = np.asarray(a, float) - np.asarray(b, float)
    k = len(d)
    var = d.var(ddof=1) * (1.0 / k + n_test / n_train)
    se = float(np.sqrt(var + 1e-18))
    t = float(d.mean() / se) if se > 0 else 0.0
    p = float(2 * (1 - stats.t.cdf(abs(t), k - 1)))
    naive = float(stats.ttest_rel(a, b).pvalue)
    half = float(stats.t.ppf(0.975, k - 1) * se)
    return float(d.mean()), t, p, naive, half


def benjamini_hochberg(pvals, alpha: float = 0.05):
    """Trả về (p hiệu chỉnh theo BH, bác bỏ hay không) giữ nguyên thứ tự đầu vào."""
    p = np.asarray(pvals, float)
    m = len(p)
    order = np.argsort(p)
    adj = p[order] * m / np.arange(1, m + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0, 1)
    out = np.empty(m)
    out[order] = adj
    return out, out <= alpha


def se_auc_hanley(A: float, n1: int, n0: int) -> float:
    """Sai số chuẩn của AUC (Hanley & McNeil 1982), xấp xỉ phân phối mũ."""
    q1 = A / (2 - A)
    q2 = 2 * A * A / (1 + A)
    return float(np.sqrt((A * (1 - A) + (n1 - 1) * (q1 - A * A)
                          + (n0 - 1) * (q2 - A * A)) / (n1 * n0)))


def n_needed(delta: float, prevalence: float, A: float = 0.945, r: float = 0.9,
             power: float = 0.80, step: int = 25, cap: int = 4_000_000) -> int:
    """Cỡ mẫu cho power cho trước khi so sánh hai AUC tương quan trên cùng ca bệnh."""
    za, zb = stats.norm.ppf(0.975), stats.norm.ppf(power)
    n = 50
    while n <= cap:
        n1 = max(int(round(n * prevalence)), 1)
        n0 = n - n1
        if n1 >= 5 and n0 >= 5:
            sed = np.sqrt(2 * se_auc_hanley(A, n1, n0) ** 2 * (1 - r))
            if (za + zb) * sed <= delta:
                return n
        n += step
    return -1


def mde(n: int, prevalence: float, A: float = 0.945, r: float = 0.9,
        power: float = 0.80) -> float:
    """Chênh AUC nhỏ nhất phát hiện được (minimum detectable effect) ở cỡ mẫu n."""
    za, zb = stats.norm.ppf(0.975), stats.norm.ppf(power)
    n1 = max(int(round(n * prevalence)), 1)
    n0 = n - n1
    sed = np.sqrt(2 * se_auc_hanley(A, n1, n0) ** 2 * (1 - r))
    return float((za + zb) * sed)
