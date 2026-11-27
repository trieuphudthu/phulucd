# -*- coding: utf-8 -*-

import sys
import warnings
from pathlib import Path

def thieu_goi(ten):
    print("\n" + "=" * 72)
    print("THIEU THU VIEN:", ten)
    print("Trong Thonny vao: Tools -> Manage packages")
    print("Tim va cai:", ten)
    print("Sau do bam F5 chay lai.")
    print("=" * 72)
    raise SystemExit(1)

try:
    import numpy as np
except ImportError:
    thieu_goi("numpy")

try:
    import pandas as pd
except ImportError:
    thieu_goi("pandas")

try:
    import matplotlib.pyplot as plt
except ImportError:
    thieu_goi("matplotlib")

try:
    import openpyxl
except ImportError:
    thieu_goi("openpyxl")

try:
    from sklearn.model_selection import (
        train_test_split,
        StratifiedKFold,
        cross_val_score,
    )
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import (
        accuracy_score,
        precision_recall_fscore_support,
        confusion_matrix,
        roc_auc_score,
        roc_curve,
        auc,
    )
    from sklearn.preprocessing import label_binarize
except ImportError:
    thieu_goi("scikit-learn")

try:
    from scipy.stats import binomtest, chi2_contingency, spearmanr
except ImportError:
    thieu_goi("scipy")

warnings.filterwarnings("ignore", category=UserWarning)

try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:
    BASE_DIR = Path.cwd()

# Cấu hình chung
OUTPUT_DIR = BASE_DIR / "HINH_CHUONG_3"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_NAMES = [
    "Du_lieu_khao_sat_500_sinh_vien.xlsx",
    "Du_lieu_khao_sat_500_sinh_vien(1).xlsx",
]

SHEET_NAME = "Du_lieu_500"
RANDOM_STATE = 42
TEST_SIZE = 0.20

FEATURES = [f"Q{i:02d}" for i in range(7, 37)]

REVERSE_FEATURES = (
    [f"Q{i:02d}" for i in range(13, 25)]
    + [f"Q{i:02d}" for i in range(31, 37)]
)

GROUPS = {
    "Chia sẻ": [f"Q{i:02d}" for i in range(7, 13)],
    "Quyền riêng tư": [f"Q{i:02d}" for i in range(13, 19)],
    "Bảo mật tài khoản": [f"Q{i:02d}" for i in range(19, 25)],
    "Tương tác rủi ro": [f"Q{i:02d}" for i in range(25, 31)],
    "Nhận thức & ứng phó": [f"Q{i:02d}" for i in range(31, 37)],
}

LABEL_NAMES = {0: "Thấp", 1: "Trung bình", 2: "Cao"}

FEATURE_LABELS = {
    "Q07": "Q07 – Công khai ngày sinh đầy đủ",
    "Q08": "Q08 – Công khai số điện thoại/email cá nhân",
    "Q09": "Q09 – Chia sẻ vị trí/check-in thời gian thực",
    "Q10": "Q10 – Đăng lịch học/làm việc/di chuyển",
    "Q11": "Q11 – Chia sẻ thông tin người khác khi chưa xin phép",
    "Q12": "Q12 – Đăng ảnh/tài liệu có thông tin nhận dạng",
    "Q13": "Q13 – Không kiểm tra/cập nhật quyền riêng tư",
    "Q14": "Q14 – Không giới hạn đối tượng xem bài đăng",
    "Q15": "Q15 – Không xem lại bài viết/thẻ gắn tên",
    "Q16": "Q16 – Không kiểm soát người có thể liên hệ",
    "Q17": "Q17 – Không kiểm tra quyền ứng dụng liên kết",
    "Q18": "Q18 – Không gỡ quyền/ứng dụng không còn sử dụng",
    "Q19": "Q19 – Không dùng mật khẩu mạnh và khác nhau",
    "Q20": "Q20 – Không thay mật khẩu khi nghi ngờ",
    "Q21": "Q21 – Không bật xác thực hai yếu tố (2FA)",
    "Q22": "Q22 – Không bật cảnh báo đăng nhập lạ",
    "Q23": "Q23 – Không bảo vệ mã xác thực/OTP",
    "Q24": "Q24 – Không đăng xuất trên thiết bị dùng chung",
    "Q25": "Q25 – Chấp nhận kết bạn/theo dõi từ tài khoản lạ",
    "Q26": "Q26 – Nhấp liên kết từ nguồn chưa xác minh",
    "Q27": "Q27 – Trả lời tin nhắn yêu cầu thông tin cá nhân",
    "Q28": "Q28 – Đăng nhập qua đường link chưa xác minh",
    "Q29": "Q29 – Tham gia trò chơi/khảo sát yêu cầu nhiều TT",
    "Q30": "Q30 – Cài ứng dụng/tiện ích từ nguồn chưa rõ",
    "Q31": "Q31 – Không nhận diện dấu hiệu lừa đảo",
    "Q32": "Q32 – Không xác minh nguồn trước khi cung cấp TT",
    "Q33": "Q33 – Không báo cáo/chặn tài khoản đáng ngờ",
    "Q34": "Q34 – Không tìm hiểu thiết lập bảo vệ dữ liệu",
    "Q35": "Q35 – Không kiểm tra lịch sử đăng nhập/hoạt động",
    "Q36": "Q36 – Không chủ động xử lý khi nghi lộ lọt thông tin",
}

BLUE = "#1f77b4"
RED = "#d62728"
ORANGE = "#ff7f0e"
GREEN = "#2ca02c"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "axes.unicode_minus": False,
})

# Đọc và kiểm tra dữ liệu
def chon_file_excel():
    for name in DATA_NAMES:
        p = BASE_DIR / name
        if p.exists():
            return p

    candidates = [
        p for p in BASE_DIR.glob("*.xlsx")
        if "500" in p.name.lower() and not p.name.startswith("~$")
    ]
    if len(candidates) == 1:
        return candidates[0]

    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        path = filedialog.askopenfilename(
            title="Chọn file Excel dữ liệu 500 sinh viên",
            initialdir=str(BASE_DIR),
            filetypes=[("Excel", "*.xlsx"), ("Tất cả", "*.*")],
        )
        root.destroy()
        if path:
            return Path(path)
    except Exception:
        pass

    print("\nKhông tìm thấy file Excel.")
    print("Hãy đặt file Excel cùng thư mục với file Python.")
    raise SystemExit(1)


def read_data(path):
    xls = pd.ExcelFile(path)
    if SHEET_NAME in xls.sheet_names:
        sheet = SHEET_NAME
    else:
        sheet = xls.sheet_names[0]
        print(
            f"CẢNH BÁO: không có sheet '{SHEET_NAME}', "
            f"đang dùng sheet đầu tiên '{sheet}'."
        )

    df = pd.read_excel(path, sheet_name=sheet)
    df.columns = [str(c).strip() for c in df.columns]
    return df, sheet


def risk_label(s):
    if s < 1.33:
        return 0
    if s < 2.67:
        return 1
    return 2


def clean_numeric(df, cols):
    z = df[cols].copy()
    for c in cols:
        z[c] = pd.to_numeric(z[c], errors="coerce")
    return z


def check_required(df):
    all_questions = [f"Q{i:02d}" for i in range(1, 37)]
    expected = ["ID"] + all_questions + [
        "G1_Chia_se", "G2_Quyen_rieng_tu", "G3_Bao_mat_tai_khoan",
        "G4_Tuong_tac_rui_ro", "G5_Nhan_thuc_ung_pho",
        "Diem_rui_ro", "Nhan_ma", "Nhan_rui_ro",
    ]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError("Thiếu cột trong Excel: " + ", ".join(missing))
    if len(df) != 500:
        raise ValueError(f"Dữ liệu phải có 500 bản ghi; đang có {len(df)}")
    if df["ID"].isna().any() or df["ID"].duplicated().any():
        raise ValueError("ID bị thiếu hoặc trùng lặp")

    raw = clean_numeric(df, all_questions)
    if raw.isna().any().any():
        raise ValueError("Cột Q01-Q36 chứa giá trị thiếu/không hợp lệ")
    for i, q in enumerate(all_questions, 1):
        max_val = 2 if i in (1, 4) else 3 if i in (2, 3) else 4
        v = raw[q]
        if ((v < 0) | (v > max_val) | (v != v.astype(int))).any():
            raise ValueError(f"{q} có giá trị ngoài miền mã hóa 0-{max_val}")
    if raw.duplicated().any():
        raise ValueError("Phát hiện bản ghi trùng cả 36 câu hỏi")
    print("\nKIỂM TRA DỮ LIỆU")
    print("- Số dòng:", len(df))
    print("- Q01-Q36: đủ, mã hợp lệ, không trùng hoàn toàn")
    return raw[FEATURES]


def make_risk_matrix(X_raw):
    X = X_raw.copy()
    for c in REVERSE_FEATURES:
        X[c] = 4 - X[c]
    return X


def get_target(df, X):
    score = X.mean(axis=1)
    y_calc = score.apply(risk_label).astype(int)
    group_columns = [
        "G1_Chia_se", "G2_Quyen_rieng_tu", "G3_Bao_mat_tai_khoan",
        "G4_Tuong_tac_rui_ro", "G5_Nhan_thuc_ung_pho",
    ]
    for i, col in enumerate(group_columns):
        v = pd.to_numeric(df[col], errors="coerce")
        expected = X.iloc[:, 6*i:6*(i+1)].mean(axis=1)
        if v.isna().any() or not np.allclose(v, expected, rtol=0, atol=1e-9):
            raise ValueError(f"Điểm nhóm {col} không khớp với Q07-Q36")
    stored_score = pd.to_numeric(df["Diem_rui_ro"], errors="coerce")
    if stored_score.isna().any() or not np.allclose(stored_score, score, rtol=0, atol=1e-9):
        raise ValueError("Cột Diem_rui_ro không khớp điểm tính lại")
    y_file = pd.to_numeric(df["Nhan_ma"], errors="coerce")
    if y_file.isna().any() or not y_file.isin([0, 1, 2]).all():
        raise ValueError("Nhan_ma có giá trị thiếu/không hợp lệ")
    y_file = y_file.astype(int)
    if not np.array_equal(y_file.values, y_calc.values):
        raise ValueError("Nhan_ma không khớp nhãn gán theo S: dừng để tránh công bố sai")
    label_names = y_file.map(LABEL_NAMES)
    if not np.array_equal(label_names.astype(str).values, df["Nhan_rui_ro"].astype(str).values):
        raise ValueError("Nhan_rui_ro không khớp Nhan_ma")
    print("- Kiểm tra G1-G5, Diem_rui_ro, Nhan_ma: khớp toàn bộ 500 dòng")
    return y_calc, score, y_calc


def tinh_bang_3_6(df, nhan):
    """Kiểm định Pearson trên bảng tần số của từng biến với ba mức rủi ro."""
    ket_qua = []
    ten_bien = {
        "Q01": "Giới tính", "Q02": "Độ tuổi", "Q03": "Năm học",
        "Q04": "Nền tảng chính", "Q05": "Thời gian dùng/ngày",
        "Q06": "Tần suất sự cố/nghi ngờ",
    }
    for ma, ten in ten_bien.items():
        bang_cheo = pd.crosstab(pd.to_numeric(df[ma]), nhan)
        so_hang, so_cot = bang_cheo.shape
        n = int(bang_cheo.to_numpy().sum())
        if min(so_hang, so_cot) < 2 or n != len(df):
            raise ValueError(f"Bảng chéo {ma} không đủ dữ liệu cho kiểm định.")
        chi2, p_value, dfree, ky_vong = chi2_contingency(bang_cheo, correction=False)
        v = np.sqrt(chi2 / (n * min(so_hang - 1, so_cot - 1)))
        ket_qua.append({
            "Ma": ma, "Bien": ten, "N": n, "Chi_square": chi2,
            "Bac_tu_do": dfree, "p_value": p_value, "Cramers_V": v,
            "Tan_so_ky_vong_nho_nhat": ky_vong.min(),
            "So_o_ky_vong_duoi_5": int((ky_vong < 5).sum()),
        })
    return pd.DataFrame(ket_qua)


def cronbach_alpha(items):
    """Alpha chưa chuẩn hóa, tính trên các biến đã quy đổi cùng chiều rủi ro."""
    values = items.to_numpy(dtype=float)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 2:
        raise ValueError("Cronbach's Alpha cần ít nhất 2 bản ghi và 2 biến.")
    if not np.isfinite(values).all():
        raise ValueError("Dữ liệu tính Alpha chứa giá trị thiếu hoặc vô hạn.")
    k = values.shape[1]
    item_variances = values.var(axis=0, ddof=1)
    total_variance = values.sum(axis=1).var(ddof=1)
    if total_variance <= 0:
        raise ValueError("Không xác định Alpha: tổng điểm không có biến thiên.")
    return float(k / (k - 1) * (1 - item_variances.sum() / total_variance))


def tinh_alpha_spearman(df, X, risk_score):
    """Thống kê toàn mẫu; không dùng để chọn biến hay mô hình học máy."""
    rows = [
        {"Thang_do": name, "So_bien": len(cols), "So_mau": len(X),
         "Cronbach_Alpha": cronbach_alpha(X[cols])}
        for name, cols in GROUPS.items()
    ]
    rows.append({"Thang_do": "Toàn bộ 30 biến", "So_bien": len(FEATURES),
                 "So_mau": len(X), "Cronbach_Alpha": cronbach_alpha(X[FEATURES])})
    alpha_df = pd.DataFrame(rows)
    q06 = pd.to_numeric(df["Q06"], errors="coerce").to_numpy(dtype=float)
    score = np.asarray(risk_score, dtype=float)
    if len(q06) != len(score) or not np.isfinite(q06).all() or not np.isfinite(score).all():
        raise ValueError("Dữ liệu Q06 và điểm rủi ro không hợp lệ.")
    if np.unique(q06).size < 2 or np.unique(score).size < 2:
        raise ValueError("Không xác định Spearman: một biến không có biến thiên.")

    rho, p_value = spearmanr(q06, score, alternative="two-sided")
    spearman_df = pd.DataFrame([{
        "Bien_1": "Q06", "Bien_2": "Diem_rui_ro", "So_mau": len(q06),
        "Spearman_rho": float(rho), "p_value": float(p_value),
        "Kiem_dinh": "Hai phía (p xấp xỉ)",
    }])
    return alpha_df, spearman_df


def savefig(fig, filename):
    path = OUTPUT_DIR / filename
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Đã tạo:", path.name)


# Xử lý dữ liệu và huấn luyện mô hình
print("=" * 76)
print("TẠO HÌNH 3.1 -> HÌNH 3.8")
print("=" * 76)

DATA_FILE = chon_file_excel()
print("File Excel:", DATA_FILE)
df, sheet_used = read_data(DATA_FILE)
print("Sheet:", sheet_used)
print("Kích thước:", df.shape)

X_raw = check_required(df)
X = make_risk_matrix(X_raw)
y, risk_score, y_calc = get_target(df, X)


bang_3_6 = tinh_bang_3_6(df, y)
bang_3_6.to_csv(OUTPUT_DIR / "Bang_3_6_Chi_square_Cramers_V.csv",
                index=False, encoding="utf-8-sig")
print(bang_3_6.to_string(index=False))
if "--bang-3-6" in sys.argv:
    raise SystemExit(0)

try:
    import shap
except ImportError:
    print("\nTHIEU THU VIEN SHAP.")
    print("Hinh 3.7 bat buoc dung SHAP, nen chuong trinh se dung lai")
    print("de tranh tao mot hinh gia hoac gan nhan SHAP khong dung.")
    thieu_goi("shap")

alpha_df, spearman_df = tinh_alpha_spearman(df, X, risk_score)
print("\nCRONBACH'S ALPHA (sau quy đổi chiều điểm)")
print(alpha_df.to_string(index=False, float_format=lambda v: f"{v:.6f}"))
print("\nSPEARMAN Q06 VỚI ĐIỂM RỦI RO")
print(f"rho = {spearman_df.loc[0, 'Spearman_rho']:.6f}")
print(f"p (hai phía, xấp xỉ) = {spearman_df.loc[0, 'p_value']:.6e}")
alpha_df.to_csv(OUTPUT_DIR / "Cronbach_Alpha.csv", index=False, encoding="utf-8-sig")
spearman_df.to_csv(OUTPUT_DIR / "Spearman_Q06.csv", index=False, encoding="utf-8-sig")

X_train, X_test, y_train, y_test = train_test_split(
    X[FEATURES],
    y,
    test_size=TEST_SIZE,
    stratify=y,
    random_state=RANDOM_STATE,
)

rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    max_features="sqrt",
    min_samples_split=5,
    min_samples_leaf=1,
    class_weight="balanced",
    random_state=RANDOM_STATE,
    n_jobs=1,
)

dt = DecisionTreeClassifier(
    class_weight="balanced",
    random_state=RANDOM_STATE,
)

print("\nHuấn luyện Random Forest 300 cây...")
rf.fit(X_train, y_train)
dt.fit(X_train, y_train)

rf_pred = rf.predict(X_test)
rf_prob = rf.predict_proba(X_test)

dt_pred = dt.predict(X_test)

# Ghép đúng từng phiếu trước khi kiểm định McNemar hai phía.
paired = pd.DataFrame({
    "ID": df.loc[y_test.index, "ID"],
    "Nhan_ma": y_test,
    "RF": rf_pred,
    "DT": dt_pred,
})
rf_correct = paired["RF"] == paired["Nhan_ma"]
dt_correct = paired["DT"] == paired["Nhan_ma"]
paired["RF_dung"] = rf_correct
paired["DT_dung"] = dt_correct
paired.to_csv(OUTPUT_DIR / "Du_doan_ghep_cap.csv", index=False, encoding="utf-8-sig")
rf_only = int((rf_correct & ~dt_correct).sum())
dt_only = int((~rf_correct & dt_correct).sum())
discordant = rf_only + dt_only
p_mcnemar = binomtest(rf_only, discordant, 0.5).pvalue if discordant else 1.0
print(pd.crosstab(rf_correct, dt_correct))
print(f"McNemar (hai phía): b={rf_only}, c={dt_only}, p={p_mcnemar:.7f}")

rf_acc = accuracy_score(y_test, rf_pred)
rf_p, rf_r, rf_f1, _ = precision_recall_fscore_support(
    y_test, rf_pred, average="macro", zero_division=0
)

y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
rf_auc_macro = roc_auc_score(
    y_test_bin, rf_prob, multi_class="ovr", average="macro"
)

dt_acc = accuracy_score(y_test, dt_pred)
dt_p, dt_r, dt_f1, _ = precision_recall_fscore_support(
    y_test, dt_pred, average="macro", zero_division=0
)

cm = confusion_matrix(y_test, rf_pred, labels=[0, 1, 2])

print("\nKẾT QUẢ CHÍNH")
print(f"Random Forest Accuracy : {rf_acc:.4f}")
print(f"Random Forest Macro-F1 : {rf_f1:.4f}")
print(f"Random Forest ROC-AUC  : {rf_auc_macro:.4f}")
print("Ma trận nhầm lẫn:")
print(cm)
print(f"Decision Tree Accuracy : {dt_acc:.4f}")
print(f"Decision Tree Macro-F1 : {dt_f1:.4f}")

print("\nĐang tạo 8 hình...")

# Hình 3.1: Nền tảng mạng xã hội sử dụng chính
if "Q04" not in df.columns:
    raise SystemExit("Thiếu Q04 nên không thể tạo Hình 3.1.")

q04 = pd.to_numeric(df["Q04"], errors="coerce")
if q04.isna().any():
    raise SystemExit("Q04 có giá trị không phải số.")

platform_counts = (
    q04.astype(int)
    .value_counts()
    .reindex([0, 1, 2], fill_value=0)
)
platform_names = ["Facebook", "Zalo", "TikTok"]
platform_values = platform_counts.values
platform_pct = 100 * platform_values / len(df)

fig, ax = plt.subplots(figsize=(7.2, 5.0))
bars = ax.bar(platform_names, platform_values, color=BLUE)
ax.set_title("Nền tảng mạng xã hội sử dụng chính")
ax.set_ylabel("Số sinh viên")
ax.set_ylim(0, max(platform_values) * 1.18)

for bar, n, pct in zip(bars, platform_values, platform_pct):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + max(platform_values) * 0.015,
        f"{int(n)}\n({pct:.1f}%)",
        ha="center",
        va="bottom",
        fontsize=9,
    )

fig.tight_layout()
savefig(fig, "Hinh_3_1_Nen_tang_mang_xa_hoi.png")

# Hình 3.2: Điểm rủi ro trung bình theo nhóm yếu tố
group_means = pd.Series({
    group: X[cols].mean(axis=1).mean()
    for group, cols in GROUPS.items()
})

fig, ax = plt.subplots(figsize=(7.5, 5.0))
bars = ax.bar(group_means.index, group_means.values, color=BLUE)
ax.set_title("Điểm rủi ro trung bình theo năm nhóm yếu tố")
ax.set_ylabel("Điểm rủi ro trung bình (0–4)")
ax.set_ylim(0, 4.0)
ax.tick_params(axis="x", rotation=17)

for bar, val in zip(bars, group_means.values):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        val + 0.05,
        f"{val:.2f}",
        ha="center",
        va="bottom",
        fontsize=9,
    )

fig.tight_layout()
savefig(fig, "Hinh_3_2_Diem_rui_ro_5_nhom.png")

# Hình 3.3: Phân bố mức độ rủi ro
risk_counts = (
    pd.Series(y)
    .value_counts()
    .reindex([0, 1, 2], fill_value=0)
)
risk_values = risk_counts.values
risk_names = [LABEL_NAMES[i] for i in [0, 1, 2]]
risk_pct = 100 * risk_values / len(y)

fig, ax = plt.subplots(figsize=(7.2, 5.0))
bars = ax.bar(risk_names, risk_values, color=BLUE)
ax.set_title(f"Phân bố mức độ rủi ro của {len(y)} sinh viên")
ax.set_ylabel("Số sinh viên")
ax.set_ylim(0, max(risk_values) * 1.18)

for bar, n, pct in zip(bars, risk_values, risk_pct):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + max(risk_values) * 0.015,
        f"{int(n)}\n({pct:.1f}%)",
        ha="center",
        va="bottom",
        fontsize=9,
    )

fig.tight_layout()
savefig(fig, "Hinh_3_3_Phan_bo_muc_do_rui_ro.png")

# Hình 3.4: Ma trận nhầm lẫn Random Forest
fig, ax = plt.subplots(figsize=(6.8, 5.7))
im = ax.imshow(
    cm,
    interpolation="nearest",
    cmap=plt.cm.Blues,
    vmin=0,
    vmax=max(1, int(cm.max())),
)
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

ax.set_title("Ma trận nhầm lẫn – Random Forest")
ax.set_xlabel("Dự đoán")
ax.set_ylabel("Thực tế")
ax.set_xticks([0, 1, 2], ["Thấp", "Trung bình", "Cao"])
ax.set_yticks([0, 1, 2], ["Thấp", "Trung bình", "Cao"])

for i in range(3):
    for j in range(3):
        ax.text(
            j,
            i,
            f"{cm[i, j]}",
            ha="center",
            va="center",
            color="black",
            fontsize=12,
        )

fig.tight_layout()
savefig(fig, "Hinh_3_4_Ma_tran_nham_lan_RF.png")

# Hình 3.5: ROC đa lớp Random Forest
fig, ax = plt.subplots(figsize=(7.2, 5.5))
roc_colors = [BLUE, ORANGE, GREEN]
class_aucs = {}

for i, cls in enumerate([0, 1, 2]):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], rf_prob[:, i])
    class_auc = auc(fpr, tpr)
    class_aucs[cls] = class_auc
    ax.plot(
        fpr,
        tpr,
        color=roc_colors[i],
        linewidth=1.8,
        label=f"{LABEL_NAMES[cls]} (AUC={class_auc:.4f})",
    )

ax.plot(
    [0, 1],
    [0, 1],
    color="red",
    linestyle="--",
    linewidth=1.5,
    label="Ngẫu nhiên",
)

ax.set_title(f"ROC One-vs-Rest – Macro AUC = {rf_auc_macro:.4f}")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(-0.02, 1.05)
ax.legend(loc="lower right", frameon=True)

fig.tight_layout()
savefig(fig, "Hinh_3_5_ROC_da_lop_RF.png")

# Hình 3.6: 15 đặc trưng quan trọng nhất
importance_df = pd.DataFrame({
    "Bien": FEATURES,
    "Importance": rf.feature_importances_,
})
importance_df["Noi_dung"] = importance_df["Bien"].map(FEATURE_LABELS)
importance_df = importance_df.sort_values(
    "Importance", ascending=False
).reset_index(drop=True)

top15 = importance_df.head(15).copy()
plot15 = top15.iloc[::-1]

fig, ax = plt.subplots(figsize=(9.0, 6.8))
ax.barh(plot15["Noi_dung"], plot15["Importance"], color=BLUE)
ax.set_title("15 đặc trưng quan trọng nhất của Random Forest")
ax.set_xlabel("Feature importance (Gini)")
ax.tick_params(axis="y", labelsize=8)

fig.tight_layout()
savefig(fig, "Hinh_3_6_Top15_Feature_Importance.png")

# Hình 3.7: 15 đặc trưng có ảnh hưởng lớn theo SHAP
print("  Đang tính SHAP cho Hình 3.7...")

explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_train.sample(n=min(100, len(X_train)), random_state=RANDOM_STATE))  # 100 mẫu training cố định; không dùng test

if isinstance(shap_values, list):
    abs_per_class = [
        np.abs(np.asarray(v)).mean(axis=0)
        for v in shap_values
    ]
    mean_abs_shap = np.mean(abs_per_class, axis=0)
else:
    arr = np.asarray(shap_values)

    if arr.ndim == 3 and arr.shape[1] == len(FEATURES):
        mean_abs_shap = np.abs(arr).mean(axis=(0, 2))
    elif arr.ndim == 3 and arr.shape[2] == len(FEATURES):
        mean_abs_shap = np.abs(arr).mean(axis=(0, 1))
    elif arr.ndim == 2 and arr.shape[1] == len(FEATURES):
        mean_abs_shap = np.abs(arr).mean(axis=0)
    else:
        raise RuntimeError(
            "Không nhận dạng được định dạng SHAP: "
            + str(arr.shape)
        )

shap_df = pd.DataFrame({
    "Bien": FEATURES,
    "Mean_abs_SHAP": mean_abs_shap,
})
shap_df["Noi_dung"] = shap_df["Bien"].map(FEATURE_LABELS)
shap_df = shap_df.sort_values(
    "Mean_abs_SHAP", ascending=False
).reset_index(drop=True)

shap_top15 = shap_df.head(15).copy()
shap_plot = shap_top15.iloc[::-1]

fig, ax = plt.subplots(figsize=(9.0, 6.8))
ax.barh(
    shap_plot["Noi_dung"],
    shap_plot["Mean_abs_SHAP"],
    color=BLUE,
)
ax.set_title("15 đặc trưng có ảnh hưởng lớn theo SHAP")
ax.set_xlabel("Mean |SHAP value|")
ax.tick_params(axis="y", labelsize=8)

fig.tight_layout()
savefig(fig, "Hinh_3_7_Top15_SHAP.png")

# Hình 3.8: Kết quả 5-fold trên 400 mẫu training (lựa chọn đặc trưng bên trong mỗi fold)

ks = [5, 10, 15, 20, 25, 30]
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
reduced_scores = {k: [] for k in ks}

def make_rf():
    return RandomForestClassifier(
        n_estimators=300, max_depth=None, max_features="sqrt",
        min_samples_split=5, min_samples_leaf=1,
        class_weight="balanced", random_state=RANDOM_STATE, n_jobs=1,
    )

for fold_id, (train_idx, valid_idx) in enumerate(cv.split(X_train, y_train), 1):
    X_fit = X_train.iloc[train_idx]
    y_fit = y_train.iloc[train_idx]
    X_valid = X_train.iloc[valid_idx]
    y_valid = y_train.iloc[valid_idx]
    # Xếp hạng đặc trưng chỉ trên tập huấn luyện của từng fold.
    rank_model = make_rf()
    rank_model.fit(X_fit, y_fit)
    ranked_fold = np.array(FEATURES)[np.argsort(rank_model.feature_importances_)[::-1]].tolist()
    for k in ks:
        subfeatures = [c for c in FEATURES if c in set(ranked_fold[:k])]
        model_k = make_rf()
        model_k.fit(X_fit[subfeatures], y_fit)
        fold_pred = model_k.predict(X_valid[subfeatures])
        _, _, fold_f1, _ = precision_recall_fscore_support(
            y_valid, fold_pred, average="macro", zero_division=0
        )
        reduced_scores[k].append(float(fold_f1))
    print(f"  Đã xong fold {fold_id}/5 (chỉ training)")

reduced_rows = []
for k in ks:
    vals = np.array(reduced_scores[k])
    reduced_rows.append({
        "So_dac_trung": k,
        "CV_Macro_F1_TB": float(vals.mean()),
        "CV_SD": float(vals.std(ddof=0)),
        **{f"Fold_{i+1}": float(vals[i]) for i in range(5)},
    })
reduced_df = pd.DataFrame(reduced_rows)
# K được chọn theo CV training, hòa điểm ưu tiên mô hình ít biến hơn.
best_k = int(reduced_df.sort_values(["CV_Macro_F1_TB", "So_dac_trung"], ascending=[False, True]).iloc[0]["So_dac_trung"])

ranked_features = importance_df["Bien"].tolist()
selected_features = [c for c in FEATURES if c in set(ranked_features[:best_k])]
selected_model = make_rf()
selected_model.fit(X_train[selected_features], y_train)
selected_pred = selected_model.predict(X_test[selected_features])
selected_accuracy = accuracy_score(y_test, selected_pred)
selected_f1 = precision_recall_fscore_support(
    y_test, selected_pred, average="macro", zero_division=0
)[2]

fig, ax = plt.subplots(figsize=(7.2, 5.1))
ax.errorbar(
    reduced_df["So_dac_trung"], reduced_df["CV_Macro_F1_TB"],
    yerr=reduced_df["CV_SD"], fmt="-o", color=BLUE, linewidth=1.7,
    markersize=5, capsize=4, label="Trung bình ± SD qua 5 fold (training)",
)
ax.axvline(best_k, color=GREEN, linestyle="--", linewidth=1.2,
           label=f"Chọn k={best_k} bằng CV training")
ax.set_title("Hiệu năng mô hình rút gọn – đánh giá chéo trên training")
ax.set_xlabel("Số đặc trưng sử dụng")
ax.set_ylabel("Macro-F1 trung bình trên 5 fold training")
ax.set_xticks(ks)
ax.set_ylim(max(0.0, float(reduced_df["CV_Macro_F1_TB"].min()) - 0.09), 1.02)
ax.legend(loc="lower right", fontsize=8)
for k, val in zip(reduced_df["So_dac_trung"], reduced_df["CV_Macro_F1_TB"]):
    ax.annotate(f"{val:.3f}", (k, val), xytext=(0, 6), textcoords="offset points",
                ha="center", fontsize=8)
fig.tight_layout()
savefig(fig, "Hinh_3_8_Hieu_nang_theo_so_dac_trung.png")

# Xác thực chéo và lưu kết quả
print("\nĐang chạy 5-fold Macro-F1...")

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE,
)
cv_scores = cross_val_score(
    rf,
    X_train[FEATURES],
    y_train,
    cv=cv,
    scoring="f1_macro",
    n_jobs=1,
)
dt_cv_scores = cross_val_score(
    dt, X_train[FEATURES], y_train, cv=cv,
    scoring="f1_macro", n_jobs=1,
)
print("RF 5-fold training:", cv_scores, "TB", float(cv_scores.mean()))
print("DT 5-fold training:", dt_cv_scores, "TB", float(dt_cv_scores.mean()))
print("Mô hình rút gọn chọn bằng CV: k=", best_k,
      " | test Accuracy=", round(selected_accuracy, 4),
      " | test Macro-F1=", round(selected_f1, 4))

metrics_df = pd.DataFrame([
    {
        "Mo_hinh": "Random Forest",
        "Accuracy": rf_acc,
        "Macro_Precision": rf_p,
        "Macro_Recall": rf_r,
        "Macro_F1": rf_f1,
        "ROC_AUC_Macro_OvR": rf_auc_macro,
    },
    {
        "Mo_hinh": "Decision Tree",
        "Accuracy": dt_acc,
        "Macro_Precision": dt_p,
        "Macro_Recall": dt_r,
        "Macro_F1": dt_f1,
        "ROC_AUC_Macro_OvR": np.nan,
    },
])

distribution_df = pd.DataFrame({
    "Nhan_ma": [0, 1, 2],
    "Muc_rui_ro": ["Thấp", "Trung bình", "Cao"],
    "So_luong": risk_values,
    "Ty_le_%": risk_pct,
})

platform_df = pd.DataFrame({
    "Nen_tang": platform_names,
    "So_luong": platform_values,
    "Ty_le_%": platform_pct,
})

group_df = pd.DataFrame({
    "Nhom": group_means.index,
    "Diem_TB": group_means.values,
})

cm_df = pd.DataFrame(
    cm,
    index=["Thực tế - Thấp", "Thực tế - Trung bình", "Thực tế - Cao"],
    columns=["Dự đoán - Thấp", "Dự đoán - Trung bình", "Dự đoán - Cao"],
)

metrics_df.to_csv(
    OUTPUT_DIR / "Ket_qua_mo_hinh.csv",
    index=False,
    encoding="utf-8-sig",
)
importance_df.to_csv(
    OUTPUT_DIR / "Feature_Importance_30_bien.csv",
    index=False,
    encoding="utf-8-sig",
)
shap_df.to_csv(
    OUTPUT_DIR / "SHAP_Importance_30_bien.csv",
    index=False,
    encoding="utf-8-sig",
)
reduced_df.to_csv(
    OUTPUT_DIR / "Mo_hinh_rut_gon.csv",
    index=False,
    encoding="utf-8-sig",
)

excel_path = OUTPUT_DIR / "Bang_ket_qua_Hinh_3_1_den_3_8.xlsx"
with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
    alpha_df.to_excel(writer, sheet_name="Cronbach_Alpha", index=False)
    spearman_df.to_excel(writer, sheet_name="Spearman_Q06", index=False)
    platform_df.to_excel(writer, sheet_name="Hinh_3_1", index=False)
    group_df.to_excel(writer, sheet_name="Hinh_3_2", index=False)
    distribution_df.to_excel(writer, sheet_name="Hinh_3_3", index=False)
    cm_df.to_excel(writer, sheet_name="Hinh_3_4")
    metrics_df.to_excel(writer, sheet_name="Ket_qua_mo_hinh", index=False)
    importance_df.to_excel(writer, sheet_name="Hinh_3_6", index=False)
    shap_df.to_excel(writer, sheet_name="Hinh_3_7", index=False)
    reduced_df.to_excel(writer, sheet_name="Hinh_3_8", index=False)
    pd.DataFrame([{"So_dac_trung_chon": best_k,
                   "Dac_trung_duoc_chon": ", ".join(selected_features),
                   "Test_Accuracy": selected_accuracy,
                   "Test_Macro_F1": selected_f1}]).to_excel(
        writer, sheet_name="Rut_gon_chon_cuoi", index=False
    )
    pd.DataFrame({
        "Fold": [1, 2, 3, 4, 5],
        "Macro_F1": cv_scores,
        "DT_Macro_F1": dt_cv_scores,
    }).to_excel(writer, sheet_name="CV_5_fold", index=False)

summary = OUTPUT_DIR / "DOI_CHIEU_KET_QUA_LUAN_VAN.txt"
with open(summary, "w", encoding="utf-8") as f:
    f.write("ĐỐI CHIẾU KẾT QUẢ CHƯƠNG 3\n")
    f.write("=" * 60 + "\n")
    f.write(f"File: {DATA_FILE}\n")
    f.write(f"Số sinh viên: {len(df)}\n\n")

    f.write("Cronbach's Alpha (toàn mẫu, sau quy đổi chiều điểm)\n")
    f.write(alpha_df.to_string(index=False, float_format=lambda v: f"{v:.6f}"))
    f.write("\n\nSpearman Q06 với điểm rủi ro (toàn mẫu)\n")
    f.write(f"rho = {spearman_df.loc[0, 'Spearman_rho']:.9f}\n")
    f.write(f"p (hai phía, xấp xỉ) = {spearman_df.loc[0, 'p_value']:.9e}\n\n")
    f.write("Hình 3.1 - Nền tảng chính\n")
    for name, n, pct in zip(
        platform_names, platform_values, platform_pct
    ):
        f.write(f"  {name}: {int(n)} ({pct:.1f}%)\n")

    f.write("\nHình 3.2 - Điểm 5 nhóm\n")
    for name, val in group_means.items():
        f.write(f"  {name}: {val:.3f}\n")

    f.write("\nHình 3.3 - Phân bố rủi ro\n")
    for name, n, pct in zip(risk_names, risk_values, risk_pct):
        f.write(f"  {name}: {int(n)} ({pct:.1f}%)\n")

    f.write("\nRandom Forest\n")
    f.write(f"  Accuracy = {rf_acc:.4f}\n")
    f.write(f"  Macro-F1 = {rf_f1:.4f}\n")
    f.write(f"  ROC-AUC macro OvR = {rf_auc_macro:.4f}\n")
    f.write("  Confusion matrix:\n")
    f.write(np.array2string(cm) + "\n")

    f.write("\nDecision Tree\n")
    f.write(f"  Accuracy = {dt_acc:.4f}\n")
    f.write(f"  Macro-F1 = {dt_f1:.4f}\n")

    f.write("\nRandom Forest 5-fold TRAINING (400 mẫu)\n")
    f.write("  Fold = " + np.array2string(cv_scores, precision=6) + "\n")
    f.write(f"  Mean = {cv_scores.mean():.6f}\n")
    f.write(f"  SD = {cv_scores.std():.6f}\n")

    f.write("\nDecision Tree 5-fold TRAINING\n")
    f.write(f"  Mean = {dt_cv_scores.mean():.6f}; SD = {dt_cv_scores.std():.6f}\n")
    f.write("\nMô hình rút gọn, xếp hạng RIÊNG trong mỗi fold trên training\n")
    f.write(reduced_df.to_string(index=False))
    f.write(f"\n\nChọn k={best_k} chỉ bằng CV training, đặc trưng: {selected_features}\n")
    f.write(f"Tập test CHỈ đánh giá cuối: Accuracy={selected_accuracy:.6f}; Macro-F1={selected_f1:.6f}\n")

print("\n" + "=" * 76)
print("HOÀN TẤT - ĐÃ TẠO HÌNH 3.1 -> HÌNH 3.8")
print("=" * 76)
print("Thư mục kết quả:")
print(OUTPUT_DIR.resolve())

print("\nDanh sách hình:")
for p in sorted(OUTPUT_DIR.glob("Hinh_3_*.png")):
    print(" -", p.name)
