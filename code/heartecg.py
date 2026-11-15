#!/usr/bin/env python3
"""
heartecg.py - Image-based 12-lead ECG analysis for heart-disease classification and
myocardial-infarction (MI) detection.  Classical ML, CNN / Transformer baselines and a
three-stream fusion network (TriStream-ECG), with patient-disjoint evaluation on PTB-XL
and external testing on real ECG printouts (Mendeley "ECG Images dataset of Cardiac Patients").

Stages (run in this order; every stage resumes from what is already on disk):
    python heartecg.py data            # download + cache PTB-XL and the Mendeley images
    python heartecg.py ml              # handcrafted descriptors + LR / SVM / RF / XGBoost
    python heartecg.py train_all       # 1-D reference, 4 image baselines, TriStream-ECG, ablations
    python heartecg.py explain         # Grad-CAM lead-localisation analysis
    python heartecg.py analyse         # all tables, statistics and figures -> results.json
    python heartecg.py smoke           # end-to-end self-test on synthetic data (no downloads)

Environment: Google Colab GPU runtime (T4 works; L4 / A100 is 3-4x faster).  Seeded; nothing
in the PTB-XL test fold (fold 10) or in the external set is used for fitting or model selection.
"""
import argparse, ast, json, math, os, subprocess, sys, time, zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2

cv2.setNumThreads(0)

# ----------------------------------------------------------------------------- configuration
CFG = dict(
    work="/content/drive/MyDrive/HeartECG", raw="/content/heart_raw",
    H=320, W=544, epochs=12, bs=32, lr=1e-4, head_lr=1e-3, wd=0.05, warmup=1,
    seeds=[0, 1, 2], workers=max(2, min(8, (os.cpu_count() or 2))), pretrained=True,
    n_boot=1000, smoke=False,
)
CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
MI = CLASSES.index("MI")
LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
LAYOUT = [["I", "aVR", "V1", "V4"], ["II", "aVL", "V2", "V5"], ["III", "aVF", "V3", "V6"]]
ROW_Y = [0.145, 0.39, 0.63, 0.875]          # baselines of the 3 lead rows + rhythm strip
X0, X1 = 0.035, 0.98                         # horizontal extent of the 10-s trace
ANT_CODES, INF_CODES = {"AMI", "ASMI", "ALMI"}, {"IMI", "ILMI", "IPMI", "IPLMI"}
BACKBONES = dict(
    resnet50="resnet50.a1_in1k", effnetv2s="tf_efficientnetv2_s.in21k_ft_in1k",
    convnext_t="convnext_tiny.fb_in22k_ft_in1k", vit_s="vit_small_patch16_224.augreg_in21k_ft_in1k")
SMOKE_BACKBONES = dict(resnet50="resnet18", effnetv2s="efficientnet_b0",
                       convnext_t="convnext_atto", vit_s="vit_tiny_patch16_224")
PERTURBS = ["rot5", "blur", "jpeg20", "noise", "nogrid", "lowcontrast", "halfres"]
MEND_API = "https://data.mendeley.com/public-api/datasets/gwbz3fsgp8"
PTBXL_ZIP = "https://physionet.org/content/ptb-xl/get-zip/1.0.3/"
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def W_(*p):
    d = Path(CFG["work"]).joinpath(*p)
    d.parent.mkdir(parents=True, exist_ok=True)
    return d


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


# ----------------------------------------------------------------------------- data
def stage_data():
    """Download PTB-XL (100 Hz) and the Mendeley ECG images; cache compact arrays in the work dir."""
    raw = Path(CFG["raw"]); raw.mkdir(parents=True, exist_ok=True)
    if not W_("ptbxl_X.npy").exists():
        import wfdb
        from scipy.signal import butter, filtfilt
        z = raw / "ptbxl.zip"
        if not z.exists():
            log("downloading PTB-XL (1.8 GB)")
            subprocess.run(["wget", "-q", "--show-progress", "-O", str(z), PTBXL_ZIP], check=True)
        with zipfile.ZipFile(z) as zf:
            names = [n for n in zf.namelist() if "/records100/" in n or n.endswith(".csv")]
            zf.extractall(raw, names)
        root = next(raw.glob("ptb-xl*/"))
        db = pd.read_csv(root / "ptbxl_database.csv", index_col="ecg_id")
        scp = pd.read_csv(root / "scp_statements.csv", index_col=0)
        scp = scp[scp.diagnostic == 1]
        codes = db.scp_codes.apply(ast.literal_eval)
        Y = np.zeros((len(db), 5), np.int8); loc = []
        for i, d in enumerate(codes):
            for k in d:
                if k in scp.index:
                    Y[i, CLASSES.index(scp.loc[k].diagnostic_class)] = 1
            a, b = bool(ANT_CODES & set(d)), bool(INF_CODES & set(d))
            loc.append("ant" if a and not b else "inf" if b and not a else "")
        keep = Y.sum(1) > 0                      # records with at least one diagnostic superclass
        db = db[keep]; Y = Y[keep]; loc = np.array(loc)[keep]
        bb, aa = butter(2, 0.5 / 50.0, "high")   # 0.5 Hz high-pass (baseline wander), as on printers

        def rd(f):
            s = wfdb.rdsamp(str(root / f))[0].astype(np.float32)
            return filtfilt(bb, aa, np.nan_to_num(s), axis=0).astype(np.float16)
        with ThreadPoolExecutor(8) as ex:
            X = np.stack(list(ex.map(rd, db.filename_lr)))
        meta = pd.DataFrame(dict(ecg_id=db.index, patient_id=db.patient_id.astype(int).values,
                                 age=db.age.values, sex=db.sex.values, fold=db.strat_fold.values, mi_loc=loc))
        for j, c in enumerate(CLASSES):
            meta[c] = Y[:, j]
        np.save(W_("ptbxl_X.npy"), X); meta.to_csv(W_("ptbxl_meta.csv"), index=False)
        log("PTB-XL cached", X.shape, "dropped (no superclass):", int((~keep).sum()))
    if not W_("mend_X.npy").exists():
        try:
            import requests
            info = requests.get(MEND_API, timeout=60).json()
            d = raw / "mendeley"; d.mkdir(exist_ok=True)

            def get(f):
                out = d / f["filename"]
                if not out.exists() or out.stat().st_size != f["size"]:
                    subprocess.run(["curl", "-s", "-L", "--retry", "5", "-o", str(out),
                                    f["content_details"]["download_url"]], check=False)
                return out
            log("downloading", len(info["files"]), "Mendeley ECG images (615 MB)")
            with ThreadPoolExecutor(8) as ex:
                paths = sorted(ex.map(get, info["files"]))
            cache_mendeley(paths)
        except Exception as e:                    # the pipeline still runs without the external set
            log("WARNING: Mendeley download failed (%s). Upload the four class folders to %s/mendeley "
                "and re-run this stage." % (e, raw))
            paths = sorted((raw / "mendeley").rglob("*.jpg"))
            if paths:
                cache_mendeley(paths)


def mend_label(name):
    n = name.upper()
    return 3 if n.startswith("PMI") else 1 if n.startswith("MI") else 2 if n.startswith("HB") else 0


def crop_printout(img):
    """Keep the trace area of the EDAN report (drops the header that carries the record ID)."""
    h, w = img.shape[:2]
    c = img[int(0.182 * h):int(0.963 * h), int(0.016 * w):int(0.983 * w)]
    return cv2.resize(c, (CFG["W"], CFG["H"]), interpolation=cv2.INTER_AREA)


def cache_mendeley(paths):
    X, y, names = [], [], []
    for p in paths:
        im = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if im is None:
            continue
        X.append(cv2.cvtColor(crop_printout(im), cv2.COLOR_BGR2RGB)); y.append(mend_label(p.name)); names.append(p.name)
    np.save(W_("mend_X.npy"), np.stack(X)); np.save(W_("mend_y.npy"), np.array(y))
    pd.Series(names).to_csv(W_("mend_names.csv"), index=False)
    log("Mendeley cached:", len(X), "images; class counts (Normal, MI, HB, PMI):", np.bincount(y, minlength=4).tolist())


def synth_data(n=420, seed=0):
    """Synthetic stand-in used only by the smoke test."""
    r = np.random.default_rng(seed); t = np.arange(1000) / 100.0
    X = np.zeros((n, 1000, 12), np.float32); Y = np.zeros((n, 5), np.int8)
    for i in range(n):
        k = r.integers(0, 5); Y[i, k] = 1
        if r.random() < 0.2:
            Y[i, r.integers(1, 5)] = 1; Y[i, 0] = 0
        hr = r.uniform(0.8, 1.6)
        for l in range(12):
            a = r.uniform(0.5, 1.4); s = np.zeros(1000, np.float32)
            for b in np.arange(0.3, 10, 1 / hr):
                s += a * np.exp(-((t - b) / 0.012) ** 2) + 0.15 * np.exp(-((t - b - 0.25) / 0.05) ** 2)
                if Y[i, MI] and l in (6, 7, 8, 9):
                    s += 0.35 * np.exp(-((t - b - 0.12) / 0.06) ** 2)
                if Y[i, 2]:
                    s -= 0.2 * np.exp(-((t - b - 0.25) / 0.05) ** 2)
                if Y[i, 4]:
                    s += 0.8 * a * np.exp(-((t - b) / 0.012) ** 2)
                if Y[i, 3]:
                    s += 0.4 * np.exp(-((t - b - 0.05) / 0.03) ** 2)
            X[i, :, l] = s + r.normal(0, 0.02, 1000)
    meta = pd.DataFrame(dict(ecg_id=np.arange(n), patient_id=np.arange(n) // 2 * 2 + (np.arange(n) % 10 == 0),
                             age=r.integers(20, 90, n), sex=r.integers(0, 2, n), fold=np.arange(n) % 10 + 1,
                             mi_loc=np.where(Y[:, MI] == 1, np.where(r.random(n) < 0.5, "ant", "inf"), "")))
    for j, c in enumerate(CLASSES):
        meta[c] = Y[:, j]
    np.save(W_("ptbxl_X.npy"), X.astype(np.float16)); meta.to_csv(W_("ptbxl_meta.csv"), index=False)
    rr = np.random.default_rng(1); ex = []; ey = []
    for i in range(48):
        ex.append(render(X[i], rr, "rand")); ey.append([0, 1, 2, 3][i % 4] if not Y[i, MI] else 1)
    np.save(W_("mend_X.npy"), np.stack(ex)); np.save(W_("mend_y.npy"), np.array(ey))


def load_ptbxl():
    X = np.load(W_("ptbxl_X.npy"), mmap_mode="r")
    meta = pd.read_csv(W_("ptbxl_meta.csv"), keep_default_na=False)
    Y = meta[CLASSES].values.astype(np.float32)
    idx = dict(train=np.where(meta.fold <= 8)[0], val=np.where(meta.fold == 9)[0], test=np.where(meta.fold == 10)[0])
    return X, Y, meta, idx


def load_mend():
    if not W_("mend_X.npy").exists():
        return None, None
    return np.load(W_("mend_X.npy"), mmap_mode="r"), np.load(W_("mend_y.npy"))


# ----------------------------------------------------------------------------- ECG -> image
def render(sig, rng=None, style="clean", H=None, W=None, grid=True):
    """Rasterise a 10-s 12-lead ECG ([1000,12], mV, 100 Hz) as a 3x4 + rhythm-strip printout.
    25 mm/s, 10 mm/mV. style='clean' is deterministic; style='rand' randomises printout appearance."""
    H = H or CFG["H"]; W = W or CFG["W"]; ss = 2; h, w = H * ss, W * ss
    rand = style == "rand"
    sig = np.asarray(sig, np.float32)
    paper = (255 - rng.integers(0, 20, 3)) if rand else np.array([255, 255, 255])
    img = np.empty((h, w, 3), np.uint8); img[:] = paper
    x0, x1 = X0 * w, X1 * w
    pxs = (x1 - x0) / 1000.0; mm = (x1 - x0) / 250.0
    mv = 10 * mm * (rng.uniform(0.92, 1.08) if rand else 1.0)
    gmode = rng.choice(3, p=[0.15, 0.35, 0.5]) if rand else 2            # 0 none, 1 major, 2 major+minor
    if grid and gmode > 0:
        gcol = np.array([250, 185, 185]) if not rand else np.array(
            [[250, 185, 185], [240, 150, 150], [200, 200, 200], [250, 200, 160]][rng.integers(0, 4)])
        ox, oy = (rng.uniform(0, 5 * mm, 2) if rand else (0.0, 0.0))
        if gmode == 2:
            ys = np.arange(oy, h, mm).astype(int); xs = np.arange(ox, w, mm).astype(int)
            img[ys[:, None], xs[None, :]] = (gcol * 0.5 + paper * 0.5).astype(np.uint8)
        img[np.arange(oy, h, 5 * mm).astype(int)] = gcol
        img[:, np.arange(ox, w, 5 * mm).astype(int)] = gcol
    ink = (0, 0, 0) if not rand else tuple(int(v) for v in rng.integers(0, 70, 3))
    th = ss if not rand else int(rng.integers(1, ss + 2))
    by = [r * h for r in ROW_Y]

    def line(xs, ys):
        pts = np.stack([xs, np.clip(ys, 1, h - 2)], 1) * 16
        cv2.polylines(img, [pts.astype(np.int32)], False, ink, th, cv2.LINE_AA, 4)
    for r in range(3):
        for c in range(4):
            li = LEADS.index(LAYOUT[r][c]); t = np.arange(250 * c, 250 * (c + 1))
            line(x0 + t * pxs, by[r] - sig[t, li] * mv)
            cv2.putText(img, LAYOUT[r][c], (int(x0 + 250 * c * pxs + 3 * ss), int(by[r] - 3 * mm)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38 * ss, ink, 1, cv2.LINE_AA)
    rl = LEADS.index("II") if not rand else LEADS.index(["II", "II", "III", "V1"][rng.integers(0, 4)])
    line(x0 + np.arange(1000) * pxs, by[3] - sig[:, rl] * mv)
    cv2.putText(img, LEADS[rl], (int(x0 + 3 * ss), int(by[3] - 3 * mm)), cv2.FONT_HERSHEY_SIMPLEX, 0.38 * ss, ink, 1, cv2.LINE_AA)
    for b in by:                                                           # 1-mV calibration pulse
        xa, xb = 0.008 * w, 0.026 * w
        line(np.array([xa, xa, xb, xb]), np.array([b, b - mv, b - mv, b]))
    if not rand or rng.random() < 0.6:                                     # column separators
        for c in (1, 2, 3):
            xc = int(x0 + 250 * c * pxs)
            cv2.line(img, (xc, int(by[0] - 3.5 * mm)), (xc, int(by[2] + 3.5 * mm)), ink, 1, cv2.LINE_AA)
    img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
    if rand:
        M = cv2.getRotationMatrix2D((W / 2, H / 2), rng.uniform(-2, 2), rng.uniform(0.93, 1.05))
        M[:, 2] += rng.uniform(-0.03, 0.03, 2) * (W, H)
        img = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderValue=tuple(int(v) for v in paper))
        if rng.random() < 0.3:
            img = cv2.GaussianBlur(img, (3, 3), rng.uniform(0.3, 1.0))
        if rng.random() < 0.5:
            a, b = rng.uniform(0.75, 1.1), rng.uniform(-15, 15)
            img = np.clip(img.astype(np.float32) * a + b, 0, 255).astype(np.uint8)
        if rng.random() < 0.3:
            img = np.clip(img + rng.normal(0, rng.uniform(2, 8), img.shape), 0, 255).astype(np.uint8)
        if rng.random() < 0.2:
            img = np.repeat(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)[..., None], 3, 2)
        if rng.random() < 0.5:
            img = cv2.imdecode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(30, 92))])[1], 1)
    return img


def perturb(img, kind):
    """Deterministic test-time corruptions used in the robustness analysis."""
    H, W = img.shape[:2]
    if kind == "rot5":
        return cv2.warpAffine(img, cv2.getRotationMatrix2D((W / 2, H / 2), 5, 1.0), (W, H), borderValue=(255, 255, 255))
    if kind == "blur":
        return cv2.GaussianBlur(img, (5, 5), 1.5)
    if kind == "jpeg20":
        return cv2.imdecode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 20])[1], 1)
    if kind == "noise":
        return np.clip(img + np.random.default_rng(0).normal(0, 12, img.shape), 0, 255).astype(np.uint8)
    if kind == "lowcontrast":
        return np.clip(img.astype(np.float32) * 0.6 + 60, 0, 255).astype(np.uint8)
    if kind == "halfres":
        return cv2.resize(cv2.resize(img, (W // 2, H // 2), interpolation=cv2.INTER_AREA), (W, H), interpolation=cv2.INTER_LINEAR)
    return img


def to_tensor(img):
    return torch.from_numpy(((img.astype(np.float32) / 255.0 - MEAN) / STD).transpose(2, 0, 1))


class ECGImages(torch.utils.data.Dataset):
    def __init__(self, X, Y, idx, style="clean", pert=None, seed=0):
        self.X, self.Y, self.idx, self.style, self.pert, self.seed, self.rng = X, Y, idx, style, pert, seed, None

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        if self.rng is None:
            wi = torch.utils.data.get_worker_info()
            self.rng = np.random.default_rng([self.seed, wi.id if wi else 0, int(time.time() * 1e3) % 100000 if self.style == "rand" else 0])
        j = self.idx[i]
        img = render(self.X[j], self.rng, self.style, grid=self.pert != "nogrid")
        if self.pert:
            img = perturb(img, self.pert)
        return to_tensor(img), torch.from_numpy(self.Y[j])


class Arr(torch.utils.data.Dataset):
    """Already-rasterised images (external printouts)."""
    def __init__(self, X):
        self.X = X

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return to_tensor(np.asarray(self.X[i])), torch.zeros(5)


class Sig(torch.utils.data.Dataset):
    def __init__(self, X, Y, idx, train=False):
        self.X, self.Y, self.idx, self.train = X, Y, idx, train

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        j = self.idx[i]; s = np.asarray(self.X[j], np.float32).T.copy()
        if self.train:
            s = np.roll(s, np.random.randint(0, 100), 1) * np.random.uniform(0.9, 1.1)
        return torch.from_numpy(s), torch.from_numpy(self.Y[j])


def loader(ds, train=False, bs=None):
    return torch.utils.data.DataLoader(ds, batch_size=bs or CFG["bs"], shuffle=train, drop_last=train,
                                       num_workers=CFG["workers"], pin_memory=DEV == "cuda",
                                       persistent_workers=False)


# ----------------------------------------------------------------------------- handcrafted descriptors (GPU)
class Handcrafted(nn.Module):
    """Parameter-free descriptor computed on the GPU: orientation histograms (HOG-style, 9 bins,
    32x32 cells), local-binary-pattern histograms and ink-geometry statistics over a 4x4 zone grid
    that follows the lead panels."""
    def __init__(self):
        super().__init__()
        self.register_buffer("mean", torch.tensor(MEAN).view(1, 3, 1, 1)); self.register_buffer("std", torch.tensor(STD).view(1, 3, 1, 1))
        self.register_buffer("sx", torch.tensor([[-1., 0, 1], [-2, 0, 2], [-1, 0, 1]]).view(1, 1, 3, 3))

    @torch.no_grad()
    def forward(self, x):
        x = x.float()
        g = ((x * self.std + self.mean).clamp(0, 1) * torch.tensor([0.299, 0.587, 0.114], device=x.device).view(1, 3, 1, 1)).sum(1, keepdim=True)
        B = g.shape[0]
        gx = F.conv2d(g, self.sx, padding=1); gy = F.conv2d(g, self.sx.transpose(2, 3), padding=1)
        mag = (gx ** 2 + gy ** 2).sqrt(); ang = torch.atan2(gy, gx) % math.pi
        b = (ang / math.pi * 9).long().clamp(0, 8)
        hog = F.avg_pool2d(F.one_hot(b.squeeze(1), 9).permute(0, 3, 1, 2).float() * mag, 32)
        hog = (hog / (hog.norm(dim=1, keepdim=True) + 1e-4)).flatten(1)
        p = F.pad(g, (1, 1, 1, 1), mode="replicate"); code = torch.zeros_like(g)
        for k, (dy, dx) in enumerate([(0, 0), (0, 1), (0, 2), (1, 2), (2, 2), (2, 1), (2, 0), (1, 0)]):
            code += (p[:, :, dy:dy + g.shape[2], dx:dx + g.shape[3]] >= g + 0.02).float() * (2 ** k)
        code = (code / 8).long().clamp(0, 31)                                # 32-bin quantised LBP
        lbp = F.adaptive_avg_pool2d(F.one_hot(code.squeeze(1), 32).permute(0, 3, 1, 2).float(), 4).flatten(1)
        ink = (g < 0.45).float(); H, W = ink.shape[2:]
        yy = torch.linspace(0, 1, H, device=x.device).view(1, 1, H, 1)
        dens = F.adaptive_avg_pool2d(ink, 4); m1 = F.adaptive_avg_pool2d(ink * yy, 4) / (dens + 1e-4)
        m2 = (F.adaptive_avg_pool2d(ink * yy ** 2, 4) / (dens + 1e-4) - m1 ** 2).clamp_min(0).sqrt()
        colext = ink.view(B, 1, 4, H // 4, W).sum(3)                          # ink pixels per column in each row band
        ext = F.adaptive_avg_pool2d(colext, (4, 4)); exm = F.adaptive_max_pool2d(colext, (4, 4))
        geo = torch.cat([dens, m1, m2, ext / H, exm / H], 1).flatten(1)
        return torch.cat([hog, lbp, geo], 1)


def hc_dim():
    return Handcrafted()(torch.zeros(1, 3, CFG["H"], CFG["W"])).shape[1]


# ----------------------------------------------------------------------------- networks
def backbone(key, **kw):
    import timm
    name = (SMOKE_BACKBONES if CFG["smoke"] else BACKBONES)[key]
    if "vit" in name:
        kw["img_size"] = (CFG["H"], CFG["W"])
    return timm.create_model(name, pretrained=CFG["pretrained"], num_classes=0, **kw)


class Single(nn.Module):
    def __init__(self, key):
        super().__init__()
        self.key = key; self.net = backbone(key); self.head = nn.Linear(self.net.num_features, 5)

    def forward(self, x, emb=False):
        z = self.net(x); o = self.head(z)
        return (o, z) if emb else o

    def groups(self):
        return list(self.net.parameters()), list(self.head.parameters())


class TriStream(nn.Module):
    """TriStream-ECG: ConvNeXt stream (local waveform morphology) + ViT stream (long-range, cross-lead
    context) + handcrafted-descriptor stream, joined by bidirectional cross-attention and a learned gate."""
    def __init__(self, use_hc=True, use_xattn=True, d=256):
        super().__init__()
        self.use_hc, self.use_xattn = use_hc, use_xattn
        self.cnn = backbone("convnext_t"); self.vit = backbone("vit_s")
        self.pc = nn.Linear(self.cnn.num_features, d); self.pv = nn.Linear(self.vit.num_features, d)
        if use_xattn:
            self.c2v = nn.MultiheadAttention(d, 4, batch_first=True, dropout=0.1); self.v2c = nn.MultiheadAttention(d, 4, batch_first=True, dropout=0.1)
            self.nc, self.nv = nn.LayerNorm(d), nn.LayerNorm(d)
        n = 2
        if use_hc:
            self.hc = Handcrafted(); self.ph = nn.Sequential(nn.BatchNorm1d(hc_dim()), nn.Linear(hc_dim(), d), nn.GELU(), nn.Dropout(0.2), nn.Linear(d, d)); n = 3
        self.gate = nn.Linear(n * d, n)
        self.head = nn.Sequential(nn.LayerNorm(n * d), nn.Dropout(0.2), nn.Linear(n * d, d), nn.GELU(), nn.Linear(d, 5))
        self.aux_c, self.aux_v = nn.Linear(d, 5), nn.Linear(d, 5)
        self.fmap = None

    def forward(self, x, emb=False, aux=False):
        f = self.cnn.forward_features(x); self.fmap = f                       # [B,C,h,w] kept for Grad-CAM
        c = self.pc(f.flatten(2).transpose(1, 2))
        v = self.pv(self.vit.forward_features(x)[:, self.vit.num_prefix_tokens:])
        if self.use_xattn:
            c2 = self.nc(c + self.c2v(c, v, v, need_weights=False)[0]); v2 = self.nv(v + self.v2c(v, c, c, need_weights=False)[0])
            c, v = c2, v2
        zs = [c.mean(1), v.mean(1)]
        if self.use_hc:
            zs.append(self.ph(self.hc(x)))
        g = torch.softmax(self.gate(torch.cat(zs, 1)), 1) * len(zs)
        z = torch.cat([zi * g[:, i:i + 1] for i, zi in enumerate(zs)], 1)
        o = self.head(z); self.last_gate = g.detach()
        if aux:
            return o, self.aux_c(zs[0]), self.aux_v(zs[1])
        return (o, z) if emb else o

    def groups(self):
        bb = list(self.cnn.parameters()) + list(self.vit.parameters()); ids = {id(p) for p in bb}
        return bb, [p for p in self.parameters() if id(p) not in ids]


class Res1d(nn.Module):
    """Compact 1-D ResNet on the raw 12-lead signal: a signal-domain reference, not an image model."""
    def __init__(self, w=64):
        super().__init__()
        def blk(i, o, s):
            return nn.Sequential(nn.Conv1d(i, o, 7, s, 3, bias=False), nn.BatchNorm1d(o), nn.ReLU(True), nn.Conv1d(o, o, 7, 1, 3, bias=False), nn.BatchNorm1d(o))
        self.stem = nn.Sequential(nn.Conv1d(12, w, 15, 2, 7, bias=False), nn.BatchNorm1d(w), nn.ReLU(True))
        ch = [w, w, 2 * w, 2 * w, 4 * w, 4 * w, 8 * w]; self.b = nn.ModuleList(); self.s = nn.ModuleList()
        for i in range(6):
            st = 2 if i % 2 else 1
            self.b.append(blk(ch[i], ch[i + 1], st))
            self.s.append(nn.Sequential(nn.Conv1d(ch[i], ch[i + 1], 1, st, bias=False), nn.BatchNorm1d(ch[i + 1])))
        self.head = nn.Linear(ch[-1] * 2, 5)

    def forward(self, x, emb=False):
        x = self.stem(x)
        for b, s in zip(self.b, self.s):
            x = F.relu(b(x) + s(x))
        z = torch.cat([x.mean(2), x.amax(2)], 1); o = self.head(z)
        return (o, z) if emb else o

    def groups(self):
        return [], list(self.parameters())


RUNS = {  # run key -> (constructor, training-image style)
    "sig1d": (lambda: Res1d(), None),
    "resnet50": (lambda: Single("resnet50"), "rand"), "effnetv2s": (lambda: Single("effnetv2s"), "rand"),
    "convnext_t": (lambda: Single("convnext_t"), "rand"), "vit_s": (lambda: Single("vit_s"), "rand"),
    "tristream": (lambda: TriStream(), "rand"),
    "tristream_nohc": (lambda: TriStream(use_hc=False), "rand"),
    "tristream_noxattn": (lambda: TriStream(use_xattn=False), "rand"),
    "tristream_cleanaug": (lambda: TriStream(), "clean"),
}
MAIN = ["resnet50", "effnetv2s", "convnext_t", "vit_s", "tristream"]
ABLATIONS = ["tristream_nohc", "tristream_noxattn", "tristream_cleanaug"]


# ----------------------------------------------------------------------------- training / inference
@torch.no_grad()
def predict(model, dl):
    model.eval(); L, E = [], []
    for x, _ in dl:
        with torch.autocast(DEV, enabled=DEV == "cuda"):
            o, z = model(x.to(DEV, non_blocking=True), emb=True)
        L.append(o.float().cpu()); E.append(z.float().cpu())
    return torch.cat(L).numpy(), torch.cat(E).numpy().astype(np.float16)


def macro_auc(y, s):
    from sklearn.metrics import roc_auc_score
    return float(np.mean([roc_auc_score(y[:, k], s[:, k]) for k in range(y.shape[1]) if 0 < y[:, k].sum() < len(y)]))


def run_train(key, seed):
    out = W_("preds", f"{key}_s{seed}.npz")
    if out.exists():
        log("skip (done):", out.name); return
    torch.manual_seed(seed); np.random.seed(seed)
    X, Y, meta, idx = load_ptbxl(); make, style = RUNS[key]
    model = make().to(DEV); sig = key == "sig1d"
    if sig:
        mk = lambda part, **k: Sig(X, Y, idx[part], train=k.get("train", False))
    else:
        mk = lambda part, **k: ECGImages(X, Y, idx[part], style=style if k.get("train") else "clean", pert=k.get("pert"), seed=seed)
    dtr = loader(mk("train", train=True), True, 128 if sig else None); dva = loader(mk("val")); dte = loader(mk("test"))
    bb, hd = model.groups(); ep = CFG["epochs"] * (3 if sig else 1)
    opt = torch.optim.AdamW([dict(params=bb, lr=CFG["lr"]), dict(params=hd, lr=CFG["head_lr"])], weight_decay=CFG["wd"])
    steps = ep * len(dtr); wu = CFG["warmup"] * len(dtr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / wu if s < wu else 0.5 * (1 + math.cos(math.pi * (s - wu) / max(1, steps - wu))))
    scaler = torch.amp.GradScaler(DEV, enabled=DEV == "cuda"); best, state, hist = -1, None, []
    t0 = time.time()
    for e in range(ep):
        model.train(); tl = 0
        for x, y in dtr:
            x, y = x.to(DEV, non_blocking=True), y.to(DEV, non_blocking=True)
            with torch.autocast(DEV, enabled=DEV == "cuda"):
                if isinstance(model, TriStream):
                    o, a1, a2 = model(x, aux=True)
                    loss = F.binary_cross_entropy_with_logits(o, y) + 0.3 * (F.binary_cross_entropy_with_logits(a1, y) + F.binary_cross_entropy_with_logits(a2, y))
                else:
                    loss = F.binary_cross_entropy_with_logits(model(x), y)
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward()
            scaler.unscale_(opt); nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step(); tl += loss.item()
        va = macro_auc(Y[idx["val"]], predict(model, dva)[0]); hist.append([tl / len(dtr), va])
        log(f"{key} s{seed} ep{e + 1}/{ep} loss {tl / len(dtr):.4f} val macro-AUROC {va:.4f} ({(time.time() - t0) / 60:.1f} min)")
        if va > best:
            best = va; state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(state)
    res = dict(hist=np.array(hist), params=sum(p.numel() for p in model.parameters()), minutes=(time.time() - t0) / 60)
    res["val"], _ = predict(model, dva); res["test"], res["test_emb"] = predict(model, dte)
    if not sig:
        t1 = time.time(); predict(model, dte); res["ms_per_img"] = (time.time() - t1) / len(idx["test"]) * 1e3
        for p in PERTURBS:
            res["rob_" + p] = predict(model, loader(mk("test", pert=p)))[0]
        MX, _ = load_mend()
        if MX is not None:
            res["ext"], res["ext_emb"] = predict(model, loader(Arr(MX)))
        if seed == 0:
            if key == "convnext_t":
                res["train_emb"] = predict(model, loader(mk("train")))[1]; res["val_emb"] = predict(model, dva)[1]
            torch.save(state, W_("ckpt", f"{key}_s0.pt"))
    np.savez_compressed(out, **res); log("saved", out.name)


def stage_train_all(only=None):
    order = [("sig1d", s) for s in CFG["seeds"]] + [(k, s) for s in CFG["seeds"] for k in MAIN] + [(k, 0) for k in ABLATIONS]
    for k, s in order:
        if only is None or k in only:
            run_train(k, s)


# ----------------------------------------------------------------------------- classical ML
def stage_ml():
    if W_("preds", "ml_xgb_s0.npz").exists():
        log("skip ML (done)"); return
    from sklearn.decomposition import PCA
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    import xgboost as xgb
    X, Y, meta, idx = load_ptbxl(); hc = Handcrafted().to(DEV)

    def feats(ds):
        return np.concatenate([hc(x.to(DEV)).cpu().numpy() for x, _ in loader(ds, bs=64)])
    Ftr = np.concatenate([feats(ECGImages(X, Y, idx["train"], "clean")), feats(ECGImages(X, Y, idx["train"], "rand", seed=7))])
    Ytr = np.concatenate([Y[idx["train"]]] * 2)
    Fva, Fte = feats(ECGImages(X, Y, idx["val"])), feats(ECGImages(X, Y, idx["test"]))
    MX, _ = load_mend(); Fex = feats(Arr(MX)) if MX is not None else None
    rob = {p: feats(ECGImages(X, Y, idx["test"], pert=p)) for p in PERTURBS}
    np.savez_compressed(W_("hc_feats.npz"), test=Fte.astype(np.float16), **({"ext": Fex.astype(np.float16)} if Fex is not None else {}))
    log("handcrafted features", Ftr.shape)
    pre = make_pipeline(StandardScaler(), PCA(min(256, Ftr.shape[0] - 1, Ftr.shape[1]), random_state=0)).fit(Ftr)
    P = lambda A: pre.transform(A)
    Ptr = P(Ftr); nsv = min(len(Ptr), 12000); sub = np.random.default_rng(0).permutation(len(Ptr))[:nsv]
    models = dict(
        ml_lr=lambda: LogisticRegression(C=0.1, max_iter=2000),
        ml_svm=lambda: SVC(C=3.0, gamma="scale", kernel="rbf"),
        ml_rf=lambda: RandomForestClassifier(500, min_samples_leaf=2, n_jobs=-1, random_state=0),
        ml_xgb=lambda: xgb.XGBClassifier(n_estimators=600, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.5,
                                         tree_method="hist", device="cuda" if DEV == "cuda" else "cpu", random_state=0))
    for name, mk in models.items():
        t0 = time.time(); res = {k: [] for k in ["val", "test", "ext"] + ["rob_" + p for p in PERTURBS]}
        for k in range(5):
            m = mk()
            if name == "ml_svm":
                m.fit(Ptr[sub], Ytr[sub, k])
            else:
                m.fit(Ptr, Ytr[:, k])

            def sc(A):                                                     # scores on the logit scale
                if name == "ml_svm":
                    return m.decision_function(P(A))
                p = np.clip(m.predict_proba(P(A))[:, 1], 1e-6, 1 - 1e-6); return np.log(p / (1 - p))
            res["val"].append(sc(Fva)); res["test"].append(sc(Fte))
            if Fex is not None:
                res["ext"].append(sc(Fex))
            for p in PERTURBS:
                res["rob_" + p].append(sc(rob[p]))
        res = {k: np.stack(v, 1) for k, v in res.items() if v}
        np.savez_compressed(W_("preds", f"{name}_s0.npz"), minutes=(time.time() - t0) / 60, **res)
        log(name, "val macro-AUROC %.4f" % macro_auc(Y[idx["val"]], res["val"]))


def stage_hybrid():
    """Deep ConvNeXt embeddings + gradient-boosted trees (hybrid DL -> ML model)."""
    src, out = W_("preds", "convnext_t_s0.npz"), W_("preds", "hyb_xgb_s0.npz")
    if out.exists() or not src.exists():
        return
    import xgboost as xgb
    d = np.load(src); X, Y, meta, idx = load_ptbxl(); res = {"val": [], "test": [], "ext": []}
    for k in range(5):
        m = xgb.XGBClassifier(n_estimators=500, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.5, tree_method="hist",
                              device="cuda" if DEV == "cuda" else "cpu", random_state=0).fit(d["train_emb"].astype(np.float32), Y[idx["train"], k])
        for part, e in (("val", "val_emb"), ("test", "test_emb"), ("ext", "ext_emb")):
            if e in d:
                p = np.clip(m.predict_proba(d[e].astype(np.float32))[:, 1], 1e-6, 1 - 1e-6); res[part].append(np.log(p / (1 - p)))
    np.savez_compressed(out, **{k: np.stack(v, 1) for k, v in res.items() if v})


# ----------------------------------------------------------------------------- Grad-CAM
def panel_masks(h, w):
    """Boolean masks (on an h x w grid) for the anterior (V1-V4) and inferior (II, III, aVF) lead panels."""
    ant = np.zeros((h, w), bool); inf = np.zeros((h, w), bool)
    ys = [0] + [int(round((ROW_Y[i] + ROW_Y[i + 1]) / 2 * h)) for i in range(3)] + [h]
    xs = [int(round((X0 + (X1 - X0) * c / 4) * w)) for c in range(5)]
    for r in range(3):
        for c in range(4):
            m = ant if LAYOUT[r][c] in ("V1", "V2", "V3", "V4") else inf if LAYOUT[r][c] in ("II", "III", "aVF") else None
            if m is not None:
                m[ys[r]:ys[r + 1], xs[c]:xs[c + 1]] = True
    return ant, inf


def gradcam(model, x, cls=MI):
    model.eval(); x = x.to(DEV).requires_grad_(False)
    if isinstance(model, TriStream):
        o = model(x); f = model.fmap
    else:
        f = model.net.forward_features(x); o = model.head(model.net.forward_head(f, pre_logits=True))
    g = torch.autograd.grad(o[:, cls].sum(), f)[0]
    cam = F.relu((f * g.mean((2, 3), keepdim=True)).sum(1))
    return cam.detach().float().cpu().numpy(), torch.sigmoid(o[:, cls]).detach().cpu().numpy()


def stage_explain():
    out = W_("cam.npz")
    if out.exists():
        return
    X, Y, meta, idx = load_ptbxl(); te = idx["test"]; res = {}
    sel = te[(meta.mi_loc.values[te] != "")]
    for key in ("tristream", "convnext_t"):
        ck = W_("ckpt", f"{key}_s0.pt")
        if not ck.exists():
            continue
        m = RUNS[key][0]().to(DEV); m.load_state_dict(torch.load(ck, map_location=DEV)); cams = []
        for x, _ in loader(ECGImages(X, Y, sel), bs=16):
            cams.append(gradcam(m, x)[0])
        cams = np.concatenate(cams); ant, inf = panel_masks(*cams.shape[1:]); tot = cams.reshape(len(cams), -1).sum(1) + 1e-8
        res[key + "_ant"] = (cams * ant).reshape(len(cams), -1).sum(1) / tot; res[key + "_inf"] = (cams * inf).reshape(len(cams), -1).sum(1) / tot
        res[key + "_cams"] = cams[:24].astype(np.float32)
    res["idx"] = sel; res["loc"] = meta.mi_loc.values[sel].astype(str)
    np.savez_compressed(out, **res)


# ----------------------------------------------------------------------------- statistics
def sig_(z):
    return 1 / (1 + np.exp(-np.clip(z, -40, 40)))


def fit_T(z, y):
    from scipy.optimize import minimize_scalar
    f = lambda t: -np.mean(y * np.log(sig_(z / t) + 1e-9) + (1 - y) * np.log(1 - sig_(z / t) + 1e-9))
    return float(minimize_scalar(f, bounds=(0.05, 20), method="bounded").x)


def ece(p, y, bins=15):
    e = 0.0
    for a in np.linspace(0, 1, bins + 1)[:-1]:
        m = (p >= a) & (p < a + 1 / bins) if a + 1 / bins < 1 else (p >= a)
        if m.any():
            e += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(e)


def delong(y, a, b):
    """Two-sided DeLong test for two correlated AUROCs (fast midrank algorithm, Sun & Xu 2014)."""
    from scipy.stats import norm, rankdata
    pos, neg = y == 1, y == 0; m, n = pos.sum(), neg.sum(); V10, V01, A = [], [], []
    for s in (a, b):
        tx, ty, tz = rankdata(s[pos]), rankdata(s[neg]), rankdata(np.concatenate([s[pos], s[neg]]))
        A.append((tz[:m].sum() / m - (m + 1) / 2) / n); V10.append((tz[:m] - tx) / n); V01.append(1 - (tz[m:] - ty) / m)
    S = np.cov(np.array(V10)) / m + np.cov(np.array(V01)) / n
    var = S[0, 0] + S[1, 1] - 2 * S[0, 1]; zs = (A[0] - A[1]) / math.sqrt(max(var, 1e-12))
    return float(A[0] - A[1]), float(2 * norm.sf(abs(zs)))


def boot_idx(groups, n, seed=0):
    """Cluster (patient-level) bootstrap index sets."""
    r = np.random.default_rng(seed); u, inv = np.unique(groups, return_inverse=True)
    members = [np.where(inv == i)[0] for i in range(len(u))]
    return [np.concatenate([members[j] for j in r.integers(0, len(u), len(u))]) for _ in range(n)]


def ci(fn, B):
    v = []
    for b in B:
        try:
            v.append(fn(b))
        except ValueError:
            pass
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def bin_metrics(y, p, thr):
    from sklearn.metrics import average_precision_score, roc_auc_score
    h = p >= thr; tp, fp, fn, tn = (h & (y == 1)).sum(), (h & (y == 0)).sum(), (~h & (y == 1)).sum(), (~h & (y == 0)).sum()
    return dict(auc=float(roc_auc_score(y, p)), auprc=float(average_precision_score(y, p)), sens=float(tp / max(tp + fn, 1)), spec=float(tn / max(tn + fp, 1)),
                ppv=float(tp / max(tp + fp, 1)), npv=float(tn / max(tn + fn, 1)), f1=float(2 * tp / max(2 * tp + fp + fn, 1)), brier=float(np.mean((p - y) ** 2)))


def youden(y, p):
    from sklearn.metrics import roc_curve
    f, t, th = roc_curve(y, p); return float(th[np.argmax(t - f)])


def stage_analyse():
    from sklearn.metrics import f1_score, roc_auc_score, roc_curve
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight"})
    stage_hybrid()
    X, Y, meta, idx = load_ptbxl(); te, va = idx["test"], idx["val"]; yt, yv = Y[te], Y[va]
    MX, my = load_mend(); B = boot_idx(meta.patient_id.values[te], CFG["n_boot"])
    R = dict(config={k: v for k, v in CFG.items() if k not in ("work", "raw")}, classes=CLASSES, models={},
             data=dict(n_records=len(meta), n_patients=int(meta.patient_id.nunique()),
                       **{p: dict(n=len(i), patients=int(meta.patient_id.iloc[i].nunique()), prevalence={c: float(Y[i, k].mean()) for k, c in enumerate(CLASSES)}) for p, i in idx.items()},
                       patient_overlap_train_test=int(len(set(meta.patient_id.iloc[idx["train"]]) & set(meta.patient_id.iloc[te]))),
                       mi_loc_test={k: int((meta.mi_loc.values[te] == k).sum()) for k in ("ant", "inf")},
                       external=None if my is None else dict(n=len(my), Normal=int((my == 0).sum()), MI=int((my == 1).sum()), HB=int((my == 2).sum()), PMI=int((my == 3).sum()))))
    names = dict(ml_lr="HC + Logistic regression", ml_svm="HC + SVM (RBF)", ml_rf="HC + Random forest", ml_xgb="HC + XGBoost", hyb_xgb="ConvNeXt features + XGBoost",
                 resnet50="ResNet-50", effnetv2s="EfficientNetV2-S", convnext_t="ConvNeXt-T", vit_s="ViT-S/16", tristream="TriStream-ECG (proposed)",
                 tristream_nohc="w/o handcrafted stream", tristream_noxattn="w/o cross-attention", tristream_cleanaug="w/o printout randomisation", sig1d="1-D ResNet (raw signal)")
    ens = {}
    for key, nm in names.items():
        runs = [np.load(f) for f in sorted(Path(CFG["work"], "preds").glob(key + "_s[0-9].npz"))]
        if not runs:
            continue
        pv = np.mean([sig_(r["val"]) for r in runs], 0); pt = np.mean([sig_(r["test"]) for r in runs], 0)
        zt, zv = np.log(pt / (1 - pt + 1e-9) + 1e-9), np.log(pv / (1 - pv + 1e-9) + 1e-9)
        T = fit_T(zv[:, MI], yv[:, MI]); thr = youden(yv[:, MI], pv[:, MI]); pc = sig_(zt[:, MI] / T)
        fthr = [max(np.linspace(0.05, 0.95, 37), key=lambda t: f1_score(yv[:, k], pv[:, k] >= t)) for k in range(5)]
        M = dict(name=nm, n_seeds=len(runs),
                 seed_macro_auc=[macro_auc(yt, r["test"]) for r in runs], seed_mi_auc=[float(roc_auc_score(yt[:, MI], r["test"][:, MI])) for r in runs],
                 macro_auc=macro_auc(yt, pt), macro_auc_ci=ci(lambda b: macro_auc(yt[b], pt[b]), B),
                 class_auc={c: float(roc_auc_score(yt[:, k], pt[:, k])) for k, c in enumerate(CLASSES)},
                 macro_f1=float(np.mean([f1_score(yt[:, k], pt[:, k] >= fthr[k]) for k in range(5)])),
                 mi=bin_metrics(yt[:, MI], pt[:, MI], thr), mi_auc_ci=ci(lambda b: roc_auc_score(yt[b, MI], pt[b, MI]), B),
                 mi_threshold=thr, mi_T=T, mi_ece_raw=ece(pt[:, MI], yt[:, MI]), mi_ece_cal=ece(pc, yt[:, MI]), mi_brier_cal=float(np.mean((pc - yt[:, MI]) ** 2)))
        for f in ("params", "minutes", "ms_per_img"):
            if f in runs[0]:
                M[f] = float(np.mean([r[f] for r in runs]))
        if "rob_rot5" in runs[0]:
            M["robust"] = {p: dict(macro_auc=macro_auc(yt, np.mean([sig_(r["rob_" + p]) for r in runs], 0)),
                                   mi_auc=float(roc_auc_score(yt[:, MI], np.mean([sig_(r["rob_" + p][:, MI]) for r in runs], 0)))) for p in PERTURBS}
        if my is not None and "ext" in runs[0]:
            pe = np.mean([sig_(r["ext"]) for r in runs], 0); E = {}
            for tag, posm, score in (("MIorPMI_vs_Normal", (my == 1) | (my == 3), pe[:, MI]), ("MI_vs_Normal", my == 1, pe[:, MI]), ("PMI_vs_Normal", my == 3, pe[:, MI]),
                                     ("Abnormal_vs_Normal", my != 0, 1 - pe[:, 0])):
                m_ = posm | (my == 0); yy = posm[m_].astype(int); ss = score[m_]; BE = boot_idx(np.arange(len(yy)), CFG["n_boot"], 1)
                E[tag] = dict(auc=float(roc_auc_score(yy, ss)), ci=ci(lambda b: roc_auc_score(yy[b], ss[b]), BE), n=int(m_.sum()),
                              seed_auc=[float(roc_auc_score(yy, sig_(r["ext"])[m_][:, MI] if "MI" in tag else 1 - sig_(r["ext"])[m_][:, 0])) for r in runs],
                              **{k: v for k, v in bin_metrics(yy, ss, thr if "MI" in tag else 0.5).items() if k in ("sens", "spec")})
            M["external"] = E
        R["models"][key] = M; ens[key] = dict(pt=pt, pc=pc, thr=thr, runs=runs)
        log(f"{nm:34s} macro-AUROC {M['macro_auc']:.4f}  MI-AUROC {M['mi']['auc']:.4f}")
    # ---- paired comparisons of the proposed model against every other model
    if "tristream" in ens:
        a = ens["tristream"]["pt"]; R["comparisons"] = {}
        for key in ens:
            if key != "tristream":
                b = ens[key]["pt"]; d = [macro_auc(yt[i], a[i]) - macro_auc(yt[i], b[i]) for i in B[:max(200, CFG["n_boot"] // 2)]]
                dl = delong(yt[:, MI], a[:, MI], b[:, MI])
                R["comparisons"][key] = dict(d_macro_auc=float(macro_auc(yt, a) - macro_auc(yt, b)), d_macro_ci=[float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                                             p_boot=float(min(1.0, 2 * min(np.mean(np.array(d) <= 0), np.mean(np.array(d) >= 0)) + 1 / len(d))), d_mi_auc=dl[0], p_delong_mi=dl[1])
    best = "tristream" if "tristream" in ens else max(ens, key=lambda k: R["models"][k]["macro_auc"]); pb = ens[best]
    # ---- subgroups (MI detection, best model)
    sub = {}; age = pd.to_numeric(meta.age, errors="coerce").values[te]; sex = pd.to_numeric(meta.sex, errors="coerce").values[te]
    for tag, m_ in (("male", sex == 0), ("female", sex == 1), ("age<50", age < 50), ("age50-69", (age >= 50) & (age < 70)), ("age>=70", age >= 70)):
        if m_.sum() > 20 and 0 < yt[m_, MI].sum() < m_.sum():
            Bs = boot_idx(meta.patient_id.values[te][m_], CFG["n_boot"], 2); ys, ps = yt[m_, MI], pb["pt"][m_, MI]
            sub[tag] = dict(n=int(m_.sum()), n_mi=int(ys.sum()), auc=float(roc_auc_score(ys, ps)), ci=ci(lambda b: roc_auc_score(ys[b], ps[b]), Bs), **{k: v for k, v in bin_metrics(ys, ps, pb["thr"]).items() if k in ("sens", "spec")})
    R["subgroups"] = sub; R["best_model"] = best
    # ---- decision-curve analysis (calibrated MI probability)
    ths = np.linspace(0.02, 0.6, 30); prev = yt[:, MI].mean()
    nb = lambda p: [float(((p >= t) & (yt[:, MI] == 1)).mean() - ((p >= t) & (yt[:, MI] == 0)).mean() * t / (1 - t)) for t in ths]
    R["dca"] = dict(thresholds=ths.tolist(), treat_all=[float(prev - (1 - prev) * t / (1 - t)) for t in ths], **{k: nb(ens[k]["pc"]) for k in ens if k in MAIN + ["ml_xgb"]})
    # ---- external adaptation: logistic probe on frozen embeddings, duplicate-aware 5-fold CV
    if my is not None:
        g = np.stack([cv2.resize(cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2GRAY), (17, 16), interpolation=cv2.INTER_AREA) for im in MX])
        hsh = (g[:, :, 1:] > g[:, :, :-1]).reshape(len(g), -1); D = (hsh[:, None] != hsh[None]).sum(2); par = np.arange(len(g))
        def find(i):
            while par[i] != i:
                par[i] = par[par[i]]; i = par[i]
            return i
        for i, j in zip(*np.where(np.triu(D <= 6, 1))):
            par[find(i)] = find(j)
        grp = np.array([find(i) for i in range(len(g))]); R["external_probe"] = dict(near_duplicate_pairs=int(np.triu(D <= 6, 1).sum()), n_groups=int(len(set(grp))))
        feats = {k: ens[k]["runs"][0]["ext_emb"].astype(np.float32) for k in ens if "ext_emb" in ens[k]["runs"][0]}
        if W_("hc_feats.npz").exists() and "ext" in np.load(W_("hc_feats.npz")):
            feats["handcrafted"] = np.load(W_("hc_feats.npz"))["ext"].astype(np.float32)
        for k, Fm in feats.items():
            acc, f1s = [], []
            for rep in range(3):
                pr = np.zeros(len(my), int)
                for tr, ts in StratifiedGroupKFold(5, shuffle=True, random_state=rep).split(Fm, my, grp):
                    pr[ts] = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=3000)).fit(Fm[tr], my[tr]).predict(Fm[ts])
                acc.append(float((pr == my).mean())); f1s.append(float(f1_score(my, pr, average="macro")))
            R["external_probe"][k] = dict(acc=float(np.mean(acc)), acc_sd=float(np.std(acc)), macro_f1=float(np.mean(f1s)))
    # ---- Grad-CAM localisation
    if W_("cam.npz").exists():
        from scipy.stats import mannwhitneyu
        c = np.load(W_("cam.npz")); loc = c["loc"]; R["cam"] = {}
        for key in ("tristream", "convnext_t"):
            if key + "_ant" in c:
                d = c[key + "_ant"] - c[key + "_inf"]; a_, i_ = d[loc == "ant"], d[loc == "inf"]
                R["cam"][key] = dict(n_ant=int(len(a_)), n_inf=int(len(i_)), ant_mass_in_ant=float(c[key + "_ant"][loc == "ant"].mean()), inf_mass_in_ant=float(c[key + "_inf"][loc == "ant"].mean()),
                                     ant_mass_in_inf=float(c[key + "_ant"][loc == "inf"].mean()), inf_mass_in_inf=float(c[key + "_inf"][loc == "inf"].mean()),
                                     p_mwu=float(mannwhitneyu(a_, i_).pvalue) if len(a_) and len(i_) else None, concordance=float(((d > 0) == (loc == "ant")).mean()))
    json.dump(R, open(W_("results", "results.json"), "w"), indent=1)
    # ------------------------------------------------------------------ figures
    fig = lambda n: str(W_("results", n)); col = dict(zip(MAIN + ["ml_xgb", "sig1d"], ["#4C72B0", "#55A868", "#DD8452", "#8172B3", "#C44E52", "#937860", "#7F7F7F"]))
    r0 = np.random.default_rng(3); f, ax = plt.subplots(1, 3 if MX is not None else 2, figsize=(11, 2.6)); s0 = X[te[int(np.argmax(yt[:, MI]))]]
    for a, im, t in zip(ax, [render(s0), render(s0, r0, "rand")] + ([np.asarray(MX[int(np.argmax(my == 1))])] if MX is not None else []), ["(a) PTB-XL, clean rendering", "(b) PTB-XL, randomised printout", "(c) Real printout (external set)"]):
        a.imshow(im); a.set_title(t, fontsize=9); a.axis("off")
    plt.savefig(fig("fig_samples.png")); plt.close()
    f, ax = plt.subplots(1, 3, figsize=(11, 3.4))
    for k in col:
        if k in ens:
            fp, tp, _ = roc_curve(yt[:, MI], ens[k]["pt"][:, MI]); ax[0].plot(fp, tp, color=col[k], lw=1.4, label="%s (%.3f)" % (names[k].replace(" (proposed)", ""), R["models"][k]["mi"]["auc"]))
            if k in R["dca"]:
                ax[2].plot(ths, R["dca"][k], color=col[k], lw=1.4, label=names[k].replace(" (proposed)", ""))
    ax[0].plot([0, 1], [0, 1], "k:", lw=0.8); ax[0].set(xlabel="1 - specificity", ylabel="Sensitivity", title="(a) MI detection, ROC (AUROC)"); ax[0].legend(fontsize=6.5, loc="lower right")
    for lab, p in (("uncalibrated", pb["pt"][:, MI]), ("temperature-scaled", pb["pc"])):
        q = np.quantile(p, np.linspace(0, 1, 11)); q[-1] += 1e-9; bi = np.digitize(p, q[1:-1])
        ax[1].plot([p[bi == i].mean() for i in range(10) if (bi == i).any()], [yt[bi == i, MI].mean() for i in range(10) if (bi == i).any()], "o-", ms=3, lw=1.2, label=lab)
    ax[1].plot([0, 1], [0, 1], "k:", lw=0.8); ax[1].set(xlabel="Predicted MI probability", ylabel="Observed MI frequency", title="(b) Calibration, " + names[best].replace(" (proposed)", "")); ax[1].legend(fontsize=7)
    ax[2].plot(ths, R["dca"]["treat_all"], "k--", lw=0.8, label="Flag all"); ax[2].axhline(0, color="k", lw=0.8, ls=":", label="Flag none")
    ax[2].set(xlabel="Threshold probability", ylabel="Net benefit", title="(c) Decision curve", ylim=(-0.02, max(0.05, prev * 1.05))); ax[2].legend(fontsize=6.5)
    plt.tight_layout(); plt.savefig(fig("fig_roc_cal_dca.png")); plt.close()
    rk = [k for k in ["ml_xgb"] + MAIN if k in R["models"] and "robust" in R["models"][k]]
    if rk:
        f, ax = plt.subplots(figsize=(8, 3)); wd = 0.8 / len(rk); cond = ["clean"] + PERTURBS
        for i, k in enumerate(rk):
            ax.bar(np.arange(len(cond)) + i * wd, [R["models"][k]["macro_auc"]] + [R["models"][k]["robust"][p]["macro_auc"] for p in PERTURBS], wd, color=col[k], label=names[k].replace(" (proposed)", ""))
        ax.set_xticks(np.arange(len(cond)) + 0.4 - wd / 2); ax.set_xticklabels(["Clean", "Rotation 5 deg", "Blur", "JPEG q20", "Noise", "No grid", "Low contrast", "Half resolution"], rotation=20, fontsize=8)
        ax.set(ylabel="Macro-AUROC", ylim=(0.5, 1.0)); ax.legend(fontsize=6.5, ncol=3); plt.savefig(fig("fig_robust.png")); plt.close()
    if sub or my is not None:
        f, ax = plt.subplots(1, 2, figsize=(10, 3))
        for i, (t, s) in enumerate(sub.items()):
            ax[0].errorbar(s["auc"], i, xerr=[[s["auc"] - s["ci"][0]], [s["ci"][1] - s["auc"]]], fmt="o", color="#C44E52", capsize=3)
        ax[0].set_yticks(range(len(sub))); ax[0].set_yticklabels(["%s (n=%d)" % (t, s["n"]) for t, s in sub.items()]); ax[0].set(xlabel="MI AUROC (95% CI)", title="(a) Subgroups, PTB-XL test fold")
        ek = [k for k in ["ml_xgb"] + MAIN + ["tristream_cleanaug"] if k in R["models"] and "external" in R["models"][k]]
        for i, k in enumerate(ek):
            e = R["models"][k]["external"]["MIorPMI_vs_Normal"]; ax[1].errorbar(e["auc"], i, xerr=[[e["auc"] - e["ci"][0]], [e["ci"][1] - e["auc"]]], fmt="s", color=col.get(k, "#333333"), capsize=3)
        ax[1].axvline(0.5, color="k", ls=":", lw=0.8); ax[1].set_yticks(range(len(ek))); ax[1].set_yticklabels([names[k].replace(" (proposed)", "") for k in ek], fontsize=7)
        ax[1].set(xlabel="AUROC (95% CI), infarction vs normal", title="(b) External real printouts, zero-shot"); plt.tight_layout(); plt.savefig(fig("fig_subgroup_external.png")); plt.close()
    if W_("cam.npz").exists():
        c = np.load(W_("cam.npz")); k = "tristream_cams" if "tristream_cams" in c else "convnext_t_cams" if "convnext_t_cams" in c else None
        if k:
            pick = [int(np.where(c["loc"][:24] == l)[0][j]) for l in ("ant", "inf") for j in range(2) if (c["loc"][:24] == l).sum() > j]
            f, ax = plt.subplots(1, len(pick), figsize=(3.6 * len(pick), 2.4), squeeze=False)
            for a, j in zip(ax[0], pick):
                im = render(X[c["idx"][j]]); cm_ = cv2.resize(c[k][j], (CFG["W"], CFG["H"])); a.imshow(im); a.imshow(cm_ / (cm_.max() + 1e-8), cmap="jet", alpha=0.35)
                a.set_title({"ant": "Anterior MI", "inf": "Inferior MI"}[str(c["loc"][j])], fontsize=9); a.axis("off")
            plt.savefig(fig("fig_gradcam.png")); plt.close()
    with zipfile.ZipFile(W_("HEART_RESULTS.zip"), "w", zipfile.ZIP_DEFLATED) as z:
        for p in Path(CFG["work"], "results").glob("*"):
            z.write(p, p.name)
    print("=====RESULTS-BEGIN====="); print(json.dumps(R)); print("=====RESULTS-END=====")
    log("wrote", W_("HEART_RESULTS.zip"))


# ----------------------------------------------------------------------------- entry point
def main():
    global DEV
    ap = argparse.ArgumentParser(); ap.add_argument("stage"); ap.add_argument("--work"); ap.add_argument("--raw"); ap.add_argument("--epochs", type=int)
    ap.add_argument("--bs", type=int); ap.add_argument("--seeds", type=int, nargs="+"); ap.add_argument("--only", nargs="+"); ap.add_argument("--workers", type=int)
    a = ap.parse_args()
    for k in ("work", "raw", "epochs", "bs", "seeds", "workers"):
        if getattr(a, k) is not None:
            CFG[k] = getattr(a, k)
    if a.stage == "smoke":
        CFG.update(smoke=True, pretrained=False, H=128, W=224, epochs=1, bs=8, seeds=[0], n_boot=40, workers=0)
        if a.work is None:
            CFG["work"] = "/tmp/heartecg_smoke"
        synth_data(); stage_ml(); stage_train_all(a.only); stage_explain(); stage_analyse(); log("SMOKE TEST PASSED"); return
    log("device:", DEV, torch.cuda.get_device_name(0) if DEV == "cuda" else "")
    dict(data=stage_data, ml=stage_ml, train_all=lambda: stage_train_all(a.only), explain=stage_explain, analyse=stage_analyse)[a.stage]()


if __name__ == "__main__":
    main()
