import argparse
import copy
import hashlib
import io
import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from report_mode_policy import build_report_mode_annotation

EPS = 1e-8
CMP_TOKEN_RE = re.compile(r"^\s*[<>≤≥＜＞]\s*[-+]?\d+(?:\.\d+)?\s*$")
STRICT_NUM_RE = re.compile(r"^\s*[-+]?\d+(?:\.\d+)?\s*$")
CN_TEXT_RE = re.compile(r"[\u4e00-\u9fff]")
CMP_PARSE_RE = re.compile(r"^\s*([<>≤≥＜＞])\s*([-+]?\d+(?:\.\d+)?)\s*$")
CMP_MMAX_RE = re.compile(r"^\s*[>≥＞]\s*Mmax\s*$", flags=re.IGNORECASE)

CLINICAL_PENALTY_RULES = [
    (
        re.compile(
            r"小圆细胞|电导率|导电率|结晶|管型|上皮细胞|鳞状上皮|黏液丝|粘液丝|细菌|镜检|HPF|尿比重|酸碱度|尿",
            flags=re.IGNORECASE,
        ),
        0.35,
        "尿沉渣或形态学代理变量",
    ),
    (re.compile(r"颜色|透明度|浊度|比重", flags=re.IGNORECASE), 0.50, "非特异体液外观指标"),
    (re.compile(r"编号|序号|id|住院号|病案号", flags=re.IGNORECASE), 0.10, "标识符噪声变量"),
    (re.compile(r"备注|说明|文本|描述|结果描述", flags=re.IGNORECASE), 0.50, "自由文本代理变量"),
]

CLINICAL_EVIDENCE_LIBRARY = [
    {
        "pattern": re.compile(r"肌钙蛋白|troponin|ctni|ctnt", flags=re.IGNORECASE),
        "mechanism": "反映心肌细胞损伤，升高通常提示急性心肌缺血/坏死负荷增加。",
        "treatment": "结合症状与心电图，可优先进入ACS路径，强化抗栓、他汀及血运重建评估。",
        "prognosis": "持续升高或动态上升与更高短中期不良心血管事件风险相关。",
        "untreated": "可能延误再灌注或二级预防，增加再梗死、心衰和死亡风险。",
        "references": [
            "ESC ACS Guideline 2023 (Eur Heart J. 2023;44:3720-3826)",
            "Fourth Universal Definition of MI (Circulation. 2018;138:e618-e651)",
        ],
    },
    {
        "pattern": re.compile(
            r"bnp|nt[-_ ]?pro[-_ ]?bnp|脑钠肽|利钠肽|b型脑钠肽|脑钠肽前体|末端b型脑钠肽前体",
            flags=re.IGNORECASE,
        ),
        "mechanism": "提示心室壁张力与容量/压力负荷增加，反映心衰或亚临床心功能受损。",
        "treatment": "提示需强化容量管理、神经内分泌阻滞与心功能随访。",
        "prognosis": "升高与心衰住院、心源性死亡风险上升相关。",
        "untreated": "可能错失心衰早期干预窗口，导致反复失代偿。",
        "references": [
            "ESC HF Guideline 2021 (Eur Heart J. 2021;42:3599-3726)",
            "Januzzi et al., Natriuretic Peptides in HF (JACC. 2019)",
        ],
    },
    {
        "pattern": re.compile(r"ldl|ldl-c|总胆固醇|胆固醇|apo\s*b|载脂蛋白b", flags=re.IGNORECASE),
        "mechanism": "反映动脉粥样硬化脂质负荷，驱动斑块形成与进展。",
        "treatment": "支持强化降脂（高强度他汀±依折麦布/PCSK9抑制剂）并设定更低靶值。",
        "prognosis": "长期控制达标可显著降低MACE；持续偏高提示残余风险。",
        "untreated": "斑块进展和事件复发风险增加。",
        "references": [
            "ESC Dyslipidaemia Guideline 2019 (Eur Heart J. 2020;41:111-188)",
            "Sabatine et al., Evolocumab (NEJM. 2017;376:1713-1722)",
        ],
    },
    {
        "pattern": re.compile(r"hdl|hdl-c|载脂蛋白a1|apo\s*a1", flags=re.IGNORECASE),
        "mechanism": "低HDL/apoA1常提示逆向胆固醇转运能力不足和代谢风险增加。",
        "treatment": "重点仍为LDL主导降脂和生活方式干预，HDL更多用于风险分层。",
        "prognosis": "持续偏低常与长期动脉粥样硬化风险相关。",
        "untreated": "综合心血管危险度维持高位。",
        "references": [
            "ESC Prevention Guideline 2021 (Eur Heart J. 2021;42:3227-3337)",
        ],
    },
    {
        "pattern": re.compile(r"crp|hs-?crp|超敏c反应蛋白", flags=re.IGNORECASE),
        "mechanism": "提示炎症激活，与斑块不稳定和事件风险升高相关。",
        "treatment": "可辅助识别高炎症残余风险，支持强化二级预防与危险因素控制。",
        "prognosis": "持续升高与再发事件风险增高相关。",
        "untreated": "炎症残余风险持续，事件复发概率上升。",
        "references": [
            "Ridker et al., JUPITER (NEJM. 2008;359:2195-2207)",
            "Ridker et al., CANTOS (NEJM. 2017;377:1119-1131)",
        ],
    },
    {
        "pattern": re.compile(
            r"d[- ]?dimer|d二聚体|二聚体|纤维蛋白原|fibrinogen|fdp|纤维蛋白\(原\)降解产物",
            flags=re.IGNORECASE,
        ),
        "mechanism": "反映凝血-纤溶系统激活，提示血栓负荷和炎症-凝血耦联增强。",
        "treatment": "提示应关注抗栓策略、血栓并发症筛查与动态复测。",
        "prognosis": "升高常关联更高短期事件风险。",
        "untreated": "血栓相关并发症风险持续增加。",
        "references": [
            "AHA/ACC Chest Pain Guideline 2021",
            "Righini et al., Diagnostic pathways for thrombosis (BMJ. 2022)",
        ],
    },
    {
        "pattern": re.compile(r"肌酐|creatinine|egfr|尿素氮|bun", flags=re.IGNORECASE),
        "mechanism": "肾功能受损与内皮功能障碍、炎症和药代动力学改变相关，放大心血管风险。",
        "treatment": "需进行肾功能分层并调整抗栓/造影/RAAS相关治疗策略。",
        "prognosis": "肾功能异常与死亡及心衰住院风险上升相关。",
        "untreated": "可加速心肾综合征进展并增加不良事件。",
        "references": [
            "KDIGO CKD Guideline 2024",
            "ESC Prevention Guideline 2021",
        ],
    },
    {
        "pattern": re.compile(r"葡萄糖|血糖|glucose|hba1c|糖化", flags=re.IGNORECASE),
        "mechanism": "高血糖/糖化负荷可促进内皮损伤、炎症和斑块进展。",
        "treatment": "提示强化代谢控制，必要时联合具有心血管获益证据的降糖策略。",
        "prognosis": "控制不良与长期MACE风险增加相关。",
        "untreated": "动脉粥样硬化和微血管并发症进展加快。",
        "references": [
            "ADA Standards of Care 2025",
            "ESC Diabetes/CVD Guideline 2023",
        ],
    },
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def parse_csv_arg(s: str) -> list[str]:
    parts = [p.strip() for p in str(s).split(",")]
    return [p for p in parts if p]


def parse_float_csv_arg(s: str) -> list[float]:
    out: list[float] = []
    for p in parse_csv_arg(s):
        try:
            out.append(float(p))
        except Exception:
            continue
    return out


def safe_float(x, default: float = 0.0) -> float:
    try:
        if x is None:
            return float(default)
        if isinstance(x, (float, int, np.floating, np.integer)):
            return float(x)
        if isinstance(x, str) and x.strip().lower() in {"nan", "none", ""}:
            return float(default)
        return float(x)
    except Exception:
        return float(default)


def apply_lightweight_profile(args) -> dict:
    before = {
        "mlp_hidden": int(args.mlp_hidden),
        "tr_d_model": int(args.tr_d_model),
        "tr_layers": int(args.tr_layers),
        "tr_ffn": int(args.tr_ffn),
        "feature_select_max_features": int(args.feature_select_max_features),
        "feature_select_min_features": int(args.feature_select_min_features),
    }

    profile = str(args.lightweight_profile)
    if profile == "balanced":
        args.mlp_hidden = min(int(args.mlp_hidden), 256)
        args.tr_d_model = min(int(args.tr_d_model), 64)
        args.tr_layers = min(int(args.tr_layers), 2)
        args.tr_ffn = min(int(args.tr_ffn), 128)
        args.feature_select_max_features = min(int(args.feature_select_max_features), 96)
        args.feature_select_min_features = min(int(args.feature_select_min_features), 32)
    elif profile == "aggressive":
        args.mlp_hidden = min(int(args.mlp_hidden), 192)
        args.tr_d_model = min(int(args.tr_d_model), 48)
        args.tr_layers = min(int(args.tr_layers), 2)
        args.tr_ffn = min(int(args.tr_ffn), 96)
        args.feature_select_max_features = min(int(args.feature_select_max_features), 72)
        args.feature_select_min_features = min(int(args.feature_select_min_features), 24)

    # Keep transformer heads divisible after profile override.
    if int(args.tr_nhead) > 0 and int(args.tr_d_model) % int(args.tr_nhead) != 0:
        args.tr_d_model = int(math.ceil(float(args.tr_d_model) / float(args.tr_nhead)) * int(args.tr_nhead))

    after = {
        "mlp_hidden": int(args.mlp_hidden),
        "tr_d_model": int(args.tr_d_model),
        "tr_layers": int(args.tr_layers),
        "tr_ffn": int(args.tr_ffn),
        "feature_select_max_features": int(args.feature_select_max_features),
        "feature_select_min_features": int(args.feature_select_min_features),
    }
    changed = any(before[k] != after[k] for k in before)
    return {
        "profile": profile,
        "changed": bool(changed),
        "before": before,
        "after": after,
    }


def model_param_count(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def model_state_dict_size_bytes(model: nn.Module) -> int:
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return int(len(buffer.getvalue()))


def quantize_dynamic_linear(model: nn.Module) -> nn.Module:
    try:
        from torch.ao.quantization import quantize_dynamic
    except Exception:
        from torch.quantization import quantize_dynamic
    return quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)


def refresh_outputs_dynamic_index(out_dir: Path, prefix: str) -> str:
    try:
        from discover_outputs import discover, to_markdown
    except Exception:
        try:
            from examples.tabular.project_multidisease_rigorous.discover_outputs import discover, to_markdown
        except Exception as ex:
            raise RuntimeError(f"cannot import discover_outputs.py: {ex}") from ex

    resolved_out = out_dir.resolve()
    base_dir = resolved_out.parent
    for candidate in [resolved_out, *resolved_out.parents]:
        if candidate.name.startswith(prefix):
            base_dir = candidate.parent
            break
    rows = discover(base_dir=base_dir, prefix=prefix)
    index_path = base_dir / "OUTPUTS_DYNAMIC_INDEX.md"
    index_path.write_text(to_markdown(rows), encoding="utf-8")
    return str(index_path)


def load_selected_sheets(excel_path: Path, sheet_names: list[str]) -> pd.DataFrame:
    xls = pd.ExcelFile(excel_path)
    actual = set(xls.sheet_names)
    missing = [s for s in sheet_names if s not in actual]
    if missing:
        raise ValueError(f"Sheets not found in Excel: {missing}. Existing: {list(xls.sheet_names)}")

    frames: list[pd.DataFrame] = []
    for s in sheet_names:
        df = pd.read_excel(excel_path, sheet_name=s)
        df = df.copy()
        df["__label__"] = s
        frames.append(df)
    return pd.concat(frames, axis=0, ignore_index=True)


def load_single_sheet_with_label_column(
    excel_path: Path,
    data_sheet: str,
    label_column: str,
    label_values: list[str],
) -> tuple[pd.DataFrame, list[str], str]:
    xls = pd.ExcelFile(excel_path)
    if data_sheet:
        if data_sheet not in set(xls.sheet_names):
            raise ValueError(f"Sheet not found: {data_sheet}. Existing: {list(xls.sheet_names)}")
        target_sheet = data_sheet
    else:
        if not xls.sheet_names:
            raise ValueError("Excel has no sheets")
        target_sheet = str(xls.sheet_names[0])

    df = pd.read_excel(excel_path, sheet_name=target_sheet).copy()
    if label_column not in df.columns:
        raise ValueError(f"label_column not found: {label_column}. Existing columns count={len(df.columns)}")

    labels_series = df[label_column].map(lambda x: "" if pd.isna(x) else str(x).strip())
    non_empty_mask = labels_series != ""
    if not bool(non_empty_mask.any()):
        raise ValueError(f"label_column={label_column} has no non-empty labels")

    df = df.loc[non_empty_mask].copy()
    labels_series = labels_series.loc[non_empty_mask]

    requested_labels = list(dict.fromkeys([s.strip() for s in label_values if str(s).strip()]))
    if requested_labels:
        keep_mask = labels_series.isin(requested_labels)
        df = df.loc[keep_mask].copy()
        labels_series = labels_series.loc[keep_mask]
        if df.empty:
            raise ValueError(
                f"No samples left after label_values filter: {requested_labels}. "
                f"Observed labels: {list(dict.fromkeys(labels_series.tolist()))}"
            )
        observed = set(labels_series.unique().tolist())
        missing_requested = [x for x in requested_labels if x not in observed]
        if missing_requested:
            raise ValueError(
                f"Requested labels not found in filtered data: {missing_requested}. "
                f"Observed labels: {sorted(observed)}"
            )
        label_names = requested_labels
    else:
        label_names = list(dict.fromkeys(labels_series.tolist()))

    if len(label_names) < 2:
        raise ValueError(f"Need at least two classes, got: {label_names}")

    df["__label__"] = labels_series.to_numpy()
    df = df.drop(columns=[label_column], errors="ignore")
    return df, label_names, target_sheet


def drop_columns_by_keywords(df: pd.DataFrame, keywords: list[str]) -> tuple[pd.DataFrame, list[str]]:
    if not keywords:
        return df, []

    normalized = [k.lower() for k in keywords if k]
    to_drop: list[str] = []
    for col in df.columns:
        c = str(col)
        c_low = c.lower()
        if any(k in c_low for k in normalized):
            to_drop.append(col)

    out = df.drop(columns=to_drop, errors="ignore")
    return out, sorted(set(to_drop))


def drop_leaky_or_useless_columns(
    df: pd.DataFrame,
    leakage_guard_mode: str = "strict",
) -> tuple[pd.DataFrame, list[str]]:
    df = df.copy()

    mode = str(leakage_guard_mode).strip().lower()
    if mode not in {"strict", "balanced", "off"}:
        raise ValueError(f"Unsupported leakage_guard_mode={leakage_guard_mode}")

    strict_keywords = ["诊断", "病种", "疾病", "label", "target", "outcome", "diagnosis", "disease"]
    balanced_exact = {
        "label",
        "target",
        "y",
        "class",
        "类别",
        "标签",
        "疾病标签",
        "病种标签",
        "出院诊断",
        "最终诊断",
        "诊断结论",
        "诊断结果",
        "临床诊断",
        "真实标签",
        "金标准",
        "final_diagnosis",
        "discharge_diagnosis",
        "ground_truth",
        "gold_standard",
        "y_true",
        "y_pred",
        "是否患病",
        "是否心脏病",
    }
    balanced_substrings = [
        "出院诊断",
        "最终诊断",
        "诊断结论",
        "诊断结果",
        "临床诊断",
        "真实标签",
        "金标准",
        "ground_truth",
        "gold_standard",
    ]
    balanced_exact_compact = {re.sub(r"\s+", "", x.lower()) for x in balanced_exact}

    pii_keywords = ["姓名", "住院", "病案", "病历", "身份证", "电话", "手机号", "地址"]

    drop_leak: list[str] = []
    if mode == "strict":
        for c in df.columns:
            c_str = str(c).strip()
            c_low = c_str.lower()
            if any(k in c_low for k in strict_keywords):
                drop_leak.append(c)
                continue
            if re.search(r"^是否.*(患病|疾病|病)$", c_str):
                drop_leak.append(c)
                continue
            if re.search(r"^(has_|is_).*(disease|diagnosis)$", c_low):
                drop_leak.append(c)
    elif mode == "balanced":
        for c in df.columns:
            c_str = str(c).strip()
            c_low = c_str.lower()
            c_compact = re.sub(r"\s+", "", c_low)
            if c_compact in balanced_exact_compact:
                drop_leak.append(c)
                continue
            if any(k in c_str for k in balanced_substrings):
                drop_leak.append(c)
                continue
            if re.search(r"(^|[_\-\s])(label|target|class|outcome|y_true|y_pred)([_\-\s]|$)", c_low):
                drop_leak.append(c)
                continue
            if re.search(r"^是否.*(患病|疾病|病)$", c_str):
                drop_leak.append(c)
                continue
            if re.search(r"^(has_|is_).*(disease|diagnosis)$", c_low):
                drop_leak.append(c)

    drop_pii = [c for c in df.columns if any(k in str(c) for k in pii_keywords)]

    missing_ratio = df.isna().mean()
    drop_missing = missing_ratio[missing_ratio > 0.95].index.tolist()

    drop_high_card: list[str] = []
    n = len(df)
    for col in df.columns:
        s = df[col]
        if s.dtype == object or pd.api.types.is_string_dtype(s):
            nunique = s.nunique(dropna=True)
            if n > 0 and nunique > 50 and (nunique / max(n, 1)) > 0.80:
                drop_high_card.append(col)

    to_drop = sorted(set(drop_leak + drop_pii + drop_missing + drop_high_card))
    out = df.drop(columns=[c for c in to_drop if c in df.columns])
    return out, to_drop


def summarize_column_types_and_anomalies(df: pd.DataFrame, numeric_like_threshold: float = 0.8) -> dict:
    dtype_counts = {str(k): int(v) for k, v in df.dtypes.astype(str).value_counts().to_dict().items()}
    column_types = {str(c): str(t) for c, t in df.dtypes.astype(str).to_dict().items()}

    impacted_object_columns: dict[str, dict] = {}
    anomaly_object_columns: dict[str, dict] = {}

    for col in df.columns:
        s = df[col]
        if not (s.dtype == object or pd.api.types.is_string_dtype(s)):
            continue

        s_non_na = s.dropna().astype(str).str.strip()
        if s_non_na.empty:
            continue

        is_num = s_non_na.str.match(STRICT_NUM_RE)
        is_cmp = s_non_na.str.match(CMP_TOKEN_RE)
        is_cn = s_non_na.str.contains(CN_TEXT_RE)
        is_negative_word = s_non_na.str.contains("阴性", na=False)

        numeric_like_ratio = float((is_num | is_cmp).mean())
        non_numeric_tokens = s_non_na[~(is_num | is_cmp)]
        top_non_numeric = non_numeric_tokens.value_counts().head(10).to_dict()
        top_cmp = s_non_na[is_cmp].value_counts().head(10).to_dict()
        top_negative = s_non_na[is_negative_word].value_counts().head(10).to_dict()

        if numeric_like_ratio >= float(numeric_like_threshold):
            impacted_object_columns[str(col)] = {
                "numeric_like_ratio": numeric_like_ratio,
                "top_non_numeric_tokens": {str(k): int(v) for k, v in top_non_numeric.items()},
                "top_cmp_tokens": {str(k): int(v) for k, v in top_cmp.items()},
                "top_negative_tokens": {str(k): int(v) for k, v in top_negative.items()},
            }

        if len(top_non_numeric) > 0 or len(top_cmp) > 0 or len(top_negative) > 0:
            anomaly_object_columns[str(col)] = {
                "numeric_like_ratio": numeric_like_ratio,
                "has_chinese_text": bool(is_cn.any()),
                "top_non_numeric_tokens": {str(k): int(v) for k, v in top_non_numeric.items()},
                "top_cmp_tokens": {str(k): int(v) for k, v in top_cmp.items()},
                "top_negative_tokens": {str(k): int(v) for k, v in top_negative.items()},
            }

    return {
        "rows": int(len(df)),
        "cols": int(df.shape[1]),
        "dtype_counts": dtype_counts,
        "column_types": column_types,
        "impacted_object_columns": impacted_object_columns,
        "anomaly_object_columns": anomaly_object_columns,
    }


def _normalize_cmp_symbol(symbol: str) -> str:
    s = str(symbol)
    if s == "＜":
        return "<"
    if s == "＞":
        return ">"
    if s == "≤":
        return "<="
    if s == "≥":
        return ">="
    return s


def _build_cell_rng(seed: int, col: str, row_idx: int, token: str) -> np.random.Generator:
    key = f"{int(seed)}|{col}|{int(row_idx)}|{token}".encode("utf-8")
    digest = hashlib.sha256(key).digest()
    seed_int = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return np.random.default_rng(seed_int)


def _sample_cmp_value_by_rule(seed: int, col: str, row_idx: int, token: str, fallback_max: float) -> float:
    raw = str(token).strip().replace("＜", "<").replace("＞", ">")
    rng = _build_cell_rng(seed, col, row_idx, raw)

    if CMP_MMAX_RE.match(raw):
        return float(int(rng.integers(600, 801)))

    m = CMP_PARSE_RE.match(raw)
    if m is None:
        upper = max(0.0, float(fallback_max))
        return float(rng.uniform(0.0, upper)) if upper > 0 else 0.0

    op = _normalize_cmp_symbol(m.group(1))
    v = float(m.group(2))

    if op in {"<", "<="}:
        if v <= 1.0 + 1e-9:
            choices = np.asarray([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], dtype=np.float64)
            return float(rng.choice(choices))
        lo = max(0.0, min(v, v - 1.0))
        hi = max(lo, v)
        return float(rng.uniform(lo, hi)) if hi > lo else float(hi)

    if v >= 500:
        return float(int(rng.integers(500, 601)))
    if v >= 240:
        return float(int(rng.integers(240, 301)))
    if v >= 180:
        return float(int(rng.integers(180, 221)))

    low = max(0, int(math.floor(v)))
    high = max(low + 1, low + 40)
    return float(int(rng.integers(low, high + 1)))


def _is_directional_fill_text_token(token: str) -> bool:
    t = str(token).strip()
    if not t:
        return False
    tl = t.lower()
    if "主动脉" in t or "口服" in t:
        return True
    if "pci" in tl:
        return True
    if re.search(r"\d+\s*mg", tl) is not None:
        return True
    if any(k in t for k in ["导丝", "球囊", "支架", "术，取", "术,取"]):
        return True
    return False


def _collect_direction_values(values: np.ndarray, idx: int, need: int = 4) -> tuple[list[float], str]:
    n = int(values.shape[0])

    up_vals: list[float] = []
    j = idx - 1
    while j >= 0 and len(up_vals) < need:
        v = values[j]
        if np.isfinite(v):
            up_vals.append(float(v))
        j -= 1
    if len(up_vals) >= need:
        return up_vals[:need], "up"

    down_vals: list[float] = []
    j = idx + 1
    while j < n and len(down_vals) < need:
        v = values[j]
        if np.isfinite(v):
            down_vals.append(float(v))
        j += 1
    if len(down_vals) >= need:
        return down_vals[:need], "down"

    merged = up_vals + down_vals
    return merged[:need], "mixed"


def fill_positions_by_directional_mean(values: np.ndarray, positions: list[int], need: int = 4) -> tuple[np.ndarray, dict]:
    arr = np.asarray(values, dtype=np.float64).copy()
    stats = {
        "targets": int(len(positions)),
        "filled": 0,
        "filled_up": 0,
        "filled_down": 0,
        "filled_mixed": 0,
        "unfilled": 0,
    }
    if len(positions) == 0:
        return arr, stats

    for pos in sorted(set(int(p) for p in positions if 0 <= int(p) < len(arr))):
        candidates, mode = _collect_direction_values(arr, pos, need=need)
        if len(candidates) > 0:
            arr[pos] = float(np.mean(np.asarray(candidates, dtype=np.float64)))
            stats["filled"] += 1
            if mode == "up":
                stats["filled_up"] += 1
            elif mode == "down":
                stats["filled_down"] += 1
            else:
                stats["filled_mixed"] += 1
        else:
            stats["unfilled"] += 1

    return arr, stats


def fill_missing_with_neighbor_mean(values: np.ndarray, window: int = 4) -> tuple[np.ndarray, int, int]:
    arr = np.asarray(values, dtype=np.float64).copy()
    n = int(arr.shape[0])
    if n == 0:
        return arr, 0, 0

    neighbor_filled = 0
    nan_idx = np.where(np.isnan(arr))[0]
    for i in nan_idx:
        left = arr[max(0, i - window) : i]
        right = arr[i + 1 : min(n, i + 1 + window)]
        neighbors = np.concatenate([left, right])
        neighbors = neighbors[~np.isnan(neighbors)]
        if neighbors.size > 0:
            arr[i] = float(np.mean(neighbors))
            neighbor_filled += 1

    remain = np.isnan(arr)
    fallback_filled = 0
    if remain.any():
        valid = arr[~np.isnan(arr)]
        fallback = float(np.mean(valid)) if valid.size > 0 else 0.0
        arr[remain] = fallback
        fallback_filled = int(remain.sum())

    return arr, int(neighbor_filled), int(fallback_filled)


def apply_user_rule_cleaning(
    df: pd.DataFrame,
    seed: int,
    neighbor_window: int,
    cmp_random_max: float,
    numeric_like_threshold: float,
    enable_missing_indicators: bool,
    missing_indicator_threshold: float,
) -> tuple[pd.DataFrame, dict]:
    out = df.copy()

    rule_replace_by_column: dict[str, dict] = {}
    total_negative_replaced = 0
    total_cmp_replaced = 0
    directional_text_positions_by_col: dict[str, list[int]] = {}
    directional_text_tokens_by_col: dict[str, dict[str, int]] = {}

    for col in out.columns:
        s = out[col]
        if not (s.dtype == object or pd.api.types.is_string_dtype(s)):
            continue

        col_negative_replaced = 0
        col_cmp_replaced = 0
        col_directional_positions: list[int] = []
        col_directional_tokens: dict[str, int] = {}
        converted: list[object] = []
        for row_idx, raw in enumerate(s.to_numpy(dtype=object)):
            if pd.isna(raw):
                converted.append(np.nan)
                continue

            token = str(raw).strip()
            token = token.replace("＜", "<").replace("＞", ">")

            if token == "" or token == "-":
                converted.append(np.nan)
                continue

            if "阴性" in token:
                converted.append(0.0)
                col_negative_replaced += 1
                continue

            if CMP_TOKEN_RE.match(token):
                converted.append(
                    _sample_cmp_value_by_rule(
                        seed=int(seed),
                        col=str(col),
                        row_idx=int(row_idx),
                        token=token,
                        fallback_max=cmp_random_max,
                    )
                )
                col_cmp_replaced += 1
                continue

            if STRICT_NUM_RE.match(token):
                converted.append(float(token))
                continue

            if _is_directional_fill_text_token(token):
                converted.append(np.nan)
                col_directional_positions.append(int(row_idx))
                col_directional_tokens[token] = int(col_directional_tokens.get(token, 0) + 1)
                continue

            converted.append(token)

        out[col] = pd.Series(converted, index=out.index, dtype="object")
        if col_directional_positions:
            directional_text_positions_by_col[str(col)] = col_directional_positions
            directional_text_tokens_by_col[str(col)] = col_directional_tokens
        if col_negative_replaced > 0 or col_cmp_replaced > 0:
            rule_replace_by_column[str(col)] = {
                "negative_to_zero": int(col_negative_replaced),
                "cmp_replaced_by_rule": int(col_cmp_replaced),
            }

        total_negative_replaced += int(col_negative_replaced)
        total_cmp_replaced += int(col_cmp_replaced)

    converted_numeric_like_cols: dict[str, dict] = {}
    directional_fill_by_column: dict[str, dict] = {}
    directional_fill_total = 0
    for col in out.columns:
        s = out[col]
        if pd.api.types.is_numeric_dtype(s):
            continue
        if not (s.dtype == object or pd.api.types.is_string_dtype(s)):
            continue

        notna = s.notna()
        n_notna = int(notna.sum())
        if n_notna == 0:
            continue

        as_num = pd.to_numeric(s, errors="coerce")
        numeric_ratio = float((as_num.notna() & notna).sum() / max(n_notna, 1))
        if numeric_ratio >= float(numeric_like_threshold):
            bad_tokens = s[notna & as_num.isna()].astype(str).str.strip()
            as_num_arr = as_num.to_numpy(dtype=np.float64)

            positions = directional_text_positions_by_col.get(str(col), [])
            if positions:
                as_num_arr, d_stats = fill_positions_by_directional_mean(as_num_arr, positions, need=4)
                directional_fill_by_column[str(col)] = {
                    "tokens": directional_text_tokens_by_col.get(str(col), {}),
                    **d_stats,
                }
                directional_fill_total += int(d_stats.get("filled", 0))

            out[col] = as_num_arr
            converted_numeric_like_cols[str(col)] = {
                "numeric_like_ratio": numeric_ratio,
                "non_numeric_token_examples": {
                    str(k): int(v) for k, v in bad_tokens.value_counts().head(8).to_dict().items()
                },
            }

    missing_fill_by_column: dict[str, dict] = {}
    missing_indicator_ratio_by_column: dict[str, float] = {}
    missing_indicator_added_cols: list[str] = []
    neighbor_fill_total = 0
    fallback_fill_total = 0

    numeric_cols, _ = split_columns(out)
    for col in numeric_cols:
        col_values = pd.to_numeric(out[col], errors="coerce").to_numpy(dtype=np.float64)
        missing_before = int(np.isnan(col_values).sum())
        if missing_before <= 0:
            continue

        if bool(enable_missing_indicators) and not str(col).endswith("__is_missing"):
            miss_ratio = float(missing_before / max(len(col_values), 1))
            if miss_ratio >= float(missing_indicator_threshold):
                ind_col = f"{col}__is_missing"
                out[ind_col] = np.isnan(col_values).astype(np.float32)
                missing_indicator_ratio_by_column[str(col)] = miss_ratio
                missing_indicator_added_cols.append(ind_col)

        filled, n_neighbor, n_fallback = fill_missing_with_neighbor_mean(col_values, window=int(neighbor_window))
        out[col] = filled

        missing_fill_by_column[str(col)] = {
            "missing_before": int(missing_before),
            "filled_by_neighbor_mean": int(n_neighbor),
            "filled_by_fallback_mean": int(n_fallback),
            "missing_after": int(np.isnan(filled).sum()),
        }
        neighbor_fill_total += int(n_neighbor)
        fallback_fill_total += int(n_fallback)

    post_profile = summarize_column_types_and_anomalies(out, numeric_like_threshold=numeric_like_threshold)
    anomaly_object_columns = post_profile.get("anomaly_object_columns", {})
    other_anomaly_columns = {
        c: v
        for c, v in anomaly_object_columns.items()
        if len(v.get("top_non_numeric_tokens", {})) > 0
    }

    info = {
        "enabled": True,
        "neighbor_window": int(neighbor_window),
        "cmp_sampling_strategy": {
            "lt_1": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            "gt_500": [500, 600],
            "gt_240": [240, 300],
            "gt_180": [180, 220],
            "gt_Mmax": [600, 800],
            "fallback_max": float(max(0.0, cmp_random_max)),
            "deterministic_by_cell": True,
        },
        "numeric_like_threshold": float(numeric_like_threshold),
        "negative_to_zero_total": int(total_negative_replaced),
        "cmp_to_random_total": int(total_cmp_replaced),
        "rule_replace_by_column": rule_replace_by_column,
        "converted_numeric_like_columns": converted_numeric_like_cols,
        "directional_fill_by_column": directional_fill_by_column,
        "directional_fill_total": int(directional_fill_total),
        "missing_fill_by_column": missing_fill_by_column,
        "missing_filled_by_neighbor_total": int(neighbor_fill_total),
        "missing_filled_by_fallback_total": int(fallback_fill_total),
        "missing_indicators": {
            "enabled": bool(enable_missing_indicators),
            "threshold": float(missing_indicator_threshold),
            "source_missing_ratio": missing_indicator_ratio_by_column,
            "added_columns": missing_indicator_added_cols,
            "added_count": int(len(missing_indicator_added_cols)),
        },
        "other_anomaly_columns": other_anomaly_columns,
    }
    return out, info


def split_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric_cols: list[str] = []
    categorical_cols: list[str] = []
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)
    return numeric_cols, categorical_cols


def coerce_categorical_to_str(df: pd.DataFrame, categorical_cols: list[str]) -> pd.DataFrame:
    if not categorical_cols:
        return df
    df = df.copy()
    for col in categorical_cols:
        df[col] = df[col].astype("object").where(~df[col].isna(), "__NA__").astype(str)
    return df


def build_preprocessor(numeric_cols: list[str], categorical_cols: list[str]) -> ColumnTransformer:
    transformers = []
    if numeric_cols:
        numeric_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("num", numeric_pipe, numeric_cols))

    if categorical_cols:
        categorical_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "to_str",
                    FunctionTransformer(
                        lambda x: x.astype(str),
                        validate=False,
                        feature_names_out="one-to-one",
                    ),
                ),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
            ]
        )
        transformers.append(("cat", categorical_pipe, categorical_cols))

    return ColumnTransformer(transformers=transformers, remainder="drop")


def score_from_metrics(metrics: dict, metric_name: str) -> float:
    if metric_name == "accuracy":
        return float(metrics["accuracy"])
    if metric_name == "macro_f1":
        return float(metrics["macro_f1"])
    raise ValueError(f"Unsupported selection metric: {metric_name}")


def build_model_strategy_settings(args) -> dict[str, dict]:
    unified = {
        "selection_metric": str(args.selection_metric),
        "enable_missing_indicators": bool(args.enable_missing_indicators),
        "missing_indicator_threshold": float(args.missing_indicator_threshold),
        "enable_feature_selection": bool(args.enable_feature_selection),
        "feature_select_max_features": int(args.feature_select_max_features),
        "feature_select_min_features": int(args.feature_select_min_features),
        "imbalance_reference": str(args.imbalance_reference),
        "sampler_weight_power": float(args.sampler_weight_power),
        "class_weight_min_clip": float(args.class_weight_min_clip),
        "class_weight_max_clip": float(args.class_weight_max_clip),
    }
    settings = {
        "mlp": dict(unified),
        "transformer": dict(unified),
    }

    if str(args.strategy_profile) == "mixed_best":
        settings["mlp"].update(
            {
                "selection_metric": "macro_f1",
                "enable_missing_indicators": True,
                "missing_indicator_threshold": 0.20,
                "enable_feature_selection": True,
                "feature_select_max_features": 120,
                "feature_select_min_features": 40,
                "imbalance_reference": "pre_augmentation",
                "sampler_weight_power": 0.75,
                "class_weight_min_clip": 0.25,
                "class_weight_max_clip": 4.0,
            }
        )
        settings["transformer"].update(
            {
                "selection_metric": "accuracy",
                "enable_missing_indicators": False,
                "missing_indicator_threshold": 0.20,
                "enable_feature_selection": False,
                "feature_select_max_features": 120,
                "feature_select_min_features": 40,
                "imbalance_reference": "post_augmentation",
                "sampler_weight_power": 1.0,
                "class_weight_min_clip": 0.0,
                "class_weight_max_clip": 1_000_000.0,
            }
        )

    return settings


def fit_light_feature_selection(
    X_train_df: pd.DataFrame,
    y_train: np.ndarray,
    seed: int,
    max_features: int,
    min_features: int,
) -> tuple[list[str], pd.DataFrame, dict]:
    cols = list(X_train_df.columns)
    info = {
        "enabled": True,
        "n_features_before": int(len(cols)),
        "n_features_after": int(len(cols)),
        "max_features": int(max_features),
        "min_features": int(min_features),
    }

    if len(cols) <= 1:
        info["skipped_reason"] = "too_few_features"
        return cols, pd.DataFrame(columns=["feature", "mutual_info"]), info

    if int(max_features) <= 0:
        info["skipped_reason"] = "max_features_non_positive"
        return cols, pd.DataFrame(columns=["feature", "mutual_info"]), info

    if len(cols) <= int(max_features):
        info["skipped_reason"] = "n_features_not_exceed_max"
        return cols, pd.DataFrame(columns=["feature", "mutual_info"]), info

    numeric_cols, categorical_cols = split_columns(X_train_df)
    parts: list[np.ndarray] = []
    feature_names: list[str] = []
    discrete_flags: list[bool] = []

    if numeric_cols:
        num_df = X_train_df[numeric_cols].apply(pd.to_numeric, errors="coerce")
        num_arr = num_df.to_numpy(dtype=np.float32)
        medians = np.zeros(num_arr.shape[1], dtype=np.float32)
        for j in range(num_arr.shape[1]):
            col_j = num_arr[:, j]
            valid = col_j[np.isfinite(col_j)]
            medians[j] = float(np.median(valid)) if valid.size > 0 else 0.0
        nan_pos = np.where(~np.isfinite(num_arr))
        if nan_pos[0].size > 0:
            num_arr[nan_pos] = medians[nan_pos[1]]
        parts.append(num_arr.astype(np.float32))
        feature_names.extend([str(c) for c in numeric_cols])
        discrete_flags.extend([False] * len(numeric_cols))

    if categorical_cols:
        cat_df = X_train_df[categorical_cols].astype("object").where(~X_train_df[categorical_cols].isna(), "__NA__")
        cat_df = cat_df.astype(str)
        cat_enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        cat_arr = cat_enc.fit_transform(cat_df).astype(np.float32)
        parts.append(cat_arr)
        feature_names.extend([str(c) for c in categorical_cols])
        discrete_flags.extend([True] * len(categorical_cols))

    if not parts:
        info["skipped_reason"] = "no_features_after_encoding"
        return cols, pd.DataFrame(columns=["feature", "mutual_info"]), info

    X_mi = np.concatenate(parts, axis=1)
    y_arr = np.asarray(y_train, dtype=np.int64)

    try:
        mi = mutual_info_classif(
            X_mi,
            y_arr,
            discrete_features=np.asarray(discrete_flags, dtype=bool),
            random_state=int(seed),
        )
    except Exception as ex:
        info["skipped_reason"] = f"mutual_info_failed: {ex}"
        return cols, pd.DataFrame(columns=["feature", "mutual_info"]), info

    mi = np.asarray(mi, dtype=np.float64)
    mi = np.where(np.isfinite(mi), mi, 0.0)
    rank_df = pd.DataFrame({"feature": feature_names, "mutual_info": mi}).sort_values(
        ["mutual_info", "feature"],
        ascending=[False, True],
    )

    keep_n = min(len(cols), max(int(min_features), int(max_features)))
    keep_n = max(1, keep_n)
    ranked_cols = rank_df["feature"].tolist()
    selected = ranked_cols[:keep_n]

    # Preserve missingness indicators because they carry independent signal beyond imputed values.
    indicator_cols = [str(c) for c in cols if str(c).endswith("__is_missing")]
    forced_keep = 0
    for ind_col in indicator_cols:
        if ind_col in selected:
            continue
        replaced = False
        for i in range(len(selected) - 1, -1, -1):
            if not str(selected[i]).endswith("__is_missing"):
                selected[i] = ind_col
                replaced = True
                forced_keep += 1
                break
        if not replaced and len(selected) < len(cols):
            selected.append(ind_col)
            forced_keep += 1

    seen: set[str] = set()
    selected_unique: list[str] = []
    for c in selected:
        if c not in seen:
            selected_unique.append(c)
            seen.add(c)
    if len(selected_unique) < keep_n:
        for c in ranked_cols:
            if c in seen:
                continue
            selected_unique.append(c)
            seen.add(c)
            if len(selected_unique) >= keep_n:
                break

    selected_cols = [c for c in cols if str(c) in seen]
    if not selected_cols:
        selected_cols = cols
        info["skipped_reason"] = "empty_after_selection_fallback_all"

    info.update(
        {
            "n_features_after": int(len(selected_cols)),
            "forced_keep_missing_indicators": int(forced_keep),
            "selected_columns_preview": selected_cols[:50],
            "score_top20": [
                {
                    "feature": str(r["feature"]),
                    "mutual_info": float(r["mutual_info"]),
                }
                for r in rank_df.head(20).to_dict("records")
            ],
        }
    )
    return selected_cols, rank_df, info


def compute_balanced_class_weights(
    y: np.ndarray,
    n_classes: int,
    power: float = 1.0,
    min_clip: float = 0.25,
    max_clip: float = 4.0,
) -> np.ndarray:
    y = np.asarray(y, dtype=np.int64)
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    counts[counts <= 0] = 1.0
    weights = (counts.sum() / counts) / max(n_classes, 1)
    weights = np.power(weights, float(max(power, 0.0)))
    if float(max_clip) > 0.0:
        lo = float(min(min_clip, max_clip))
        hi = float(max(min_clip, max_clip))
        weights = np.clip(weights, lo, hi)
    weights = weights / max(weights.mean(), EPS)
    return weights.astype(np.float32)


def build_weighted_sampler(y: np.ndarray, class_weights: np.ndarray) -> WeightedRandomSampler:
    sample_weights = class_weights[np.asarray(y, dtype=np.int64)].astype(np.float64)
    return WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights),
        num_samples=len(sample_weights),
        replacement=True,
    )


def detect_label_noise_by_oof_rf(
    X_dense: np.ndarray,
    y: np.ndarray,
    seed: int,
    prob_threshold: float,
    cv_folds: int,
    max_drop_ratio: float,
) -> tuple[np.ndarray, dict]:
    y = np.asarray(y, dtype=np.int64)
    n = int(len(y))
    n_classes = int(np.max(y) + 1) if n > 0 else 0

    info = {
        "enabled": True,
        "n_train_before": n,
        "prob_threshold": float(prob_threshold),
        "cv_folds": int(cv_folds),
        "max_drop_ratio": float(max_drop_ratio),
        "n_flagged": 0,
        "n_dropped": 0,
        "n_train_after": n,
    }

    if n < max(60, cv_folds * 8) or n_classes < 2:
        info["skipped_reason"] = "not_enough_samples_or_classes"
        return np.zeros(n, dtype=bool), info

    class_counts = np.bincount(y, minlength=n_classes)
    if np.min(class_counts[class_counts > 0]) < cv_folds:
        info["skipped_reason"] = "minority_class_too_small_for_cv"
        return np.zeros(n, dtype=bool), info

    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    oof_prob = np.zeros((n, n_classes), dtype=np.float32)

    for tr_idx, va_idx in skf.split(X_dense, y):
        rf = RandomForestClassifier(
            n_estimators=500,
            random_state=seed,
            n_jobs=-1,
            class_weight="balanced_subsample",
        )
        rf.fit(X_dense[tr_idx], y[tr_idx])
        fold_prob_raw = rf.predict_proba(X_dense[va_idx])
        fold_prob = np.zeros((len(va_idx), n_classes), dtype=np.float32)
        for j, cls in enumerate(rf.classes_):
            fold_prob[:, int(cls)] = fold_prob_raw[:, j]
        oof_prob[va_idx] = fold_prob

    pred = oof_prob.argmax(axis=1)
    true_prob = oof_prob[np.arange(n), y]
    flagged = (pred != y) & (true_prob < float(prob_threshold))
    info["n_flagged"] = int(flagged.sum())

    drop_mask = np.zeros(n, dtype=bool)
    for cls in range(n_classes):
        cls_idx = np.where((y == cls) & flagged)[0]
        if len(cls_idx) == 0:
            continue
        cls_total = int((y == cls).sum())
        cls_cap_by_ratio = int(np.floor(cls_total * float(max_drop_ratio)))
        cls_cap_by_min_keep = max(0, cls_total - max(12, int(np.ceil(0.5 * cls_total))))
        cls_cap = max(0, min(cls_cap_by_ratio, cls_cap_by_min_keep))
        if cls_cap <= 0:
            continue
        order = cls_idx[np.argsort(true_prob[cls_idx])]
        take = order[:cls_cap]
        drop_mask[take] = True

    info["n_dropped"] = int(drop_mask.sum())
    info["n_train_after"] = int(n - drop_mask.sum())
    return drop_mask, info


def augment_minority_bootstrap(
    X_train_df: pd.DataFrame,
    y_train: np.ndarray,
    numeric_cols: list[str],
    seed: int,
    target_ratio: float,
    noise_std_ratio: float,
) -> tuple[pd.DataFrame, np.ndarray, dict]:
    y_train = np.asarray(y_train, dtype=np.int64)
    counts = pd.Series(y_train).value_counts().sort_index()
    max_count = int(counts.max())
    target_count = int(np.ceil(max_count * float(target_ratio)))
    target_count = max(target_count, int(counts.min()))

    rng = np.random.default_rng(seed)
    parts = [X_train_df.copy()]
    y_parts = [y_train.copy()]
    aug_rows = 0

    numeric_std: dict[str, float] = {}
    for col in numeric_cols:
        v = pd.to_numeric(X_train_df[col], errors="coerce").to_numpy(dtype=np.float32)
        s = float(np.nanstd(v))
        numeric_std[col] = s if np.isfinite(s) else 0.0

    for cls, cnt in counts.items():
        need = max(0, target_count - int(cnt))
        if need <= 0:
            continue
        cls_idx = np.where(y_train == int(cls))[0]
        sampled = rng.choice(cls_idx, size=need, replace=True)
        aug = X_train_df.iloc[sampled].copy().reset_index(drop=True)
        for col in numeric_cols:
            std = numeric_std.get(col, 0.0)
            if std <= 0.0:
                continue
            col_num = pd.to_numeric(aug[col], errors="coerce")
            noise = rng.normal(loc=0.0, scale=float(noise_std_ratio) * std, size=len(aug))
            aug[col] = col_num + noise
        parts.append(aug)
        y_parts.append(np.full(need, int(cls), dtype=np.int64))
        aug_rows += int(need)

    out_df = pd.concat(parts, axis=0, ignore_index=True)
    out_y = np.concatenate(y_parts, axis=0)
    info = {
        "enabled": True,
        "target_ratio": float(target_ratio),
        "noise_std_ratio": float(noise_std_ratio),
        "rows_added": int(aug_rows),
        "train_before": int(len(X_train_df)),
        "train_after": int(len(out_df)),
    }
    return out_df, out_y, info


def fit_teacher_catboost(
    X_train_df: pd.DataFrame,
    y_train: np.ndarray,
    X_val_df: pd.DataFrame,
    y_val: np.ndarray,
    X_test_df: pd.DataFrame,
    y_test: np.ndarray,
    categorical_cols: list[str],
    seed: int,
    depth: int,
    learning_rate: float,
    l2_leaf_reg: float,
    iterations: int,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, dict]:
    info = {
        "enabled": True,
        "applied": False,
        "teacher_model": "catboost",
    }

    try:
        from catboost import CatBoostClassifier
    except Exception:
        info["skipped_reason"] = "catboost_not_installed"
        return None, None, None, info

    n_classes = int(np.max(y_train) + 1)

    if n_classes <= 2:
        loss_function = "Logloss"
        eval_metric = "F1"
    else:
        loss_function = "MultiClass"
        eval_metric = "TotalF1:average=Macro"

    model = CatBoostClassifier(
        loss_function=loss_function,
        eval_metric=eval_metric,
        depth=int(depth),
        learning_rate=float(learning_rate),
        l2_leaf_reg=float(l2_leaf_reg),
        iterations=int(iterations),
        random_seed=int(seed),
        allow_writing_files=False,
        verbose=False,
    )

    model.fit(
        X_train_df,
        y_train,
        cat_features=categorical_cols,
        eval_set=(X_val_df, y_val),
        use_best_model=True,
        early_stopping_rounds=100,
    )

    def _predict_prob(df: pd.DataFrame) -> np.ndarray:
        raw = np.asarray(model.predict_proba(df), dtype=np.float32)
        out = np.zeros((len(df), n_classes), dtype=np.float32)
        for j, cls in enumerate(model.classes_):
            out[:, int(cls)] = raw[:, j]
        row_sum = out.sum(axis=1, keepdims=True)
        out = out / np.clip(row_sum, EPS, None)
        return out

    train_prob = _predict_prob(X_train_df)
    val_prob = _predict_prob(X_val_df)
    test_prob = _predict_prob(X_test_df)

    test_metrics = compute_metrics(y_test, test_prob)
    info.update(
        {
            "applied": True,
            "best_iteration": int(model.get_best_iteration()),
            "test_accuracy": float(test_metrics["accuracy"]),
            "test_macro_f1": float(test_metrics["macro_f1"]),
        }
    )
    return train_prob, val_prob, test_prob, info


def combine_hard_soft_loss(
    logits: torch.Tensor,
    y_true: torch.Tensor,
    teacher_prob: torch.Tensor | None,
    loss_fn,
    hard_weight: float,
    temperature: float,
) -> torch.Tensor:
    ce = loss_fn(logits, y_true)
    if teacher_prob is None:
        return ce

    hw = float(np.clip(hard_weight, 0.0, 1.0))
    t = max(float(temperature), 1e-3)
    tp = teacher_prob / teacher_prob.sum(dim=1, keepdim=True).clamp_min(EPS)
    kd = F.kl_div(
        F.log_softmax(logits / t, dim=1),
        tp,
        reduction="batchmean",
    ) * (t * t)
    return hw * ce + (1.0 - hw) * kd


class SparseMLPDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, teacher_prob: np.ndarray | None = None):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()
        self.teacher_prob = torch.from_numpy(teacher_prob).float() if teacher_prob is not None else None

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        if self.teacher_prob is None:
            return self.X[idx], self.y[idx]
        return self.X[idx], self.y[idx], self.teacher_prob[idx]


class MLP(nn.Module):
    def __init__(self, in_dim: int, num_classes: int, hidden: int = 384, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class TabTransformerConfig:
    d_model: int = 96
    nhead: int = 4
    num_layers: int = 3
    dim_feedforward: int = 192
    dropout: float = 0.05


class TabTransformer(nn.Module):
    def __init__(
        self,
        cat_cardinalities: list[int],
        num_numeric: int,
        num_classes: int,
        cfg: TabTransformerConfig,
    ):
        super().__init__()
        self.num_cat = len(cat_cardinalities)
        self.num_num = num_numeric

        self.cat_embeddings = nn.ModuleList([nn.Embedding(card + 1, cfg.d_model) for card in cat_cardinalities])
        self.num_proj = nn.Linear(1, cfg.d_model)

        n_tokens = self.num_cat + self.num_num
        self.feature_emb = nn.Parameter(torch.zeros(1, n_tokens, cfg.d_model))
        nn.init.normal_(self.feature_emb, mean=0.0, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=cfg.num_layers)
        self.norm = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, num_classes)

    def forward(self, cat_x: torch.Tensor, num_x: torch.Tensor) -> torch.Tensor:
        tokens: list[torch.Tensor] = []
        if self.num_cat:
            for i, emb in enumerate(self.cat_embeddings):
                tokens.append(emb(cat_x[:, i]).unsqueeze(1))
        if self.num_num:
            tokens.append(self.num_proj(num_x.unsqueeze(-1)))

        if not tokens:
            raise ValueError("No tokens for TabTransformer")

        x = torch.cat(tokens, dim=1) if len(tokens) > 1 else tokens[0]
        x = x + self.feature_emb
        x = self.encoder(x)
        x = self.norm(x)
        pooled = x.mean(dim=1)
        return self.head(pooled)


class TabTensorDataset(Dataset):
    def __init__(
        self,
        cat_x: np.ndarray,
        num_x: np.ndarray,
        y: np.ndarray,
        teacher_prob: np.ndarray | None = None,
    ):
        self.cat_x = torch.from_numpy(cat_x).long()
        self.num_x = torch.from_numpy(num_x).float()
        self.y = torch.from_numpy(y).long()
        self.teacher_prob = torch.from_numpy(teacher_prob).float() if teacher_prob is not None else None

    def __len__(self) -> int:
        return self.y.shape[0]

    def __getitem__(self, idx: int):
        if self.teacher_prob is None:
            return self.cat_x[idx], self.num_x[idx], self.y[idx]
        return self.cat_x[idx], self.num_x[idx], self.y[idx], self.teacher_prob[idx]


def train_epoch(
    model,
    loader,
    optimizer,
    device,
    loss_fn=None,
    distill_hard_weight: float = 1.0,
    distill_temperature: float = 1.0,
) -> float:
    model.train()
    if loss_fn is None:
        loss_fn = nn.CrossEntropyLoss()
    losses: list[float] = []
    for batch in loader:
        optimizer.zero_grad(set_to_none=True)

        teacher_prob = None
        if len(batch) == 2:
            x, y = batch
            logits = model(x.to(device))
        elif len(batch) == 3:
            b0, b1, b2 = batch
            if torch.is_floating_point(b1):
                cat_x, num_x, y = b0, b1, b2
                logits = model(cat_x.to(device), num_x.to(device))
            else:
                x, y, teacher_prob = b0, b1, b2
                logits = model(x.to(device))
        elif len(batch) == 4:
            cat_x, num_x, y, teacher_prob = batch
            logits = model(cat_x.to(device), num_x.to(device))
        else:
            raise ValueError(f"Unexpected batch size: {len(batch)}")

        teacher_prob_t = teacher_prob.to(device) if teacher_prob is not None else None
        loss = combine_hard_soft_loss(
            logits=logits,
            y_true=y.to(device),
            teacher_prob=teacher_prob_t,
            loss_fn=loss_fn,
            hard_weight=distill_hard_weight,
            temperature=distill_temperature,
        )
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
    return float(np.mean(losses)) if losses else math.nan


@torch.no_grad()
def predict_proba_mlp_dense(model: nn.Module, X_dense: np.ndarray, device, batch: int = 1024) -> np.ndarray:
    model.eval()
    out: list[np.ndarray] = []
    n = len(X_dense)
    for i in range(0, n, batch):
        xb = torch.from_numpy(X_dense[i : i + batch]).float().to(device)
        logits = model(xb)
        out.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.vstack(out) if out else np.zeros((0, 0), dtype=np.float32)


@torch.no_grad()
def predict_proba_transformer_arrays(
    model: nn.Module,
    cat_x: np.ndarray,
    num_x: np.ndarray,
    device,
    batch: int = 1024,
) -> np.ndarray:
    model.eval()
    out: list[np.ndarray] = []
    n = len(cat_x)
    for i in range(0, n, batch):
        cat_t = torch.from_numpy(cat_x[i : i + batch]).long().to(device)
        num_t = torch.from_numpy(num_x[i : i + batch]).float().to(device)
        logits = model(cat_t, num_t)
        out.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.vstack(out) if out else np.zeros((0, 0), dtype=np.float32)


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    y_pred = y_prob.argmax(axis=1)
    n_classes = y_prob.shape[1]

    # label_binarize returns shape (n_samples, 1) for binary case;
    # build one-vs-rest matrix explicitly to keep shape (n_samples, n_classes).
    y_true_bin = np.zeros((len(y_true), n_classes), dtype=np.int64)
    for i in range(n_classes):
        y_true_bin[:, i] = (y_true == i).astype(np.int64)

    aucs: list[float] = []
    for i in range(n_classes):
        yi = y_true_bin[:, i]
        if np.unique(yi).size < 2:
            continue
        fpr_i, tpr_i, _ = roc_curve(yi, y_prob[:, i])
        aucs.append(float(auc(fpr_i, tpr_i)))
    auc_macro = float(np.mean(aucs)) if aucs else math.nan

    try:
        fpr_micro, tpr_micro, _ = roc_curve(y_true_bin.ravel(), y_prob.ravel())
        auc_micro = float(auc(fpr_micro, tpr_micro))
    except Exception:
        auc_micro = math.nan

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "roc_auc_micro": auc_micro,
        "roc_auc_macro": auc_macro,
    }


def compute_macro_pr_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    if y_prob.ndim != 2 or len(y_true) != y_prob.shape[0] or y_prob.shape[0] == 0:
        return math.nan

    n_classes = int(y_prob.shape[1])
    y_true_bin = np.zeros((len(y_true), n_classes), dtype=np.int64)
    for i in range(n_classes):
        y_true_bin[:, i] = (y_true == i).astype(np.int64)

    aps: list[float] = []
    for i in range(n_classes):
        yi = y_true_bin[:, i]
        if np.unique(yi).size < 2:
            continue
        try:
            aps.append(float(average_precision_score(yi, y_prob[:, i])))
        except Exception:
            continue
    return float(np.mean(aps)) if aps else math.nan


def compute_imbalance_diagnostics(y_true: np.ndarray, y_prob: np.ndarray, label_names: list[str]) -> dict:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    if y_prob.ndim != 2 or y_prob.shape[0] == 0:
        return {
            "valid": False,
            "reason": "empty_probabilities",
            "per_class": [],
            "worst_class_recall": 0.0,
            "worst_class_f1": 0.0,
            "macro_pr_auc": math.nan,
        }

    y_pred = y_prob.argmax(axis=1)
    n_classes = int(y_prob.shape[1])
    labels = np.arange(n_classes, dtype=np.int64)
    p, r, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average=None,
        zero_division=0,
    )

    per_class: list[dict] = []
    for i in range(n_classes):
        cname = str(label_names[i]) if i < len(label_names) else str(i)
        per_class.append(
            {
                "class_id": int(i),
                "class_name": cname,
                "support": int(support[i]),
                "precision": float(p[i]),
                "recall": float(r[i]),
                "f1": float(f1[i]),
            }
        )

    support_arr = np.asarray(support, dtype=np.float64)
    positive_support = support_arr[support_arr > 0]
    minority_cutoff = float(np.percentile(positive_support, 40.0)) if positive_support.size > 0 else 0.0
    minority_ids = [int(i) for i in range(n_classes) if float(support_arr[i]) > 0 and float(support_arr[i]) <= minority_cutoff]

    if minority_ids:
        minority_recall = float(np.mean([safe_float(r[i]) for i in minority_ids]))
        minority_f1 = float(np.mean([safe_float(f1[i]) for i in minority_ids]))
        minority_classes = [str(label_names[i]) if i < len(label_names) else str(i) for i in minority_ids]
    else:
        minority_recall = float(np.mean(r)) if len(r) > 0 else 0.0
        minority_f1 = float(np.mean(f1)) if len(f1) > 0 else 0.0
        minority_classes = []

    worst_recall_idx = int(np.argmin(r)) if len(r) > 0 else 0
    worst_f1_idx = int(np.argmin(f1)) if len(f1) > 0 else 0

    return {
        "valid": True,
        "macro_pr_auc": compute_macro_pr_auc(y_true, y_prob),
        "worst_class_recall": float(r[worst_recall_idx]) if len(r) > 0 else 0.0,
        "worst_class_recall_class": per_class[worst_recall_idx]["class_name"] if per_class else "",
        "worst_class_f1": float(f1[worst_f1_idx]) if len(f1) > 0 else 0.0,
        "worst_class_f1_class": per_class[worst_f1_idx]["class_name"] if per_class else "",
        "minority_support_cutoff": minority_cutoff,
        "minority_classes": minority_classes,
        "minority_macro_recall": minority_recall,
        "minority_macro_f1": minority_f1,
        "per_class": per_class,
    }


def apply_prior_adjustment(y_prob: np.ndarray, class_priors: np.ndarray, tau: float) -> np.ndarray:
    y_prob = np.asarray(y_prob, dtype=np.float64)
    priors = np.asarray(class_priors, dtype=np.float64)
    priors = np.clip(priors, EPS, None)
    tau = float(max(0.0, tau))

    adjusted = y_prob / np.power(priors[None, :], tau)
    adjusted = adjusted / np.clip(adjusted.sum(axis=1, keepdims=True), EPS, None)
    return adjusted.astype(np.float32)


def tune_prior_adjustment_tau(
    y_true: np.ndarray,
    y_prob_val: np.ndarray,
    class_priors: np.ndarray,
    label_names: list[str],
    metric_name: str,
    tau_grid: list[float],
    worst_class_recall_floor: float,
) -> tuple[float, dict, dict, list[dict]]:
    if not tau_grid:
        tau_grid = [0.0]

    best_tau = 0.0
    best_score = -1e18
    best_metrics = compute_metrics(y_true, y_prob_val)
    best_diag = compute_imbalance_diagnostics(y_true, y_prob_val, label_names)
    history: list[dict] = []

    for tau in tau_grid:
        tau_f = float(max(0.0, tau))
        adjusted = apply_prior_adjustment(y_prob_val, class_priors, tau_f)
        metrics = compute_metrics(y_true, adjusted)
        diag = compute_imbalance_diagnostics(y_true, adjusted, label_names)
        base_score = score_from_metrics(metrics, metric_name)
        recall_penalty = max(0.0, float(worst_class_recall_floor) - safe_float(diag.get("worst_class_recall", 0.0)))
        score = float(base_score - 2.0 * recall_penalty)
        history.append(
            {
                "tau": tau_f,
                "metric": float(base_score),
                "worst_class_recall": float(diag.get("worst_class_recall", 0.0)),
                "score_after_penalty": score,
            }
        )
        if score > best_score:
            best_score = score
            best_tau = tau_f
            best_metrics = metrics
            best_diag = diag

    return best_tau, best_metrics, best_diag, history


def estimate_feature_direction_for_class(feature_values: pd.Series, y_true: np.ndarray, class_id: int) -> dict:
    y_true = np.asarray(y_true, dtype=np.int64)
    x_num = pd.to_numeric(feature_values, errors="coerce").to_numpy(dtype=np.float64)
    valid = np.isfinite(x_num)
    n_valid = int(valid.sum())

    if n_valid < 12:
        return {
            "method": "insufficient_numeric_values",
            "valid_n": n_valid,
            "direction": "unknown",
            "effect_size": 0.0,
            "pos_median": math.nan,
            "neg_median": math.nan,
        }

    pos = x_num[(y_true == int(class_id)) & valid]
    neg = x_num[(y_true != int(class_id)) & valid]
    if len(pos) < 4 or len(neg) < 4:
        return {
            "method": "insufficient_class_partition",
            "valid_n": n_valid,
            "direction": "unknown",
            "effect_size": 0.0,
            "pos_median": math.nan,
            "neg_median": math.nan,
        }

    pos_median = float(np.median(pos))
    neg_median = float(np.median(neg))
    spread = float(np.std(x_num[valid]))
    spread = max(spread, EPS)
    effect = float((pos_median - neg_median) / spread)

    if effect > 0:
        direction = "higher_towards_disease"
    elif effect < 0:
        direction = "lower_towards_disease"
    else:
        direction = "neutral"

    return {
        "method": "median_shift_over_std",
        "valid_n": n_valid,
        "direction": direction,
        "effect_size": effect,
        "pos_median": pos_median,
        "neg_median": neg_median,
    }


def build_disease_feature_direction_table(
    disease_intersection_df: pd.DataFrame,
    label_names: list[str],
    y_test: np.ndarray,
    topk: int,
    mlp_test_df: pd.DataFrame,
    tr_test_df: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict] = []
    for class_id, class_name in enumerate(label_names):
        sub = disease_intersection_df[disease_intersection_df["class_name"] == class_name].head(int(topk))
        for feature in sub["feature"].tolist():
            if feature in tr_test_df.columns:
                series = tr_test_df[feature]
                source_branch = "transformer"
            elif feature in mlp_test_df.columns:
                series = mlp_test_df[feature]
                source_branch = "mlp"
            else:
                rows.append(
                    {
                        "class_id": int(class_id),
                        "class_name": str(class_name),
                        "feature": str(feature),
                        "source_branch": "missing",
                        "method": "feature_not_found",
                        "valid_n": 0,
                        "direction": "unknown",
                        "effect_size": 0.0,
                        "pos_median": math.nan,
                        "neg_median": math.nan,
                    }
                )
                continue

            direction_info = estimate_feature_direction_for_class(series, y_test, class_id)
            rows.append(
                {
                    "class_id": int(class_id),
                    "class_name": str(class_name),
                    "feature": str(feature),
                    "source_branch": source_branch,
                    "method": str(direction_info.get("method", "")),
                    "valid_n": int(direction_info.get("valid_n", 0)),
                    "direction": str(direction_info.get("direction", "unknown")),
                    "effect_size": float(direction_info.get("effect_size", 0.0)),
                    "pos_median": safe_float(direction_info.get("pos_median", math.nan), default=math.nan),
                    "neg_median": safe_float(direction_info.get("neg_median", math.nan), default=math.nan),
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=[
                "class_id",
                "class_name",
                "feature",
                "source_branch",
                "method",
                "valid_n",
                "direction",
                "effect_size",
                "pos_median",
                "neg_median",
            ]
        )

    out = pd.DataFrame(rows)
    return out.sort_values(["class_id", "effect_size"], ascending=[True, False]).reset_index(drop=True)


def format_direction_note(direction_row: dict | None) -> str:
    if not direction_row:
        return "方向证据缺失。"

    direction = str(direction_row.get("direction", "unknown"))
    effect = safe_float(direction_row.get("effect_size", 0.0), default=0.0)
    method = str(direction_row.get("method", ""))
    valid_n = int(direction_row.get("valid_n", 0))

    if direction == "higher_towards_disease":
        d_text = "数值升高更偏向该病种"
    elif direction == "lower_towards_disease":
        d_text = "数值降低更偏向该病种"
    elif direction == "neutral":
        d_text = "方向接近中性"
    else:
        d_text = "方向不确定"

    return f"{d_text}（effect_size={effect:+.3f}, valid_n={valid_n}, method={method}）"


def permutation_importance(
    predict_proba_fn,
    X_test_df: pd.DataFrame,
    y_test: np.ndarray,
    metric_name: str,
    n_repeats: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    base_prob = predict_proba_fn(X_test_df)
    base_metrics = compute_metrics(y_test, base_prob)
    base = float(base_metrics[metric_name])

    importances = []
    cols = list(X_test_df.columns)

    for col in cols:
        drops = []
        for _ in range(n_repeats):
            X_perm = X_test_df.copy()
            shuffled = X_perm[col].to_numpy(copy=True)
            rng.shuffle(shuffled)
            X_perm[col] = shuffled
            prob = predict_proba_fn(X_perm)
            m = compute_metrics(y_test, prob)[metric_name]
            drops.append(base - float(m))
        importances.append(
            {
                "feature": col,
                "importance_mean": float(np.mean(drops)),
                "importance_std": float(np.std(drops, ddof=1)) if len(drops) > 1 else 0.0,
            }
        )

    df_imp = pd.DataFrame(importances).sort_values("importance_mean", ascending=False)
    df_imp.insert(0, "baseline_metric", base)
    return df_imp


def compute_ovr_metric(y_true: np.ndarray, y_prob: np.ndarray, class_id: int, metric_name: str) -> float:
    y_bin = (np.asarray(y_true, dtype=np.int64) == int(class_id)).astype(np.int64)
    if np.unique(y_bin).size < 2:
        return math.nan

    class_prob = np.asarray(y_prob, dtype=np.float64)[:, int(class_id)]

    if metric_name == "roc_auc":
        fpr_i, tpr_i, _ = roc_curve(y_bin, class_prob)
        return float(auc(fpr_i, tpr_i))
    if metric_name == "average_precision":
        return float(average_precision_score(y_bin, class_prob))
    if metric_name == "f1":
        y_pred_bin = (class_prob >= 0.5).astype(np.int64)
        return float(f1_score(y_bin, y_pred_bin, zero_division=0))

    raise ValueError(f"Unsupported OVR metric: {metric_name}")


def permutation_importance_ovr(
    predict_proba_fn,
    X_test_df: pd.DataFrame,
    y_test: np.ndarray,
    class_id: int,
    metric_name: str,
    n_repeats: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    base_prob = predict_proba_fn(X_test_df)
    base = compute_ovr_metric(y_test, base_prob, class_id=int(class_id), metric_name=metric_name)

    importances = []
    cols = list(X_test_df.columns)
    for col in cols:
        drops = []
        for _ in range(n_repeats):
            X_perm = X_test_df.copy()
            shuffled = X_perm[col].to_numpy(copy=True)
            rng.shuffle(shuffled)
            X_perm[col] = shuffled
            prob = predict_proba_fn(X_perm)
            score = compute_ovr_metric(y_test, prob, class_id=int(class_id), metric_name=metric_name)
            if math.isnan(base) or math.isnan(score):
                drops.append(math.nan)
            else:
                drops.append(base - float(score))

        valid = [x for x in drops if not math.isnan(x)]
        if valid:
            imp_mean = float(np.mean(valid))
            imp_std = float(np.std(valid, ddof=1)) if len(valid) > 1 else 0.0
        else:
            imp_mean = math.nan
            imp_std = math.nan

        importances.append(
            {
                "class_id": int(class_id),
                "feature": col,
                "importance_mean": imp_mean,
                "importance_std": imp_std,
            }
        )

    df_imp = pd.DataFrame(importances).sort_values("importance_mean", ascending=False, na_position="last")
    df_imp.insert(0, "baseline_metric", base)
    return df_imp


def normalize_positive_scores(scores: np.ndarray) -> np.ndarray:
    s = np.maximum(np.asarray(scores, dtype=np.float64), 0.0)
    if s.size == 0:
        return s
    total = float(np.sum(s))
    if total <= EPS:
        return np.full(shape=s.shape, fill_value=1.0 / float(len(s)), dtype=np.float64)
    return s / total


def get_clinical_penalty(feature_name: str) -> tuple[float, str]:
    feature_name = str(feature_name)
    penalty = 1.0
    reasons: list[str] = []
    for pattern, weight, reason in CLINICAL_PENALTY_RULES:
        if pattern.search(feature_name):
            penalty *= float(weight)
            reasons.append(str(reason))
    penalty = max(0.05, float(penalty))
    return penalty, ";".join(reasons)


def build_joint_intersection_table(
    imp_mlp: pd.DataFrame,
    imp_tr: pd.DataFrame,
    apply_clinical_penalty: bool,
) -> pd.DataFrame:
    mlp_cols = imp_mlp[["feature", "importance_mean"]].copy()
    tr_cols = imp_tr[["feature", "importance_mean"]].copy()

    mlp_cols = mlp_cols.rename(columns={"importance_mean": "mlp_importance"})
    tr_cols = tr_cols.rename(columns={"importance_mean": "transformer_importance"})

    mlp_cols["mlp_prob"] = normalize_positive_scores(mlp_cols["mlp_importance"].to_numpy())
    tr_cols["transformer_prob"] = normalize_positive_scores(tr_cols["transformer_importance"].to_numpy())

    merged = mlp_cols.merge(tr_cols, on="feature", how="inner")
    if merged.empty:
        merged["joint_prob"] = []
        merged["joint_importance_geo"] = []
        merged["clinical_penalty"] = []
        merged["penalty_reasons"] = []
        merged["joint_prob_adjusted"] = []
        merged["joint_prob_adjusted_norm"] = []
        return merged

    merged["joint_prob"] = merged["mlp_prob"] * merged["transformer_prob"]
    merged["joint_importance_geo"] = np.sqrt(
        np.maximum(merged["mlp_importance"].to_numpy(), 0.0)
        * np.maximum(merged["transformer_importance"].to_numpy(), 0.0)
    )

    penalties = [get_clinical_penalty(f) for f in merged["feature"].tolist()]
    merged["clinical_penalty"] = [p[0] for p in penalties]
    merged["penalty_reasons"] = [p[1] for p in penalties]
    if apply_clinical_penalty:
        merged["joint_prob_adjusted"] = merged["joint_prob"] * merged["clinical_penalty"]
    else:
        merged["joint_prob_adjusted"] = merged["joint_prob"]

    merged["joint_prob_adjusted_norm"] = normalize_positive_scores(merged["joint_prob_adjusted"].to_numpy())
    merged = merged.sort_values(
        ["joint_prob_adjusted_norm", "joint_importance_geo"],
        ascending=False,
    ).reset_index(drop=True)
    return merged


def build_clinical_note(feature_name: str, disease_name: str) -> dict:
    fname = str(feature_name)
    for item in CLINICAL_EVIDENCE_LIBRARY:
        if item["pattern"].search(fname):
            return {
                "feature": fname,
                "disease": str(disease_name),
                "mechanism": item["mechanism"],
                "treatment": item["treatment"],
                "prognosis": item["prognosis"],
                "untreated": item["untreated"],
                "references": item["references"],
                "matched": True,
            }

    return {
        "feature": fname,
        "disease": str(disease_name),
        "mechanism": f"{fname} 与 {disease_name} 可能存在风险分层关联，但需结合原始化验定义确认生物学含义。",
        "treatment": "建议将其作为辅助分层信号，与指南推荐的一线指标联合判断，不单独作为治疗决策依据。",
        "prognosis": "若在双模型中稳定高权重，通常提示其对不良结局风险分层有贡献。",
        "untreated": "忽略该信号可能导致风险评估不足，影响随访与二级预防强度。",
        "references": ["ESC/AHA 最新疾病相关指南", "请结合科室本地检验解释区间进行二次校验"],
        "matched": False,
    }


def build_inference_table(
    *,
    idx_test: np.ndarray,
    y_test: np.ndarray,
    label_names: list[str],
    mlp_prob: np.ndarray,
    tr_prob: np.ndarray,
) -> pd.DataFrame:
    ensemble_prob = 0.5 * (mlp_prob + tr_prob)
    joint_raw = mlp_prob * tr_prob
    joint_denom = np.maximum(joint_raw.sum(axis=1, keepdims=True), EPS)
    joint_prob = joint_raw / joint_denom

    df = pd.DataFrame(
        {
            "sample_index": idx_test.astype(int),
            "y_true": y_test.astype(int),
            "y_true_name": [label_names[int(i)] for i in y_test],
            "mlp_pred": mlp_prob.argmax(axis=1).astype(int),
            "mlp_pred_name": [label_names[int(i)] for i in mlp_prob.argmax(axis=1)],
            "transformer_pred": tr_prob.argmax(axis=1).astype(int),
            "transformer_pred_name": [label_names[int(i)] for i in tr_prob.argmax(axis=1)],
            "ensemble_pred": ensemble_prob.argmax(axis=1).astype(int),
            "ensemble_pred_name": [label_names[int(i)] for i in ensemble_prob.argmax(axis=1)],
            "joint_pred": joint_prob.argmax(axis=1).astype(int),
            "joint_pred_name": [label_names[int(i)] for i in joint_prob.argmax(axis=1)],
        }
    )

    for i, disease in enumerate(label_names):
        df[f"mlp_prob_{i}_{disease}"] = mlp_prob[:, i]
        df[f"transformer_prob_{i}_{disease}"] = tr_prob[:, i]
        df[f"ensemble_prob_{i}_{disease}"] = ensemble_prob[:, i]
        df[f"joint_prob_{i}_{disease}"] = joint_prob[:, i]
    return df


def build_disease_clinical_markdown(
    disease_intersection_df: pd.DataFrame,
    label_names: list[str],
    topk: int,
    direction_df: pd.DataFrame | None = None,
) -> str:
    lines: list[str] = []
    lines.append("# 分病种 Top 特征临床解释（one-vs-rest + 双模型交集）\n\n")

    for disease in label_names:
        lines.append(f"## {disease}\n")
        sub = disease_intersection_df[disease_intersection_df["class_name"] == disease].head(topk)
        if sub.empty:
            lines.append("- 无可用交集特征（可能由样本量或指标稀疏导致）。\n\n")
            continue

        lines.append("| 排名 | 特征 | 联合概率(降权后) | MLP重要性 | Transformer重要性 | 降权原因 |\n")
        lines.append("|---:|---|---:|---:|---:|---|\n")
        for i, r in enumerate(sub.itertuples(index=False), start=1):
            lines.append(
                f"| {i} | {r.feature} | {r.joint_prob_adjusted_norm:.4f} | {r.mlp_importance:.4f} | {r.transformer_importance:.4f} | {r.penalty_reasons or '-'} |\n"
            )

        lines.append("\n")
        for i, r in enumerate(sub.itertuples(index=False), start=1):
            note = build_clinical_note(str(r.feature), disease)
            direction_row = None
            if direction_df is not None and not direction_df.empty:
                match = direction_df[
                    (direction_df["class_name"] == str(disease))
                    & (direction_df["feature"] == str(r.feature))
                ]
                if not match.empty:
                    direction_row = match.iloc[0].to_dict()
            lines.append(f"{i}. 指标: {r.feature}\n")
            lines.append(f"   - 方向证据: {format_direction_note(direction_row)}\n")
            lines.append(f"   - 机制: {note['mechanism']}\n")
            lines.append(f"   - 治疗启示: {note['treatment']}\n")
            lines.append(f"   - 预后提示: {note['prognosis']}\n")
            lines.append(f"   - 不干预后果: {note['untreated']}\n")
            lines.append("   - 参考: " + "; ".join(note["references"]) + "\n")
        lines.append("\n")

    return "".join(lines)


@dataclass
class BranchData:
    name: str
    X_all_df: pd.DataFrame
    X_train_df: pd.DataFrame
    X_val_df: pd.DataFrame
    X_test_df: pd.DataFrame
    y_train: np.ndarray
    numeric_cols: list[str]
    categorical_cols: list[str]
    type_profile_before: dict
    type_profile_after: dict
    cleaning_info: dict
    noise_filter_info: dict
    augmentation_info: dict
    feature_selection_info: dict
    y_train_for_imbalance: np.ndarray


def build_training_branch(
    *,
    branch_name: str,
    X_base_df: pd.DataFrame,
    idx_train: np.ndarray,
    idx_val: np.ndarray,
    idx_test: np.ndarray,
    y_train_base: np.ndarray,
    seed: int,
    enable_user_rule_cleaning: bool,
    neighbor_window: int,
    cmp_random_max: float,
    numeric_like_threshold: float,
    enable_missing_indicators: bool,
    missing_indicator_threshold: float,
    enable_label_noise_filter: bool,
    noise_prob_threshold: float,
    noise_cv_folds: int,
    noise_max_drop_ratio: float,
    enable_bootstrap_augmentation: bool,
    augment_target_ratio: float,
    augment_noise_std_ratio: float,
    enable_feature_selection: bool,
    feature_select_max_features: int,
    feature_select_min_features: int,
) -> BranchData:
    type_profile_before = summarize_column_types_and_anomalies(
        X_base_df,
        numeric_like_threshold=numeric_like_threshold,
    )

    X_all_df = X_base_df.copy()
    cleaning_info = {"enabled": False}
    if enable_user_rule_cleaning:
        X_all_df, cleaning_info = apply_user_rule_cleaning(
            df=X_all_df,
            seed=seed,
            neighbor_window=neighbor_window,
            cmp_random_max=cmp_random_max,
            numeric_like_threshold=numeric_like_threshold,
            enable_missing_indicators=enable_missing_indicators,
            missing_indicator_threshold=missing_indicator_threshold,
        )

    type_profile_after = summarize_column_types_and_anomalies(
        X_all_df,
        numeric_like_threshold=numeric_like_threshold,
    )

    numeric_cols, categorical_cols = split_columns(X_all_df)
    if not numeric_cols and not categorical_cols:
        raise ValueError(f"[{branch_name}] no features left after filtering/cleaning")

    X_train_df = X_all_df.iloc[idx_train].reset_index(drop=True)
    X_val_df = X_all_df.iloc[idx_val].reset_index(drop=True)
    X_test_df = X_all_df.iloc[idx_test].reset_index(drop=True)

    X_train_df = coerce_categorical_to_str(X_train_df, categorical_cols)
    X_val_df = coerce_categorical_to_str(X_val_df, categorical_cols)
    X_test_df = coerce_categorical_to_str(X_test_df, categorical_cols)

    y_train = np.asarray(y_train_base, dtype=np.int64).copy()

    noise_filter_info = {
        "enabled": False,
        "n_train_before": int(len(X_train_df)),
        "n_dropped": 0,
        "n_train_after": int(len(X_train_df)),
    }
    if enable_label_noise_filter:
        pre_noise = build_preprocessor(numeric_cols, categorical_cols)
        X_noise = pre_noise.fit_transform(X_train_df)
        X_noise_dense = X_noise.toarray() if hasattr(X_noise, "toarray") else np.asarray(X_noise)
        drop_mask, noise_filter_info = detect_label_noise_by_oof_rf(
            X_dense=X_noise_dense,
            y=y_train,
            seed=seed,
            prob_threshold=noise_prob_threshold,
            cv_folds=noise_cv_folds,
            max_drop_ratio=noise_max_drop_ratio,
        )
        if drop_mask.any():
            X_train_df = X_train_df.loc[~drop_mask].reset_index(drop=True)
            y_train = y_train[~drop_mask]

    y_train_for_imbalance = np.asarray(y_train, dtype=np.int64).copy()

    augmentation_info = {
        "enabled": False,
        "rows_added": 0,
        "train_before": int(len(X_train_df)),
        "train_after": int(len(X_train_df)),
    }
    if enable_bootstrap_augmentation:
        X_train_df, y_train, augmentation_info = augment_minority_bootstrap(
            X_train_df=X_train_df,
            y_train=y_train,
            numeric_cols=numeric_cols,
            seed=seed,
            target_ratio=augment_target_ratio,
            noise_std_ratio=augment_noise_std_ratio,
        )

    feature_selection_info = {
        "enabled": bool(enable_feature_selection),
        "n_features_before": int(X_train_df.shape[1]),
        "n_features_after": int(X_train_df.shape[1]),
    }
    if enable_feature_selection:
        selected_cols, _, feature_selection_info = fit_light_feature_selection(
            X_train_df=X_train_df,
            y_train=y_train,
            seed=seed,
            max_features=feature_select_max_features,
            min_features=feature_select_min_features,
        )
        X_all_df = X_all_df[selected_cols].copy()
        X_train_df = X_train_df[selected_cols].copy()
        X_val_df = X_val_df[selected_cols].copy()
        X_test_df = X_test_df[selected_cols].copy()

    numeric_cols, categorical_cols = split_columns(X_all_df)

    return BranchData(
        name=branch_name,
        X_all_df=X_all_df,
        X_train_df=X_train_df,
        X_val_df=X_val_df,
        X_test_df=X_test_df,
        y_train=y_train,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        type_profile_before=type_profile_before,
        type_profile_after=type_profile_after,
        cleaning_info=cleaning_info,
        noise_filter_info=noise_filter_info,
        augmentation_info=augmentation_info,
        feature_selection_info=feature_selection_info,
        y_train_for_imbalance=y_train_for_imbalance,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--excel", type=str, default="心肌梗死患者生化指标分析.xlsx")
    parser.add_argument(
        "--out",
        type=str,
        default="examples/tabular/outputs_project_multiclass_mlp_transformer",
    )
    parser.add_argument(
        "--sheets",
        type=str,
        default="冠状动脉粥样硬化性心脏病,急性非ST段抬高型心肌梗死,冠状动脉粥样硬化",
        help="Comma-separated exact sheet names for multiclass training; ignored when label_column is set",
    )
    parser.add_argument(
        "--label_column",
        type=str,
        default="",
        help="Enable single-sheet multiclass mode by specifying label column name",
    )
    parser.add_argument(
        "--data_sheet",
        type=str,
        default="",
        help="Sheet name for single-sheet mode; defaults to first sheet when empty",
    )
    parser.add_argument(
        "--label_values",
        type=str,
        default="",
        help="Optional comma-separated labels to keep in single-sheet mode",
    )
    parser.add_argument(
        "--project_name",
        type=str,
        default="",
        help="Optional project title used in report markdown",
    )
    parser.add_argument(
        "--exclude_keywords",
        type=str,
        default="姓名,名字,性别,年龄,身高,体重,Name,Gender,Age,Height,Weight",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--mlp_hidden", type=int, default=384)
    parser.add_argument("--mlp_dropout", type=float, default=0.2)
    parser.add_argument("--tr_d_model", type=int, default=96)
    parser.add_argument("--tr_nhead", type=int, default=4)
    parser.add_argument("--tr_layers", type=int, default=3)
    parser.add_argument("--tr_ffn", type=int, default=192)
    parser.add_argument("--tr_dropout", type=float, default=0.05)
    parser.add_argument(
        "--lightweight_profile",
        type=str,
        default="off",
        choices=["off", "balanced", "aggressive"],
        help="Optional profile to reduce model size and feature width.",
    )
    parser.add_argument("--enable_lightweight_eval", type=int, default=1, choices=[0, 1])
    parser.add_argument("--enable_quantized_eval", type=int, default=1, choices=[0, 1])
    parser.add_argument("--strategy_profile", type=str, default="unified", choices=["unified", "mixed_best"])
    parser.add_argument("--selection_metric", type=str, default="macro_f1", choices=["accuracy", "macro_f1"])
    parser.add_argument("--enable_imbalance_validation", type=int, default=1, choices=[0, 1])
    parser.add_argument("--enable_prior_adjustment_tuning", type=int, default=1, choices=[0, 1])
    parser.add_argument(
        "--prior_adjustment_metric",
        type=str,
        default="macro_f1",
        choices=["accuracy", "macro_f1"],
    )
    parser.add_argument(
        "--prior_adjustment_tau_grid",
        type=str,
        default="0.0,0.2,0.4,0.6,0.8,1.0",
        help="Comma-separated tau grid for prior-adjustment probability calibration.",
    )
    parser.add_argument(
        "--worst_class_recall_floor",
        type=float,
        default=0.0,
        help="Penalty floor for worst-class recall during tau tuning.",
    )
    parser.add_argument("--importance_metric", type=str, default="macro_f1", choices=["accuracy", "macro_f1"])
    parser.add_argument("--perm_repeats", type=int, default=2)
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument(
        "--ovr_metric",
        type=str,
        default="roc_auc",
        choices=["roc_auc", "average_precision", "f1"],
        help="one-vs-rest importance metric",
    )
    parser.add_argument("--clinical_topk", type=int, default=10)
    parser.add_argument("--disease_topk", type=int, default=5)
    parser.add_argument("--downweight_noise_features", type=int, default=1, choices=[0, 1])
    parser.add_argument("--use_class_weight", type=int, default=1, choices=[0, 1])
    parser.add_argument("--use_weighted_sampler", type=int, default=1, choices=[0, 1])
    parser.add_argument(
        "--imbalance_reference",
        type=str,
        default="pre_augmentation",
        choices=["pre_augmentation", "post_augmentation"],
    )
    parser.add_argument("--class_weight_power", type=float, default=1.0)
    parser.add_argument("--sampler_weight_power", type=float, default=0.75)
    parser.add_argument("--class_weight_min_clip", type=float, default=0.25)
    parser.add_argument("--class_weight_max_clip", type=float, default=4.0)
    parser.add_argument("--enable_label_noise_filter", type=int, default=1, choices=[0, 1])
    parser.add_argument("--noise_prob_threshold", type=float, default=0.20)
    parser.add_argument("--noise_cv_folds", type=int, default=5)
    parser.add_argument("--noise_max_drop_ratio", type=float, default=0.08)
    parser.add_argument("--enable_bootstrap_augmentation", type=int, default=1, choices=[0, 1])
    parser.add_argument("--augment_target_ratio", type=float, default=1.0)
    parser.add_argument("--augment_noise_std_ratio", type=float, default=0.01)
    parser.add_argument("--enable_distillation", type=int, default=0, choices=[0, 1])
    parser.add_argument(
        "--distill_target",
        type=str,
        default="both",
        choices=["both", "mlp", "transformer"],
        help="Distillation target when enable_distillation=1",
    )
    parser.add_argument("--distill_hard_weight", type=float, default=0.6)
    parser.add_argument("--distill_temperature", type=float, default=2.5)
    parser.add_argument("--teacher_depth", type=int, default=8)
    parser.add_argument("--teacher_lr", type=float, default=0.05)
    parser.add_argument("--teacher_l2", type=float, default=5.0)
    parser.add_argument("--teacher_iterations", type=int, default=2000)
    parser.add_argument("--enable_user_rule_cleaning", type=int, default=1, choices=[0, 1])
    parser.add_argument("--enable_model_specific_cleaning", type=int, default=1, choices=[0, 1])
    parser.add_argument("--enable_missing_indicators", type=int, default=1, choices=[0, 1])
    parser.add_argument("--missing_indicator_threshold", type=float, default=0.20)
    parser.add_argument("--enable_feature_selection", type=int, default=1, choices=[0, 1])
    parser.add_argument("--feature_select_max_features", type=int, default=120)
    parser.add_argument("--feature_select_min_features", type=int, default=40)
    parser.add_argument("--neighbor_window", type=int, default=4)
    parser.add_argument("--cmp_random_max", type=float, default=100.0)
    parser.add_argument("--numeric_like_threshold", type=float, default=0.80)
    parser.add_argument("--mlp_numeric_like_threshold", type=float, default=1.00)
    parser.add_argument("--tr_numeric_like_threshold", type=float, default=0.80)
    parser.add_argument(
        "--report_mode",
        type=str,
        default="off",
        choices=["off", "on"],
        help="on=在summary/report中写入可发表口径与风险提示; off=关闭",
    )
    parser.add_argument(
        "--leakage_guard_mode",
        type=str,
        default="strict",
        choices=["strict", "balanced", "off"],
        help="strict=强过滤(去除含诊断/病种/疾病字段); balanced=仅去除显式标签/结论字段; off=关闭泄漏字段名过滤",
    )
    parser.add_argument("--refresh_outputs_index", type=int, default=1, choices=[0, 1])
    parser.add_argument("--outputs_prefix", type=str, default="outputs_project_multiclass")
    args = parser.parse_args()

    smoke_run = bool(int(args.epochs) <= 3)
    if smoke_run:
        print(
            "[WARN] epochs<=3: 当前属于测速/冒烟配置，指标不建议用于正式性能结论。"
        )

    lightweight_profile_info = apply_lightweight_profile(args)

    if args.tr_d_model % args.tr_nhead != 0:
        raise ValueError(f"tr_d_model={args.tr_d_model} must be divisible by tr_nhead={args.tr_nhead}")

    set_seed(args.seed)
    excel_path = Path(args.excel).expanduser().resolve()
    if not excel_path.exists():
        raise FileNotFoundError(f"Excel not found: {excel_path}")

    out_dir = Path(args.out)
    safe_mkdir(out_dir)

    selected_sheets = parse_csv_arg(args.sheets)
    label_column = str(args.label_column).strip()
    data_sheet = str(args.data_sheet).strip()
    requested_label_values = parse_csv_arg(args.label_values)
    dataset_mode = "multi_sheet"
    data_sheet_used = ""

    if label_column:
        dataset_mode = "single_sheet_label_column"
        selected_sheets = []
        df_all, label_names, data_sheet_used = load_single_sheet_with_label_column(
            excel_path=excel_path,
            data_sheet=data_sheet,
            label_column=label_column,
            label_values=requested_label_values,
        )
    else:
        if len(selected_sheets) < 2:
            raise ValueError("Need at least two sheets for multiclass classification")
        df_all = load_selected_sheets(excel_path, selected_sheets)
        label_names = list(selected_sheets)

    label_to_id = {n: i for i, n in enumerate(label_names)}
    y = df_all["__label__"].map(label_to_id).astype(int).to_numpy()

    X_df = df_all.drop(columns=["__label__"])
    exclude_keywords = parse_csv_arg(args.exclude_keywords)
    X_df, dropped_by_request = drop_columns_by_keywords(X_df, exclude_keywords)
    X_df, dropped_auto = drop_leaky_or_useless_columns(
        X_df,
        leakage_guard_mode=args.leakage_guard_mode,
    )

    idx = np.arange(len(X_df))
    idx_train, idx_tmp, y_train, y_tmp = train_test_split(
        idx,
        y,
        test_size=0.30,
        random_state=args.seed,
        stratify=y,
    )
    idx_val, idx_test, y_val, y_test = train_test_split(
        idx_tmp,
        y_tmp,
        test_size=0.50,
        random_state=args.seed,
        stratify=y_tmp,
    )

    enable_label_noise_filter = bool(args.enable_label_noise_filter)
    enable_bootstrap_augmentation = bool(args.enable_bootstrap_augmentation)
    use_class_weight = bool(args.use_class_weight)
    use_weighted_sampler = bool(args.use_weighted_sampler)
    enable_distillation = bool(args.enable_distillation)
    enable_distillation_mlp = enable_distillation and args.distill_target in ("both", "mlp")
    enable_distillation_tr = enable_distillation and args.distill_target in ("both", "transformer")
    enable_user_rule_cleaning = bool(args.enable_user_rule_cleaning)
    enable_model_specific_cleaning = bool(args.enable_model_specific_cleaning)
    enable_missing_indicators = bool(args.enable_missing_indicators)
    enable_feature_selection = bool(args.enable_feature_selection)
    downweight_noise_features = bool(args.downweight_noise_features)
    enable_imbalance_validation = bool(args.enable_imbalance_validation)
    enable_prior_adjustment_tuning = bool(args.enable_prior_adjustment_tuning)
    enable_lightweight_eval = bool(args.enable_lightweight_eval)
    enable_quantized_eval = bool(args.enable_quantized_eval)
    prior_adjustment_tau_grid = parse_float_csv_arg(args.prior_adjustment_tau_grid)
    if not prior_adjustment_tau_grid:
        prior_adjustment_tau_grid = [0.0]

    strategy_by_model = build_model_strategy_settings(args)
    mlp_strategy = strategy_by_model["mlp"]
    tr_strategy = strategy_by_model["transformer"]

    mlp_clean_threshold = float(args.mlp_numeric_like_threshold)
    tr_clean_threshold = float(args.tr_numeric_like_threshold)
    if not enable_model_specific_cleaning:
        mlp_clean_threshold = float(args.numeric_like_threshold)
        tr_clean_threshold = float(args.numeric_like_threshold)

    mlp_branch = build_training_branch(
        branch_name="mlp",
        X_base_df=X_df,
        idx_train=idx_train,
        idx_val=idx_val,
        idx_test=idx_test,
        y_train_base=y_train,
        seed=args.seed,
        enable_user_rule_cleaning=enable_user_rule_cleaning,
        neighbor_window=args.neighbor_window,
        cmp_random_max=args.cmp_random_max,
        numeric_like_threshold=mlp_clean_threshold,
        enable_missing_indicators=bool(mlp_strategy["enable_missing_indicators"]),
        missing_indicator_threshold=float(mlp_strategy["missing_indicator_threshold"]),
        enable_label_noise_filter=enable_label_noise_filter,
        noise_prob_threshold=args.noise_prob_threshold,
        noise_cv_folds=args.noise_cv_folds,
        noise_max_drop_ratio=args.noise_max_drop_ratio,
        enable_bootstrap_augmentation=enable_bootstrap_augmentation,
        augment_target_ratio=args.augment_target_ratio,
        augment_noise_std_ratio=args.augment_noise_std_ratio,
        enable_feature_selection=bool(mlp_strategy["enable_feature_selection"]),
        feature_select_max_features=int(mlp_strategy["feature_select_max_features"]),
        feature_select_min_features=int(mlp_strategy["feature_select_min_features"]),
    )

    tr_branch = build_training_branch(
        branch_name="transformer",
        X_base_df=X_df,
        idx_train=idx_train,
        idx_val=idx_val,
        idx_test=idx_test,
        y_train_base=y_train,
        seed=args.seed,
        enable_user_rule_cleaning=enable_user_rule_cleaning,
        neighbor_window=args.neighbor_window,
        cmp_random_max=args.cmp_random_max,
        numeric_like_threshold=tr_clean_threshold,
        enable_missing_indicators=bool(tr_strategy["enable_missing_indicators"]),
        missing_indicator_threshold=float(tr_strategy["missing_indicator_threshold"]),
        enable_label_noise_filter=enable_label_noise_filter,
        noise_prob_threshold=args.noise_prob_threshold,
        noise_cv_folds=args.noise_cv_folds,
        noise_max_drop_ratio=args.noise_max_drop_ratio,
        enable_bootstrap_augmentation=enable_bootstrap_augmentation,
        augment_target_ratio=args.augment_target_ratio,
        augment_noise_std_ratio=args.augment_noise_std_ratio,
        enable_feature_selection=bool(tr_strategy["enable_feature_selection"]),
        feature_select_max_features=int(tr_strategy["feature_select_max_features"]),
        feature_select_min_features=int(tr_strategy["feature_select_min_features"]),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_classes = len(label_names)

    pre_mlp = build_preprocessor(mlp_branch.numeric_cols, mlp_branch.categorical_cols)
    X_train_mlp = pre_mlp.fit_transform(mlp_branch.X_train_df)
    X_val_mlp = pre_mlp.transform(mlp_branch.X_val_df)
    X_test_mlp = pre_mlp.transform(mlp_branch.X_test_df)

    X_train_mlp_dense = X_train_mlp.toarray() if hasattr(X_train_mlp, "toarray") else np.asarray(X_train_mlp)
    X_val_mlp_dense = X_val_mlp.toarray() if hasattr(X_val_mlp, "toarray") else np.asarray(X_val_mlp)
    X_test_mlp_dense = X_test_mlp.toarray() if hasattr(X_test_mlp, "toarray") else np.asarray(X_test_mlp)

    distillation_info_mlp = {
        "enabled": bool(enable_distillation_mlp),
        "applied": False,
    }
    teacher_train_prob_mlp = None
    teacher_val_prob_mlp = None
    if enable_distillation_mlp:
        (
            teacher_train_prob_mlp,
            teacher_val_prob_mlp,
            _,
            distillation_info_mlp,
        ) = fit_teacher_catboost(
            X_train_df=mlp_branch.X_train_df,
            y_train=mlp_branch.y_train,
            X_val_df=mlp_branch.X_val_df,
            y_val=y_val,
            X_test_df=mlp_branch.X_test_df,
            y_test=y_test,
            categorical_cols=mlp_branch.categorical_cols,
            seed=args.seed,
            depth=args.teacher_depth,
            learning_rate=args.teacher_lr,
            l2_leaf_reg=args.teacher_l2,
            iterations=args.teacher_iterations,
        )

    distill_hard_weight_mlp = float(args.distill_hard_weight) if teacher_train_prob_mlp is not None else 1.0
    distill_temperature_mlp = float(args.distill_temperature) if teacher_train_prob_mlp is not None else 1.0

    mlp = MLP(
        in_dim=X_train_mlp_dense.shape[1],
        num_classes=n_classes,
        hidden=args.mlp_hidden,
        dropout=args.mlp_dropout,
    ).to(device)
    opt = torch.optim.AdamW(mlp.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    imbalance_y_mlp = (
        mlp_branch.y_train_for_imbalance
        if str(mlp_strategy["imbalance_reference"]) == "pre_augmentation"
        else mlp_branch.y_train
    )
    class_weights_mlp = compute_balanced_class_weights(
        imbalance_y_mlp,
        n_classes,
        power=args.class_weight_power,
        min_clip=float(mlp_strategy["class_weight_min_clip"]),
        max_clip=float(mlp_strategy["class_weight_max_clip"]),
    )
    sampler_weights_mlp = compute_balanced_class_weights(
        imbalance_y_mlp,
        n_classes,
        power=float(mlp_strategy["sampler_weight_power"]),
        min_clip=float(mlp_strategy["class_weight_min_clip"]),
        max_clip=float(mlp_strategy["class_weight_max_clip"]),
    )
    imbalance_counts_mlp = np.bincount(imbalance_y_mlp, minlength=n_classes).astype(int).tolist()
    class_priors_mlp = np.asarray(imbalance_counts_mlp, dtype=np.float64)
    class_priors_mlp = class_priors_mlp / max(float(class_priors_mlp.sum()), 1.0)
    train_counts_mlp = np.bincount(mlp_branch.y_train, minlength=n_classes).astype(int).tolist()
    class_weight_t_mlp = torch.from_numpy(class_weights_mlp).float().to(device)
    mlp_loss_fn = nn.CrossEntropyLoss(weight=class_weight_t_mlp) if use_class_weight else nn.CrossEntropyLoss()

    sampler_mlp = build_weighted_sampler(mlp_branch.y_train, sampler_weights_mlp) if use_weighted_sampler else None
    train_loader = DataLoader(
        SparseMLPDataset(X_train_mlp_dense, mlp_branch.y_train, teacher_prob=teacher_train_prob_mlp),
        batch_size=args.batch,
        shuffle=(sampler_mlp is None),
        sampler=sampler_mlp,
    )

    best_val = -1.0
    best_state = None
    bad = 0
    for _ in range(args.epochs):
        _ = train_epoch(
            mlp,
            train_loader,
            opt,
            device,
            loss_fn=mlp_loss_fn,
            distill_hard_weight=distill_hard_weight_mlp,
            distill_temperature=distill_temperature_mlp,
        )
        val_prob = predict_proba_mlp_dense(mlp, X_val_mlp_dense, device)
        val_metric = score_from_metrics(compute_metrics(y_val, val_prob), str(mlp_strategy["selection_metric"]))
        if val_metric > best_val:
            best_val = val_metric
            best_state = {k: v.detach().cpu().clone() for k, v in mlp.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= args.patience:
                break

    if best_state is not None:
        mlp.load_state_dict(best_state)

    mlp_val_prob_final = predict_proba_mlp_dense(mlp, X_val_mlp_dense, device)
    mlp_prob_raw = predict_proba_mlp_dense(mlp, X_test_mlp_dense, device)
    mlp_prior_adjustment = {
        "enabled": bool(enable_prior_adjustment_tuning),
        "metric": str(args.prior_adjustment_metric),
        "tau_grid": [float(x) for x in prior_adjustment_tau_grid],
        "worst_class_recall_floor": float(args.worst_class_recall_floor),
        "best_tau": 0.0,
        "history": [],
    }
    if enable_prior_adjustment_tuning:
        best_tau_mlp, _, _, tau_history_mlp = tune_prior_adjustment_tau(
            y_true=y_val,
            y_prob_val=mlp_val_prob_final,
            class_priors=class_priors_mlp,
            label_names=label_names,
            metric_name=str(args.prior_adjustment_metric),
            tau_grid=prior_adjustment_tau_grid,
            worst_class_recall_floor=float(args.worst_class_recall_floor),
        )
        mlp_prior_adjustment["best_tau"] = float(best_tau_mlp)
        mlp_prior_adjustment["history"] = tau_history_mlp
        mlp_prob = apply_prior_adjustment(mlp_prob_raw, class_priors_mlp, best_tau_mlp)
    else:
        mlp_prob = mlp_prob_raw

    mlp_metrics_raw = compute_metrics(y_test, mlp_prob_raw)
    mlp_metrics = compute_metrics(y_test, mlp_prob)
    mlp_imbalance_diag = compute_imbalance_diagnostics(y_test, mlp_prob, label_names)
    y_pred_mlp = mlp_prob.argmax(axis=1)

    distillation_info_tr = {
        "enabled": bool(enable_distillation_tr),
        "applied": False,
    }
    teacher_train_prob_tr = None
    teacher_val_prob_tr = None
    if enable_distillation_tr:
        (
            teacher_train_prob_tr,
            teacher_val_prob_tr,
            _,
            distillation_info_tr,
        ) = fit_teacher_catboost(
            X_train_df=tr_branch.X_train_df,
            y_train=tr_branch.y_train,
            X_val_df=tr_branch.X_val_df,
            y_val=y_val,
            X_test_df=tr_branch.X_test_df,
            y_test=y_test,
            categorical_cols=tr_branch.categorical_cols,
            seed=args.seed,
            depth=args.teacher_depth,
            learning_rate=args.teacher_lr,
            l2_leaf_reg=args.teacher_l2,
            iterations=args.teacher_iterations,
        )

    distill_hard_weight_tr = float(args.distill_hard_weight) if teacher_train_prob_tr is not None else 1.0
    distill_temperature_tr = float(args.distill_temperature) if teacher_train_prob_tr is not None else 1.0

    cat_maps: dict[str, dict[str, int]] = {}
    for col in tr_branch.categorical_cols:
        uniq = pd.unique(tr_branch.X_train_df[col].astype(str))
        cat_maps[col] = {v: i + 1 for i, v in enumerate(uniq)}

    if tr_branch.numeric_cols:
        num_imputer = SimpleImputer(strategy="median")
        num_scaler = StandardScaler()
        train_num = num_imputer.fit_transform(tr_branch.X_train_df[tr_branch.numeric_cols])
        _ = num_scaler.fit_transform(train_num)
    else:
        num_imputer = None
        num_scaler = None

    def encode_df(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        cat_arr = np.zeros((len(df), len(tr_branch.categorical_cols)), dtype=np.int64)
        for j, col in enumerate(tr_branch.categorical_cols):
            mapping = cat_maps[col]
            cat_arr[:, j] = [mapping.get(v, 0) for v in df[col].astype(str)]

        if tr_branch.numeric_cols:
            num = num_imputer.transform(df[tr_branch.numeric_cols])
            num = num_scaler.transform(num).astype(np.float32)
        else:
            num = np.zeros((len(df), 0), dtype=np.float32)
        return cat_arr, num

    cat_train, num_train = encode_df(tr_branch.X_train_df)
    cat_val, num_val = encode_df(tr_branch.X_val_df)
    cat_test, num_test = encode_df(tr_branch.X_test_df)

    tr_cfg = TabTransformerConfig(
        d_model=args.tr_d_model,
        nhead=args.tr_nhead,
        num_layers=args.tr_layers,
        dim_feedforward=args.tr_ffn,
        dropout=args.tr_dropout,
    )

    cat_cardinalities = [max(m.values(), default=0) for m in cat_maps.values()]
    transformer = TabTransformer(
        cat_cardinalities=cat_cardinalities,
        num_numeric=len(tr_branch.numeric_cols),
        num_classes=n_classes,
        cfg=tr_cfg,
    ).to(device)

    opt2 = torch.optim.AdamW(transformer.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    imbalance_y_tr = (
        tr_branch.y_train_for_imbalance
        if str(tr_strategy["imbalance_reference"]) == "pre_augmentation"
        else tr_branch.y_train
    )
    class_weights_tr = compute_balanced_class_weights(
        imbalance_y_tr,
        n_classes,
        power=args.class_weight_power,
        min_clip=float(tr_strategy["class_weight_min_clip"]),
        max_clip=float(tr_strategy["class_weight_max_clip"]),
    )
    sampler_weights_tr = compute_balanced_class_weights(
        imbalance_y_tr,
        n_classes,
        power=float(tr_strategy["sampler_weight_power"]),
        min_clip=float(tr_strategy["class_weight_min_clip"]),
        max_clip=float(tr_strategy["class_weight_max_clip"]),
    )
    imbalance_counts_tr = np.bincount(imbalance_y_tr, minlength=n_classes).astype(int).tolist()
    class_priors_tr = np.asarray(imbalance_counts_tr, dtype=np.float64)
    class_priors_tr = class_priors_tr / max(float(class_priors_tr.sum()), 1.0)
    train_counts_tr = np.bincount(tr_branch.y_train, minlength=n_classes).astype(int).tolist()
    class_weight_t_tr = torch.from_numpy(class_weights_tr).float().to(device)
    tr_loss_fn = nn.CrossEntropyLoss(weight=class_weight_t_tr) if use_class_weight else nn.CrossEntropyLoss()
    sampler_tr = build_weighted_sampler(tr_branch.y_train, sampler_weights_tr) if use_weighted_sampler else None
    train_loader2 = DataLoader(
        TabTensorDataset(cat_train, num_train, tr_branch.y_train, teacher_prob=teacher_train_prob_tr),
        batch_size=args.batch,
        shuffle=(sampler_tr is None),
        sampler=sampler_tr,
    )

    best_val2 = -1.0
    best_state2 = None
    bad2 = 0
    for _ in range(args.epochs):
        _ = train_epoch(
            transformer,
            train_loader2,
            opt2,
            device,
            loss_fn=tr_loss_fn,
            distill_hard_weight=distill_hard_weight_tr,
            distill_temperature=distill_temperature_tr,
        )
        val_prob2 = predict_proba_transformer_arrays(transformer, cat_val, num_val, device)
        val_metric2 = score_from_metrics(compute_metrics(y_val, val_prob2), str(tr_strategy["selection_metric"]))
        if val_metric2 > best_val2:
            best_val2 = val_metric2
            best_state2 = {k: v.detach().cpu().clone() for k, v in transformer.state_dict().items()}
            bad2 = 0
        else:
            bad2 += 1
            if bad2 >= args.patience:
                break

    if best_state2 is not None:
        transformer.load_state_dict(best_state2)

    tr_val_prob_final = predict_proba_transformer_arrays(transformer, cat_val, num_val, device)
    tr_prob_raw = predict_proba_transformer_arrays(transformer, cat_test, num_test, device)
    tr_prior_adjustment = {
        "enabled": bool(enable_prior_adjustment_tuning),
        "metric": str(args.prior_adjustment_metric),
        "tau_grid": [float(x) for x in prior_adjustment_tau_grid],
        "worst_class_recall_floor": float(args.worst_class_recall_floor),
        "best_tau": 0.0,
        "history": [],
    }
    if enable_prior_adjustment_tuning:
        best_tau_tr, _, _, tau_history_tr = tune_prior_adjustment_tau(
            y_true=y_val,
            y_prob_val=tr_val_prob_final,
            class_priors=class_priors_tr,
            label_names=label_names,
            metric_name=str(args.prior_adjustment_metric),
            tau_grid=prior_adjustment_tau_grid,
            worst_class_recall_floor=float(args.worst_class_recall_floor),
        )
        tr_prior_adjustment["best_tau"] = float(best_tau_tr)
        tr_prior_adjustment["history"] = tau_history_tr
        tr_prob = apply_prior_adjustment(tr_prob_raw, class_priors_tr, best_tau_tr)
    else:
        tr_prob = tr_prob_raw

    tr_metrics_raw = compute_metrics(y_test, tr_prob_raw)
    tr_metrics = compute_metrics(y_test, tr_prob)
    tr_imbalance_diag = compute_imbalance_diagnostics(y_test, tr_prob, label_names)
    y_pred_tr = tr_prob.argmax(axis=1)

    def mlp_predict_from_df(df_in: pd.DataFrame) -> np.ndarray:
        df_tmp = coerce_categorical_to_str(df_in.copy(), mlp_branch.categorical_cols)
        Xt = pre_mlp.transform(df_tmp)
        Xd = Xt.toarray() if hasattr(Xt, "toarray") else np.asarray(Xt)
        return predict_proba_mlp_dense(mlp, Xd, device)

    def tr_predict_from_df(df_in: pd.DataFrame) -> np.ndarray:
        df_tmp = coerce_categorical_to_str(df_in.copy(), tr_branch.categorical_cols)
        cat_tmp, num_tmp = encode_df(df_tmp)
        return predict_proba_transformer_arrays(transformer, cat_tmp, num_tmp, device)

    imp_mlp = permutation_importance(
        mlp_predict_from_df,
        mlp_branch.X_test_df,
        y_test,
        metric_name=args.importance_metric,
        n_repeats=args.perm_repeats,
        seed=args.seed,
    )
    imp_tr = permutation_importance(
        tr_predict_from_df,
        tr_branch.X_test_df,
        y_test,
        metric_name=args.importance_metric,
        n_repeats=args.perm_repeats,
        seed=args.seed + 11,
    )

    ovr_mlp_frames: list[pd.DataFrame] = []
    ovr_tr_frames: list[pd.DataFrame] = []
    disease_intersections: list[pd.DataFrame] = []

    for class_id, class_name in enumerate(label_names):
        cls_imp_mlp = permutation_importance_ovr(
            mlp_predict_from_df,
            mlp_branch.X_test_df,
            y_test,
            class_id=class_id,
            metric_name=args.ovr_metric,
            n_repeats=args.perm_repeats,
            seed=args.seed + 101 + class_id,
        )
        cls_imp_tr = permutation_importance_ovr(
            tr_predict_from_df,
            tr_branch.X_test_df,
            y_test,
            class_id=class_id,
            metric_name=args.ovr_metric,
            n_repeats=args.perm_repeats,
            seed=args.seed + 211 + class_id,
        )
        cls_imp_mlp.insert(1, "class_name", class_name)
        cls_imp_tr.insert(1, "class_name", class_name)
        ovr_mlp_frames.append(cls_imp_mlp)
        ovr_tr_frames.append(cls_imp_tr)

        cls_intersection = build_joint_intersection_table(
            cls_imp_mlp,
            cls_imp_tr,
            apply_clinical_penalty=downweight_noise_features,
        )
        if not cls_intersection.empty:
            cls_intersection.insert(0, "class_name", class_name)
            cls_intersection.insert(0, "class_id", class_id)
            disease_intersections.append(cls_intersection)

    imp_ovr_mlp = pd.concat(ovr_mlp_frames, axis=0, ignore_index=True)
    imp_ovr_tr = pd.concat(ovr_tr_frames, axis=0, ignore_index=True)

    global_intersection = build_joint_intersection_table(
        imp_mlp,
        imp_tr,
        apply_clinical_penalty=downweight_noise_features,
    )
    disease_intersection_df = (
        pd.concat(disease_intersections, axis=0, ignore_index=True)
        if disease_intersections
        else pd.DataFrame(
            columns=[
                "class_id",
                "class_name",
                "feature",
                "mlp_importance",
                "mlp_prob",
                "transformer_importance",
                "transformer_prob",
                "joint_prob",
                "joint_importance_geo",
                "clinical_penalty",
                "penalty_reasons",
                "joint_prob_adjusted",
                "joint_prob_adjusted_norm",
            ]
        )
    )

    inference_df = build_inference_table(
        idx_test=idx_test,
        y_test=y_test,
        label_names=label_names,
        mlp_prob=mlp_prob,
        tr_prob=tr_prob,
    )
    ensemble_prob = 0.5 * (mlp_prob + tr_prob)
    joint_raw = mlp_prob * tr_prob
    joint_prob = joint_raw / np.maximum(joint_raw.sum(axis=1, keepdims=True), EPS)

    y_pred_joint = inference_df["joint_pred"].to_numpy(dtype=np.int64)
    y_pred_ensemble = inference_df["ensemble_pred"].to_numpy(dtype=np.int64)
    joint_metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred_joint)),
        "macro_precision": float(precision_score(y_test, y_pred_joint, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_test, y_pred_joint, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_test, y_pred_joint, average="macro", zero_division=0)),
    }
    ensemble_metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred_ensemble)),
        "macro_precision": float(precision_score(y_test, y_pred_ensemble, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_test, y_pred_ensemble, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_test, y_pred_ensemble, average="macro", zero_division=0)),
    }

    if enable_imbalance_validation:
        ensemble_imbalance_diag = compute_imbalance_diagnostics(y_test, ensemble_prob, label_names)
        joint_imbalance_diag = compute_imbalance_diagnostics(y_test, joint_prob, label_names)
    else:
        ensemble_imbalance_diag = {
            "valid": False,
            "reason": "disabled",
            "per_class": [],
            "worst_class_recall": 0.0,
            "worst_class_f1": 0.0,
            "macro_pr_auc": math.nan,
        }
        joint_imbalance_diag = {
            "valid": False,
            "reason": "disabled",
            "per_class": [],
            "worst_class_recall": 0.0,
            "worst_class_f1": 0.0,
            "macro_pr_auc": math.nan,
        }

    lightweight_summary = {
        "enabled": bool(enable_lightweight_eval),
        "quantized_eval_enabled": bool(enable_quantized_eval),
        "profile": lightweight_profile_info,
        "models": {
            "mlp": {
                "param_count": model_param_count(mlp),
                "fp32_state_dict_bytes": model_state_dict_size_bytes(mlp),
            },
            "transformer": {
                "param_count": model_param_count(transformer),
                "fp32_state_dict_bytes": model_state_dict_size_bytes(transformer),
            },
        },
    }
    if enable_lightweight_eval and enable_quantized_eval:
        try:
            mlp_q = quantize_dynamic_linear(copy.deepcopy(mlp).cpu().eval())
            mlp_q_prob = predict_proba_mlp_dense(mlp_q, X_test_mlp_dense, torch.device("cpu"))
            mlp_q_metrics = compute_metrics(y_test, mlp_q_prob)
            lightweight_summary["models"]["mlp"]["quantized"] = {
                "ok": True,
                "state_dict_bytes": model_state_dict_size_bytes(mlp_q),
                "size_reduction_ratio": float(
                    model_state_dict_size_bytes(mlp_q)
                    / max(model_state_dict_size_bytes(mlp), 1)
                ),
                "metrics": mlp_q_metrics,
            }
        except Exception as ex:
            lightweight_summary["models"]["mlp"]["quantized"] = {
                "ok": False,
                "error": str(ex),
            }

        try:
            tr_q = quantize_dynamic_linear(copy.deepcopy(transformer).cpu().eval())
            tr_q_prob = predict_proba_transformer_arrays(tr_q, cat_test, num_test, torch.device("cpu"))
            tr_q_metrics = compute_metrics(y_test, tr_q_prob)
            lightweight_summary["models"]["transformer"]["quantized"] = {
                "ok": True,
                "state_dict_bytes": model_state_dict_size_bytes(tr_q),
                "size_reduction_ratio": float(
                    model_state_dict_size_bytes(tr_q)
                    / max(model_state_dict_size_bytes(transformer), 1)
                ),
                "metrics": tr_q_metrics,
            }
        except Exception as ex:
            lightweight_summary["models"]["transformer"]["quantized"] = {
                "ok": False,
                "error": str(ex),
            }

    imp_mlp.to_csv(out_dir / "feature_importance_mlp.csv", index=False, encoding="utf-8")
    imp_tr.to_csv(out_dir / "feature_importance_transformer.csv", index=False, encoding="utf-8")
    imp_ovr_mlp.to_csv(out_dir / "feature_importance_mlp_ovr.csv", index=False, encoding="utf-8")
    imp_ovr_tr.to_csv(out_dir / "feature_importance_transformer_ovr.csv", index=False, encoding="utf-8")
    global_intersection.to_csv(out_dir / "feature_intersection_joint_global.csv", index=False, encoding="utf-8")
    disease_intersection_df.to_csv(out_dir / "feature_intersection_joint_by_disease.csv", index=False, encoding="utf-8")
    global_intersection.head(args.clinical_topk).to_csv(
        out_dir / "clinical_interpretable_top_global.csv",
        index=False,
        encoding="utf-8",
    )

    pd.DataFrame({"y_true": y_test, "y_pred": y_pred_mlp}).to_csv(
        out_dir / "predictions_mlp.csv", index=False, encoding="utf-8"
    )
    pd.DataFrame({"y_true": y_test, "y_pred": y_pred_tr}).to_csv(
        out_dir / "predictions_transformer.csv", index=False, encoding="utf-8"
    )
    inference_df.to_csv(out_dir / "unseen_multiclass_inference.csv", index=False, encoding="utf-8")

    direction_df = build_disease_feature_direction_table(
        disease_intersection_df=disease_intersection_df,
        label_names=label_names,
        y_test=y_test,
        topk=args.disease_topk,
        mlp_test_df=mlp_branch.X_test_df,
        tr_test_df=tr_branch.X_test_df,
    )
    direction_df.to_csv(out_dir / "clinical_feature_direction_by_disease.csv", index=False, encoding="utf-8")

    imbalance_bundle = {
        "enabled": bool(enable_imbalance_validation),
        "mlp": mlp_imbalance_diag,
        "transformer": tr_imbalance_diag,
        "ensemble": ensemble_imbalance_diag,
        "joint": joint_imbalance_diag,
    }
    (out_dir / "imbalance_diagnostics.json").write_text(
        json.dumps(imbalance_bundle, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    (out_dir / "model_lightweight_summary.json").write_text(
        json.dumps(lightweight_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    clinical_md = build_disease_clinical_markdown(
        disease_intersection_df=disease_intersection_df,
        label_names=label_names,
        topk=args.disease_topk,
        direction_df=direction_df,
    )
    (out_dir / "clinical_dossier_by_disease.md").write_text(clinical_md, encoding="utf-8")

    project_name = str(args.project_name).strip() or "心谱智析：多病种智能分型系统"

    class_distribution_idx = pd.Series(y).value_counts().sort_index().to_dict()
    class_distribution_by_label = {
        str(label_names[int(k)]): int(v)
        for k, v in class_distribution_idx.items()
        if int(k) < len(label_names)
    }
    report_annotation = build_report_mode_annotation(
        report_mode=args.report_mode,
        leakage_guard_mode=args.leakage_guard_mode,
        smoke_run=smoke_run,
    )

    summary = {
        "project_name": project_name,
        "excel": str(excel_path),
        "dataset_mode": dataset_mode,
        "data_sheet": data_sheet_used,
        "label_column": label_column,
        "label_values_requested": requested_label_values,
        "selected_sheets": selected_sheets,
        "label_names": label_names,
        "exclude_keywords": exclude_keywords,
        "rows_total": int(len(df_all)),
        "class_distribution": {int(k): int(v) for k, v in class_distribution_idx.items()},
        "class_distribution_by_label": class_distribution_by_label,
        "label_mapping": {name: int(i) for i, name in enumerate(label_names)},
        "run_config": {
            "seed": int(args.seed),
            "epochs": int(args.epochs),
            "patience": int(args.patience),
            "batch": int(args.batch),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "report_mode": str(args.report_mode),
            "leakage_guard_mode": str(args.leakage_guard_mode),
            "is_smoke_run": bool(smoke_run),
        },
        "dropped_columns_by_request": dropped_by_request,
        "dropped_columns_auto": dropped_auto,
        "n_features_used": {
            "mlp": int(mlp_branch.X_all_df.shape[1]),
            "transformer": int(tr_branch.X_all_df.shape[1]),
        },
        "n_numeric": {
            "mlp": int(len(mlp_branch.numeric_cols)),
            "transformer": int(len(tr_branch.numeric_cols)),
        },
        "n_categorical": {
            "mlp": int(len(mlp_branch.categorical_cols)),
            "transformer": int(len(tr_branch.categorical_cols)),
        },
        "data_cleaning": {
            "enabled": bool(enable_user_rule_cleaning),
            "model_specific_cleaning": bool(enable_model_specific_cleaning),
            "leakage_guard_mode": str(args.leakage_guard_mode),
            "enable_missing_indicators": bool(enable_missing_indicators),
            "missing_indicator_threshold": float(args.missing_indicator_threshold),
            "mlp_numeric_like_threshold": float(mlp_clean_threshold),
            "tr_numeric_like_threshold": float(tr_clean_threshold),
            "mlp": mlp_branch.cleaning_info,
            "transformer": tr_branch.cleaning_info,
        },
        "type_profile_before_cleaning": {
            "mlp": mlp_branch.type_profile_before,
            "transformer": tr_branch.type_profile_before,
        },
        "type_profile_after_cleaning": {
            "mlp": mlp_branch.type_profile_after,
            "transformer": tr_branch.type_profile_after,
        },
        "split": {
            "train": int(len(idx_train)),
            "val": int(len(idx_val)),
            "test": int(len(idx_test)),
        },
        "robust_training": {
            "strategy_profile": args.strategy_profile,
            "selection_metric": args.selection_metric,
            "selection_metric_per_model": {
                "mlp": str(mlp_strategy["selection_metric"]),
                "transformer": str(tr_strategy["selection_metric"]),
            },
            "imbalance_validation_enabled": bool(enable_imbalance_validation),
            "prior_adjustment_tuning": {
                "enabled": bool(enable_prior_adjustment_tuning),
                "metric": str(args.prior_adjustment_metric),
                "tau_grid": [float(x) for x in prior_adjustment_tau_grid],
                "worst_class_recall_floor": float(args.worst_class_recall_floor),
            },
            "lightweight_profile": lightweight_profile_info,
            "lightweight_eval_enabled": bool(enable_lightweight_eval),
            "quantized_eval_enabled": bool(enable_quantized_eval),
            "use_class_weight": bool(use_class_weight),
            "use_weighted_sampler": bool(use_weighted_sampler),
            "imbalance_reference": args.imbalance_reference,
            "imbalance_reference_per_model": {
                "mlp": str(mlp_strategy["imbalance_reference"]),
                "transformer": str(tr_strategy["imbalance_reference"]),
            },
            "class_weight_power": float(args.class_weight_power),
            "sampler_weight_power": float(args.sampler_weight_power),
            "sampler_weight_power_per_model": {
                "mlp": float(mlp_strategy["sampler_weight_power"]),
                "transformer": float(tr_strategy["sampler_weight_power"]),
            },
            "class_weight_clip": {
                "min": float(args.class_weight_min_clip),
                "max": float(args.class_weight_max_clip),
            },
            "class_weight_clip_per_model": {
                "mlp": {
                    "min": float(mlp_strategy["class_weight_min_clip"]),
                    "max": float(mlp_strategy["class_weight_max_clip"]),
                },
                "transformer": {
                    "min": float(tr_strategy["class_weight_min_clip"]),
                    "max": float(tr_strategy["class_weight_max_clip"]),
                },
            },
            "feature_selection": {
                "enabled": bool(enable_feature_selection),
                "max_features": int(args.feature_select_max_features),
                "min_features": int(args.feature_select_min_features),
            },
            "feature_selection_per_model": {
                "mlp": {
                    "enabled": bool(mlp_strategy["enable_feature_selection"]),
                    "max_features": int(mlp_strategy["feature_select_max_features"]),
                    "min_features": int(mlp_strategy["feature_select_min_features"]),
                },
                "transformer": {
                    "enabled": bool(tr_strategy["enable_feature_selection"]),
                    "max_features": int(tr_strategy["feature_select_max_features"]),
                    "min_features": int(tr_strategy["feature_select_min_features"]),
                },
            },
            "missing_indicators_per_model": {
                "mlp": {
                    "enabled": bool(mlp_strategy["enable_missing_indicators"]),
                    "threshold": float(mlp_strategy["missing_indicator_threshold"]),
                },
                "transformer": {
                    "enabled": bool(tr_strategy["enable_missing_indicators"]),
                    "threshold": float(tr_strategy["missing_indicator_threshold"]),
                },
            },
            "distill_target": args.distill_target,
            "per_model": {
                "mlp": {
                    "strategy": mlp_strategy,
                    "class_weights": [float(x) for x in class_weights_mlp.tolist()],
                    "sampler_class_weights": [float(x) for x in sampler_weights_mlp.tolist()],
                    "imbalance_reference_distribution": imbalance_counts_mlp,
                    "train_distribution_after_augmentation": train_counts_mlp,
                    "label_noise_filter": mlp_branch.noise_filter_info,
                    "bootstrap_augmentation": mlp_branch.augmentation_info,
                    "feature_selection": mlp_branch.feature_selection_info,
                    "prior_adjustment": mlp_prior_adjustment,
                    "metrics_raw_before_prior_adjustment": mlp_metrics_raw,
                    "imbalance_diagnostics": mlp_imbalance_diag,
                    "train_rows_effective": int(len(mlp_branch.y_train)),
                    "distillation": {
                        "enabled": bool(enable_distillation_mlp),
                        "applied": bool(distillation_info_mlp.get("applied", False)),
                        "hard_weight": float(distill_hard_weight_mlp),
                        "temperature": float(distill_temperature_mlp),
                        "teacher_info": distillation_info_mlp,
                        "teacher_val_prob_ready": bool(teacher_val_prob_mlp is not None),
                    },
                },
                "transformer": {
                    "strategy": tr_strategy,
                    "class_weights": [float(x) for x in class_weights_tr.tolist()],
                    "sampler_class_weights": [float(x) for x in sampler_weights_tr.tolist()],
                    "imbalance_reference_distribution": imbalance_counts_tr,
                    "train_distribution_after_augmentation": train_counts_tr,
                    "label_noise_filter": tr_branch.noise_filter_info,
                    "bootstrap_augmentation": tr_branch.augmentation_info,
                    "feature_selection": tr_branch.feature_selection_info,
                    "prior_adjustment": tr_prior_adjustment,
                    "metrics_raw_before_prior_adjustment": tr_metrics_raw,
                    "imbalance_diagnostics": tr_imbalance_diag,
                    "train_rows_effective": int(len(tr_branch.y_train)),
                    "distillation": {
                        "enabled": bool(enable_distillation_tr),
                        "applied": bool(distillation_info_tr.get("applied", False)),
                        "hard_weight": float(distill_hard_weight_tr),
                        "temperature": float(distill_temperature_tr),
                        "teacher_info": distillation_info_tr,
                        "teacher_val_prob_ready": bool(teacher_val_prob_tr is not None),
                    },
                },
            },
        },
        "mlp_raw": mlp_metrics_raw,
        "mlp": mlp_metrics,
        "transformer_raw": tr_metrics_raw,
        "transformer": tr_metrics,
        "ensemble": ensemble_metrics,
        "joint": joint_metrics,
        "imbalance_diagnostics": imbalance_bundle,
        "lightweight_summary": lightweight_summary,
        "selection_metric": (
            str(mlp_strategy["selection_metric"])
            if str(mlp_strategy["selection_metric"]) == str(tr_strategy["selection_metric"])
            else "per_model"
        ),
        "selection_metric_per_model": {
            "mlp": str(mlp_strategy["selection_metric"]),
            "transformer": str(tr_strategy["selection_metric"]),
        },
        "importance_metric": args.importance_metric,
        "ovr_metric": args.ovr_metric,
        "clinical_ranking": {
            "downweight_noise_features": bool(downweight_noise_features),
            "clinical_topk": int(args.clinical_topk),
            "disease_topk": int(args.disease_topk),
        },
        "clinical_directionality": {
            "method": "median_shift_over_std",
            "rows": int(len(direction_df)),
            "output_file": str(out_dir / "clinical_feature_direction_by_disease.csv"),
        },
        "mlp_top_features": imp_mlp.head(args.topk)["feature"].tolist(),
        "transformer_top_features": imp_tr.head(args.topk)["feature"].tolist(),
        "global_intersection_top_features": global_intersection.head(args.clinical_topk)["feature"].tolist(),
        "disease_top_features": {
            disease: disease_intersection_df[disease_intersection_df["class_name"] == disease]
            .head(args.disease_topk)["feature"]
            .tolist()
            for disease in label_names
        },
    }
    if report_annotation is not None:
        summary["report_annotation"] = report_annotation

    if bool(args.refresh_outputs_index):
        try:
            dynamic_index_path = refresh_outputs_dynamic_index(out_dir=out_dir, prefix=args.outputs_prefix)
            summary["dynamic_outputs_index"] = dynamic_index_path
        except Exception as ex:
            summary["dynamic_outputs_index_error"] = str(ex)

    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    md = []
    md.append(f"# {project_name}: {len(label_names)}类分类 (MLP vs Transformer)\n\n")
    if dataset_mode == "single_sheet_label_column":
        md.append("- 数据模式: 单Sheet标签列\n")
        md.append(f"- 数据Sheet: {data_sheet_used}\n")
        md.append(f"- 标签列: {label_column}\n")
        if requested_label_values:
            md.append(f"- 限定标签: {', '.join(requested_label_values)}\n")
        md.append(f"- 标签类别: {', '.join(label_names)}\n")
    else:
        md.append("- 数据模式: 多Sheet拼接（sheet名即标签）\n")
        md.append(f"- 标签类别: {', '.join(label_names)}\n")
    md.append(f"- 忽略关键词: {', '.join(exclude_keywords)}\n")
    md.append(f"- 泄漏字段防护模式: {args.leakage_guard_mode}\n")
    md.append(f"- 样本数: {len(df_all)}\n")
    if report_annotation is not None:
        md.append("\n## 报告口径提示\n")
        md.append(f"- 报告模式: {args.report_mode}\n")
        md.append(f"- 可发表判定: {report_annotation.get('publishability', '-')}; 分层: {report_annotation.get('tier', '-')}\n")
        md.append(f"- 风险等级: {report_annotation.get('risk_level', '-')}\n")
        for i, note in enumerate(report_annotation.get("risk_notes", []), start=1):
            md.append(f"  {i}. {note}\n")
        md.append(f"- 建议披露语句: {report_annotation.get('disclosure', '-')}\n")
    md.append(
        f"- 特征数(MLP): {mlp_branch.X_all_df.shape[1]} (数值={len(mlp_branch.numeric_cols)}, 类别={len(mlp_branch.categorical_cols)})\n"
    )
    md.append(
        f"- 特征数(Transformer): {tr_branch.X_all_df.shape[1]} (数值={len(tr_branch.numeric_cols)}, 类别={len(tr_branch.categorical_cols)})\n"
    )
    md.append("\n## 数据清洗（按用户规则）\n")
    md.append(f"- 启用清洗: {enable_user_rule_cleaning}; 启用分模型清洗: {enable_model_specific_cleaning}\n")
    md.append(f"- 策略配置档: {args.strategy_profile}\n")
    md.append(
        f"- numeric-like阈值: MLP={mlp_clean_threshold:.3f}, Transformer={tr_clean_threshold:.3f}; 邻居窗口={args.neighbor_window}\n"
    )
    md.append(
        f"- 缺失指示特征: MLP(enabled={mlp_strategy['enable_missing_indicators']}, threshold={float(mlp_strategy['missing_indicator_threshold']):.3f}); "
        f"Transformer(enabled={tr_strategy['enable_missing_indicators']}, threshold={float(tr_strategy['missing_indicator_threshold']):.3f})\n"
    )
    md.append(
        f"- 轻量特征筛选: MLP(enabled={mlp_strategy['enable_feature_selection']}, max_features={int(mlp_strategy['feature_select_max_features'])}, min_features={int(mlp_strategy['feature_select_min_features'])}); "
        f"Transformer(enabled={tr_strategy['enable_feature_selection']}, max_features={int(tr_strategy['feature_select_max_features'])}, min_features={int(tr_strategy['feature_select_min_features'])})\n"
    )
    md.append("- 比较符采样: 使用固定随机序列（按seed+列名+行号+token确定），避免每次运行漂移\n")

    mlp_clean = mlp_branch.cleaning_info
    tr_clean = tr_branch.cleaning_info
    md.append(
        f"- MLP清洗: 阴性->0 {mlp_clean.get('negative_to_zero_total', 0)} 条；"
        f"比较符替换 {mlp_clean.get('cmp_to_random_total', 0)} 条；"
        f"文本定向填补 {mlp_clean.get('directional_fill_total', 0)} 条\n"
    )
    md.append(
        f"- Transformer清洗: 阴性->0 {tr_clean.get('negative_to_zero_total', 0)} 条；"
        f"比较符替换 {tr_clean.get('cmp_to_random_total', 0)} 条；"
        f"文本定向填补 {tr_clean.get('directional_fill_total', 0)} 条\n"
    )
    md.append(
        f"- MLP空值填补: 邻居均值 {mlp_clean.get('missing_filled_by_neighbor_total', 0)} 条；"
        f"后备均值 {mlp_clean.get('missing_filled_by_fallback_total', 0)} 条\n"
    )
    md.append(
        f"- Transformer空值填补: 邻居均值 {tr_clean.get('missing_filled_by_neighbor_total', 0)} 条；"
        f"后备均值 {tr_clean.get('missing_filled_by_fallback_total', 0)} 条\n"
    )
    md.append(
        f"- MLP缺失指示列: {mlp_clean.get('missing_indicators', {}).get('added_count', 0)} 列；"
        f"Transformer缺失指示列: {tr_clean.get('missing_indicators', {}).get('added_count', 0)} 列\n"
    )
    md.append(
        f"- MLP特征筛选: {mlp_branch.feature_selection_info.get('n_features_before', mlp_branch.X_all_df.shape[1])} -> {mlp_branch.feature_selection_info.get('n_features_after', mlp_branch.X_all_df.shape[1])}; "
        f"Transformer特征筛选: {tr_branch.feature_selection_info.get('n_features_before', tr_branch.X_all_df.shape[1])} -> {tr_branch.feature_selection_info.get('n_features_after', tr_branch.X_all_df.shape[1])}\n"
    )

    before_types_mlp = mlp_branch.type_profile_before.get("dtype_counts", {})
    after_types_mlp = mlp_branch.type_profile_after.get("dtype_counts", {})
    before_types_tr = tr_branch.type_profile_before.get("dtype_counts", {})
    after_types_tr = tr_branch.type_profile_after.get("dtype_counts", {})
    md.append(f"- MLP清洗前后列类型: {before_types_mlp} -> {after_types_mlp}\n")
    md.append(f"- Transformer清洗前后列类型: {before_types_tr} -> {after_types_tr}\n")

    impacted_before = tr_branch.type_profile_before.get("impacted_object_columns", {})
    if impacted_before:
        md.append("- 清洗前受异常值影响而保持object的列（前10个，按Transformer分支统计）:\n")
        for i, col in enumerate(list(impacted_before.keys())[:10], start=1):
            top_non_numeric = impacted_before[col].get("top_non_numeric_tokens", {})
            md.append(f"  {i}. {col} | 非数值示例: {top_non_numeric}\n")
    other_anomalies = tr_clean.get("other_anomaly_columns", {})
    if other_anomalies:
        md.append("- 清洗后仍有异常文本的列（前10个，按Transformer分支统计）:\n")
        for i, col in enumerate(list(other_anomalies.keys())[:10], start=1):
            tokens = other_anomalies[col].get("top_non_numeric_tokens", {})
            md.append(f"  {i}. {col} | 文本示例: {tokens}\n")

    md.append("\n## 噪声与样本处理\n")
    md.append(
        f"- MLP: 标签噪声过滤删除 {mlp_branch.noise_filter_info.get('n_dropped', 0)} 条；"
        f"自举增强新增 {mlp_branch.augmentation_info.get('rows_added', 0)} 条\n"
    )
    md.append(
        f"- Transformer: 标签噪声过滤删除 {tr_branch.noise_filter_info.get('n_dropped', 0)} 条；"
        f"自举增强新增 {tr_branch.augmentation_info.get('rows_added', 0)} 条\n"
    )
    md.append(
        f"- 类别不平衡处理: class_weight={use_class_weight}, weighted_sampler={use_weighted_sampler}, "
        f"reference(MLP)={mlp_strategy['imbalance_reference']}, reference(Transformer)={tr_strategy['imbalance_reference']}, "
        f"class_weight_power={args.class_weight_power}, sampler_weight_power(MLP)={float(mlp_strategy['sampler_weight_power'])}, sampler_weight_power(Transformer)={float(tr_strategy['sampler_weight_power'])}\n"
    )
    md.append(
        f"- 训练选模指标: MLP={mlp_strategy['selection_metric']}, Transformer={tr_strategy['selection_metric']} "
        f"(early-stopping 按各模型配置监控 val 指标)\n"
    )
    md.append(
        f"- 先验校正调参: enabled={enable_prior_adjustment_tuning}, metric={args.prior_adjustment_metric}, "
        f"tau_grid={[float(x) for x in prior_adjustment_tau_grid]}, recall_floor={float(args.worst_class_recall_floor):.3f}, "
        f"best_tau(MLP)={float(mlp_prior_adjustment.get('best_tau', 0.0)):.3f}, "
        f"best_tau(Transformer)={float(tr_prior_adjustment.get('best_tau', 0.0)):.3f}\n"
    )
    md.append(
        f"- 教师蒸馏: enabled={enable_distillation}, target={args.distill_target}, "
        f"MLP_applied={distillation_info_mlp.get('applied', False)}, Transformer_applied={distillation_info_tr.get('applied', False)}\n"
    )
    md.append("\n## 测试集指标\n")
    md.append("| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 | ROC-AUC Macro |\n")
    md.append("|---|---:|---:|---:|---:|---:|---:|\n")
    md.append(
        "| MLP | {a:.4f} | {p:.4f} | {r:.4f} | {f1:.4f} | {wf1:.4f} | {auc:.4f} |\n".format(
            a=mlp_metrics["accuracy"],
            p=mlp_metrics["macro_precision"],
            r=mlp_metrics["macro_recall"],
            f1=mlp_metrics["macro_f1"],
            wf1=mlp_metrics["weighted_f1"],
            auc=mlp_metrics["roc_auc_macro"] if not math.isnan(mlp_metrics["roc_auc_macro"]) else float("nan"),
        )
    )
    md.append(
        "| Transformer | {a:.4f} | {p:.4f} | {r:.4f} | {f1:.4f} | {wf1:.4f} | {auc:.4f} |\n".format(
            a=tr_metrics["accuracy"],
            p=tr_metrics["macro_precision"],
            r=tr_metrics["macro_recall"],
            f1=tr_metrics["macro_f1"],
            wf1=tr_metrics["weighted_f1"],
            auc=tr_metrics["roc_auc_macro"] if not math.isnan(tr_metrics["roc_auc_macro"]) else float("nan"),
        )
    )

    if enable_imbalance_validation:
        md.append("\n## 类别不均衡验证\n")
        md.append(
            "| Model | Macro PR-AUC | Worst-class Recall | Worst-class F1 | Minority Macro Recall | Minority Macro F1 |\n"
        )
        md.append("|---|---:|---:|---:|---:|---:|\n")
        md.append(
            "| MLP | {pr:.4f} | {wr:.4f} ({wrc}) | {wf1:.4f} ({wf1c}) | {mr:.4f} | {mf1:.4f} |\n".format(
                pr=safe_float(mlp_imbalance_diag.get("macro_pr_auc", math.nan), default=math.nan),
                wr=safe_float(mlp_imbalance_diag.get("worst_class_recall", 0.0), default=0.0),
                wrc=str(mlp_imbalance_diag.get("worst_class_recall_class", "-")),
                wf1=safe_float(mlp_imbalance_diag.get("worst_class_f1", 0.0), default=0.0),
                wf1c=str(mlp_imbalance_diag.get("worst_class_f1_class", "-")),
                mr=safe_float(mlp_imbalance_diag.get("minority_macro_recall", 0.0), default=0.0),
                mf1=safe_float(mlp_imbalance_diag.get("minority_macro_f1", 0.0), default=0.0),
            )
        )
        md.append(
            "| Transformer | {pr:.4f} | {wr:.4f} ({wrc}) | {wf1:.4f} ({wf1c}) | {mr:.4f} | {mf1:.4f} |\n".format(
                pr=safe_float(tr_imbalance_diag.get("macro_pr_auc", math.nan), default=math.nan),
                wr=safe_float(tr_imbalance_diag.get("worst_class_recall", 0.0), default=0.0),
                wrc=str(tr_imbalance_diag.get("worst_class_recall_class", "-")),
                wf1=safe_float(tr_imbalance_diag.get("worst_class_f1", 0.0), default=0.0),
                wf1c=str(tr_imbalance_diag.get("worst_class_f1_class", "-")),
                mr=safe_float(tr_imbalance_diag.get("minority_macro_recall", 0.0), default=0.0),
                mf1=safe_float(tr_imbalance_diag.get("minority_macro_f1", 0.0), default=0.0),
            )
        )
        md.append(
            "| Ensemble | {pr:.4f} | {wr:.4f} ({wrc}) | {wf1:.4f} ({wf1c}) | {mr:.4f} | {mf1:.4f} |\n".format(
                pr=safe_float(ensemble_imbalance_diag.get("macro_pr_auc", math.nan), default=math.nan),
                wr=safe_float(ensemble_imbalance_diag.get("worst_class_recall", 0.0), default=0.0),
                wrc=str(ensemble_imbalance_diag.get("worst_class_recall_class", "-")),
                wf1=safe_float(ensemble_imbalance_diag.get("worst_class_f1", 0.0), default=0.0),
                wf1c=str(ensemble_imbalance_diag.get("worst_class_f1_class", "-")),
                mr=safe_float(ensemble_imbalance_diag.get("minority_macro_recall", 0.0), default=0.0),
                mf1=safe_float(ensemble_imbalance_diag.get("minority_macro_f1", 0.0), default=0.0),
            )
        )
        md.append(
            "| Joint | {pr:.4f} | {wr:.4f} ({wrc}) | {wf1:.4f} ({wf1c}) | {mr:.4f} | {mf1:.4f} |\n".format(
                pr=safe_float(joint_imbalance_diag.get("macro_pr_auc", math.nan), default=math.nan),
                wr=safe_float(joint_imbalance_diag.get("worst_class_recall", 0.0), default=0.0),
                wrc=str(joint_imbalance_diag.get("worst_class_recall_class", "-")),
                wf1=safe_float(joint_imbalance_diag.get("worst_class_f1", 0.0), default=0.0),
                wf1c=str(joint_imbalance_diag.get("worst_class_f1_class", "-")),
                mr=safe_float(joint_imbalance_diag.get("minority_macro_recall", 0.0), default=0.0),
                mf1=safe_float(joint_imbalance_diag.get("minority_macro_f1", 0.0), default=0.0),
            )
        )

    md.append("\n## 模型轻量化评估\n")
    md.append(
        f"- 轻量化profile: {lightweight_profile_info.get('profile')} (changed={lightweight_profile_info.get('changed')})\n"
    )
    md.append(
        f"- 量化评估: enabled={enable_quantized_eval}; 详细见 model_lightweight_summary.json\n"
    )
    md.append("| Model | Params | FP32 Size(MB) | Quantized Size(MB) | Compression Ratio | Quantized Macro-F1 |\n")
    md.append("|---|---:|---:|---:|---:|---:|\n")

    mlp_lw = lightweight_summary.get("models", {}).get("mlp", {})
    tr_lw = lightweight_summary.get("models", {}).get("transformer", {})
    mlp_q = mlp_lw.get("quantized", {}) if isinstance(mlp_lw, dict) else {}
    tr_q = tr_lw.get("quantized", {}) if isinstance(tr_lw, dict) else {}

    mlp_q_size_mb = safe_float(mlp_q.get("state_dict_bytes", 0), default=0.0) / (1024.0 * 1024.0)
    tr_q_size_mb = safe_float(tr_q.get("state_dict_bytes", 0), default=0.0) / (1024.0 * 1024.0)
    mlp_q_f1 = safe_float(mlp_q.get("metrics", {}).get("macro_f1", math.nan), default=math.nan) if mlp_q.get("ok") else math.nan
    tr_q_f1 = safe_float(tr_q.get("metrics", {}).get("macro_f1", math.nan), default=math.nan) if tr_q.get("ok") else math.nan

    md.append(
        "| MLP | {params} | {fp32:.3f} | {qsz:.3f} | {cr:.3f} | {qf1:.4f} |\n".format(
            params=int(mlp_lw.get("param_count", 0)),
            fp32=safe_float(mlp_lw.get("fp32_state_dict_bytes", 0), default=0.0) / (1024.0 * 1024.0),
            qsz=mlp_q_size_mb,
            cr=safe_float(mlp_q.get("size_reduction_ratio", math.nan), default=math.nan),
            qf1=mlp_q_f1,
        )
    )
    md.append(
        "| Transformer | {params} | {fp32:.3f} | {qsz:.3f} | {cr:.3f} | {qf1:.4f} |\n".format(
            params=int(tr_lw.get("param_count", 0)),
            fp32=safe_float(tr_lw.get("fp32_state_dict_bytes", 0), default=0.0) / (1024.0 * 1024.0),
            qsz=tr_q_size_mb,
            cr=safe_float(tr_q.get("size_reduction_ratio", math.nan), default=math.nan),
            qf1=tr_q_f1,
        )
    )

    md.append("\n### 陌生样本多分类推理（测试集模拟）\n")
    md.append("| Fusion | Accuracy | Macro Precision | Macro Recall | Macro F1 |\n")
    md.append("|---|---:|---:|---:|---:|\n")
    md.append(
        "| Ensemble(mean prob) | {a:.4f} | {p:.4f} | {r:.4f} | {f1:.4f} |\n".format(
            a=ensemble_metrics["accuracy"],
            p=ensemble_metrics["macro_precision"],
            r=ensemble_metrics["macro_recall"],
            f1=ensemble_metrics["macro_f1"],
        )
    )
    md.append(
        "| Joint(product prob) | {a:.4f} | {p:.4f} | {r:.4f} | {f1:.4f} |\n".format(
            a=joint_metrics["accuracy"],
            p=joint_metrics["macro_precision"],
            r=joint_metrics["macro_recall"],
            f1=joint_metrics["macro_f1"],
        )
    )

    md.append(f"\n## 关键性指标 Top{args.topk} (基于置换重要性: {args.importance_metric})\n")
    md.append("\n### MLP\n")
    for i, r in enumerate(imp_mlp.head(args.topk).itertuples(index=False), start=1):
        md.append(f"{i}. {r.feature} (drop={r.importance_mean:.4f}±{r.importance_std:.4f})\n")

    md.append("\n### Transformer\n")
    for i, r in enumerate(imp_tr.head(args.topk).itertuples(index=False), start=1):
        md.append(f"{i}. {r.feature} (drop={r.importance_mean:.4f}±{r.importance_std:.4f})\n")

    md.append(f"\n## one-vs-rest 病种级重要性 (metric={args.ovr_metric})\n")
    for disease in label_names:
        md.append(f"\n### {disease}\n")
        sub_mlp = imp_ovr_mlp[imp_ovr_mlp["class_name"] == disease].head(args.disease_topk)
        sub_tr = imp_ovr_tr[imp_ovr_tr["class_name"] == disease].head(args.disease_topk)
        md.append("- MLP Top:\n")
        for i, r in enumerate(sub_mlp.itertuples(index=False), start=1):
            md.append(f"  {i}. {r.feature} (drop={r.importance_mean:.4f}±{r.importance_std:.4f})\n")
        md.append("- Transformer Top:\n")
        for i, r in enumerate(sub_tr.itertuples(index=False), start=1):
            md.append(f"  {i}. {r.feature} (drop={r.importance_mean:.4f}±{r.importance_std:.4f})\n")

    md.append("\n## 临床可解释交集特征（双模型联合概率）\n")
    md.append(
        f"- 降权噪声代理变量: {downweight_noise_features}; 全局Top={args.clinical_topk}; 病种Top={args.disease_topk}\n"
    )
    md.append("\n### 全局交集 Top\n")
    for i, r in enumerate(global_intersection.head(args.clinical_topk).itertuples(index=False), start=1):
        md.append(
            f"{i}. {r.feature} | joint={r.joint_prob_adjusted_norm:.4f} | "
            f"MLP={r.mlp_importance:.4f} | Transformer={r.transformer_importance:.4f} | "
            f"penalty={r.clinical_penalty:.3f} ({r.penalty_reasons or '-'})\n"
        )

    md.append("\n### 分病种交集 Top\n")
    for disease in label_names:
        md.append(f"\n#### {disease}\n")
        sub = disease_intersection_df[disease_intersection_df["class_name"] == disease].head(args.disease_topk)
        if sub.empty:
            md.append("- 无可用交集特征\n")
            continue
        for i, r in enumerate(sub.itertuples(index=False), start=1):
            md.append(
                f"{i}. {r.feature} | joint={r.joint_prob_adjusted_norm:.4f} | "
                f"MLP={r.mlp_importance:.4f} | Transformer={r.transformer_importance:.4f} | "
                f"penalty={r.clinical_penalty:.3f} ({r.penalty_reasons or '-'})\n"
            )

    md.append("\n## 输出文件\n")
    md.append("- summary.json\n")
    md.append("- feature_importance_mlp.csv\n")
    md.append("- feature_importance_transformer.csv\n")
    md.append("- feature_importance_mlp_ovr.csv\n")
    md.append("- feature_importance_transformer_ovr.csv\n")
    md.append("- feature_intersection_joint_global.csv\n")
    md.append("- feature_intersection_joint_by_disease.csv\n")
    md.append("- clinical_interpretable_top_global.csv\n")
    md.append("- clinical_feature_direction_by_disease.csv\n")
    md.append("- unseen_multiclass_inference.csv\n")
    md.append("- clinical_dossier_by_disease.md\n")
    md.append("- imbalance_diagnostics.json\n")
    md.append("- model_lightweight_summary.json\n")
    md.append("- predictions_mlp.csv\n")
    md.append("- predictions_transformer.csv\n")
    if summary.get("dynamic_outputs_index"):
        md.append(f"- {summary['dynamic_outputs_index']} (auto-refreshed)\n")
    if summary.get("dynamic_outputs_index_error"):
        md.append(f"- OUTPUTS_DYNAMIC_INDEX refresh error: {summary['dynamic_outputs_index_error']}\n")

    (out_dir / "report.md").write_text("".join(md), encoding="utf-8")

    print("Saved to:", out_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
