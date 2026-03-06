"""
BQ EDA Module
- BQ 테이블의 타입(CDC/LOG/SNAPSHOT)을 자동 분류하고, 타입별 EDA를 수행한다.
- 컬럼 분류: numeric, categorical, datetime, json, text
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta

import yaml
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from google.cloud import bigquery

matplotlib.rcParams["font.family"] = "AppleGothic"
matplotlib.rcParams["axes.unicode_minus"] = False
sns.set_theme(style="whitegrid", font_scale=0.9, rc={"font.family": "AppleGothic", "axes.unicode_minus": False})


# ---------------------------------------------------------------------------
# 1. BQ Helper
# ---------------------------------------------------------------------------

# CDC 테이블의 PK 매핑 (자동 탐지 실패 시 fallback)
_CDC_PK_MAP = {
    "product_collection.product_collections": "id",
    "product_collection.product_collection_items": "id",
    "product_collection.product_categories": "id",
    "product_collection.site_category_groups": "id",
    "product_collection.gd_goods_link": "sno",
    "pms.contents_products_view": "_id",
    "cms.mk_member": "m_no",
    "cms.mk_member_grp": "sno",
    "cms.mk_goods_view": "gv_idx",
    "cms.mk_addressbook": "seq",
    "cms.dsms_contents_product": "no",
    "review.gd_goods_review": "sno",
}

# 테이블 타입별 기본 기간 (일)
DEFAULT_DATE_RANGES = {
    "CDC": 7,        # 1주일 (deduplicate 후 최신 스냅샷)
    "LOG": 1,        # 1일
    "SNAPSHOT": 1,   # 1일
}

# 날짜 필터 적용 후 최소 row 수 (이하이면 날짜 필터 없이 재시도)
MIN_ROWS_THRESHOLD = 30


def load_table_list(yaml_path: str) -> list[dict]:
    """yaml에서 project.dataset.table 목록을 파싱한다."""
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    tables = []
    for project, datasets in data.items():
        for dataset, table_names in datasets.items():
            for t in table_names:
                tables.append({
                    "project": project,
                    "dataset": dataset,
                    "table": t,
                    "full_id": f"{project}.{dataset}.{t}",
                })
    return tables


def get_bq_schema(client: bigquery.Client, full_table_id: str) -> dict:
    """BQ 스키마 정보를 dict로 반환한다. {col_name: bq_type}"""
    table = client.get_table(full_table_id)
    return {field.name: field.field_type for field in table.schema}


def detect_table_type(bq_schema: dict, dataset: str) -> str:
    """테이블 타입을 자동 분류한다.

    - CDC: operation 컬럼이 있는 테이블 (변경 이력)
    - LOG: kalog 데이터셋 (이벤트 로그)
    - SNAPSHOT: 그 외 (스냅샷/트랜잭션)
    """
    if "operation" in bq_schema:
        return "CDC"
    if dataset == "kalog":
        return "LOG"
    return "SNAPSHOT"


def detect_date_col(bq_schema: dict) -> str | None:
    """BQ 스키마에서 날짜 필터에 사용할 컬럼을 자동 탐지한다.

    우선순위: pdt > DATE 타입 > DATETIME 타입 > TIMESTAMP 타입
    """
    if "pdt" in bq_schema:
        return "pdt"

    # delete 관련 컬럼은 제외
    _EXCLUDE_PATTERNS = {"delete", "deleted", "expire", "expired"}

    def _is_excluded(col_name: str) -> bool:
        col_lower = col_name.lower()
        return any(pat in col_lower for pat in _EXCLUDE_PATTERNS)

    # CDC 테이블은 timestamp 컬럼이 가장 적합 (CDC 이벤트 시각)
    if "timestamp" in bq_schema and bq_schema["timestamp"] == "TIMESTAMP":
        return "timestamp"

    priority = {"DATE": 1, "DATETIME": 2, "TIMESTAMP": 3}
    candidates = [
        (col, priority[bq_type])
        for col, bq_type in bq_schema.items()
        if bq_type in priority and not _is_excluded(col)
    ]
    if candidates:
        candidates.sort(key=lambda x: x[1])
        return candidates[0][0]
    return None


def detect_cdc_pk(bq_schema: dict, dataset_table: str) -> str | None:
    """CDC 테이블의 PK를 탐지한다. 매핑 우선, 없으면 휴리스틱."""
    if dataset_table in _CDC_PK_MAP:
        return _CDC_PK_MAP[dataset_table]

    # 휴리스틱: id, _id, sno, no, seq 순으로 탐색
    for candidate in ["id", "_id", "sno", "no", "seq"]:
        if candidate in bq_schema:
            return candidate
    return None


def compute_date_range(
    end_date: str,
    table_type: str,
    start_date: str | None = None,
) -> tuple[str, str]:
    """테이블 타입에 맞는 날짜 범위를 계산한다.

    start_date가 주어지면 그대로 사용, 없으면 table_type별 기본 기간 적용.
    """
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    if start_date:
        return start_date, end_date
    days = DEFAULT_DATE_RANGES.get(table_type, 7)
    start_dt = end_dt - timedelta(days=days - 1)
    return start_dt.strftime("%Y-%m-%d"), end_date


def fetch_data(
    client: bigquery.Client,
    full_table_id: str,
    date_col: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    sample_size: int | None = None,
) -> pd.DataFrame:
    """BQ 테이블에서 데이터를 가져온다.

    Args:
        date_col: 날짜 필터 컬럼명.
        start_date: 시작 날짜 (inclusive).
        end_date: 종료 날짜 (inclusive).
        sample_size: 최대 row 수. None이면 전체 데이터.
    """
    where = ""
    if date_col and start_date and end_date:
        where = (
            f"WHERE CAST(`{date_col}` AS DATE) >= '{start_date}' "
            f"AND CAST(`{date_col}` AS DATE) <= '{end_date}'"
        )

    limit = f"LIMIT {sample_size}" if sample_size else ""

    query = f"""
        SELECT *
        FROM `{full_table_id}`
        {where}
        {limit}
    """

    job_config = bigquery.QueryJobConfig(
        allow_large_results=True,
        use_legacy_sql=False,
    )
    return client.query(query, job_config=job_config).to_dataframe()


def deduplicate_cdc(df: pd.DataFrame, pk_col: str) -> pd.DataFrame:
    """CDC 테이블에서 PK 기준 최신 row만 남긴다.

    timestamp 컬럼 기준 최신 row를 선택하고,
    operation='DELETE'인 row는 제거한다.
    """
    if pk_col not in df.columns or "timestamp" not in df.columns:
        return df

    # timestamp 기준 최신 row만 남김
    df_sorted = df.sort_values("timestamp", ascending=False)
    df_dedup = df_sorted.drop_duplicates(subset=[pk_col], keep="first")

    # DELETE 제거
    if "operation" in df_dedup.columns:
        before = len(df_dedup)
        df_dedup = df_dedup[df_dedup["operation"].str.upper() != "DELETE"]
        deleted = before - len(df_dedup)
        if deleted > 0:
            print(f"    Removed {deleted} DELETE rows")

    return df_dedup.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2. Column Type Classifier
# ---------------------------------------------------------------------------

_DATETIME_TYPES = {"DATE", "DATETIME", "TIMESTAMP"}
_NUMERIC_TYPES = {"INTEGER", "INT64", "FLOAT", "FLOAT64", "NUMERIC", "BIGNUMERIC"}

CATEGORICAL_MAX_UNIQUE = 100
CATEGORICAL_UNIQUE_RATIO = 0.05
TEXT_AVG_LENGTH_THRESHOLD = 30
JSON_SAMPLE_CHECK_COUNT = 20
IDENTIFIER_UNIQUE_RATIO = 0.95  # unique 비율이 이 이상이면 identifier 후보

# 식별자(PK) 컬럼명 패턴 (code, key는 비즈니스 의미가 있을 수 있어 제외)
_ID_NAME_PATTERNS = {"id", "_id", "no", "sno", "seq", "idx", "uuid"}

# FK(외래키) 접미사 패턴 — INTEGER인데 _id로 끝나면 FK → categorical
_FK_SUFFIXES = {"_id", "_no", "_sno", "_seq", "_idx", "_cd"}


def _is_json_like(series: pd.Series) -> bool:
    """STRING 컬럼이 JSON 형태인지 샘플로 판별한다."""
    samples = series.dropna().head(JSON_SAMPLE_CHECK_COUNT)
    if len(samples) == 0:
        return False
    json_count = 0
    for v in samples:
        v_stripped = str(v).strip()
        if v_stripped.startswith(("{", "[")):
            try:
                json.loads(v_stripped)
                json_count += 1
            except (json.JSONDecodeError, TypeError):
                pass
    return json_count / len(samples) >= 0.5


def _is_identifier_name(col: str) -> bool:
    """컬럼명이 식별자 패턴인지 확인한다."""
    col_lower = col.lower()
    # 정확히 일치하거나, _id / _no / _sno 등으로 끝나는 경우
    if col_lower in _ID_NAME_PATTERNS:
        return True
    for pat in _ID_NAME_PATTERNS:
        if col_lower.endswith(f"_{pat}") or col_lower.endswith(f"_{pat}s"):
            return True
    return False


_SKIP_COLUMNS = {"partitionkey", "operation", "timestamp"}


def classify_columns(df: pd.DataFrame, bq_schema: dict) -> dict:
    """각 컬럼을 numeric / categorical / datetime / json / text / identifier 로 분류한다.

    Returns: {col_name: col_type}
    """
    result = {}
    for col in df.columns:
        if col.lower() in _SKIP_COLUMNS:
            continue
        bq_type = bq_schema.get(col, "STRING").upper()
        series = df[col]
        non_null = series.dropna()

        # datetime
        if bq_type in _DATETIME_TYPES:
            result[col] = "datetime"
            continue

        # numeric (BQ 타입 기준)
        if bq_type in _NUMERIC_TYPES:
            col_lower = col.lower()
            nunique = non_null.nunique()
            count = len(non_null)

            # 컬럼명 자체가 _id → identifier (PK)
            if col_lower == "_id":
                result[col] = "identifier"
            # FK 참조 컬럼 (_id, _no 등으로 끝남) → categorical
            elif any(col_lower.endswith(suffix) for suffix in _FK_SUFFIXES):
                result[col] = "categorical"
            # 낮은 cardinality → categorical
            elif count > 0 and nunique <= 20 and (nunique / count) < CATEGORICAL_UNIQUE_RATIO:
                result[col] = "categorical"
            # 거의 모든 값이 unique + 이름이 식별자 패턴 → identifier
            elif count > 0 and (nunique / count) >= IDENTIFIER_UNIQUE_RATIO and _is_identifier_name(col):
                result[col] = "identifier"
            else:
                result[col] = "numeric"
            continue

        # STRING 계열 -> json / identifier / categorical / text 구분
        if _is_json_like(series):
            result[col] = "json"
            continue

        nunique = non_null.nunique()
        count = len(non_null)
        unique_ratio = nunique / count if count > 0 else 0
        avg_len = non_null.astype(str).str.len().mean() if count > 0 else 0

        # STRING이지만 거의 unique + 이름이 식별자 패턴 → identifier
        if count > 0 and unique_ratio >= IDENTIFIER_UNIQUE_RATIO and _is_identifier_name(col):
            result[col] = "identifier"
        elif nunique <= CATEGORICAL_MAX_UNIQUE or unique_ratio < CATEGORICAL_UNIQUE_RATIO:
            result[col] = "categorical"
        elif avg_len >= TEXT_AVG_LENGTH_THRESHOLD:
            result[col] = "text"
        else:
            result[col] = "categorical"

        continue

    return result


# ---------------------------------------------------------------------------
# 3. EDA per column type
# ---------------------------------------------------------------------------

def eda_identifier(df: pd.DataFrame, col: str) -> dict:
    """식별자 컬럼 EDA: count + cardinality + null count만 출력."""
    s = df[col]
    non_null = s.dropna()
    return {
        "column": col,
        "type": "identifier",
        "total_count": len(s),
        "cardinality": int(non_null.nunique()),
        "null_count": int(s.isna().sum()),
        "null_ratio": round(s.isna().mean(), 4),
        "is_unique": int(non_null.nunique()) == len(non_null),
        "sample_values": non_null.head(5).tolist(),
    }


def _safe_float(val, default=0) -> float:
    """pd.NA / NaN / NAType 을 안전하게 float으로 변환한다."""
    if pd.isna(val):
        return default
    return float(val)


def eda_numeric(df: pd.DataFrame, col: str) -> dict:
    """수치형 컬럼 EDA: 기본 통계량 + skewness + null/zero count"""
    s = pd.to_numeric(df[col], errors="coerce")
    non_null_count = int(s.notna().sum())

    if non_null_count == 0:
        return {
            "column": col, "type": "numeric",
            "count": 0, "mean": 0, "median": 0, "std": 0,
            "min": 0, "max": 0, "p1": 0, "p5": 0, "p25": 0, "p75": 0, "p95": 0, "p99": 0,
            "skewness": 0,
            "null_count": len(s), "zero_count": 0,
            "null_ratio": 1.0, "zero_ratio": 0,
        }

    desc = s.describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    result = {
        "column": col,
        "type": "numeric",
        "count": int(desc["count"]),
        "mean": _safe_float(desc["mean"]),
        "median": _safe_float(desc["50%"]),
        "std": _safe_float(desc["std"]),
        "min": _safe_float(desc["min"]),
        "max": _safe_float(desc["max"]),
        "p1": _safe_float(desc["1%"]),
        "p5": _safe_float(desc["5%"]),
        "p25": _safe_float(desc["25%"]),
        "p75": _safe_float(desc["75%"]),
        "p95": _safe_float(desc["95%"]),
        "p99": _safe_float(desc["99%"]),
        "skewness": _safe_float(s.skew()),
        "null_count": int(s.isna().sum()),
        "zero_count": int((s == 0).sum()),
        "null_ratio": round(_safe_float(s.isna().mean()), 4),
        "zero_ratio": round(_safe_float((s == 0).mean()), 4),
    }
    return result


def eda_categorical(df: pd.DataFrame, col: str, top_n: int = 20) -> dict:
    """범주형 컬럼 EDA: cardinality + top N frequency + 누적 분포 + null count"""
    s = df[col]
    vc = s.value_counts(dropna=False)
    total = len(s)
    cardinality = int(s.nunique(dropna=True))

    if total == 0 or len(vc) == 0:
        return {
            "column": col, "type": "categorical",
            "total_count": 0, "cardinality": 0,
            "null_count": 0, "null_ratio": 0,
            "top_n_values": {}, "top_n": top_n, "top_n_coverage": 0,
            "cumulative_distribution": {},
        }

    top = vc.head(top_n)

    # 누적 분포: 상위 N% 카테고리가 전체 데이터의 몇 %를 차지하는지
    cumsum = vc.cumsum()
    cum_ratios = {}
    for pct in [1, 5, 10, 20, 50]:
        n_cats = max(1, int(cardinality * pct / 100))
        coverage = cumsum.iloc[min(n_cats, len(cumsum)) - 1] / total
        cum_ratios[f"top_{pct}pct_categories_coverage"] = round(float(coverage), 4)

    result = {
        "column": col,
        "type": "categorical",
        "total_count": total,
        "cardinality": cardinality,
        "null_count": int(s.isna().sum()),
        "null_ratio": round(s.isna().mean(), 4),
        "top_n_values": {
            str(k): {"count": int(v), "ratio": round(v / total, 4)}
            for k, v in top.items()
        },
        "top_n": top_n,
        "top_n_coverage": round(top.sum() / total, 4),
        "cumulative_distribution": cum_ratios,
    }
    return result


def eda_datetime(df: pd.DataFrame, col: str) -> dict:
    """시계열 컬럼 EDA: min/max date + 일별 건수 통계"""
    s = pd.to_datetime(df[col], errors="coerce")
    non_null = s.dropna()

    result = {
        "column": col,
        "type": "datetime",
        "total_count": len(s),
        "null_count": int(s.isna().sum()),
        "null_ratio": round(s.isna().mean(), 4),
    }

    if len(non_null) == 0:
        result["min_date"] = None
        result["max_date"] = None
        return result

    result["min_date"] = str(non_null.min())
    result["max_date"] = str(non_null.max())

    # 일별 건수 통계
    daily = non_null.dt.date.value_counts().sort_index()
    result["daily_count_stats"] = {
        "mean": round(daily.mean(), 2),
        "median": round(daily.median(), 2),
        "std": round(daily.std(), 2) if len(daily) > 1 else 0,
        "min": int(daily.min()),
        "max": int(daily.max()),
        "num_days": len(daily),
    }

    # 시간대별 건수 (hour)
    if hasattr(non_null.dt, "hour"):
        hourly = non_null.dt.hour.value_counts().sort_index()
        result["hourly_distribution"] = hourly.to_dict()

    return result


def _flatten_json(obj, prefix: str = "") -> dict:
    """JSON 객체를 recursive하게 펼쳐서 {dotted_path: value} 딕셔너리로 반환한다."""
    items = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            new_key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, (dict, list)):
                items.update(_flatten_json(v, new_key))
            else:
                items[new_key] = v
    elif isinstance(obj, list):
        if len(obj) > 0 and isinstance(obj[0], dict):
            # list of dicts → 첫 번째 원소 기준으로 구조 파악
            for k, v in obj[0].items():
                new_key = f"{prefix}[].{k}" if prefix else f"[].{k}"
                if isinstance(v, (dict, list)):
                    items.update(_flatten_json(v, new_key))
                else:
                    items[new_key] = v
            items[f"{prefix}.__list_length__"] = len(obj)
        else:
            items[prefix] = obj
    else:
        items[prefix] = obj
    return items


def _infer_value_type(values: list) -> str:
    """값 리스트에서 대표 타입을 추론한다."""
    type_counts = {}
    for v in values:
        if v is None:
            t = "null"
        elif isinstance(v, bool):
            t = "boolean"
        elif isinstance(v, (int, float)):
            t = "numeric"
        elif isinstance(v, list):
            t = "array"
        elif isinstance(v, dict):
            t = "object"
        else:
            t = "string"
        type_counts[t] = type_counts.get(t, 0) + 1
    if not type_counts:
        return "unknown"
    return max(type_counts, key=type_counts.get)


def _summarize_field_values(values: list) -> dict:
    """펼쳐진 필드의 값 리스트에 대한 요약 통계를 반환한다."""
    non_null = [v for v in values if v is not None]
    total = len(values)
    null_count = total - len(non_null)
    vtype = _infer_value_type(non_null)

    summary = {
        "count": total,
        "null_count": null_count,
        "null_ratio": round(null_count / total, 4) if total > 0 else 0,
        "inferred_type": vtype,
    }

    if vtype == "numeric" and non_null:
        nums = [float(v) for v in non_null if isinstance(v, (int, float))]
        if nums:
            s = pd.Series(nums)
            summary["mean"] = round(_safe_float(s.mean()), 4)
            summary["median"] = round(_safe_float(s.median()), 4)
            summary["min"] = _safe_float(s.min())
            summary["max"] = _safe_float(s.max())
            summary["zero_ratio"] = round(sum(1 for x in nums if x == 0) / len(nums), 4)

    elif vtype == "string" and non_null:
        str_vals = [str(v) for v in non_null]
        nunique = len(set(str_vals))
        summary["cardinality"] = nunique
        summary["avg_length"] = round(sum(len(v) for v in str_vals) / len(str_vals), 2)
        # top 5 빈도
        vc = pd.Series(str_vals).value_counts().head(5)
        summary["top_values"] = {str(k): int(v) for k, v in vc.items()}

    elif vtype == "boolean" and non_null:
        true_count = sum(1 for v in non_null if v is True)
        summary["true_ratio"] = round(true_count / len(non_null), 4)

    return summary


def eda_json(df: pd.DataFrame, col: str, sample_size: int = 1000) -> dict:
    """JSON 컬럼 EDA: recursive하게 모든 필드를 펼쳐서 타입별 요약 통계를 생성한다."""
    s = df[col].dropna().astype(str)
    result = {
        "column": col,
        "type": "json",
        "total_count": len(df[col]),
        "null_count": int(df[col].isna().sum()),
        "null_ratio": round(df[col].isna().mean(), 4),
    }

    # JSON 파싱 + 펼치기
    field_values = {}  # {dotted_path: [values]}
    parse_count = 0
    root_types = {"dict": 0, "list": 0, "other": 0}

    for v in s.head(sample_size):
        try:
            parsed = json.loads(v.strip())
            parse_count += 1
            if isinstance(parsed, dict):
                root_types["dict"] += 1
            elif isinstance(parsed, list):
                root_types["list"] += 1
            else:
                root_types["other"] += 1
                continue
            flat = _flatten_json(parsed)
            for path, val in flat.items():
                field_values.setdefault(path, []).append(val)
        except (json.JSONDecodeError, TypeError):
            pass

    result["parseable_count"] = parse_count
    result["root_type_distribution"] = root_types

    # 필드별 요약
    fields_summary = {}
    for path, values in sorted(field_values.items()):
        fields_summary[path] = _summarize_field_values(values)
    result["fields"] = fields_summary
    result["field_count"] = len(fields_summary)
    result["sample_values"] = s.head(3).tolist()
    return result


def eda_text(df: pd.DataFrame, col: str) -> dict:
    """텍스트 컬럼 EDA: 길이 분포 + null count"""
    s = df[col]
    non_null = s.dropna().astype(str)
    lengths = non_null.str.len()
    result = {
        "column": col,
        "type": "text",
        "total_count": len(s),
        "cardinality": int(non_null.nunique()),
        "null_count": int(s.isna().sum()),
        "null_ratio": round(s.isna().mean(), 4),
        "length_stats": {
            "mean": round(lengths.mean(), 2) if len(lengths) > 0 else 0,
            "median": round(lengths.median(), 2) if len(lengths) > 0 else 0,
            "min": int(lengths.min()) if len(lengths) > 0 else 0,
            "max": int(lengths.max()) if len(lengths) > 0 else 0,
        },
        "sample_values": non_null.head(3).tolist(),
    }
    return result


# ---------------------------------------------------------------------------
# 4. Visualization
# ---------------------------------------------------------------------------

def plot_numeric(df: pd.DataFrame, col: str, n_quantiles: int = 10):
    """수치형: 히스토그램 (log scale) + 박스플롯 + Quantile Binning"""
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    if len(s) == 0:
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    ax_hist, ax_box = axes[0]
    ax_qbin, ax_qcum = axes[1]

    # 1) 히스토그램
    ax_hist.hist(s, bins=50, edgecolor="black", alpha=0.7)
    ax_hist.set_title(f"{col} - Distribution")
    ax_hist.set_ylabel("Count")
    if s.max() / (s.median() + 1e-9) > 10:
        ax_hist.set_yscale("log")
        ax_hist.set_title(f"{col} - Distribution (log scale)")

    # 2) 박스플롯
    ax_box.boxplot(s, vert=True)
    ax_box.set_title(f"{col} - Boxplot")

    # 3) Quantile Binning + 4) Cumulative Distribution
    try:
        qcut = pd.qcut(s, q=n_quantiles, duplicates="drop")
        qcounts = qcut.value_counts().sort_index()
        labels = [str(interval) for interval in qcounts.index]

        # Quantile Binning bar chart
        ax_qbin.barh(labels, qcounts.values, edgecolor="black", alpha=0.7)
        ax_qbin.set_title(f"{col} - Quantile Binning ({len(qcounts)} bins)")
        ax_qbin.set_xlabel("Count")

        # Cumulative Distribution
        total = qcounts.sum()
        cum_ratio = qcounts.cumsum() / total
        x = range(1, len(cum_ratio) + 1)
        ax_qcum.plot(x, cum_ratio.values, marker="o", linewidth=2)
        ax_qcum.axhline(y=0.8, color="red", linestyle="--", alpha=0.5, label="80%")
        ax_qcum.axhline(y=0.5, color="orange", linestyle="--", alpha=0.5, label="50%")
        ax_qcum.set_xticks(list(x))
        ax_qcum.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax_qcum.set_ylabel("Cumulative Ratio")
        ax_qcum.set_title(f"{col} - Quantile Cumulative Distribution")
        ax_qcum.set_ylim(0, 1.05)
        ax_qcum.legend()
    except Exception:
        for ax_fail in (ax_qbin, ax_qcum):
            ax_fail.text(0.5, 0.5, "Quantile binning failed\n(too few unique values)",
                         ha="center", va="center", transform=ax_fail.transAxes)
            ax_fail.set_title(f"{col} - Quantile Binning")

    plt.tight_layout()


def plot_categorical(df: pd.DataFrame, col: str, top_n: int = 20, ax=None):
    """범주형: Top N 바 차트 + 누적 분포 곡선"""
    vc = df[col].value_counts(dropna=False)
    if len(vc) == 0:
        return

    if ax is None:
        _, axes = plt.subplots(1, 2, figsize=(16, max(4, min(len(vc), top_n) * 0.35)))
        ax_bar, ax_cum = axes
    else:
        ax_bar = ax
        ax_cum = None

    # 1) Top N 바 차트
    top = vc.head(top_n)
    top_sorted = top.iloc[::-1]
    labels = [str(x)[:40] for x in top_sorted.index]
    counts = [int(v) for v in top_sorted.values]
    ax_bar.barh(labels, counts, edgecolor="black", alpha=0.7)
    ax_bar.set_title(f"{col} - Top {top_n} Values")
    ax_bar.set_xlabel("Count")
    ax_bar.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))

    # 2) 누적 분포 곡선
    if ax_cum is not None:
        total = vc.sum()
        cum_ratio = (vc.cumsum() / total).values
        x = range(1, len(cum_ratio) + 1)
        ax_cum.plot(x, cum_ratio, linewidth=1.5)
        ax_cum.set_title(f"{col} - Cumulative Distribution")
        ax_cum.set_xlabel("Number of Categories (ranked)")
        ax_cum.set_ylabel("Cumulative Ratio")
        ax_cum.axhline(y=0.9, color="red", linestyle="--", alpha=0.5, label="90%")
        ax_cum.axhline(y=0.5, color="orange", linestyle="--", alpha=0.5, label="50%")
        ax_cum.legend()
        ax_cum.set_ylim(0, 1.05)

    plt.tight_layout()


def plot_datetime(df: pd.DataFrame, col: str, ax=None):
    """시계열: 일별 건수 추이"""
    s = pd.to_datetime(df[col], errors="coerce").dropna()
    if len(s) == 0:
        return

    daily = s.dt.date.value_counts().sort_index()

    if ax is None:
        _, ax = plt.subplots(figsize=(12, 4))

    ax.plot(daily.index, daily.values, marker=".", markersize=3, linewidth=0.8)
    ax.set_title(f"{col} - Daily Count Trend")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=45)

    mean_val = daily.mean()
    ax.axhline(y=mean_val, color="red", linestyle="--", alpha=0.5, label=f"mean={mean_val:.0f}")
    ax.legend()
    plt.tight_layout()


def plot_text_lengths(df: pd.DataFrame, col: str, ax=None):
    """텍스트: 길이 분포 히스토그램"""
    lengths = df[col].dropna().astype(str).str.len()
    if len(lengths) == 0:
        return

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))

    ax.hist(lengths, bins=50, edgecolor="black", alpha=0.7)
    ax.set_title(f"{col} - Text Length Distribution")
    ax.set_xlabel("Length")
    ax.set_ylabel("Count")
    plt.tight_layout()


def flatten_json_column(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """JSON 컬럼을 펼쳐서 각 내부 필드를 독립 컬럼으로 가진 DataFrame을 반환한다.

    컬럼명은 '{원본컬럼}.{dotted_path}' 형태가 된다.
    """
    s = df[col].dropna().astype(str)
    rows = []
    for idx, v in s.items():
        try:
            parsed = json.loads(v.strip())
            flat = _flatten_json(parsed)
            flat["__idx__"] = idx
            rows.append(flat)
        except (json.JSONDecodeError, TypeError):
            pass

    if not rows:
        return pd.DataFrame()

    flat_df = pd.DataFrame(rows).set_index("__idx__")
    # 컬럼명에 원본 JSON 컬럼명을 prefix로 붙임
    flat_df.columns = [f"{col}.{c}" for c in flat_df.columns]
    return flat_df


def _classify_flat_json_columns(flat_df: pd.DataFrame) -> dict:
    """펼쳐진 JSON DataFrame의 각 컬럼을 타입 분류한다."""
    result = {}
    for c in flat_df.columns:
        series = flat_df[c]
        non_null = series.dropna()
        if len(non_null) == 0:
            result[c] = "categorical"
            continue

        # __list_length__ 필드는 numeric
        if c.endswith(".__list_length__"):
            result[c] = "numeric"
            continue

        # 값 타입 추론
        vtype = _infer_value_type(non_null.tolist())

        if vtype == "numeric":
            nunique = non_null.nunique()
            count = len(non_null)
            if count > 0 and nunique <= 20 and (nunique / count) < CATEGORICAL_UNIQUE_RATIO:
                result[c] = "categorical"
            else:
                result[c] = "numeric"
        elif vtype == "boolean":
            result[c] = "categorical"
        elif vtype == "string":
            str_vals = non_null.astype(str)
            nunique = str_vals.nunique()
            count = len(str_vals)
            avg_len = str_vals.str.len().mean()
            if nunique <= CATEGORICAL_MAX_UNIQUE or (count > 0 and nunique / count < CATEGORICAL_UNIQUE_RATIO):
                result[c] = "categorical"
            elif avg_len >= TEXT_AVG_LENGTH_THRESHOLD:
                result[c] = "text"
            else:
                result[c] = "categorical"
        else:
            result[c] = "categorical"

    return result


# ---------------------------------------------------------------------------
# 5. Full Table EDA Runner
# ---------------------------------------------------------------------------

_EDA_FUNCS = {
    "identifier": eda_identifier,
    "numeric": eda_numeric,
    "categorical": eda_categorical,
    "datetime": eda_datetime,
    "json": eda_json,
    "text": eda_text,
}

_PLOT_FUNCS = {
    "numeric": plot_numeric,
    "categorical": plot_categorical,
    "datetime": plot_datetime,
    "text": plot_text_lengths,
}


def run_table_eda(
    client: bigquery.Client,
    full_table_id: str,
    end_date: str | None = None,
    start_date: str | None = None,
    sample_size: int | None = None,
    plot: bool = True,
    top_n_categorical: int = 20,
) -> dict:
    """테이블 전체 EDA를 수행한다.

    - 테이블 타입(CDC/LOG/SNAPSHOT)을 스키마에서 자동 분류한다.
    - 날짜 범위는 타입별 기본값이 적용된다 (CDC: 90일, LOG: 7일, SNAPSHOT: 1일).
    - CDC 테이블은 PK 기준 deduplicate 후 EDA를 수행한다.

    Args:
        end_date: 분석 종료 날짜 (예: '2026-03-01'). None이면 날짜 필터 없음.
        start_date: 분석 시작 날짜. None이면 테이블 타입별 기본 기간 적용.
        sample_size: 최대 row 수. None이면 전체 데이터.
    """
    # 테이블 메타 파싱
    parts = full_table_id.split(".")
    dataset = parts[1] if len(parts) >= 3 else ""
    dataset_table = ".".join(parts[1:3]) if len(parts) >= 3 else ""

    print(f"=== EDA: {full_table_id} ===")

    # 1) 스키마 & 테이블 타입 분류
    bq_schema = get_bq_schema(client, full_table_id)
    table_type = detect_table_type(bq_schema, dataset)
    print(f"  Schema: {len(bq_schema)} columns | Type: {table_type}")

    # 2) 날짜 범위 결정
    date_col = detect_date_col(bq_schema) if end_date else None
    actual_start, actual_end = None, None
    if end_date and date_col:
        actual_start, actual_end = compute_date_range(end_date, table_type, start_date)
        print(f"  Date filter: {date_col} [{actual_start} ~ {actual_end}]")
    elif end_date and not date_col:
        print(f"  [WARN] No date column found in schema. Skipping date filter.")

    # 3) 데이터 로드
    df = fetch_data(client, full_table_id, date_col, actual_start, actual_end, sample_size)
    print(f"  Raw data: {df.shape[0]} rows x {df.shape[1]} cols")

    # 건수가 너무 적으면 날짜 필터 없이 재시도
    if len(df) < MIN_ROWS_THRESHOLD and date_col:
        print(f"  [WARN] {len(df)} rows (< {MIN_ROWS_THRESHOLD}) with date filter. Retrying without date filter...")
        df = fetch_data(client, full_table_id, sample_size=sample_size)
        actual_start, actual_end = None, None
        print(f"  Raw data (no filter): {df.shape[0]} rows x {df.shape[1]} cols")

    # 4) CDC deduplicate
    if table_type == "CDC" and len(df) > 0:
        pk_col = detect_cdc_pk(bq_schema, dataset_table)
        if pk_col:
            before = len(df)
            df = deduplicate_cdc(df, pk_col)
            print(f"  Dedup (PK={pk_col}): {before} -> {len(df)} rows")
        else:
            print(f"  [WARN] CDC table but no PK detected. Skipping dedup.")

    # 5) 컬럼 분류
    col_types = classify_columns(df, bq_schema)
    type_summary = {t: sum(1 for v in col_types.values() if v == t) for t in set(col_types.values())}
    print(f"  Column types: {type_summary}")

    # 5-1) JSON 컬럼 펼치기 → 내부 필드를 독립 컬럼으로 추가
    json_cols = [c for c, t in col_types.items() if t == "json"]
    json_flat_df = pd.DataFrame(index=df.index)
    json_flat_col_types = {}
    for jcol in json_cols:
        flat = flatten_json_column(df, jcol)
        if len(flat) > 0:
            # reindex to align with df
            flat = flat.reindex(df.index)
            json_flat_df = pd.concat([json_flat_df, flat], axis=1)
            flat_types = _classify_flat_json_columns(flat)
            json_flat_col_types.update(flat_types)
            n_fields = len(flat_types)
            flat_type_summary = {}
            for t in set(flat_types.values()):
                flat_type_summary[t] = sum(1 for v in flat_types.values() if v == t)
            print(f"  JSON '{jcol}' → {n_fields} fields expanded: {flat_type_summary}")

    # 6) 컬럼별 EDA (일반 컬럼)
    eda_results = []
    for col, col_type in col_types.items():
        eda_func = _EDA_FUNCS.get(col_type)
        if eda_func:
            try:
                if col_type == "categorical":
                    result = eda_func(df, col, top_n=top_n_categorical)
                else:
                    result = eda_func(df, col)
                eda_results.append(result)
            except Exception as e:
                print(f"  [WARN] EDA failed for {col} ({col_type}): {e}")
                eda_results.append({"column": col, "type": col_type, "error": str(e)})

    # 6-1) JSON 펼쳐진 필드별 EDA
    json_eda_results = []
    for col, col_type in json_flat_col_types.items():
        eda_func = _EDA_FUNCS.get(col_type)
        if eda_func:
            try:
                if col_type == "categorical":
                    result = eda_func(json_flat_df, col, top_n=top_n_categorical)
                else:
                    result = eda_func(json_flat_df, col)
                json_eda_results.append(result)
            except Exception as e:
                print(f"  [WARN] JSON field EDA failed for {col} ({col_type}): {e}")
                json_eda_results.append({"column": col, "type": col_type, "error": str(e)})
    eda_results.extend(json_eda_results)

    # 7) 시각화 (일반 컬럼)
    if plot:
        for col, col_type in col_types.items():
            if col_type == "json":
                continue  # JSON은 펼쳐진 필드로 시각화
            plot_func = _PLOT_FUNCS.get(col_type)
            if plot_func:
                try:
                    plot_func(df, col)
                    plt.show()
                except Exception as e:
                    print(f"  [WARN] Plot failed for {col} ({col_type}): {e}")

        # 7-1) JSON 펼쳐진 필드 시각화
        if json_flat_col_types:
            print(f"\n  --- JSON Expanded Fields Visualization ---")
            for col, col_type in json_flat_col_types.items():
                plot_func = _PLOT_FUNCS.get(col_type)
                if plot_func:
                    try:
                        plot_func(json_flat_df, col)
                        plt.show()
                    except Exception as e:
                        print(f"  [WARN] JSON field plot failed for {col} ({col_type}): {e}")

    return {
        "table": full_table_id,
        "table_type": table_type,
        "date_range": (actual_start, actual_end),
        "row_count": len(df),
        "column_types": col_types,
        "eda_results": eda_results,
        "dataframe": df,
    }


def print_eda_summary(eda_results: list[dict]):
    """EDA 결과를 요약 출력한다."""
    for r in eda_results:
        col = r.get("column", "?")
        col_type = r.get("type", "?")
        print(f"\n{'='*60}")
        print(f"  [{col_type.upper()}] {col}")
        print(f"{'='*60}")

        if "error" in r:
            print(f"  ERROR: {r['error']}")
            continue

        if col_type == "identifier":
            unique_str = "UNIQUE" if r.get("is_unique") else "NOT UNIQUE"
            print(f"  Total: {r['total_count']}  |  Cardinality: {r['cardinality']}  |  {unique_str}  |  Null: {r['null_count']} ({r['null_ratio']:.1%})")
            print(f"  Samples: {r.get('sample_values', [])}")
            continue

        if col_type == "numeric":
            print(f"  Count: {r['count']}  |  Null: {r['null_count']} ({r['null_ratio']:.1%})  |  Zero: {r['zero_count']} ({r['zero_ratio']:.1%})")
            print(f"  Mean: {r['mean']:.4f}  |  Median: {r['median']:.4f}  |  Std: {r['std']:.4f}")
            print(f"  Min: {r['min']}  |  Max: {r['max']}  |  Skewness: {r['skewness']:.4f}")
            print(f"  Percentiles: p1={r['p1']:.2f}  p5={r['p5']:.2f}  p25={r['p25']:.2f}  p75={r['p75']:.2f}  p95={r['p95']:.2f}  p99={r['p99']:.2f}")

        elif col_type == "categorical":
            print(f"  Total: {r['total_count']}  |  Cardinality: {r['cardinality']}  |  Null: {r['null_count']} ({r['null_ratio']:.1%})")
            print(f"  Top {r.get('top_n', '?')} coverage: {r['top_n_coverage']:.1%}")
            for val, info in list(r["top_n_values"].items())[:10]:
                print(f"    {val}: {info['count']} ({info['ratio']:.1%})")
            if "cumulative_distribution" in r:
                cd = r["cumulative_distribution"]
                parts = [f"Top {k.split('_')[1]}: {v:.1%}" for k, v in cd.items()]
                print(f"  Cumulative: {' | '.join(parts)}")

        elif col_type == "datetime":
            print(f"  Total: {r['total_count']}  |  Null: {r['null_count']} ({r['null_ratio']:.1%})")
            print(f"  Range: {r['min_date']} ~ {r['max_date']}")
            if "daily_count_stats" in r:
                ds = r["daily_count_stats"]
                print(f"  Daily counts - Mean: {ds['mean']}  Median: {ds['median']}  Std: {ds['std']}  Min: {ds['min']}  Max: {ds['max']}  Days: {ds['num_days']}")

        elif col_type == "json":
            print(f"  Total: {r['total_count']}  |  Null: {r['null_count']} ({r['null_ratio']:.1%})  |  Parseable: {r['parseable_count']}")
            root_dist = r.get("root_type_distribution", {})
            print(f"  Root types: dict={root_dist.get('dict',0)}  list={root_dist.get('list',0)}  other={root_dist.get('other',0)}")
            print(f"  Fields ({r.get('field_count', 0)}):")
            for path, info in r.get("fields", {}).items():
                vtype = info.get("inferred_type", "?")
                null_r = info.get("null_ratio", 0)
                line = f"    {path} ({vtype}) - count: {info['count']}, null: {null_r:.1%}"
                if vtype == "numeric":
                    line += f", mean: {info.get('mean','')}, median: {info.get('median','')}, min: {info.get('min','')}, max: {info.get('max','')}"
                elif vtype == "string":
                    line += f", cardinality: {info.get('cardinality','')}, avg_len: {info.get('avg_length','')}"
                    top = info.get("top_values", {})
                    if top:
                        top_str = ", ".join(f"{k}({v})" for k, v in list(top.items())[:3])
                        line += f", top: [{top_str}]"
                elif vtype == "boolean":
                    line += f", true_ratio: {info.get('true_ratio', 0):.1%}"
                print(line)

        elif col_type == "text":
            print(f"  Total: {r['total_count']}  |  Cardinality: {r['cardinality']}  |  Null: {r['null_count']} ({r['null_ratio']:.1%})")
            ls = r["length_stats"]
            print(f"  Length - Mean: {ls['mean']}  Median: {ls['median']}  Min: {ls['min']}  Max: {ls['max']}")
