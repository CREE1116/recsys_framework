#!/usr/bin/env python
import argparse
import json
import os
from collections import Counter

import numpy as np
import pandas as pd


# --------- 유틸: 구분자/헤더 감지 --------- #

def detect_delimiter(filepath, default=","):
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        first = f.readline()
    if not first:
        return default
    candidates = [",", "\t", ";", "|"]
    counts = {d: first.count(d) for d in candidates}
    best = max(counts, key=counts.get)
    return best if counts[best] > 0 else default


def looks_like_header(line, delimiter):
    parts = line.rstrip("\n").split(delimiter)
    if not parts:
        return False
    num_like = 0
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if p.replace(".", "", 1).isdigit():
            num_like += 1
    return num_like < len(parts) / 2


# --------- 유틸: 지니/엔트로피 --------- #

def gini_from_counts(counts: np.ndarray) -> float:
    """evaluation.py의 get_gini_index_from_recs와 같은 정의를 데이터 분포에 적용."""
    if counts.size == 0:
        return 0.0
    x = np.sort(counts)
    n = len(x)
    index = np.arange(1, n + 1)
    return float((np.sum((2 * index - n - 1) * x)) / (n * np.sum(x)))


def entropy_from_counts(counts: np.ndarray) -> float:
    if counts.size == 0:
        return 0.0
    p = counts / counts.sum()
    return float(-(p * np.log2(p + 1e-12)).sum())


# --------- 메인 분석 함수 --------- #

def analyze_dataset(
    input_path: str,
    user_col: int,
    item_col: int,
    header_mode: str = "auto",
    max_rows: int | None = None,
):
    """
    CF 데이터셋에 대한 기본 통계량을 계산한다.
    - user_col, item_col: 0-based 인덱스
    - header_mode: 'auto' | 'yes' | 'no'
    - max_rows: 너무 클 때 상위 N행만 읽고 샘플링 분석(옵션)
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(input_path)

    delim = detect_delimiter(input_path)
    print(f"[analyze] delimiter='{delim}'")

    # 헤더/컬럼 이름 결정
    with open(input_path, "r", encoding="utf-8", errors="ignore") as f:
        first = f.readline()
    if header_mode == "auto":
        has_header = looks_like_header(first, delim)
    elif header_mode == "yes":
        has_header = True
    else:
        has_header = False

    print(f"[analyze] header_detected={has_header}")

    if has_header:
        header = 0
        # 실제 헤더 이름을 쓰든, 그냥 generically 쓰든 상관 없음
        df = pd.read_csv(
            input_path,
            sep=delim,
            header=header,
            nrows=max_rows,
            engine="python",
        )
        # user_col, item_col는 인덱스로 접근
        cols = df.columns.tolist()
        user_col_name = cols[user_col]
        item_col_name = cols[item_col]
    else:
        df = pd.read_csv(
            input_path,
            sep=delim,
            header=None,
            nrows=max_rows,
            engine="python",
        )
        num_cols = df.shape[1]
        cols = [f"col{i}" for i in range(num_cols)]
        df.columns = cols
        user_col_name = cols[user_col]
        item_col_name = cols[item_col]

    # 기본 통계
    n_rows = len(df)
    n_users = df[user_col_name].nunique()
    n_items = df[item_col_name].nunique()
    density = n_rows / (n_users * n_items) if n_users and n_items else 0.0
    sparsity = 1.0 - density

    user_counts = df[user_col_name].value_counts()
    item_counts = df[item_col_name].value_counts()

    # 인기도 분포 기반 지니/엔트로피
    item_pop_counts = item_counts.to_numpy()
    gini_items = gini_from_counts(item_pop_counts)
    entropy_items = entropy_from_counts(item_pop_counts)

    # 상위 인기 아이템이 차지하는 비율 (ex: top 1%, 5%, 10%)
    sorted_counts = np.sort(item_pop_counts)[::-1]
    total_inter = sorted_counts.sum()
    def top_share(pct):
        k = max(1, int(len(sorted_counts) * pct))
        return float(sorted_counts[:k].sum() / total_inter) if total_inter > 0 else 0.0

    top1_share = top_share(0.01)
    top5_share = top_share(0.05)
    top10_share = top_share(0.10)

    stats = {
        "file": os.path.basename(input_path),
        "delimiter": delim,
        "has_header": has_header,
        "num_rows": int(n_rows),
        "num_users": int(n_users),
        "num_items": int(n_items),
        "density": float(density),
        "sparsity": float(sparsity),
        "user_interactions": {
            "mean": float(user_counts.mean()),
            "median": float(user_counts.median()),
            "min": int(user_counts.min()),
            "max": int(user_counts.max()),
        },
        "item_interactions": {
            "mean": float(item_counts.mean()),
            "median": float(item_counts.median()),
            "min": int(item_counts.min()),
            "max": int(item_counts.max()),
        },
        "item_popularity_gini": gini_items,
        "item_popularity_entropy": entropy_items,
        "item_popularity_top_share": {
            "top1pct": top1_share,
            "top5pct": top5_share,
            "top10pct": top10_share,
        },
    }

    return stats


def save_stats(stats: dict, output_dir: str, prefix: str | None = None):
    os.makedirs(output_dir, exist_ok=True)
    base = prefix or os.path.splitext(stats["file"])[0]

    json_path = os.path.join(output_dir, f"{base}_stats.json")
    txt_path = os.path.join(output_dir, f"{base}_stats.txt")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    # 사람이 보기 좋은 텍스트 요약
    lines = []
    lines.append(f"file: {stats['file']}")
    lines.append(f"delimiter: {stats['delimiter']}, has_header: {stats['has_header']}")
    lines.append(
        f"rows={stats['num_rows']:,}, users={stats['num_users']:,}, "
        f"items={stats['num_items']:,}"
    )
    lines.append(
        f"density={stats['density']:.6e}, sparsity={stats['sparsity']:.6e}"
    )
    ui = stats["user_interactions"]
    ii = stats["item_interactions"]
    lines.append(
        f"user_interactions: mean={ui['mean']:.2f}, "
        f"median={ui['median']:.2f}, min={ui['min']}, max={ui['max']}"
    )
    lines.append(
        f"item_interactions: mean={ii['mean']:.2f}, "
        f"median={ii['median']:.2f}, min={ii['min']}, max={ii['max']}"
    )
    lines.append(
        f"item_popularity_gini={stats['item_popularity_gini']:.4f}, "
        f"entropy={stats['item_popularity_entropy']:.4f}"
    )
    ts = stats["item_popularity_top_share"]
    lines.append(
        "item_popularity_top_share: "
        f"top1%={ts['top1pct']:.3f}, top5%={ts['top5pct']:.3f}, "
        f"top10%={ts['top10pct']:.3f}"
    )

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"[analyze] Saved stats to:\n  {json_path}\n  {txt_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CF 데이터셋의 기본 통계량을 계산하고 저장합니다."
    )
    parser.add_argument("--input", type=str, required=True,
                        help="입력 데이터셋 파일 경로 (예: data/ml-20m/ratings.csv)")
    parser.add_argument("--user-col", type=int, default=0,
                        help="user id 컬럼 인덱스 (0-based)")
    parser.add_argument("--item-col", type=int, default=1,
                        help="item id 컬럼 인덱스 (0-based)")
    parser.add_argument("--header", type=str,
                        choices=["auto", "yes", "no"], default="auto",
                        help="헤더 자동 감지 또는 강제 지정")
    parser.add_argument("--max-rows", type=int, default=None,
                        help="상위 N행만 읽어서 샘플링 분석 (None이면 전체)")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="통계 파일을 저장할 디렉토리 "
                             "(기본: 입력 파일이 있는 디렉토리)")

    args = parser.parse_args()

    stats = analyze_dataset(
        input_path=args.input,
        user_col=args.user_col,
        item_col=args.item_col,
        header_mode=args.header,
        max_rows=args.max_rows,
    )

    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.input))
    prefix = None  # 기본은 파일 이름 기반
    save_stats(stats, out_dir, prefix=prefix)

