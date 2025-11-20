#!/usr/bin/env python
import argparse
import os
import random
from collections import defaultdict


def detect_delimiter(filepath, default=","):
    """첫 줄을 보고 구분자를 추정한다."""
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        first = f.readline()
    if not first:
        return default

    candidates = [",", "\t", ";", "|"]
    counts = {d: first.count(d) for d in candidates}
    best = max(counts, key=counts.get)
    return best if counts[best] > 0 else default


def looks_like_header(line, delimiter):
    """첫 줄이 헤더처럼 보이는지 대략 판별한다."""
    parts = line.rstrip("\n").split(delimiter)
    if not parts:
        return False

    num_like = 0
    for p in parts:
        p = p.strip()
        if not p:
            continue
        # 거의 숫자/점만 있으면 숫자 컬럼으로 본다.
        if p.replace(".", "", 1).isdigit():
            num_like += 1

    # 숫자 컬럼이 절반보다 적으면 헤더일 확률이 높다.
    return num_like < len(parts) / 2


def count_lines(filepath):
    """헤더 포함 전체 라인 수를 센다."""
    lines = 0
    with open(filepath, "rb") as f:
        for _ in f:
            lines += 1
    return lines


def create_line_subset(input_path, output_path, percentage,
                       has_header="auto", seed=42):
    """
    라인 단위 랜덤 서브셋 생성.
    - 헤더(있으면) 항상 첫 줄에 유지.
    - 나머지 데이터 라인에서 percentage% 샘플.
    """
    if not os.path.exists(input_path):
        print(f"오류: 입력 파일 '{input_path}'를 찾을 수 없습니다.")
        return

    random.seed(seed)

    print(f"[line-mode] '{input_path}' 파일의 {percentage}% 서브셋 생성을 시작합니다...")

    # 구분자 / 헤더 자동 감지
    delim = detect_delimiter(input_path)
    with open(input_path, "r", encoding="utf-8", errors="ignore") as f:
        first = f.readline()
    if has_header == "auto":
        header_flag = looks_like_header(first, delim)
    else:
        header_flag = bool(has_header)

    # 전체 라인 수
    total_lines = count_lines(input_path)
    print(f"총 {total_lines:,} 라인 (헤더 추정: {header_flag})")

    if total_lines <= int(header_flag):
        print("데이터 라인이 없습니다. 작업을 중단합니다.")
        return

    data_lines = total_lines - int(header_flag)
    num_to_sample = max(1, int(data_lines * (percentage / 100.0)))
    print(f"데이터 라인 {data_lines:,}개 중 {num_to_sample:,}개를 랜덤 샘플링합니다.")

    # 데이터 라인 인덱스 기준으로 샘플링 (0 ~ data_lines-1)
    sample_indices = set(random.sample(range(data_lines), num_to_sample))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    lines_written = 0
    with open(input_path, "r", encoding="utf-8", errors="ignore") as infile, \
         open(output_path, "w", encoding="utf-8") as outfile:

        # 헤더 처리
        if header_flag:
            outfile.write(first)
        else:
            # 헤더 없음이면 첫 줄도 데이터로 다시 처리
            infile.seek(0)

        # 데이터 라인 스트리밍
        data_idx = 0
        for line in infile:
            if data_idx in sample_indices:
                outfile.write(line)
                lines_written += 1
            data_idx += 1

    print("\n[line-mode] 서브셋 생성이 완료되었습니다.")
    print(f"  - 원본 데이터 라인 수 (헤더 제외): {data_lines:,}")
    print(f"  - 저장된 데이터 라인 수: {lines_written:,}")


def create_user_subset(input_path, output_path, percentage,
                       user_col=0, has_header="auto", seed=42):
    """
    유저 단위 서브셋 생성:
    - 1패스: 유저별로 해당 라인들을 모두 모음
    - 유저 중 percentage%만 랜덤 선택
    - 선택된 유저의 모든 라인을 서브셋으로 저장
    """
    if not os.path.exists(input_path):
        print(f"오류: 입력 파일 '{input_path}'를 찾을 수 없습니다.")
        return

    random.seed(seed)

    print(f"[user-mode] '{input_path}' 파일의 {percentage}% (유저 기준) 서브셋 생성을 시작합니다...")

    # 구분자 / 헤더 자동 감지
    delim = detect_delimiter(input_path)
    with open(input_path, "r", encoding="utf-8", errors="ignore") as f:
        first = f.readline()
    if has_header == "auto":
        header_flag = looks_like_header(first, delim)
    else:
        header_flag = bool(has_header)

    # 1패스: 유저별 라인 수집
    user_to_lines = defaultdict(list)
    header_line = first if header_flag else None

    with open(input_path, "r", encoding="utf-8", errors="ignore") as f:
        if header_flag:
            _ = f.readline()  # 이미 first 읽었으므로 한 줄 버림
        for line in f:
            parts = line.rstrip("\n").split(delim)
            if len(parts) <= user_col:
                continue
            user_id = parts[user_col]
            user_to_lines[user_id].append(line)

    num_users = len(user_to_lines)
    print(f"총 유저 수: {num_users:,}")

    if num_users == 0:
        print("유저가 없습니다. 작업을 중단합니다.")
        return

    num_users_to_sample = max(1, int(num_users * (percentage / 100.0)))
    print(f"{num_users_to_sample:,}명의 유저를 랜덤 샘플링합니다.")

    sampled_users = set(random.sample(list(user_to_lines.keys()), num_users_to_sample))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    total_lines_written = 0
    with open(output_path, "w", encoding="utf-8") as out:
        if header_line is not None:
            out.write(header_line)

        for u in sampled_users:
            for line in user_to_lines[u]:
                out.write(line)
                total_lines_written += 1

    print("\n[user-mode] 서브셋 생성이 완료되었습니다.")
    print(f"  - 샘플링된 유저 수: {len(sampled_users):,}")
    print(f"  - 저장된 라인 수: {total_lines_written:,}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("대용량 데이터셋의 랜덤 서브셋을 생성합니다.")
    parser.add_argument("--input", type=str, required=True,
                        help="입력 데이터셋 파일 경로. 예: data/ml-20m/ratings.csv")
    parser.add_argument("--output", type=str, required=True,
                        help="출력 서브셋 파일 경로. 예: data/ml-2m/ratings_subset.csv")
    parser.add_argument("--percentage", type=float, required=True,
                        help="샘플링할 비율 (0-100). 예: 10")
    parser.add_argument("--mode", type=str, choices=["line", "user"], default="line",
                        help="'line'은 라인 단위 샘플, 'user'는 유저 일부만 선택해 그 유저의 모든 라인 사용")
    parser.add_argument("--user-col", type=int, default=0,
                        help="user id가 위치한 컬럼 인덱스 (0-based)")
    parser.add_argument("--header", type=str, choices=["auto", "yes", "no"], default="auto",
                        help="'auto'면 자동 감지, 'yes'면 첫 줄을 헤더로 강제, 'no'면 헤더 없음으로 간주")
    parser.add_argument("--seed", type=int, default=42,
                        help="랜덤 시드(재현성 용)")

    args = parser.parse_args()

    if not 0 < args.percentage <= 100:
        print("오류: percentage는 0보다 크고 100보다 작거나 같아야 합니다.")
        raise SystemExit(1)

    header_flag = args.header
    if header_flag == "yes":
        header_flag = True
    elif header_flag == "no":
        header_flag = False  # 'auto'는 그대로 문자열로 전달

    if args.mode == "line":
        create_line_subset(
            args.input,
            args.output,
            args.percentage,
            has_header=header_flag,
            seed=args.seed,
        )
    else:
        create_user_subset(
            args.input,
            args.output,
            args.percentage,
            user_col=args.user_col,
            has_header=header_flag,
            seed=args.seed,
        )
