"""Measure how many queries written without the catalog can be expressed in its vocabulary.

The v3 training data can only teach what the condition catalog can express. To see how much of
what people search for lies outside it, queries that were written without the catalog are
parsed back into the catalog vocabulary with the same parser that verifies generated
paraphrases (generate_condition_queries.verify). No image labels are needed.

The default query sets are the v1 and v2 test queries, which the teachers wrote freely for
images. A query is
  - vocabulary-expressible when the parser puts nothing into `other` and names every object
    kind; the method could then produce data for it by adding a condition, and
  - a catalog condition when its parsed structure also equals a catalog condition.
Everything the parser puts into `other` is grouped by rules in `GAP_RULES`. The parser can also
drop a word it has no value for (「緑色のバン」 comes back as a van with no colour), so
`TEXT_RULES` checks the query text itself for the out-of-vocabulary words seen in these sets.

    python scripts/measure_catalog_coverage.py parse
    python scripts/measure_catalog_coverage.py report

With --extension the parser also knows the extension catalog's vocabulary (yellow/green and
parked; docs/search-condition-catalog-extension.md), so the gain from the extension can be read
off the same query sets.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import time

try:
    from .build_condition_dataset import append, now, request_json
    from .condition_data import ROOT, SCENES, load_conditions, load_extension_conditions, read_jsonl, write_json
    from .generate_condition_queries import (DEFAULT_TEACHER, PARSE_PROMPT, canonical_parse, canonical_target, chat,
                                             glossary, parse_schema)
except ImportError:
    from build_condition_dataset import append, now, request_json
    from condition_data import ROOT, SCENES, load_conditions, load_extension_conditions, read_jsonl, write_json
    from generate_condition_queries import (DEFAULT_TEACHER, PARSE_PROMPT, canonical_parse, canonical_target, chat,
                                            glossary, parse_schema)

OUT = ROOT/'datasets/catalog_coverage'
OUT_EXTENSION = ROOT/'datasets/catalog_coverage_extension'
QUERY_SETS = {'v1_test': ROOT/'datasets/dashcam_reranker_ft_v1/pairs.test.jsonl',
              'v2_test': ROOT/'datasets/dashcam_reranker_ft_v2_qwen38/pairs.test.jsonl'}


def load_queries():
    rows = []
    for name, path in QUERY_SETS.items():
        texts = {}
        for pair in read_jsonl(path): texts.setdefault(pair['query_id'], pair['query_text'])
        rows += [{'set': name, 'query_id': q, 'text': t} for q, t in texts.items()]
    return rows


def run_parse(args):
    args.out.mkdir(parents=True, exist_ok=True)
    tags = request_json(f'{args.ollama_url}/api/tags')['models']
    config = {'teacher': args.teacher, 'teacher_digest': next(m['digest'] for m in tags if m['name'] == args.teacher),
              'parse_prompt_sha256': hashlib.sha256((PARSE_PROMPT+glossary(args.extension, parse=True)).encode()).hexdigest(),
              'parse_schema_sha256': hashlib.sha256(json.dumps(parse_schema(args.extension), sort_keys=True).encode()).hexdigest(),
              'query_sets': {n: hashlib.sha256(p.read_bytes()).hexdigest() for n, p in QUERY_SETS.items()}}
    config_path = args.out/'config.json'
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise SystemExit(f'{config_path} differs from the current settings; use a different --out')
    write_json(config_path, config)
    log = args.out/'parsed.jsonl'
    done = {(r['set'], r['query_id']) for r in read_jsonl(log)} if log.exists() else set()
    queries = load_queries()
    for index, q in enumerate(queries, 1):
        if (q['set'], q['query_id']) in done: continue
        started = time.monotonic()
        # Same call as generate_condition_queries.verify: deterministic, text only.
        parsed = chat(args.ollama_url, args.teacher,
                      PARSE_PROMPT.format(query=q['text'], glossary=glossary(args.extension, parse=True), scenes=', '.join(SCENES)),
                      parse_schema(args.extension), temperature=0, seed=0, num_predict=1024)
        append(log, {**q, 'parsed': parsed, 'created_at': now(), 'elapsed_seconds': time.monotonic()-started})
        print(f"[{index}/{len(queries)}] {q['text']} -> other={parsed['other']}", flush=True)
    run_report(args)


# Checked in order; the first matching rule names the gap. Written after reading the parser's
# `other` entries, so that each group is one kind of condition the catalog does not cover.
GAP_RULES = [
    ('停止・駐車', r'停|止ま|止し|駐車'),
    ('走行などの動き', r'走|進|移動|渡|歩い|歩く|曲が|追い越|追越|発進|接近|近づ|離れ|向かっ|逆走|右折|左折|車線変更|通過|待|すれ違|横断|横切|来る|来て'),
    ('台数', r'[0-9０-９一二三四五六七八九十数複]+(台|人)|複数|多く|たくさん|人々|大勢|群|渋滞'),
    ('否定', r'ない|なし|無い|以外|いない'),
    ('一覧にない車種・色', r'セダン|SUV|ワゴン|軽|タクシー|トレーラー|ダンプ|ミニバン|ピックアップ|スクーター|銀|シルバー|灰|グレー|黄|緑|茶|オレンジ|紫|ベージュ|金|色'),
    ('自車より後ろ・対象同士の位置関係', r'後方|後ろ|手前|奥|横|並|間|挟|付近|そば|近く|脇|隣に|直前|前に|前を|先'),
    ('一覧にない場所・設備', r'駐車場|路肩|中央分離帯|標識|看板|ガードレール|建物|高架|踏切|ロータリー|料金所|街灯|電柱|ゴミ|店|ビル|木|植|壁|柵|バス停|ホーム|線路|合流|分岐|車線数|[0-9一二三四五六七八九]車線'),
]
# Entries that name something the glossary already maps (「前方」 is 自車と同じ車線, 都市部 is 市街地).
MAPPED = re.compile(r'^(自車の?)?前方(の車線|にある)?$|^(都市部|都市|街中|市街|街|街路|住宅街)(の道路)?$|^(夜間|昼間)$')


# Matched against the query text, because the parser may silently drop these words.
TEXT_RULES = [
    ('停止・駐車', r'停車|停止|停ま|止ま|駐車'),
    ('走行などの動き', r'走|渡っ|渡る|歩い|歩く|曲が|追い越|発進|進ん|進む|進行|向かっ|逆走|右折|左折|車線変更|通過|待っ|接近|近づ|離れ|すれ違|横断|横切|来る|来て|来てい'),
    ('一覧にない車種・色', r'銀|シルバー|灰|グレー|黄色|緑|茶色|オレンジ|紫|ベージュ|金色|セダン|SUV|ワゴン|軽自動車|タクシー|トレーラー|ダンプ|ミニバン|スクーター|ピックアップ'),
    ('台数', r'[0-9０-９一二三四五六七八九十数複]+(台|人)|複数|多くの|たくさん|大勢|渋滞|列をなす'),
]


def gap_of(item):
    return next((name for name, pattern in GAP_RULES if re.search(pattern, item)), 'その他')


# With the extension vocabulary, these words are gaps only if the parser did not map them.
EXTENSION_COLORS = re.compile(r'黄色|緑')
EXTENSION_PARKED = re.compile(r'駐車|駐輪|停めて|停めら')
# 「路肩に停車」 is parking under the extension definition; 停車 alone stays a gap (signal waits).
ROADSIDE = re.compile(r'路肩|駐車場|道路脇|路上')
# The parser often sets parked=yes and also lists the same phrase in `other`.
PARKING_RESTATEMENT = re.compile(r'^(道路の)?([左右]側の)?(路肩|駐車場|道路脇|路上)?に?(駐車|停車|駐輪)(している|した|されている|中)?$')


def gaps_of(row, extension=False):
    parsed = row['parsed']
    parked = extension and any(o.get('parked') == 'yes' for o in parsed['objects'])
    groups = {gap_of(item) for item in parsed['other']
              if not MAPPED.search(item) and not (parked and PARKING_RESTATEMENT.search(item))}
    for name, pattern in TEXT_RULES:
        words = set(re.findall(pattern, row['text']))
        if extension and name == '一覧にない車種・色':
            mapped = any(o['color'] in ('yellow', 'green') for o in parsed['objects'])
            words = {w for w in words if not (mapped and EXTENSION_COLORS.search(w))}
        if extension and name == '停止・駐車':
            mapped = any(o.get('parked') == 'yes' for o in parsed['objects'])
            roadside = bool(ROADSIDE.search(row['text']))
            words = {w for w in words if not (mapped and (EXTENSION_PARKED.search(w) or roadside))}
        if words: groups.add(name)
    if any(o['kind'] == 'other' for o in parsed['objects']): groups.add('種類があいまいな対象')
    return groups


# Gap groups set aside one after another, to show how far each extension of the catalog would go.
CUMULATIVE = [['停止・駐車', '走行などの動き'], ['一覧にない車種・色']]


def run_report(args):
    rows = read_jsonl(args.out/'parsed.jsonl')
    conditions = load_conditions() + (load_extension_conditions() if args.extension else [])
    catalog = {json.dumps(t) for c in conditions if (t := canonical_target(c['expression'])) is not None}
    summary, examples = {}, {}
    for name in QUERY_SETS:
        subset = [r for r in rows if r['set'] == name]
        counts, gaps, only_gap = Counter(), Counter(), Counter()
        ignoring = [0]*len(CUMULATIVE)
        for r in subset:
            parsed = r['parsed']
            groups = gaps_of(r, args.extension)
            gaps.update(groups)
            if len(groups) == 1: only_gap[next(iter(groups))] += 1
            for i in range(len(CUMULATIVE)):
                if groups <= {g for step in CUMULATIVE[:i+1] for g in step}: ignoring[i] += 1
            if not groups:
                counts['vocabulary'] += 1
                if json.dumps(canonical_parse(parsed)) in catalog: counts['catalog'] += 1
            for item in parsed['other']:
                if not MAPPED.search(item): examples.setdefault(gap_of(item), Counter())[item] += 1
        n = len(subset)
        summary[name] = {'queries': n, 'vocabulary_expressible': counts['vocabulary']/n,
                         'catalog_condition': counts['catalog']/n,
                         'expressible_ignoring': [{'ignored': sum(CUMULATIVE[:i+1], []), 'share': ignoring[i]/n}
                                                  for i in range(len(CUMULATIVE))],
                         'queries_with_gap': {k: v/n for k, v in gaps.most_common()},
                         'queries_whose_only_gap_is': {k: v/n for k, v in only_gap.most_common()}}
    summary['gap_examples'] = {k: v.most_common(15) for k, v in examples.items()}
    write_json(args.out/'report.json', summary)
    print(json.dumps(summary, ensure_ascii=False, indent=1))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', choices=['parse', 'report'])
    parser.add_argument('--out', type=Path)
    parser.add_argument('--extension', action='store_true', help='parse with the extension catalog vocabulary')
    parser.add_argument('--teacher', default=DEFAULT_TEACHER)
    parser.add_argument('--ollama-url', default='http://localhost:11434')
    args = parser.parse_args()
    args.out = args.out or (OUT_EXTENSION if args.extension else OUT)
    run_parse(args) if args.command == 'parse' else run_report(args)


if __name__ == '__main__':
    main()
