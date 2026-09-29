"""Add the facts that the extended catalog needs to the saved v3 facts, without redoing them.

The coverage check (measure_catalog_coverage.py) found that queries written without the catalog
often ask for parked vehicles and for yellow or green vehicles, which the v3 facts cannot express:
the colour field has only white/black/red/blue, and there is no parked field. This pass keeps
every v3 fact and asks the same teacher only about those two things.

Each vehicle object already recorded in the v3 facts is drawn on the image as a numbered box,
and the teacher answers per number:
  - parked: yes = left outside the travelled lanes (road shoulder, parking bay, a row of parked
    cars along the kerb); no = in a travelled lane, including waiting at a signal or in a queue;
    unknown otherwise.
  - color: yellow, green, other (any other body colour), or unknown.

The answers are stored under new keys (`parked`, `color_ext`), so every v3 condition keeps its
label. `color_ext` takes the v3 colour into account: an object that v3 recorded as white, black,
red or blue is `other` here whatever the teacher says (see merge_facts).

    python scripts/annotate_extension_facts.py annotate
    python scripts/annotate_extension_facts.py report
"""
from __future__ import annotations

import argparse
import base64
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import time
import urllib.error

from PIL import Image, ImageDraw, ImageFont

try:
    from .build_condition_dataset import append, now, request_json
    from .condition_data import ROOT, STATES, digest, enum, obj, read_jsonl, validate_schema, write_json
except ImportError:
    from build_condition_dataset import append, now, request_json
    from condition_data import ROOT, STATES, digest, enum, obj, read_jsonl, validate_schema, write_json

VERSION = 'extension-facts-v1'
SOURCE = ROOT/'datasets/dashcam_reranker_v3_conditions'
OUT = ROOT/'datasets/dashcam_reranker_v4_extension'
DEFAULT_TEACHER = 'qwen3.8-27b-mtp:UD-Q3_K_XL'
VEHICLES = ('car', 'bus', 'truck', 'van', 'motorcycle', 'bicycle')
EXT_COLORS = ['yellow', 'green', 'other', 'unknown']
FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
# Boxes shorter than this (per mille of image height, 20 px on 720 px images) are too small to
# judge parking or colour; the trial answered them as confidently as large ones. The same
# threshold is used for BDD boxes in build_bdd_eval.py.
MIN_BOX_HEIGHT = 28

PROMPT = '''車載静止画に、記録済みの乗り物を番号付きの枠で示しています。枠と番号は目印です。枠の色や線を対象の色として扱わないでください。
番号ごとに、枠の中の乗り物について次の2点を判定してください。

parked（駐車しているか）:
- yes: 走行する車線の外に停めてある。路肩・駐車場・駐車枠・道路脇の駐車の列にいる、歩道側に寄せて停めてある、駐輪されている等の根拠がある。
- no: 走行する車線上にいて、交通の流れの中にいる。信号待ちや渋滞で止まっている場合もno。
- unknown: 小さい・遮蔽・車線の境界が見えない等で判別できない。止まっているか動いているかを推測で決めない。

color（車体の主な塗装色）:
- yellow: 黄色（タクシー等の黄色を含む）。green: 緑色。other: それ以外の色（白・黒・赤・青・銀・灰など）。unknown: 暗い・小さい・遮蔽・照明で色を判別できない。
- 自転車はフレーム、バイクは車体・外装の色。乗り手の服・ライト・反射の色を転用しない。

対象の一覧: {targets}
evidenceは簡潔な日本語で根拠を書いてください。出力は指定のJSONのみ。'''


def targets(facts):
    """Vehicle objects that can be marked on the image."""
    return [o for o in facts['objects'] if o['kind'] in VEHICLES and o['bbox'] is not None]


def schema(ids):
    item = obj({'id': enum(ids), 'parked': enum(STATES), 'color': enum(EXT_COLORS),
                'evidence': {'type': 'string', 'minLength': 1, 'maxLength': 200}})
    return obj({'objects': {'type': 'array', 'items': item, 'minItems': len(ids), 'maxItems': len(ids)}})


def marked_image(path, objects):
    """JPEG bytes with each object's box and number; thin black/white lines keep colours readable."""
    image = Image.open(path).convert('RGB')
    w, h = image.size
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(FONT, max(14, h//36))
    for o in objects:
        x1, y1, x2, y2 = (o['bbox'][0]*w/1000, o['bbox'][1]*h/1000, o['bbox'][2]*w/1000, o['bbox'][3]*h/1000)
        # Degenerate teacher boxes (zero width or height) still get a visible mark.
        x2, y2 = max(x2, x1 + 2), max(y2, y1 + 2)
        draw.rectangle([x1, y1, x2, y2], outline='black', width=3)
        draw.rectangle([x1+1, y1+1, x2-1, y2-1], outline='white', width=1)
        label = o['id'].lstrip('o')
        box = draw.textbbox((x1, y1), label, font=font)
        top = max(0, y1 - (box[3]-box[1]) - 4)
        draw.rectangle([x1, top, x1 + (box[2]-box[0]) + 6, top + (box[3]-box[1]) + 4], fill='white', outline='black')
        draw.text((x1 + 3, top), label, fill='black', font=font)
    buffer = io.BytesIO()
    image.save(buffer, format='JPEG', quality=92)
    return buffer.getvalue()


def describe(objects):
    kinds = {'car': '乗用車', 'bus': 'バス', 'truck': 'トラック', 'van': 'バン', 'motorcycle': 'バイク', 'bicycle': '自転車'}
    return '、'.join(f"{o['id'].lstrip('o')}={o['id']}（{kinds[o['kind']]}）" for o in objects)


def ask(args, config, record, objects):
    ids = [o['id'] for o in objects]
    encoded = base64.b64encode(marked_image(ROOT/record['image_path'], objects)).decode()
    prompt = PROMPT.format(targets=describe(objects))
    error = None
    for attempt in (1, 2):
        text = prompt if error is None else prompt + f'\n前回の出力に誤りがありました（{error}）。すべての番号に1件ずつ答えてください。'
        payload = {'model': config['teacher'], 'stream': False, 'think': False, 'format': schema(ids),
                   'options': {'temperature': 0, 'num_ctx': 8192, 'num_predict': 2048},
                   'messages': [{'role': 'user', 'content': text, 'images': [encoded]}]}
        try:
            response = request_json(args.ollama_url + '/api/chat', payload, args.timeout)
            if response.get('done_reason') != 'stop': raise ValueError('output did not finish')
            answer = json.loads(response['message']['content'])
            validate_schema(answer, schema(ids))
            if sorted(a['id'] for a in answer['objects']) != sorted(ids): raise ValueError('ids do not match the marked objects')
            return answer, attempt
        except (ValueError, KeyError, json.JSONDecodeError) as exc:
            error = exc
            append(args.out/'annotations/extension_errors.jsonl', {'image_id': record['image_id'], 'attempt': attempt,
                                                                   'error': str(exc), 'created_at': now()})
    return None, 2


def run_annotate(args):
    source = read_jsonl(args.source/'annotations/facts.jsonl')
    if args.limit: source = source[:args.limit]
    tags = request_json(args.ollama_url + '/api/tags')['models']
    config = {'version': VERSION, 'teacher': args.teacher,
              'teacher_digest': next(m['digest'] for m in tags if m['name'] == args.teacher),
              'prompt_sha256': hashlib.sha256(PROMPT.encode()).hexdigest(),
              'source_facts_sha256': digest(args.source/'annotations/facts.jsonl'),
              'vehicles': list(VEHICLES), 'temperature': 0}
    config_path = args.out/'annotations/extension_config.json'
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise SystemExit(f'{config_path} differs from the current settings; use a different --out')
    write_json(config_path, config)
    log = args.out/'annotations/extension_facts.jsonl'
    done = {r['image_id'] for r in read_jsonl(log)}
    started, generated = time.monotonic(), 0
    for index, record in enumerate(source, 1):
        if record['image_id'] in done: continue
        objects = targets(record['facts'])
        begin = time.monotonic()
        answer, attempts = (({'objects': []}, 0) if not objects else ask(args, config, record, objects))
        row = {'version': VERSION, 'image_id': record['image_id'], 'split': record['split'],
               'created_at': now(), 'elapsed_seconds': time.monotonic()-begin, 'attempts': attempts,
               'status': 'ok' if answer is not None else 'failed',
               'objects': {a['id']: {k: a[k] for k in ('parked', 'color', 'evidence')} for a in (answer or {'objects': []})['objects']}}
        append(log, row)
        generated += bool(objects)
        elapsed = time.monotonic() - started
        eta = elapsed/max(generated, 1)*(len(source)-index)/3600
        print(f"[{index}/{len(source)}] {record['image_id']} objects={len(objects)} status={row['status']} "
              f"seconds={row['elapsed_seconds']:.1f} ETA={eta:.2f}h", flush=True)
    run_report(args)


def merge_facts(facts, extension):
    """v3 facts plus `parked` and `color_ext` on every object; unanswered or small objects stay unknown."""
    merged = json.loads(json.dumps(facts))
    for o in merged['objects']:
        answer = (extension or {}).get(o['id'])
        if o['bbox'] is not None and o['bbox'][3] - o['bbox'][1] < MIN_BOX_HEIGHT: answer = None
        o['parked'] = answer['parked'] if answer else 'unknown'
        if o['color'] in ('white', 'black', 'red', 'blue'): o['color_ext'] = 'other'
        else: o['color_ext'] = answer['color'] if answer else 'unknown'
    return merged


def load_merged(source=SOURCE, out=OUT):
    """{image_id: record with merged facts} for every v3 image; fails if the pass is incomplete."""
    extension = {r['image_id']: r for r in read_jsonl(out/'annotations/extension_facts.jsonl')}
    records = read_jsonl(source/'annotations/facts.jsonl')
    missing = [r['image_id'] for r in records if r['image_id'] not in extension]
    if missing: raise SystemExit(f'extension facts missing for {len(missing)} images')
    return {r['image_id']: {**r, 'facts': merge_facts(r['facts'], extension[r['image_id']]['objects']
                                                       if extension[r['image_id']]['status'] == 'ok' else None)}
            for r in records}


def run_report(args):
    rows = read_jsonl(args.out/'annotations/extension_facts.jsonl')
    parked, color, by_kind = Counter(), Counter(), Counter()
    source = {r['image_id']: r for r in read_jsonl(args.source/'annotations/facts.jsonl')}
    for r in rows:
        kinds = {o['id']: o['kind'] for o in source[r['image_id']]['facts']['objects']}
        for oid, a in r['objects'].items():
            parked[a['parked']] += 1; color[a['color']] += 1
            if a['parked'] == 'yes': by_kind[('parked', kinds[oid])] += 1
            if a['color'] in ('yellow', 'green'): by_kind[(a['color'], kinds[oid])] += 1
    summary = {'images': len(rows), 'status': dict(Counter(r['status'] for r in rows)),
               'objects': sum(len(r['objects']) for r in rows), 'parked': dict(parked), 'color': dict(color),
               'positives_by_kind': {f'{a}:{k}': v for (a, k), v in sorted(by_kind.items())},
               'seconds_per_image': sum(r['elapsed_seconds'] for r in rows)/max(1, len(rows))}
    write_json(args.out/'annotations/extension_report.json', summary)
    print(json.dumps(summary, ensure_ascii=False, indent=1))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', choices=['annotate', 'report'])
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--teacher', default=DEFAULT_TEACHER)
    parser.add_argument('--ollama-url', default='http://localhost:11434')
    parser.add_argument('--timeout', type=float, default=300)
    parser.add_argument('--limit', type=int, help='annotate only the first N images (trial run)')
    args = parser.parse_args()
    run_annotate(args) if args.command == 'annotate' else run_report(args)


if __name__ == '__main__':
    main()
