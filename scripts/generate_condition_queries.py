"""Generate paraphrased queries for fixed conditions from text only, then verify them.

The teacher never sees an image here. Each condition's scene sentence is rendered into
several Japanese queries, and every query is parsed back into the condition vocabulary
by a separate call that sees only the query text. A query passes only when the parsed
structure equals the target expression and nothing outside the vocabulary was added.
Labels are not touched: pairs keep using the saved condition judgments.
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
    from .condition_data import ROOT, SCENES, load_conditions, read_jsonl, stable_key, write_json
    from .build_condition_dataset import append, now, request_json
except ImportError:
    from condition_data import ROOT, SCENES, load_conditions, read_jsonl, stable_key, write_json
    from build_condition_dataset import append, now, request_json

VERSION = 'condition-queries-v3'
DEFAULT_OUT = ROOT/'datasets/dashcam_reranker_v3_paraphrase'
DEFAULT_TEACHER = 'qwen3.8-27b-mtp:UD-Q3_K_XL'
# An upper bound only: the prompt lets the teacher stop early instead of padding with extra conditions.
MAX_COUNT = 10
# X conditions use relations, counts, negation or OR, which the parse schema does not express.
UNSUPPORTED_PREFIXES = ('X', 'D')

GLOSSARY = '''用語の意味と、使ってよい言い換え:
- 車: 乗用車のこと。言い換え可「乗用車」「自動車」。バン・トラック・バスを含む「車両」は不可。
- バン／トラック／バス／バイク（言い換え可「オートバイ」。自転車とは別の種類で、「自転車」「二輪車」と書かない）／自転車／歩行者（言い換え可「人」）／列車（言い換え可「電車」）／動物／緊急車両（救急車・消防車・パトカーなどの総称。個別名だけにしない）
- 色（白・黒・赤・青）は車体の色。
- 画面左／画面中央／画面右: 画像を横に3等分した位置。言い換え可「左側」「画面の左」「真ん中」「右側」など。
- 自車と同じ車線: 言い換え可「同じ車線」「自車線」「前方」。自車の左隣の車線: 「左隣の車線」「左の車線」。自車の右隣の車線: 「右隣の車線」「右の車線」。対向車線: 「反対車線」。
- 歩道上／横断歩道上／車道上: 対象がその上にいること。
- こちらに正面を向けた: 言い換え可「正面を向いた」「こちらを向いた」。後ろ姿が見える: 「後ろ姿の」「背面が見える」。
- 市街地／住宅街／高速道路／田舎道／トンネル内／橋の上／交差点／横断歩道（が見える）／工事区間／赤信号／青信号／黄信号／濡れた路面／路面の雪（言い換え可「雪道」「雪の積もった道路」）／カーブ／坂道
- 昼／夜／薄明（言い換え可「薄暮」「夕暮れや明け方」。「夕暮れ」「明け方」の片方だけにしない）／晴れ／曇り／雨が降っている（言い換え可「雨の中」「雨天」）／雪が降っている（言い換え可「降雪中」）／霧'''

GENERATE_PROMPT = '''ドライブレコーダー画像の検索システムに入力する日本語クエリを作ります。
次の情景を表すクエリを最大{count}件書いてください。

情景: {scene}

{glossary}

守ること:
- 情景に書かれた条件はすべて含め、それ以外の情報（天候・時間帯・色・車種・場所・位置・台数・動き）は足さない。
- 「走る」「停まる」「渡る」「歩く」など動きを表す語は使わない。つなぎには「いる」「ある」「見える」「写る」を使うか、名詞句で終える。
- 上の言い換えは使ってよい。言い換えの一覧にない語で条件の意味を広げたり狭めたりしない。
- 8〜40文字程度の、人が実際に検索窓へ打ち込みそうな自然な名詞句または短い文にする。「探して」「〜の画像」などの依頼・前置きは付けない。
- 2割程度は短いキーワード風、残りは名詞句または短い文にする。不自然な語順にしてまで言い回しを変えなくてよい。同じ文を繰り返さない。
- 情景にない対象物（車など）や条件を足して件数を埋めない。自然な言い換えが尽きたら{count}件に満たなくてよい。

良い例（情景: 夜の交差点に自転車がいる）: 「夜の交差点の自転車」「夜、交差点に自転車がいる」「交差点 自転車 夜」
悪い例: 「自転車がいる交差点の夜」（不自然な語順）「夜の交差点を走る自転車」（動きを追加）「夜の交差点に赤い自転車」（色を追加）

出力はJSON {{"queries": ["...", ...]}} のみ。'''

PARSE_PROMPT = '''ドライブレコーダー画像の検索クエリを読み、クエリが要求している条件だけを構造化してください。

クエリ: {query}

{glossary}

書き方:
- scene: クエリが要求する場面・道路設備・路面・時間帯・天候。値は次から選ぶ: {scenes}
  urban市街地/residential住宅街/highway高速道路/rural田舎道/tunnelトンネル内/bridge橋の上/intersection交差点/crosswalk横断歩道/roadwork工事区間/red_signal赤信号/green_signal青信号/yellow_signal黄信号/wet_road濡れた路面/snow_on_road路面の雪/curveカーブ/slope坂/day昼/night夜/twilight薄明/clear晴れ/overcast曇り/rain雨が降っている/snowfall雪が降っている/fog霧
- objects: クエリが言及する対象ごとに1件。kind は種類: car車・乗用車・自動車/bus バス/truck トラック/van バン/pedestrian 歩行者・人/bicycle 自転車/motorcycle バイク・オートバイ/train 列車・電車/animal 動物/emergency_vehicle 緊急車両/other それ以外（「車両」「二輪車」など種類を特定しない語を含む）。color・position・lane・place・orientation はクエリがその対象について明示した場合だけ値を入れ、明示していなければ none。
  position: left/center/right（画面上の左・中央・右）。lane: same/left_adjacent/right_adjacent/oncoming。place: sidewalk歩道上/crosswalk横断歩道上/roadway車道上。orientation: front正面/rear背面。
- other: 上で表せない条件をすべて日本語で列挙する（例: 動き、上にない色や車種、台数、否定、「または」、対象同士の位置関係、上にない場所）。なければ空配列。
- 「道路」「画像」「映像」「場面」「様子」など条件を加えない一般語は、どこにも書かない。
- クエリに書かれていないことを補わない。'''


def parse_schema():
    none_or = lambda values: {'type':'string','enum':values+['none']}
    obj = {'type':'object','properties':{
        'kind':{'type':'string','enum':['car','bus','truck','van','pedestrian','bicycle','motorcycle','train','animal','emergency_vehicle','other']},
        'color':none_or(['white','black','red','blue']),
        'position':none_or(['left','center','right']),
        'lane':none_or(['same','left_adjacent','right_adjacent','oncoming']),
        'place':none_or(['sidewalk','crosswalk','roadway']),
        'orientation':none_or(['front','rear']),
    },'required':['kind','color','position','lane','place','orientation']}
    return {'type':'object','properties':{
        'scene':{'type':'array','items':{'type':'string','enum':SCENES}},
        'objects':{'type':'array','items':obj,'maxItems':4},
        'other':{'type':'array','items':{'type':'string'}},
    },'required':['scene','objects','other']}


def generate_schema(count):
    return {'type':'object','properties':{'queries':{'type':'array','items':{'type':'string'},
        'minItems':1,'maxItems':count}},'required':['queries']}

# Requests and meta words are forbidden in generated queries; the parser ignores them, so check lexically.
META_WORDS = re.compile(r'画像|映像|写真|動画|探し|検索|ください|見せて|教えて')
PLACE_KEYS = {'sidewalk':'sidewalk','crosswalk':'on_crosswalk','roadway':'roadway'}


def scene_sentence(query):
    """Strip the catalog's '...画像' suffix so the teacher sees the scene, not a request."""
    return re.sub(r'(の)?画像$', '', query)


def canonical_target(expr):
    """(scenes, objects) for AND/scene/exists expressions; None for anything else."""
    scenes, objects = set(), []
    terms = expr['terms'] if expr['op'] == 'all' else [expr]
    for term in terms:
        if term['op'] == 'scene': scenes.add(term['name'])
        elif term['op'] == 'exists': objects += term['objects']
        else: return None
    return normalize(scenes, objects)


def canonical_parse(parsed):
    objects = []
    for o in parsed['objects']:
        spec = {'kind':o['kind']}
        for key in ('color','position','lane','orientation'):
            if o[key] != 'none': spec[key] = o[key]
        if o['place'] != 'none': spec[PLACE_KEYS[o['place']]] = 'yes'
        objects.append(spec)
    return normalize(set(parsed['scene']), objects)


def normalize(scenes, objects):
    # An object on a crosswalk implies that a crosswalk is visible; do not count it twice.
    if any(o.get('on_crosswalk') == 'yes' for o in objects): scenes = scenes - {'crosswalk'}
    return sorted(scenes), sorted(json.dumps(o, sort_keys=True) for o in objects)


def chat(url, model, prompt, schema, *, temperature, seed, num_predict):
    payload = {'model':model,'stream':False,'think':False,'format':schema,
               'options':{'temperature':temperature,'seed':seed,'num_ctx':8192,'num_predict':num_predict},
               'messages':[{'role':'user','content':prompt}]}
    for attempt in range(3):
        response = request_json(f'{url}/api/chat', payload)
        try:
            return json.loads(response['message']['content'])
        except json.JSONDecodeError:
            payload['options']['seed'] += 1000
    raise RuntimeError(f'invalid JSON after retries: {response}')


def normalize_text(text):
    return re.sub(r'[\s、。,.]+', '', text)


def verify(url, model, text, target):
    parsed = chat(url, model, PARSE_PROMPT.format(query=text, glossary=GLOSSARY, scenes=', '.join(SCENES)),
                  parse_schema(), temperature=0, seed=0, num_predict=1024)
    actual = canonical_parse(parsed)
    reasons = []
    if actual[0] != target[0]: reasons.append('scene_mismatch')
    if actual[1] != target[1]: reasons.append('object_mismatch')
    if parsed['other']: reasons.append('extra_condition')
    return parsed, reasons


def run(args):
    out = args.out/'queries'
    out.mkdir(parents=True, exist_ok=True)
    if args.conditions:
        # Extra conditions in the catalog's expression format, e.g. from build_combo_conditions.py.
        conditions = read_jsonl(args.conditions)
    else:
        conditions = [c for c in load_conditions() if not c['id'].startswith(UNSUPPORTED_PREFIXES)]
    if args.only: conditions = [c for c in conditions if c['id'] in set(args.only)]
    tags = request_json(f'{args.ollama_url}/api/tags')['models']
    digest = next(m['digest'] for m in tags if m['name'] == args.teacher)
    config = {'version':VERSION,'teacher':args.teacher,'teacher_digest':digest,'max_count':args.max_count,
              'generate_temperature':args.temperature,'seed':args.seed,
              'generate_prompt_sha256':hashlib.sha256((GENERATE_PROMPT+GLOSSARY).encode()).hexdigest(),
              'parse_prompt_sha256':hashlib.sha256((PARSE_PROMPT+GLOSSARY).encode()).hexdigest(),
              'parse_schema_sha256':hashlib.sha256(json.dumps(parse_schema(),sort_keys=True).encode()).hexdigest()}
    if args.conditions: config['conditions_sha256'] = hashlib.sha256(args.conditions.read_bytes()).hexdigest()
    config_path = out/'config.json'
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise SystemExit(f'{config_path} differs from the current settings; use a different --out')
    write_json(config_path, config)

    log = out/'queries.jsonl'
    done = {r['condition_id'] for r in read_jsonl(log)} if log.exists() else set()
    for index, c in enumerate(conditions, 1):
        if c['id'] in done: continue
        started = time.monotonic()
        target = canonical_target(c['expression'])
        scene = scene_sentence(c['query'])
        count = args.max_count
        generated = chat(args.ollama_url, args.teacher,
                         GENERATE_PROMPT.format(count=count, scene=scene, glossary=GLOSSARY),
                         generate_schema(count), temperature=args.temperature,
                         seed=int(stable_key(args.seed, c['id']), 16) % 2**31, num_predict=2048)['queries']
        seen, rows = set(), []
        # The catalog query is kept as a reference phrasing and verified the same way.
        for source, text in [('catalog', c['query'])] + [('generated', q.strip()) for q in generated]:
            key = normalize_text(text)
            if key in seen:
                rows.append({'text':text,'source':source,'passed':False,'reasons':['duplicate'],'parsed':None})
                continue
            seen.add(key)
            parsed, reasons = verify(args.ollama_url, args.teacher, text, target)
            if source == 'generated' and META_WORDS.search(text): reasons.append('meta_words')
            rows.append({'text':text,'source':source,'passed':not reasons,'reasons':reasons,'parsed':parsed})
        append(log, {'condition_id':c['id'],'scene':scene,'target':target,'requested':count,'created_at':now(),
                     'elapsed_seconds':time.monotonic()-started,'queries':rows})
        passed = sum(r['passed'] for r in rows if r['source'] == 'generated')
        print(f'[{index}/{len(conditions)}] {c["id"]} {scene}: 合格 {passed}/{len(generated)} (要求 {count})', flush=True)
    report(args)


def report(args):
    rows = read_jsonl(args.out/'queries/queries.jsonl')
    by_category, reasons = Counter(), Counter()
    passed_by_category = Counter(); coverage = Counter()
    catalog_failures = []
    for r in rows:
        category = re.match(r'[A-Z]+', r['condition_id']).group(0)
        generated = [q for q in r['queries'] if q['source'] == 'generated']
        by_category[category] += len(generated)
        passed = sum(q['passed'] for q in generated)
        passed_by_category[category] += passed
        coverage['>=6' if passed >= 6 else '3-5' if passed >= 3 else '<3'] += 1
        for q in generated: reasons.update(q['reasons'])
        catalog = next(q for q in r['queries'] if q['source'] == 'catalog')
        if not catalog['passed']: catalog_failures.append((r['condition_id'], catalog['text'], catalog['reasons']))
    summary = {
        'conditions': len(rows),
        'generated': sum(by_category.values()),
        'passed': sum(passed_by_category.values()),
        'pass_rate_by_category': {k: round(passed_by_category[k]/by_category[k], 3) for k in sorted(by_category)},
        'conditions_by_passed_count': dict(coverage),
        'rejection_reasons': dict(reasons),
        'catalog_query_failures': catalog_failures,
    }
    write_json(args.out/'queries/report.json', summary)
    print(json.dumps(summary, ensure_ascii=False, indent=1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['generate','report'])
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--teacher', default=DEFAULT_TEACHER)
    parser.add_argument('--ollama-url', default='http://localhost:11434')
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--seed', type=int, default=20260926)
    parser.add_argument('--only', nargs='*', help='condition IDs for a trial run')
    parser.add_argument('--conditions', type=Path, help='conditions JSONL used instead of the catalog')
    parser.add_argument('--max-count', type=int, default=MAX_COUNT)
    args = parser.parse_args()
    run(args) if args.command == 'generate' else report(args)


if __name__ == '__main__':
    main()
