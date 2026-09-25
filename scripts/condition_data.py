"""Versioned, three-valued facts and condition evaluation for the new dataset.

No old scene cards or generated queries are read by this module.
"""
from __future__ import annotations

import csv
from collections import Counter
import hashlib
import itertools
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION = 'condition-facts-v1'
STATES = ['yes', 'no', 'unknown']
KINDS = ['car', 'bus', 'truck', 'van', 'pedestrian', 'bicycle', 'motorcycle', 'train', 'animal', 'other', 'unknown']
INVENTORY_KINDS = KINDS[:9] + ['emergency_vehicle']
SCENES = ['urban', 'residential', 'highway', 'rural', 'tunnel', 'bridge', 'intersection', 'crosswalk', 'roadwork', 'red_signal', 'green_signal', 'yellow_signal', 'wet_road', 'snow_on_road', 'curve', 'slope', 'day', 'night', 'twilight', 'clear', 'overcast', 'rain', 'snowfall', 'fog']
KIND_JA = dict(zip(['車','バス','トラック','バン','歩行者','自転車','バイク','列車','動物'], KINDS[:9]))
SCENE_JA = dict(zip(['市街地','住宅街','高速道路','田舎道','トンネル内','橋上','交差点','横断歩道','工事区間','赤信号','青信号','黄信号','濡れた路面','積雪路面','カーブ','坂','昼','夜','薄明','晴れ','曇り','降雨','降雪','霧'], SCENES))
ATTR_JA = {
    '白': ('color','white'), '黒': ('color','black'), '赤': ('color','red'), '青': ('color','blue'),
    '画面左': ('position','left'), '画面中央': ('position','center'), '画面右': ('position','right'),
    '自車と同じ車線': ('lane','same'), '自車の左隣の車線': ('lane','left_adjacent'),
    '自車の右隣の車線': ('lane','right_adjacent'), '対向車線': ('lane','oncoming'),
    '歩道上': ('sidewalk','yes'), '横断歩道上': ('on_crosswalk','yes'), '車道上': ('roadway','yes'),
    '正面がカメラ側': ('orientation','front'), '背面がカメラ側': ('orientation','rear'),
}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def stable_key(seed, value):
    return hashlib.sha256(f'{seed}:{value}'.encode()).hexdigest()


def write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    tmp.replace(path)


def read_jsonl(path: Path):
    if not path.exists():
        return []
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def enum(values):
    return {'type':'string', 'enum':values}


def obj(properties):
    return {'type':'object', 'properties':properties, 'required':list(properties), 'additionalProperties':False}


def fact_schema():
    text = {'type':'string', 'minLength':1, 'maxLength':250}
    entity = obj({
        'id': {'type':'string', 'minLength':1}, 'kind':enum(KINDS),
        'bbox':{'anyOf':[{'type':'null'}, {'type':'array','items':{'type':'number','minimum':0,'maximum':1000},'minItems':4,'maxItems':4}]},
        'color':enum(['white','black','red','blue','other','unknown']),
        'orientation':enum(['front','rear','side','unknown']),
        'lane':enum(['same','left_adjacent','right_adjacent','oncoming','other','unknown']),
        'sidewalk':enum(STATES), 'on_crosswalk':enum(STATES), 'roadway':enum(STATES),
        'emergency_vehicle':enum(STATES), 'evidence':text,
    })
    return obj({
        'scene':obj({s:enum(STATES) for s in SCENES}),
        'scene_evidence':text,
        'objects':{'type':'array','items':entity,'maxItems':24},
        'inventory':obj({k:obj({'presence':enum(STATES),'complete':{'type':'boolean'}}) for k in INVENTORY_KINDS}),
        'visibility_notes':text,
    })


FACT_PROMPT = '''車載静止画1枚を観察し、JSONスキーマに従って可視の事実だけを記録してください。検索クエリは作成しません。
yes=視覚的根拠あり、no=成立しないことを確認、unknown=不鮮明・遮蔽・未確認で判別不能。推測で埋めない。
scene: urban市街地/residential住宅街/highway高速道路/rural田舎道/tunnelカメラがトンネル内/bridgeカメラが橋上/intersection交差点/crosswalk横断歩道/roadwork道路工事/red_signal,green_signal,yellow_signal車両用信号/wet_road濡れた路面/snow_on_road路面の積雪/curveカーブ/slope坂/day昼/night夜/twilight薄明/clear晴れ/overcast曇り/rain降雨中/snowfall降雪中/fog霧。
積雪と降雪、濡れた路面と降雨は別。乾いた明るい道路を雪にしない。空が見えない時の天候はunknown。信号の適用車線・違反・実移動方向は推測しない。
objects: 視認可能な対象を個体別に最大24件。idはo1,o2等で一意。kindのcarは乗用車、van/bus/truckは区別。自転車・バイクの乗り手はpedestrianに数えず、押して歩く人はpedestrian。kind不明の像はunknown。
bboxは対象の可視範囲の[xmin,ymin,xmax,ymax]を画像幅・高さそれぞれ0〜1000へ正規化。矩形不明ならnull。色は車体主色（自転車はフレーム）。服やタイヤの色を転用しない。白/銀や多色が曖昧ならunknown。
orientationはカメラ側に見えるfront正面/rear背面/side側面/unknown。向きから移動方向を推測しない。
laneはカメラ搭載車基準のsame同一、left_adjacent左隣、right_adjacent右隣、oncoming対向、otherその他、unknown不明。画像左右と車線は別。車線が不明ならunknown。
sidewalk歩道上/on_crosswalk横断歩道上/roadway車道上は接地点からそれぞれ判定（重複可）。足元が隠れ境界不明ならunknown。emergency_vehicleは救急/消防/警察等の車両の視覚的識別。緊急走行とは無関係。
inventory: 各種類を全画面確認しpresenceを記録。presence=yesの種類は必ずobjectsに少なくとも1件記録する。complete=trueはその種類の全個体を列挙し、小さな像/遮蔽による追加候補もない場合だけ。24件で打ち切る種類はcomplete=false。不在は全画面で確認できた場合だけpresence=no,complete=true。それ以外はunknown,false。
scene_evidence/各objectのevidence/visibility_notesは簡潔な日本語。確かめられない事実を無理に断定せず理由を残す。'''


def validate_schema(value, schema, path='$'):
    if 'anyOf' in schema:
        for choice in schema['anyOf']:
            try:
                validate_schema(value, choice, path)
                return
            except ValueError:
                pass
        raise ValueError(f'{path}: no matching type')
    kind = schema['type']
    valid = {'object':lambda: isinstance(value,dict), 'array':lambda:isinstance(value,list),
             'string':lambda:isinstance(value,str), 'boolean':lambda:type(value) is bool,
             'number':lambda:type(value) in (int,float), 'null':lambda:value is None}[kind]()
    if not valid:
        raise ValueError(f'{path}: expected {kind}')
    if 'enum' in schema and value not in schema['enum']:
        raise ValueError(f'{path}: invalid enum {value!r}')
    if kind == 'object':
        if set(value) != set(schema['required']):
            raise ValueError(f'{path}: unexpected/missing keys')
        for key,v in value.items(): validate_schema(v,schema['properties'][key],f'{path}.{key}')
    elif kind == 'array':
        if not schema.get('minItems',0) <= len(value) <= schema.get('maxItems',10**9):
            raise ValueError(f'{path}: invalid array length')
        for i,v in enumerate(value): validate_schema(v,schema['items'],f'{path}[{i}]')
    elif kind == 'number':
        if not schema.get('minimum',float('-inf')) <= value <= schema.get('maximum',float('inf')):
            raise ValueError(f'{path}: invalid number')
    elif kind == 'string':
        if not schema.get('minLength',0) <= len(value) <= schema.get('maxLength',10**9):
            raise ValueError(f'{path}: invalid text length')


def validate_facts(facts):
    validate_schema(facts,fact_schema())
    ids=[o['id'] for o in facts['objects']]
    if len(ids) != len(set(ids)): raise ValueError('duplicate object IDs')
    for o in facts['objects']:
        b=o['bbox']
        if b is not None and not (b[0]<b[2] and b[1]<b[3]): raise ValueError('invalid bounding box')
    for k,inv in facts['inventory'].items():
        known=[o for o in facts['objects'] if (o['emergency_vehicle']=='yes' if k=='emergency_vehicle' else o['kind']==k)]
        if inv['presence']=='no' and (known or not inv['complete']):
            message=f'inconsistent absence: {k}: inventory.presence=noですが、その種類のobjectsがあるか、全数確認が未完了です。画像を再確認して整合させてください。'
            if k=='pedestrian':
                message+=' 自転車・バイクの乗り手は歩行者ではありません。乗り手をpedestrianとして別個体にせず、乗り物だけを記録してください。歩いている人は別です。'
            raise ValueError(message)
        if inv['presence']=='yes' and not known: raise ValueError(f'presence without object: {k}')
        if inv['presence']=='unknown' and (inv['complete'] or known): raise ValueError(f'inconsistent inventory: {k}')
        if known and inv['presence']!='yes': raise ValueError(f'unrecorded presence: {k}')
    return facts


def quality_flags(facts):
    """Conservative review triggers, not automatic claims that an image is mislabeled."""
    flags = []
    objects = facts['objects']
    if len(objects) >= 24:
        flags.append('object_limit_reached')
    evidence = Counter(o['evidence'].strip() for o in objects)
    if any(count >= 4 for count in evidence.values()):
        flags.append('repeated_object_evidence')
    boxes = [(o['kind'], tuple(o['bbox'])) for o in objects if o['bbox'] is not None]
    if len(boxes) != len(set(boxes)):
        flags.append('duplicate_object_box')
    if sum(facts['scene'][k] == 'yes' for k in ('day','night','twilight')) > 1:
        flags.append('conflicting_time_of_day')
    rider_pattern=r'(?:自転車|バイク|二輪車).{0,12}(?:乗っ|乗る|乗車)|乗り手|乗車者|ライダー|サイクリスト'
    if any(o['kind']=='pedestrian' and re.search(rider_pattern,o['evidence']) for o in objects):
        flags.append('possible_rider_as_pedestrian')
    return flags


def load_conditions(root=ROOT):
    """Compile the fixed catalog, failing on unrecognized predicates or allocation drift."""
    with (root/'docs/search-condition-allocation.csv').open() as f:
        allocations={r['condition_id']:r for r in csv.DictReader(f)}
    conditions=[]
    for line in (root/'docs/search-condition-catalog.md').read_text().splitlines():
        if not re.match(r'^\| (?:OBJ|COL|POS|LOC|ORI|ROAD|ENV|CO|CE|CA|CB|H|R|X|D)\d{2} \|',line): continue
        id,query,facts,seed=[s.strip() for s in line.strip('|').split('|')]
        terms=[]
        if id.startswith('D'): expr={'op':'deferred'}
        elif id=='R01': expr={'op':'exists','objects':[{'kind':'emergency_vehicle'}]}
        elif id in ('R02','R03'): expr={'op':'exists','objects':[{'kind':'train' if id=='R02' else 'animal'}]}
        elif id=='X01': expr={'op':'left_of','left':'pedestrian','right':'truck'}
        elif id=='X02': expr={'op':'count','kind':'car','count':2}
        elif id=='X03': expr={'op':'not','term':{'op':'exists','objects':[{'kind':'pedestrian'}]}}
        elif id=='X04': expr={'op':'any','terms':[{'op':'exists','objects':[{'kind':k}]} for k in ('bus','truck')]}
        else:
            objects=[]
            for part in facts.split('; '):
                label,value=part.split(': ',1)
                if label=='場面': terms.append({'op':'scene','name':SCENE_JA[value]})
                else:
                    if label not in ('対象A','対象B'): raise ValueError(label)
                    kind,*attrs=value.split('・'); spec={'kind':KIND_JA[kind]}
                    for attr in attrs:
                        key,val=ATTR_JA[attr]; spec[key]=val
                    objects.append(spec)
            if objects: terms.append({'op':'exists','objects':objects})
            expr=terms[0] if len(terms)==1 else {'op':'all','terms':terms}
        allocation=allocations.pop(id)
        if allocation['query'] != query: raise ValueError(f'query drift: {id}')
        conditions.append({'id':id,'query':query,'expression':expr,'allocation':allocation})
    if allocations or len(conditions)!=233 or len({c['id'] for c in conditions})!=233:
        raise ValueError('catalog/allocation mismatch')
    return conditions


def conjunction(values):
    values=list(values)
    return 'no' if 'no' in values else 'unknown' if 'unknown' in values else 'yes'


def disjunction(values):
    values=list(values)
    return 'yes' if 'yes' in values else 'unknown' if 'unknown' in values else 'no'


def kind_match(o, kind):
    if kind=='emergency_vehicle': return o['emergency_vehicle']
    return 'unknown' if o['kind']=='unknown' else 'yes' if o['kind']==kind else 'no'


def position(o):
    b=o['bbox']
    if b is None: return 'unknown'
    x=(b[0]+b[2])/2
    # A small tolerance protects rounded teacher coordinates at thirds boundaries.
    if min(abs(x-1000/3),abs(x-2000/3)) <= 2: return 'unknown'
    return 'left' if x<1000/3 else 'center' if x<2000/3 else 'right'


def object_match(o, spec):
    states=[kind_match(o,spec['kind'])]
    for k,v in spec.items():
        if k=='kind': continue
        actual=position(o) if k=='position' else o[k]
        states.append('unknown' if actual=='unknown' else 'yes' if actual==v else 'no')
    return conjunction(states)


def complete_for(facts,kind):
    # Ambiguous objects could be extra members even when a teacher marked complete.
    return facts['inventory'][kind]['complete'] and not any(kind_match(o,kind)=='unknown' for o in facts['objects'])


def evaluate(expr, facts):
    op=expr['op']
    if op=='deferred': return 'unknown'
    if op=='scene': return facts['scene'][expr['name']]
    if op in ('all','any'):
        fn=conjunction if op=='all' else disjunction
        return fn(evaluate(e,facts) for e in expr['terms'])
    if op=='not': return {'yes':'no','no':'yes','unknown':'unknown'}[evaluate(expr['term'],facts)]
    if op=='exists':
        specs=expr['objects']; observed=facts['objects']
        def matching(allow_unknown):
            edges=[[i for i,o in enumerate(observed) if object_match(o,s) in (('yes','unknown') if allow_unknown else ('yes',))] for s in specs]
            if allow_unknown:
                for j,s in enumerate(specs):
                    if not complete_for(facts,s['kind']): edges[j].append(len(observed)+j)
            def visit(j,used):
                if j==len(edges): return True
                return any(visit(j+1,used|{i}) for i in edges[j] if i not in used)
            return visit(0,set())
        return 'yes' if matching(False) else 'unknown' if matching(True) else 'no'
    if op=='count':
        kind=expr['kind']; count=sum(kind_match(o,kind)=='yes' for o in facts['objects'])
        if count>expr['count']: return 'no'
        if not complete_for(facts,kind): return 'unknown'
        return 'yes' if count==expr['count'] else 'no'
    if op=='left_of':
        states=[]
        for a,b in itertools.permutations(facts['objects'],2):
            relation='unknown' if a['bbox'] is None or b['bbox'] is None else 'yes' if (a['bbox'][0]+a['bbox'][2])<(b['bbox'][0]+b['bbox'][2])-4 else 'no' if (a['bbox'][0]+a['bbox'][2])>(b['bbox'][0]+b['bbox'][2])+4 else 'unknown'
            states.append(conjunction([kind_match(a,expr['left']),kind_match(b,expr['right']),relation]))
        if 'yes' in states: return 'yes'
        # A proven absence of either kind makes the relation false even if the other is incomplete.
        presence=conjunction(evaluate({'op':'exists','objects':[{'kind':k}]},facts) for k in (expr['left'],expr['right']))
        if presence=='no': return 'no'
        return 'no' if 'unknown' not in states and all(complete_for(facts,k) for k in (expr['left'],expr['right'])) else 'unknown'
    raise ValueError(f'unknown operator: {op}')


def contains_heldout(expr, heldout):
    """AND/exists implication for the fixed positive catalog, preserving object binding."""
    def unpack(e):
        if e['op']=='all':
            scenes=set(); specs=[]
            for term in e['terms']:
                s,o=unpack(term); scenes|=s; specs+=o
            return scenes,specs
        if e['op']=='scene': return {e['name']},[]
        if e['op']=='exists': return set(),e['objects']
        return set(),[]
    s1,o1=unpack(expr); s2,o2=unpack(heldout)
    if not s2 and not o2: return False
    return s2<=s1 and any(all(all(obj.get(k)==v for k,v in spec.items()) for obj,spec in zip(assignment,o2)) for assignment in itertools.permutations(o1,len(o2)))
