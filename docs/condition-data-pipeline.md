# 条件指定データの実装と50枚の試行

実装日: 2026-09-24

新規データの出力先は`datasets/dashcam_reranker_v3_conditions/`です。
以前の生成済みシーンカード・クエリ・ペアを入力にしません。
画像索引と元画像は候補選定に使い、画像の事実はQwen3.8で新しく生成します。
50枚の初回試行は完了し、[実測時間と品質確認の結果](reports/condition-pilot-50.md)を保存しました。12枚の確認事項は記録として保持します。ユーザー方針により追加の不整合対策・50枚の再試行は行いません。
[5,000枚の選定](reports/condition-selection-5000.md)は完了しました。2026-09-25に5,000枚の本生成・条件判定・学習用ペア作成を完了しました。同日にQLoRA学習と[test評価](reports/condition-eval.md)を完了しました。

## 1. 実行方法

このワークスペースでは、既存の`/home/takami/anaconda3/envs/image/bin/python`に必要な画像検索ライブラリが入っています。
以下はリポジトリのルートから実行します。各段階は明示的に実行し、5,000枚の本生成や学習を自動で開始しません。

```bash
/home/takami/anaconda3/envs/image/bin/python scripts/build_condition_dataset.py inventory
/home/takami/anaconda3/envs/image/bin/python scripts/build_condition_dataset.py retrieve
/home/takami/anaconda3/envs/image/bin/python scripts/build_condition_dataset.py prepare
```

- `inventory`: 確認用3画像を除く10万枚を読み込み、内容SHA-256・知覚ハッシュ・サイズを保存。完全重複をまとめ、無作為1,000枚を700/100/200に分割して予約します。
- `retrieve`: 画像索引の件数・ID・次元・正規化を検証。3画像を再埋め込みして対応を確認し、227条件の上位1,000件ずつを保存します。GPUの埋め込みモデルはこのプロセス終了時に解放されます。
- `prepare`: 順位帯ごとの候補抽出と候補台帳を作り、試行用の条件指定40枚・無作為10枚と確認用一覧画像を出力します。

`pilot/contact_sheets/`と必要な元画像を確認し、`pilot/review.jsonl`に各画像の確認を保存します。
全50枚について`image_id`・`decision: accept`・`reviewer`・`notes`が必要です。検索条件の正例と認めた記録ではなく、試行対象として採用した理由です。
人が確認したのか、アシスタントが画像を見たのかを`reviewer`と`review_scope`で明示します。

```bash
/home/takami/anaconda3/envs/image/bin/python scripts/build_condition_dataset.py annotate
/home/takami/anaconda3/envs/image/bin/python scripts/build_condition_dataset.py report
```

`annotate`は50枚の採用台帳を固定し、1枚ごとに教師の応答と検証済みJSONを保存します。
再実行では保存済みの成功画像を読み直して検証し、残りだけを処理します。
教師のdigest、プロンプト、スキーマ、画像台帳、条件一覧が変わった場合は同じ注釈runへ追記しません。
変更時は別の`--out`を使い、同じ試行画像を再利用する場合も元の出力を残します。
同時書き込みはOSのファイルロックで防ぎ、各実行の所要時間を記録します。

### 確定5,000枚の本生成（ユーザー側で実行）

`annotate-full`を追加しました。`annotate`は従来どおり試行50枚用です。
2026-09-24に、整合性チェックの有無が混在した旧実行を停止・退避しました。`--no-reuse-pilot`で、試行50枚も含めた5,000枚すべてを整合性チェックなしで新しく生成しました。旧ラベル・旧応答は新実行へ取り込みません。画像の選定とsplitは固定済みのままです。
旧結果と旧ログは`archives/mixed_annotation_20260924T042729Z/`に保存しています。新実行の開始時の状態は保存済み0枚・再利用0枚・未生成5,000枚です。

```bash
cd /home/takami/Desktop/image-vector-search
nohup /home/takami/anaconda3/envs/image/bin/python -u scripts/build_condition_dataset.py annotate-full --no-reuse-pilot >> datasets/dashcam_reranker_v3_conditions/annotation_full.log 2>&1 &
```

ログの確認:

```bash
tail -f /home/takami/Desktop/image-vector-search/datasets/dashcam_reranker_v3_conditions/annotation_full.log
```

`tail -f`はCtrl+Cで閉じられ、本生成はバックグラウンドで継続します。途中で本生成が終了した場合も、同じ`nohup`コマンドで再開できます。同時起動はファイルロックで防ぎます。

- 今回の新実行では試行50枚を含むすべての画像を新しく処理します。途中で中断した場合は、この新実行で保存済みの結果だけを再利用し、未完了の画像から続けます。再利用しない設定は保存され、再開時にフラグを省略しても旧試行を取り込みません。
- 1枚ごとに結果を保存・同期します。中断で最後のJSONL行が書きかけになった場合、その断片を別ファイルへ保存してから復旧します。完成した行の破損は黙って捨てません。
- 既存と同じプロンプト・スキーマ・教師モデル・生成設定を使用します。対象の存在判定と個体一覧の矛盾、全数確認との矛盾、矩形の大小関係、個体IDの重複など、内容の整合性チェックは実行しません。生成・再開・集計のいずれでも、それらを理由に再生成・除外しません。
- JSONの読み込みと指定データ構造の確認は行います。応答の途切れ・読み込めない形式の場合のみ、1回の実行で最大2回試します。未完了が残った場合は同じコマンドで再開できます。
- Ollamaへの接続失敗が2回続いた場合は停止します。Ollamaの復旧後に同じコマンドで再開します。
- 品質・不整合フラグの検出も行いません。旧試行の確認事項は旧記録に残し、新しいラベル・集計には混ぜません。`unknown`とH/Dの除外規約は保持します。
- 旧チェックで弾かれた応答や旧実行で採用済みのラベルも、今回は再利用しません。中断時に現在の新実行の生応答だけが保存されていた場合は、その応答を再利用できます。
- 完了時に全画像の条件判定とsplit別の正例・負例・判定不能・目標に対する不足を集計します。教師由来の集計であり、独立に確認済みの正解とは区別します。学習は自動実行しません。

主な出力は`datasets/dashcam_reranker_v3_conditions/annotations/`の`facts.jsonl`・`raw_responses.jsonl`・`errors.jsonl`・`annotation_runs.jsonl`・`condition_judgments.jsonl`・`report.json`です。
画像生成を伴わない事前確認、または保存済み事実からの集計だけを行う場合は次のコマンドを使います。

```bash
/home/takami/anaconda3/envs/image/bin/python scripts/build_condition_dataset.py annotate-full --no-reuse-pilot --check-only
/home/takami/anaconda3/envs/image/bin/python scripts/build_condition_dataset.py report-full
```

事前確認は教師推論や注釈の書き込みを行いません。見積もりは試行に基づく約24〜28時間で、画像構成や再試行率によって変わります。

### 出力上限で未完了になった2枚の再生成

2026-09-25に、4,096トークンで出力が途切れた2枚を対象として、ユーザー指定により出力上限だけを8,192へ変更しました。

```bash
/home/takami/anaconda3/envs/image/bin/python -u scripts/build_condition_dataset.py annotate-full --no-reuse-pilot --num-predict 8192
```

このオプションは未完了画像へのリクエストだけに適用します。教師・プロンプト・スキーマ・温度・コンテキスト長を維持し、保存済みの4,998枚には再推論しません。整合性チェックは無効のままです。
基本設定の`annotation_config.json`は4,096の記録を保持し、変更後の値を各リクエストの`request_options`、新しい画像事実の`generation_options`、実行記録に保存します。集計には`output_token_cap_counts`を出力し、2枚の生成設定の違いを明示します。
同オプションの動作を含む67件のテストが通っています。

2枚とも1回で正常終了しました。実際の出力長は4,212トークンと4,119トークンでした。既存4,998枚がバイト単位で変わっていないこと、全5,000枚が一意に揃い、学習3,500・検証500・評価1,000枚と1,165,000件の条件判定が保存されたことを確認しました。[完了確認記録](reports/condition-annotation-5000.json)に保存しています。

## 2. 新しい画像事実の形式

スキーマの定義元は[scripts/condition_data.py](../scripts/condition_data.py)です。
実行時には`schema/facts.schema.json`・`schema/prompt.txt`・`schema/conditions.json`として固定した内容も保存します。

| 項目 | 意味 |
|---|---|
| `scene` | 道路・設備・路面・時間帯・天候の24項目。各値は`yes` / `no` / `unknown` |
| `objects` | 対象ID、種類、可視範囲の矩形、車体色、向き、車線、歩道・横断歩道・車道上の判定、緊急用途、根拠 |
| `inventory` | 対象種類別の存在判定と、全個体を列挙できたかを表す`complete` |
| `scene_evidence` | 場面の根拠。簡潔な日本語 |
| `visibility_notes` | 暗さ、遮蔽、小さい対象などの制約 |

矩形は画像の幅・高さをそれぞれ0〜1000へ正規化した`[xmin, ymin, xmax, ymax]`です。
矩形が不明なら`null`とし、画面位置も判定不能にします。
左・中央・右は矩形の中心から計算し、3等分境界から2/1000以内は座標の丸めを考慮して判定不能にします。
対象間の左右関係でも同じ座標誤差の幅を設けます。

対象の記載上限は24個体です。上限に達したからといって全対象が列挙済みとは扱いません。
その種類の追加個体を見落としている可能性がある場合は`complete: false`とし、種類不明の対象があればプログラム側でも完全性を保守的に扱います。
例えば白い車が1台記録されていても、全乗用車の確認が済んでいなければ「赤い車がない」とは判定しません。
`complete`自体も教師の出力であり、独立した確認済みの事実とは区別します。

## 3. 条件判定と保存範囲

[固定した検索条件](search-condition-catalog.md)の必須条件を、対象ごとの制約とAND/OR/否定/数量/位置関係に変換します。
未知の表現やCSVとの不一致があれば停止し、黙って条件を落としません。
教師へ検索文を自由に作らせず、基準クエリと対応する必須条件をそのまま使います。

- 同じ対象に種類・色・位置等を束ね、複数対象は別の個体へ割り当てます。
- 記載がないだけの対象・属性は負例にしません。全数確認または確認済み不在が必要です。
- 数量は全数確認が必要ですが、指定数を超える個体が既に確認できれば不適合です。
- OR・AND・否定は判定不能を保持します。
- D01〜D06は静止画から常に判定不能です。
- H01〜H20、およびその条件を含む強いクエリはtrain/valで使用不可です。試行の診断集計には出せますが、学習用に輸出しません。

`pilot/condition_judgments.jsonl`は教師事実からの**候補判定**です。
`review_status: teacher_only`とし、この試行処理から`pairs.train.jsonl`等を自動生成しません。
画像の選定確認を、そのまま全検索条件の正解確認へ昇格させることもしません。
24個体の上限到達、同じ根拠文の4回以上の反復、同種対象の矩形の完全重複、昼夜の矛盾、歩行者の根拠文に乗り手を示す記述を検出した画像は`review_required`に分離します。
これらは確認を促す規則であり、検出されないラベルが正しいことを保証するものではありません。
元の教師出力と候補判定は残し、ラベルを黙って修正しません。

## 4. この実装で確認することと本生成への残作業

今回の実装は、元画像の確認、全条件の候補検索、50枚の試行と条件判定、5,000枚の最終選定までです。
本選定は検索条件と順位帯に基づく候補配分で、候補全件の目視ラベル作成を前提にしていません。
撮影系列情報がないため、画像内容の重複確認と撮影系列の独立性は区別して記録します。
5,000枚全体を処理する`annotate-full`は実装・事前確認済みです。本生成と学習用ペア出力は完了しており、次はQLoRA学習です。既存の`annotate`コマンドの対象は引き続き試行50枚です。

試行では候補上位から20種類×2枚を選び、無作為枠の10枚を加えています。
これは処理の動作と失敗例を調べるための小規模な学習側のセットであり、227条件の網羅性や最終5,000枚の配分を代表する標本ではありません。
元画像の処理、検索、確認、初回モデル読込、成功した注釈、失敗・再試行の時間を区別して見積もります。

### 本選定用の割り当て処理

```bash
python3 scripts/select_condition_images.py --method retrieval
```

無作為1,000枚と試行50枚の所属を保ち、test・val・trainの順に条件枠4,000枚を配分します。
各splitで検索順位1〜40位を50%、41〜200位を25%、201〜1,000位を25%に分けます。
候補が未確保の条件を優先し、その後は配分CSVの正例目標を重みとして検索候補の不足率と順位から選びます。Hはtestのみの選定目標です。
これは検索候補の確保であり、検索ヒットを正例・負例ラベルに変換しません。
元画像の内容ハッシュを5,000枚すべて再照合して、`manifests/selected_5000.jsonl`と`selection/allocation_report.json`を保存します。固定済み台帳と異なる選定結果での上書きは拒否します。

従来の暫定目視確認から割り当てる方式は`--method reviewed`で残しています。この方式のみ`selection/reviews.jsonl`（image_id、decision、reviewer、notes、judgments）が必要です。今回の本選定では使用していません。

## 5. 学習・検証・評価用ペアの作成

```bash
python3 scripts/build_condition_pairs.py
```

保存済みの全5,000枚の条件判定と固定配分CSVから、既存の学習・評価スクリプト用に`pairs.train.jsonl`・`pairs.val.jsonl`・`pairs.test.jsonl`を出力します。LLMの呼び出しや内容の整合性チェックは行いません。
判定不能は採用せず、Hの学習側への除外規約とDの対象外規約を保持します。正例・負例が両方ある条件を採用し、不足は`reports/condition_pair_coverage.csv`に残します。

作成済み件数は学習11,072ペア・検証2,241ペア・評価4,964ペアです。5,000枚すべてがペアに含まれます。[作成結果と条件別不足](reports/condition-pairs.md)を参照してください。
captionは含めていないため、学習・評価では画像のみを入力し、`--use-caption`を付けません。2026-09-25にQLoRA学習を実行しました。設定と進捗確認方法は[学習実行記録](reports/condition-qlora.md)を参照してください。

## 6. テスト

```bash
python3 -m unittest discover -s tests
```

属性の個体への対応、複数対象の割り当て、不在・数量の完全性、AND/OR/否定、座標不明、危険行動の除外、評価専用条件の混入防止、候補抽出の再現性をテストします。
本生成では、試行50枚の再利用、中断からの再開、設定・画像変更の検出、書きかけの最終行の復旧、旧チェックで弾かれた応答の再利用も検証しています。存在判定と一覧が矛盾する出力、逆順の矩形、重複IDが再生成・除外されず保存・再開・集計できることを確認し、全画像の新規生成、旧ラベルの混入防止、再開時の再利用禁止設定の保持も含め、既存分を含む66件が通りました。
これらは判定処理の検証であり、教師の視覚認識の正しさの保証ではありません。試行画像との照合を別途記録します。

ペア出力の三値判定の扱い、条件目標に沿った抽出、評価専用クエリとそれを含む条件の除外、再現性、分割の保持を含め、全73件が通りました。全18,277ペアの元判定との一致と、既存の読み込み・評価関数への互換性も確認済みです。
