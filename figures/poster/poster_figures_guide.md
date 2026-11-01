# ポスター掲載図：意味・実験設定・解釈ガイド

この文書は、`figures/poster/` に置いた次の3図について、研究質問、図の読み方、実験設定、主要結果、解釈上の注意をまとめたものである。

1. [学習済み群とランダム群の適応曲線](./adaptation_controls_trained_vs_random_v2.pdf)
2. [学習済み−ランダムのpaired適応差](./trained_minus_random_paired_iia.pdf)
3. [共有写像の未見モデル転移](./shared_mapping_heldout_transfer.pdf)

3図は同じ指標（IIA）を使うが、同じ実験ではない。1・2は同じ適応対照実験の絶対値と群間差を示し、3は別の共有写像主試験で、介入例を使った未見側適応を行わずに転移を調べる。

## 研究上の背景と全体の問い

先行研究 *The Non-Linear Representation Dilemma: Is Causal Abstraction Enough for Mechanistic Interpretability?* のFigure 3右は、階層的等価性課題におけるMLPの学習進行と、アラインメント写像の複雑さによるIIAの変化を示す。特に、複雑な非線型写像ではランダム初期化に近いMLPでも高いIIAが得られうる、という注意点を提示する。今回のポスター図はそのFigure 3右を再掲したものではなく、別のseed・実験手順で行った追加分析である。

本プロジェクトの問いは、モデル専用または複数モデル共有の写像で高いIIAが得られるかだけでなく、その対応がタスク学習済みモデルに結びついているか、共有写像が別モデルに転移するか、未見モデル側を適応させるとどうなるかである。

### IIAとは

IIA（interchange intervention accuracy）は、介入を加えた条件で、ニューラルモデルの出力が対象アルゴリズムの出力と一致する割合である。値が高いほど介入下での出力対応がよい。二値分類のchance-levelはおよそ0.5である。IIAは通常のタスク正答率そのものではなく、介入後のアルゴリズムとモデルの一致を測る。

実験対象の「Both Equality Relations」課題では、入力を4つのベクトル \(x_1,\ldots,x_4\)（各4次元）に分け、対象アルゴリズムは \((x_1=x_2)\) と \((x_3=x_4)\) を計算して両者が等しいかを判定する。モデルは隠れ次元16のMLPで、介入対象の内部表現は第1層（コードでは `layer=0`、0始まり）である。

### 写像記号 \(\phi\) とアラインメント \(A_m\)

\(\phi\) は、モデル内部表現（必要ならモデル別アラインメント後の表現）をアルゴリズム側の共通表現へ写す、学習可能な可逆写像である。モデル \(m\) の隠れ表現を \(h_m\) とすると、対応づける表現は概念的に \(\phi(A_m h_m)\) と書ける。ここで \(A_m\) はモデル \(m\) の内部座標を参照／共通座標へ整列する写像で、\(\phi\) とは役割が異なる。恒等写像条件では \(\phi\) を恒等写像とし、single-model／shared conditionsでは \(\phi\) をRevNetまたは線形写像として学習する。RevNetは可逆な残差ネットワークであり、\(\phi\) の初期化は写像パラメータの初期値、\(A_m\) の推定・適応はモデル座標の整列を指す。

基礎MLPは訓練入力1,048,576例、dev/test各10,000例で学習する。設定はAdam、学習率0.001、batch size 1,024、最大20 epoch、dev基準のpatience 3で、採用checkpointはdev基準で選ぶ。random状態は、この学習済みseedに対応するタスク学習前の保存済み初期重みである。

## 図1：学習済み群とランダム群の適応曲線

**図：** [adaptation_controls_trained_vs_random_v2.pdf](./adaptation_controls_trained_vs_random_v2.pdf)  
**生成コード：** [`plot_adaptation_state_comparison.py`](../../notebooks/plot_adaptation_state_comparison.py)  
**元データ：** [`adaptation_controls_v2/results.json`](../../Results/SharedMapping/full/adaptation_controls_v2/results.json)、[`adaptation_controls_random/results.json`](../../Results/SharedMapping/full/adaptation_controls_random/results.json)

### 何を問う図か

写像を学習したときには使っていないモデル（held-out model）に、そのモデル用の介入例を与えてアラインメントを適応させる。適応例を増やしたときのIIAを、タスク学習済みMLP群とランダム初期化MLP群で比較する。ランダム群はタスク学習前に保存した同じseedの初期重みであり、別のランダムネットワークを作った群ではない。

### 図の読み方

- **横軸：** 未見側アラインメントの適応に使う介入例数（0、256、1,024、4,096、10,000）。
- **縦軸：** 適応後のheld-out IIA。
- **左右のパネル：** 左はtask-trained MLP、右はrandom-initialized MLP。両パネルのy軸スケールは異なる。ランダム側は0.48–0.515に拡大されているため、パネル間の線の高さや見かけの変化幅を目視で直接比較しない。
- **色：** 条件。共有なし（恒等写像）、単一モデル由来の写像、共有写像（fixed alignment）、共有写像（joint alignment）。
- **細線：** 確認用held-out foldごとの曲線。各点はそのfoldの2つの最適化反復の平均。
- **太線：** 4つの確認用foldの平均。
- **水平破線：** IIA=0.5のchance-level。

### 実験条件

- 5つのモデルseed（4287、3837、9097、2635、5137）でleave-one-model-outのfoldを作った。
- fold 0（held-out seed 4287）は探索用として扱い、図の確認集計から除外した。図にはfold 1–4（held-out seed 3837、9097、2635、5137）を表示する。
- 1 fold・モデル状態ごとに2つの写像最適化反復を行った。反復0/1は独立なモデルseedではなく、写像学習および適応の最適化反復である。
- 4条件は次の通り。
  - **No shared mapping:** 恒等写像。
  - **Single-model mapping:** そのfoldの参照モデル1つで学習済みの個別RevNet写像を使う。
  - **Shared mapping (fixed alignment):** 参照モデルに対する通常活動由来のアラインメントを学習元モデル側で固定し、共有非線型写像を学習する。
  - **Shared mapping (joint alignment):** 共有写像と学習元モデル側のアラインメントを共同学習する。
- `fixed`と`joint`の違いは**共有写像を学習元モデル群で学習するときの方法**である。どちらもheld-out側では写像 \(\phi\) を固定し、held-outモデルの直交アラインメント \(A_m\) だけを適応する。
- 各状態で学習元に使う4モデルも同じ状態である。task-trained曲線の写像はtask-trained学習元モデルから、random曲線の写像はrandom初期化学習元モデルから学習する。
- **写像 \(\phi\) の初期化とcheckpoint：**
  - 学習可能な非線型写像はRevNet（10ブロック、幅128）。生成前にseedを設定するので、同じfold・状態・反復に対応する初期パラメータを再現できる。各写像はその初期値から介入train/devデータで学習し、testデータはcheckpoint選択に使わない。
  - **Single-model mapping** は、そのfoldの参照モデル1つで学習した個別RevNet checkpointを使う。初期化seedは `stable_seed("individual", reference_seed, state, repeat)`。
  - **Shared mapping (fixed alignment)** は、この適応実験で新たに初期化・学習したものではない。主試験で作成した、対応するfold・state・repeatの直交整列用共有RevNet checkpointを読み込む。初期化seedは `stable_seed("shared_nonlinear", "orthogonal", fold, state, repeat)`。
  - **Shared mapping (joint alignment)** は、この適応対照内で同じseed式により新規初期化し、学習元4モデル上で共有写像と学習元側のアラインメントを共同学習する。同一seedで作成された既存joint-PoC checkpointがある場合は再利用する。
  - 反復0/1では異なる初期seedを使う。seedはRevNetのランダム初期パラメータを決めるもので、同じrepeat番号でもfixedとjointの学習手順やcheckpointが同一という意味ではない。
- **学習元側アラインメント：** Fixedでは通常活動由来の直交アラインメントを固定する。Jointでは各モデルの同じ通常活動由来アラインメントから始め、参照モデルのアラインメントは固定、残り3モデルの直交更新を学習する。
- held-out側の初期 \(A_m\) は、参照モデルとheld-outモデルの通常活動から推定した直交アラインメントである。通常活動のキャリブレーション入力は10,000例。介入例は初期アラインメントの推定に使わない。
- **適応例数0**では、この通常活動由来の初期 \(A_m\) をそのまま使う。0例は「alignmentなし」ではない。
- 適応例数が正の場合は、共通のネストした介入train subsetを用い、\(\phi\)を凍結してheld-outの \(A_m\) のみを更新する。Adam、学習率0.001、batch size 256、最大2,000更新。10,000例のadapt-devで200更新ごとにcheckpointを選び、最終IIAは独立のtest介入データ10,000例で評価する。adapt-train/adapt-devのseedはそれぞれ20261011/20261012。
- 学習済み群とランダム群には同じheld-out seed、適応subset、適応条件、適応例数、反復対応を用いている。

### 主要結果（確認用foldの平均）

下表は各条件で、4 fold×2反復のIIAを平均した値。図は各群の曲線を別々に示す。

| 条件 | 適応例数0 | 256 | 1,024 | 4,096 | 10,000 |
| --- | ---: | ---: | ---: | ---: | ---: |
| **Task-trained: No shared mapping** | 0.556 | 0.710 | 0.760 | 0.772 | 0.779 |
| **Task-trained: Single-model mapping** | 0.606 | 0.760 | 0.845 | 0.878 | 0.887 |
| **Task-trained: Shared mapping (fixed)** | 0.579 | 0.780 | 0.853 | 0.893 | 0.902 |
| **Task-trained: Shared mapping (joint)** | 0.549 | 0.772 | 0.855 | 0.898 | 0.914 |
| **Random: No shared mapping** | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 |
| **Random: Single-model mapping** | 0.502 | 0.502 | 0.500 | 0.498 | 0.500 |
| **Random: Shared mapping (fixed)** | 0.500 | 0.500 | 0.501 | 0.502 | 0.499 |
| **Random: Shared mapping (joint)** | 0.500 | 0.501 | 0.500 | 0.499 | 0.500 |

学習済み群では介入例数の増加とともにIIAが上がる。一方、このrandom群の平均曲線は全条件でchance-level近辺にとどまる。したがって、介入例を与えること自体だけでなく、モデルがタスク学習を通じて得た性質が適応成績に関係している可能性がある。ただし、これだけで特定の表現機構が原因だと特定することはできない。

## 図2：学習済み−ランダムのpaired適応差

**図：** [trained_minus_random_paired_iia.pdf](./trained_minus_random_paired_iia.pdf)  
**生成コード：** [`plot_adaptation_diagnostics.py`](../../notebooks/plot_adaptation_diagnostics.py)  
**元データ：** 図1と同じ2つの適応対照 `results.json`

### 何を問う図か

図1では学習済み群とランダム群が別パネルかつ異なるy軸スケールで示される。図2は、同じ条件・held-out fold・最適化反復・適応例数を対応づけて、IIA差

\[
\Delta\mathrm{IIA}
= \mathrm{IIA}_{\mathrm{trained}}-\mathrm{IIA}_{\mathrm{random}}
\]

を直接示す。これは図1のデータから計算したpaired contrastであり、新しい独立実験ではない。

### 図の読み方と差の計算

- **横軸：** 適応例数（0、256、1,024、4,096、10,000）。
- **縦軸：** 学習済み−ランダムのIIA差。0より上ならtask-trained群の値が高く、0より下ならrandom群が高い。0.1はIIAで10 percentage pointsの差に相当する。
- **4パネル：** 図1と同じ4つの写像条件。
- **細い灰色線：** 各確認用foldのpaired差。各例数で同じfold・同じ反復の学習済み値からrandom値を引き、2反復の差をfold内で平均している。
- **太い色線：** 4 foldのfold-level差を平均したもの。
- **破線：** 差が0の位置であり、信頼区間や有意差境界ではない。

### 写像の初期化について

この図は図1と同じ適応実験の結果JSONから差を計算したもので、写像の再初期化・再学習はしていない。同じheld-out fold・条件・最適化反復・適応例数におけるtask-trainedとrandomのIIAを対応させて引き算している。したがって、single-model／shared-fixed／shared-joint各写像の初期化とcheckpoint由来は、図1の「写像 \(\phi\) の初期化とcheckpoint」と同じである。

### 主要結果

| 条件 | 0例 | 256例 | 1,024例 | 4,096例 | 10,000例 |
| --- | ---: | ---: | ---: | ---: | ---: |
| No shared mapping | +0.056 | +0.211 | +0.260 | +0.272 | +0.280 |
| Single-model mapping | +0.103 | +0.258 | +0.345 | +0.380 | +0.387 |
| Shared mapping (fixed alignment) | +0.080 | +0.280 | +0.352 | +0.391 | +0.403 |
| Shared mapping (joint alignment) | +0.049 | +0.272 | +0.354 | +0.399 | +0.414 |

平均差は全条件・全例数で正で、適応例数とともにおおむね増加する。これは確認用4 foldにおける学習済み群とrandom群の差が、適応例数を増やすほど大きかったことを示す。例数10,000では差は約0.28–0.41 IIAである。

この図が直接示すのは**群の差**であり、共有写像がsingle-model写像より優れるかどうかではない。たとえば10,000例でjoint sharedとsingle-modelの群間差はそれぞれ約0.414と約0.387で近い。共有方式の上積みを主張するには、写像条件間の直接paired contrastとその不確実性を別途検討する必要がある。

## 図3：共有写像の未見モデル転移

**図：** [shared_mapping_heldout_transfer.pdf](./shared_mapping_heldout_transfer.pdf)  
**生成コード：** [`plot_shared_mapping_transfer.py`](../../notebooks/plot_shared_mapping_transfer.py)  
**元データ：** [`permutation/results.jsonl`](../../Results/SharedMapping/full/permutation/results.jsonl)、[`orthogonal/results.jsonl`](../../Results/SharedMapping/full/orthogonal/results.jsonl)  
**実験記録：** [`shared_mapping_summary.md`](../../../mapping%20evaluation/shared_mapping_summary.md)、[`shared_mapping_experiment_record.md`](../../../mapping%20evaluation/shared_mapping_experiment_record.md)

### 何を問う図か

複数の学習元モデルで学習した写像、または参照モデル1つで学習した写像が、写像学習に参加していないモデルへ転移するかを調べる。未見モデルのアラインメントは通常活動だけから推定し、**未見モデルの介入ラベルで写像やアラインメントを調整しない**。したがって、図1・2の介入例によるheld-out適応とは異なり、こちらは介入ラベル適応なしの転移評価である。

### 図の読み方

- **列：** Permutation alignment（置換）とOrthogonal alignment（直交）。
- **行：** task-trained MLPとrandom-initialized MLP。
- **横軸の条件：**
  - **Alignment only:** \(\phi\)を恒等写像とする基準。通常活動由来の整列だけを利用する。
  - **Shared linear map:** 複数の学習元モデルに共有する線形写像。
  - **Shared nonlinear map:** 複数の学習元モデルで学習するRevNet共有写像。
  - **Single-model map transfer:** 参照モデル1つで学習した写像を未見モデルに適用する。
- **縦軸：** held-outモデルのIIA。表示範囲は0.45–0.72で、0.5の破線がchance-level。
- **色点：** 評価対象モデルseed 4287、3837、9097、2635、5137。写像条件では各seedあたり2初期化のIIAを平均した点。
- **黒いダイヤ：** seed平均。ひげは5 seedの最小値から最大値までの記述範囲であり、信頼区間ではない。
- **例外：** Alignment onlyは写像最適化を伴わないため、seedあたり1値。

### 実験条件

- 同じ「Both Equality Relations」課題、hidden size 16、第1層表現、4次元入力ブロック2つを使う。
- Main seedsは4287、3837、9097、2635、5137。モデル状態はtask-trainedと、同seedの学習前random checkpoint。
- 通常活動のキャリブレーション入力10,000例からseed間アラインメントを推定する。置換条件と直交条件を別々に評価する。
- 介入データはtrain 1,280,000例、dev 10,000例、test 10,000例。訓練側の共有写像では同じ介入IDを複数モデルで用いる。共有RevNetは10ブロック、幅128。共有写像の学習は最大50 epoch、batch size 6,400（4モデル合計）、学習率0.001、開発損失によるearly stopping（patience 5、改善閾値0.001）。
- 写像の学習元モデルは、評価対象targetを除く同一状態の4モデルである。共有バッチは1回につき介入ID 1,600件を各モデルに適用し、4モデルの損失平均で共有写像を学習する。devは共有学習元4モデルの平均dev lossでcheckpoint選択に使う。
- **Shared fixed/nonlinear main-transfer figureの固定写像学習は、適応対照図の“fixed alignment”条件と完全に同じプロトコルという意味ではない。** この図は主試験の置換・直交条件ごとに実施した転移評価である。
- 共有非線型／線形写像とsingle-model写像は原則2つの写像初期化を評価し、各seedの点はその2値の平均。転移評価のhold-out条件では、写像学習に当該target seedを含めない。
- single-model transferは、各targetに対する指定参照seedの写像のみを選んでいる。対応はtarget 4287←reference 3837、他の4 target←reference 4287。これは無制限の全参照モデル転移の平均ではない。
- IIAは独立したtest介入データで測る。held-outモデルの介入ラベルを写像・アラインメントの調整に使わない。

### \(\phi\) の初期化・学習checkpointと \(A_m\)

- **写像の種類：** Shared nonlinearは10ブロック・幅128のRevNet、Shared linearは16次元の直交Rotation、Alignment onlyは学習可能パラメータのない恒等写像である。
- **初期化：** RevNet／Rotationを生成する前に決定的seedを設定する。Shared nonlinearは `stable_seed("shared_nonlinear", alignment_kind, fold, state, repeat)`、Shared linearは `stable_seed("shared_linear", alignment_kind, fold, state, repeat)` を使う。`alignment_kind` は `permutation` または `orthogonal`。fold・state（trained/random）・repeatごとに異なる初期値になる。
- **学習元と更新：** 各foldでtargetを除いた4モデルが学習元となる。学習元モデルの通常活動から推定した \(A_m\) は、この主試験では固定し、写像 \(\phi\) のみを介入trainで学習する。Joint alignment適応対照のように、学習元 \(A_m\) と \(\phi\) を同時更新する条件ではない。
- **学習設定：** 介入train 1,280,000例、dev 10,000例。共有学習では同一介入IDを4モデルで共有し、4モデルの損失平均で最適化する。Adam、学習率0.001、最大50 epoch、batch size 6,400（4モデル合計）、dev lossでbest checkpointを選択し、patience 5・改善閾値0.001でearly stoppingする。各repeatで別初期値から学習したcheckpointを評価する。
- **Single-model transferの写像：** 参照モデル1つに対して学習済みの個別RevNet checkpointを使う。初期seedは `stable_seed("individual", reference_seed, state, repeat)`。評価対象target自身の個別写像を使うself-fitではなく、別seedの参照モデル由来写像である。
- **Held-out側 \(A_m\)：** 参照モデルとtargetモデルの通常活動10,000例から、指定した置換または直交alignmentを推定する。参照seedとtarget seedが同じ場合は恒等alignment。targetの介入ラベルでalignmentまたは写像を更新せず、推定した \(A_m\) と学習済み \(\phi\) を固定してtest介入10,000例で評価する。

### 主要結果（5 target seed平均、括弧内はseed平均の最小–最大）

| Alignment | Model state | Alignment only | Shared linear | Shared nonlinear | Single-model transfer |
| --- | --- | ---: | ---: | ---: | ---: |
| Permutation | Task-trained | 0.583 (0.508–0.667) | 0.586 (0.501–0.634) | 0.534 (0.499–0.582) | 0.559 (0.529–0.605) |
| Permutation | Random | 0.500 (0.499–0.501) | 0.500 (0.499–0.501) | 0.499 (0.497–0.502) | 0.500 (0.496–0.506) |
| Orthogonal | Task-trained | 0.549 (0.523–0.598) | 0.611 (0.523–0.687) | 0.580 (0.499–0.661) | 0.616 (0.530–0.671) |
| Orthogonal | Random | 0.500 (0.499–0.501) | 0.500 (0.499–0.501) | 0.500 (0.498–0.501) | 0.502 (0.498–0.506) |

学習済みモデルでは一部の方法でchance-levelを上回る平均があるが、共有非線型写像の未見側IIAは置換0.534、直交0.580であり、個別参照写像の転移・共有線形などを一貫して上回っていない。random群はほぼchance-levelである。この主試験からは、共有非線型写像が未見モデルへ強く汎化する、または課題学習済み対応だけを選別する、という強い証拠は得られていない。

## 3図を合わせた解釈

1. **図1（適応曲線）：** 未見モデルのアラインメントを介入例で適応させると、task-trained群ではIIAが改善する一方、random群ではこのデータではほぼchance-levelのままである。
2. **図2（paired差）：** 図1で別パネルに分かれていた2群の差を対応づけて示し、学習済み群の優位が適応例数とともに広がる傾向を直接確認する。
3. **図3（転移）：** 未見側の介入適応を許さない通常活動整列だけの設定では、共有写像の転移は限定的である。図1・2で見えた適応曲線の改善と矛盾せず、両者は「適応データを利用するか」が異なる問いである。

なお、1・2は同一データに対する絶対値と差の可視化なので、両方をポスターに載せる場合は重複がある。スペースが限られる場合、群間差を主張の中心にするなら図2を主図にし、図1は絶対IIAの水準も示す必要があるときに併載する。図3は介入ラベルを使わない転移という別の問いを扱うため、補完的な役割がある。

## 共通の解釈上の限界・発表時の注意

- fold数・モデルseed数は少なく、foldは完全に独立な標本とは限らない。図1・2に信頼区間や有意差検定はなく、図3のmin–maxひげも信頼区間ではない。記述的傾向として話す。
- 図1・2は確認用4 fold（held-out seed 3837、9097、2635、5137）。図3は5 target seed（4287を含む）。seed集合が異なるので、図間の平均値を直接差し引かない。
- 図1・2の横軸0例は通常活動に基づく初期アラインメントを持つ。介入例がゼロであることは、アラインメントが一切ないことを意味しない。
- 図1・2では未見側に介入例を使うため「完全なゼロショット転移」ではない。写像は固定だが、未見側 \(A_m\) を適応している。
- 図3ではheld-out介入ラベルによる調整を行わないが、単一モデル写像の参照seed選定やモデル・課題の前提がある。すべての意味で無条件の汎化を表すわけではない。
- Figure 3右（先行研究）では5 seedの最大IIAを報告しているのに対し、本ポスター図はfold平均、paired差、seed平均など別の集計である。数値の直接比較ではなく、先行研究の問題提起を今回の検証につなぐ。
- モデル専用写像を各target自身に当てた `individual_self_iia` は、適応対照の結果JSONに記録されているが、上記ポスター3図には含めていない。同じモデル用に学習・評価したself-fit基準であり、未見モデル転移の測定とは区別して別図にする。

## 発表用の短いまとめ

> 複雑な写像ではランダム初期化モデルにも高いIIAが得られる可能性がある。今回の適応実験では、同じ介入例数で比べるとtask-trainedモデルがrandomモデルを上回り、その差は例数とともに大きくなった。一方、通常活動由来のアラインメントだけを用いる共有写像の未見モデル転移は限定的だった。したがって、高いself-fitや学習済み群の適応成績と、共有写像の未見モデルへの転移は別の評価軸であり、IIAだけから強い転移可能性を結論しない。
