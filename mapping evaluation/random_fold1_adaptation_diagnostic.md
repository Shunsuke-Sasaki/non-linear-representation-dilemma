# ランダム群 fold 1：未見側適応の失敗箇所診断

対象：fold 1、heldout seed 3837、共同共有写像 init 1  
適応・評価コード：[Alignment_Adaptation_Diagnostic.py](../notebooks/Alignment_Adaptation_Diagnostic.py)、[Alignment_Adaptation_Joint_Diagnostic.py](../notebooks/Alignment_Adaptation_Joint_Diagnostic.py)  
固定写像・直交整列のみの全曲線：[診断結果JSON](../Results/SharedMapping/full/adaptation_diagnostic_random_fold1_init1_v2/results.json)  
写像と整列の同時更新：[診断結果JSON](../Results/SharedMapping/full/adaptation_diagnostic_random_fold1_joint_update_10k/results.json)

## 対象選択と分割

このfoldでは、random共同共有写像が学習元4モデル上でIIA平均約0.744、未見モデルseed 3837の個別写像が2初期化平均約0.732だった。そこで、共有写像の学習失敗と未見モデルの個別対応の不存在という二つの交絡を避ける候補として選んだ。

- adapt-train 10,000例、adapt-dev 10,000例、既存test 10,000例を使用。全条件で同じ保存済みadapt-train/test IDを使い、devは独立split。
- 写像・整列のcheckpoint選択はadapt-devのcross-entropy最小値による。testは選択済みcheckpointの最終評価に一度だけ使った。
- 通常活動整列からの2,000更新、同整列からの固定10,000更新、seedを固定した3個のランダム直交初期値からの各2,000更新を比較した。
- 学習済み状態の同じ未見seed 3837にも同条件を適用した。こちらはモデルと共有写像のstateも異なるため、実装・最適化パイプラインの陽性対照であり、random条件との差を適応初期値だけの因果効果とは解釈しない。
- さらにrandom条件では、固定写像の適応でtrain IIAがchance付近にとどまったため、同じrandom共有写像checkpointから \(\phi\) と直交整列 \(A\) の両方を更新する診断を追加した。これはadapt-trainラベルを共有写像にも使うため、未見転移の成功評価ではない。

## 1. φ固定・直交整列のみ更新

各行はadapt-dev cross-entropyで選んだcheckpointのtrain/dev/test IIA。ランダム直交初期化3回のseedは別々に固定した。

| MLP state | 初期整列 | 予算 | dev選択更新 | train IIA | dev IIA | test IIA |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| random | 通常活動 | 2,000 | 1,000 | 0.5268 | 0.5020 | 0.5027 |
| random | ランダム直交 11001 | 2,000 | 1,600 | 0.5138 | 0.5027 | 0.5038 |
| random | ランダム直交 11002 | 2,000 | 1,200 | 0.5224 | 0.5064 | 0.5026 |
| random | ランダム直交 11003 | 2,000 | 1,600 | 0.5201 | 0.5059 | 0.5050 |
| random | 通常活動 | 10,000 | 6,400 | 0.5250 | 0.5106 | 0.5105 |
| trained | 通常活動 | 2,000 | 1,200 | 0.9881 | 0.9866 | 0.9853 |
| trained | ランダム直交 11001 | 2,000 | 2,000 | 0.8842 | 0.8647 | 0.8725 |
| trained | ランダム直交 11002 | 2,000 | 2,000 | 0.9724 | 0.9643 | 0.9624 |
| trained | ランダム直交 11003 | 2,000 | 2,000 | 0.8829 | 0.8739 | 0.8677 |
| trained | 通常活動 | 10,000 | 9,400 | 0.9964 | 0.9914 | 0.9927 |

randomの通常活動初期整列では、初期train/dev IIAが0.5015/0.4940だった。2,000更新内のtrain IIA最高値は0.5320、dev IIA最高値は0.5104で、いずれもchance近辺だった。10,000更新ではtrain IIAは最大0.5395、dev IIAは最大0.5145まで上がったが、dev-loss選択checkpointのtest IIAは0.5105にとどまった。dev IIA最大値のcheckpointをtestで選ぶことはしていない。

ランダム直交初期値でもdev選択後のtest IIAは0.5026–0.5050だった。いずれの初期値でも、更新は起きている。勾配ノルムは非ゼロで、通常活動初期値からの整列行列のFrobenius移動量は10,000更新時に約2.93、ランダム初期値からは約1.28–1.44だった。直交誤差は最大 \(5.96\times10^{-7}\)。従って「勾配が全く流れない」「整列が更新されない」「直交制約が破れた」という説明とは整合しない。

同じ手順のtrained対照では、通常活動初期値で高いtrain/dev/test IIAを得ており、適応コードと評価経路が一般に機能していないわけではない。一方、ランダム直交初期値ではtrainedでもtest IIAが0.868–0.962と通常初期値より低い。初期化依存性はtrained側にも存在する。

## 2. φとAの同時更新：randomのみ

固定10,000更新、batch size 256とし、初期値は同じfold1 random共有写像checkpointと通常活動整列。写像学習率は既存設定、整列学習率は0.001。checkpointはadapt-dev loss最小値で選択した。

| 更新 | train loss | train IIA | dev loss | dev IIA |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 9.458 | 0.502 | 9.495 | 0.494 |
| 2,000 | 0.627 | 0.630 | **0.668** | 0.580 |
| 3,400 | 0.485 | 0.754 | 0.759 | **0.598** |
| 5,000 | 0.296 | 0.867 | 1.043 | 0.588 |
| 10,000 | 0.091 | 0.966 | 1.979 | 0.577 |

dev loss最小の2,000更新checkpointのtest IIAは **0.5642**（test loss 0.6819）。10,000更新全体を通したdev IIAの最大は3,400更新時の0.5983だったが、その時点ではdev lossが0.7587に悪化している。このdev IIA最大checkpointをtestで追加評価・選択してはいない。

両方を更新するとadapt-trainには高く適合できる一方、更新を続けるほどtrainとdevの隔たりが拡大した。これは固定φの下で整列だけを更新する条件よりtrain fitを大きく改善するが、dev/testに再現する高IIA対応を得たわけではない。**この条件はadapt-trainでφも更新する診断であり、共有写像の転移成功とは呼べない。**

## 解釈

1. **整列だけの適応の失敗点**：この特定ケースでは、固定共有写像のもとで直交整列を2,000～10,000更新しても、train・dev・testのいずれも高IIAに達しなかった。複数のランダム直交初期値も同様だった。勾配と行列更新は確認できたため、単純な更新停止や直交制約逸脱では説明できない。
2. **通常整列初期化だけが原因とは考えにくい**：3個のランダム直交初期値でもtest IIAは約0.50のまま。10,000更新でも通常初期値のtestは0.5105だった。ただし数個の初期値・この最適化法だけでは、初期化依存性を否定できない。
3. **固定写像の制限と過適合の双方が残る**：\(\phi\) と \(A\) を同時に更新するとtrain IIAは0.966まで上がるが、dev lossは後半に悪化し、dev IIAは約0.58、dev選択testは0.564だった。これは固定写像がtrain fitの制約になっている可能性を示唆するが、同時更新で得た自由度はadapt-trainに過適合しており、表現能力の不足や「共有対応の不存在」を証明しない。
4. **陽性対照**：trainedの同じ未見seedでは通常初期化・2,000更新でtest IIA 0.985、10,000更新で0.993。データ分割と評価パイプライン全体が必ずchanceになるわけではない。ただしstateごとにモデルと共有写像が異なるため、randomとの差を原因一つに帰属する比較ではない。

現段階で言えるのは、「fold 1 randomの固定共有写像＋直交整列適応では、初期化を3通りに変え、予算を10,000更新まで増やしても、test IIAの回復はほぼなかった。写像と整列の両方を更新すればtrainには適合するが、dev/testへの一般化は限定的だった」という診断的な観察までである。他のfoldへの拡張や「対応を表現できない」という結論は、追加の独立した初期化・最適化設定で検証する必要がある。

## 再現

```bash
.venv/bin/python notebooks/Alignment_Adaptation_Diagnostic.py
.venv/bin/python notebooks/Alignment_Adaptation_Joint_Diagnostic.py
```
