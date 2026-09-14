# AI高速化技術

## 目次

- [概要](#概要)
- [用語定義（前提知識）](#用語定義前提知識)
  - [NVIDIAソフトウェア・ツール](#nvidiaソフトウェアツール)
  - [Jetsonハードウェア](#jetsonハードウェア)
  - [TensorRTのビルドに関する用語](#tensorrtのビルドに関する用語)
  - [評価指標・関連用語](#評価指標関連用語)
- [技術紹介](#技術紹介)
  - [量子化（Quantization）](#量子化quantization)
    - [量子化の技術概要](#量子化の技術概要)
    - [量子化の実装ガイド](#量子化の実装ガイド)
  - [プルーニング（Pruning）](#プルーニングpruning)
    - [プルーニングの技術概要](#プルーニングの技術概要)
    - [プルーニングの実装ガイド](#プルーニングの実装ガイド)
  - [Deep Learning Accelerator（DLA）](#deep-learning-acceleratordla)
    - [DLAの技術概要](#dlaの技術概要)
    - [DLAの実装ガイド](#dlaの実装ガイド)
  - [演算命令最適化](#演算命令最適化)
    - [演算命令最適化の技術概要](#演算命令最適化の技術概要)
    - [演算命令最適化の実装ガイド](#演算命令最適化の実装ガイド)
- [技術評価](#技術評価)
  - [評価対象モデル](#評価対象モデル)
  - [評価軸](#評価軸)
  - [技術別比較（Jetson実装手段ベース）](#技術別比較jetson実装手段ベース)
  - [プロファイリング手法](#プロファイリング手法)
    - [前提条件（測定環境の固定）](#前提条件測定環境の固定)
    - [計測・プロファイリングツール](#計測プロファイリングツール)
  - [評価のポイント](#評価のポイント)
  - [まとめ](#まとめ)

---

## 概要

深層学習モデルは高精度化に伴いパラメータ数・演算量が増大し続けており、推論時のレイテンシ、消費電力、メモリ使用量が実用上のボトルネックになっている。特に本ドキュメントが対象とする**NVIDIA Jetsonシリーズ**（Nano、Xavier NX/AGX、Orin NX/AGX等）のようなエッジデバイスでは、消費電力・搭載メモリ・熱設計（TDP）に制約があるため、クラウドGPU向けに学習したモデルをそのまま動作させることは難しく、高速化技術の適用がほぼ前提となる。

**AI高速化技術**とは、モデルの推論精度をできる限り維持しながら、計算量・メモリ量・データ転送量を削減し、実行速度と電力効率を向上させるための技術群を指す。アプローチは大きく以下の観点に分類できる。

| アプローチ | 概要 | 代表技術 |
| --- | --- | --- |
| モデル圧縮 | パラメータ数・情報量そのものを削減する | 量子化、プルーニング、蒸留 |
| ハードウェア最適化 | 専用回路・並列演算器を用いて演算効率を上げる | Deep Learning Accelerator（NPU/TPU等） |
| ソフトウェア最適化 | 演算の実装方法・命令列を最適化する | 演算命令最適化（カーネル融合等） |

これらは互いに排他的ではなく、実際のシステムでは「量子化したモデルをDLA上でカーネル融合済みの命令として実行する」といったように組み合わせて用いられることが一般的である。

---

## 用語定義（前提知識）

技術紹介・技術評価で頻出する用語を、Jetson/TensorRTの基礎知識としてあらかじめ整理する。既に知っている用語は読み飛ばしてよい。

### NVIDIAソフトウェア・ツール

| 用語 | 説明 |
| --- | --- |
| **CUDA** | NVIDIA GPU上で並列計算を行うためのプログラミングモデル・実行環境。TensorRTやcuDNN等、本ドキュメントに登場するほぼ全てのソフトウェアの土台となる。 |
| **cuDNN / cuBLAS** | NVIDIAが提供する低レベル演算ライブラリ。cuDNNは畳み込み・プーリング等の深層学習向け演算、cuBLASは行列演算（GEMM）を高速に実行するカーネル実装を提供する。TensorRTは内部でこれら（および独自カーネル）をタクティクスの候補として利用する。 |
| **TensorRT** | NVIDIA製の推論最適化SDK。学習済みモデル（ONNX等）を読み込み、レイヤー融合・量子化・タクティクス選択などの最適化を行った上で、対象GPU/DLAに特化した実行形式（TensorRTエンジン）を生成する。 |
| **TensorRTエンジン** | TensorRTのビルド処理によって生成される、特定のGPU/DLA・TensorRTバージョン・入力形状に最適化された実行可能なバイナリ形式のモデル。ビルドした環境以外（異なるGPU世代やTensorRTバージョン）への互換性はなく、対象デバイス上で再ビルドするのが基本。 |
| **trtexec** | TensorRTに付属するCLIツール。ONNXからのエンジンビルド、レイテンシ/スループットのベンチマーク、レイヤー単位のプロファイル取得などをコマンド一つで行える。本ドキュメントのコマンド例の多くはこれを使用する。 |
| **ONNX（Open Neural Network Exchange）** | PyTorch/TensorFlow等、学習フレームワークに依存しないモデル交換フォーマット。TensorRTはONNXを主要な入力形式として受け付ける。 |
| **ONNX GraphSurgeon** | ONNXの計算グラフをPythonから編集（ノードの追加・削除・入出力の変更等）するためのライブラリ。後処理のGPU実装でNMSノードを追加する際などに使用する。 |
| **TensorRTプラグイン** | TensorRTが標準でサポートしない演算を、カスタムCUDA実装として組み込む仕組み。`EfficientNMS_TRT`はNMSを実装した代表的な公式プラグイン。 |
| **NVIDIA TensorRT Model Optimizer（modelopt）** | PyTorchモデルに対しPTQ/QAT/プルーニング/スパース化を共通APIで適用できるPythonライブラリ。`mtq`（量子化）、`mtp`（プルーニング）、`mts`（スパース化）等のサブモジュールで構成される。 |

### Jetsonハードウェア

| 用語 | 説明 |
| --- | --- |
| **GPU（CUDAコア／Tensorコア）** | 汎用の並列演算ユニット。Tensorコアは行列積演算に特化した回路で、Ampere世代（Orin）以降のJetsonに搭載される。 |
| **DLA（Deep Learning Accelerator／NVDLA）** | Xavier系・Orin系SoCに搭載される推論専用の固定機能アクセラレータ。GPUとは独立した回路で、対応演算子は限定的。 |
| **統合メモリ（UMA: Unified Memory Architecture）** | JetsonのCPUとGPUが同一の物理メモリ（LPDDR）を共有する構成。デスクトップGPUのような専用VRAMを持たないため、CPU/GPU間でメモリ帯域を奪い合う点に注意が必要。 |
| **DVFS（Dynamic Voltage and Frequency Scaling）** | 消費電力・発熱に応じてCPU/GPUの動作クロックと電圧を動的に変更する仕組み。ベンチマーク結果がぶれる主要因であり、計測前に`jetson_clocks`で固定するのはこのため。 |
| **パワーモード（`nvpmodel`）** | Jetsonが持つ、CPU/GPUコア数や最大クロックの組み合わせをプリセット化した動作モード。バッテリー駆動時の省電力モードから最大性能のMAXNモードまで機種ごとに複数用意されている。 |
| **GPUアーキテクチャ世代（Volta / Ampere等）** | JetsonのGPU設計世代を表す名称。Xavier系はVolta世代、Orin系はAmpere世代であり、2:4構造化スパース性（Sparse Tensor Core）などAmpere以降でしか使えない機能がある。 |

### TensorRTのビルドに関する用語

| 用語 | 説明 |
| --- | --- |
| **ビルド（エンジンビルド）** | ONNX等の入力モデルから、対象ハードウェア向けに最適化されたTensorRTエンジンを生成する処理。レイヤー融合・精度モードの決定・タクティクス選択などがこの過程で行われる。 |
| **キャリブレーション（Calibration）** | PTQにおいて、少量の代表データをモデルに流し、各層の値の分布からINT8等の量子化スケールを決定する処理。 |
| **レイヤー融合（Layer/Kernel Fusion）** | 複数の演算を1つのGPUカーネルにまとめる最適化。TensorRTのビルド時に自動的に行われる。 |
| **タクティクス（Tactics）** | ある演算を実行する際にTensorRTが選択できる、複数のカーネル実装の候補（cuDNN実装、cuBLAS実装、TensorRT独自実装等）。ビルド時にどれが最速かを実測し選定する。 |
| **ワークスペースメモリ（Workspace）** | エンジンビルド時にタクティクスの候補が一時的に使用できるGPUメモリの上限。大きく確保するほど選択肢が広がり、より高速なタクティクスが選ばれる可能性がある。 |
| **タイミングキャッシュ（Timing Cache）** | ビルド時に行ったタクティクスの実測結果を保存するファイル。同一GPU上での再ビルド時に再利用することでビルド時間を短縮できる。 |
| **最適化プロファイル（Optimization Profile）** | 動的な入力形状（バッチサイズや解像度が可変）に対応するために、TensorRTがビルド時に指定する入力サイズの範囲（最小/最適/最大）。 |

### 評価指標・関連用語

| 用語 | 説明 |
| --- | --- |
| **FLOPs（Floating Point Operations）** | モデル1回の推論に必要な浮動小数点演算回数。プルーニングの削減目標（例:「FLOPs比60%」）などモデルの計算量を表す指標として使う。 |
| **レイテンシ／スループット** | レイテンシは1回の推論にかかる時間（ms）、スループットは単位時間あたりに処理できる推論回数（FPS等）。バッチサイズによって最適な指標が異なる。 |
| **mAP（mean Average Precision）** | 物体検出タスクの代表的な精度指標。本ドキュメントではYOLOXの精度評価に用いる。 |
| **IoU（Intersection over Union）** | 予測領域と正解領域の重なり度合いを表す指標。セグメンテーション（UNet）の精度評価や、検出のNMS処理におけるボックスの重複判定にも使われる。 |
| **NMS（Non-Maximum Suppression）** | 物体検出モデルが出力する重複したバウンディングボックスを、IoUに基づいて間引く後処理。 |

---

## 技術紹介

各技術は「技術概要（What/Why）」と「実装ガイド（How）」に分けて記載する。概念だけを把握したい場合は技術概要のみ、実際にJetson上で試す場合は実装ガイドを参照する。

### 量子化（Quantization）

モデルの重みや活性化値を、通常使われるFP32（32bit浮動小数点）から、より少ないビット数（INT8、INT4、FP16、BF16など）で表現し直す技術。

#### 量子化の技術概要

- **メリット**
  - メモリ使用量・帯域幅を削減（例: FP32→INT8で理論上1/4）
  - 整数演算・低精度演算に特化したハードウェアで高速化
  - モデルサイズ縮小によりエッジデバイスへの搭載が容易になる
- **デメリット**
  - 表現できる数値の範囲・分解能が下がるため精度が劣化しやすい
  - 極端な低ビット化（INT4以下）では精度劣化が顕著になる場合がある

- **主な手法**
  - **PTQ（Post-Training Quantization）**: 学習済みモデルに対して事後的に量子化を適用する。実装が容易で追加学習が不要だが、精度劣化のリスクがやや高い。
  - **QAT（Quantization-Aware Training）**: 学習時から量子化による誤差を考慮してファインチューニングを行う。精度は維持しやすいが追加の学習コストがかかる。
  - **混合精度量子化**: レイヤーごとに感度が異なることを利用し、重要な層は高精度、それ以外は低精度にする。

#### 量子化の実装ガイド

##### TensorRT（trtexec）によるPTQ/QAT

```bash
# ONNXモデルをFP16でビルド(参考: 量子化前のベースライン)
trtexec --onnx=model.onnx --saveEngine=model_fp16.engine --fp16

# PTQ: キャリブレーションキャッシュを使ってINT8エンジンをビルド
trtexec --onnx=model.onnx --saveEngine=model_int8.engine \
  --int8 --calib=calibration.cache

# QAT: 事前にModel OptimizerでQAT学習済みのONNXをそのままINT8ビルド
# (量子化ノードがONNXに埋め込まれているためcalibファイル不要)
trtexec --onnx=model_qat.onnx --saveEngine=model_qat_int8.engine --int8
```

##### NVIDIA TensorRT Model Optimizer（modelopt）によるPTQ/QAT

NVIDIA製のPythonライブラリ**TensorRT Model Optimizer**（`pip install nvidia-modelopt`、`modelopt.torch.quantization`を慣例的に`mtq`としてimport）を使うと、PyTorchモデルに対してPTQ/QATを共通のAPIで適用し、そのままONNX経由でTensorRT化できる。

```python
import torch
import modelopt.torch.quantization as mtq

model = get_model()  # 学習済みPyTorchモデル
calib_loader = get_calibration_dataloader(num_samples=512)  # キャリブレーション用データ(数百枚程度で十分)

# キャリブレーション時にモデルへデータを流すループを定義
def forward_loop(model):
    for batch in calib_loader:
        model(batch)

# --- PTQ ---
# INT8設定でキャリブレーションを実行(この時点で量子化ノードが挿入されスケールが計算される)
model = mtq.quantize(model, mtq.INT8_DEFAULT_CFG, forward_loop)  # SmoothQuant系は mtq.INT8_SMOOTHQUANT_CFG

# 量子化サマリ(各レイヤーのビット幅・量子化状況)を確認
mtq.print_quant_summary(model)

# ONNXへエクスポート(以降は前述のtrtexec --int8でTensorRTエンジン化する)
torch.onnx.export(model, dummy_input, "model_ptq.onnx", opset_version=17)
```

```python
# --- QAT ---
# PTQと同じmtq.quantize()で量子化ノードを挿入した後、通常の学習ループでファインチューニングする
model = mtq.quantize(model, mtq.INT8_DEFAULT_CFG, forward_loop)

optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
for epoch in range(num_qat_epochs):  # 目安: 通常学習の10%程度のエポック数から
    train_one_epoch(model, train_loader, optimizer)

torch.onnx.export(model, dummy_input, "model_qat.onnx", opset_version=17)
```

### プルーニング（Pruning）

モデル内の重要度が低いパラメータ（重み、チャネル、ニューロン、レイヤーなど）を削除し、モデルを疎（スパース）化する技術。

#### プルーニングの技術概要

- **メリット**
  - パラメータ数・演算量の削減によるモデルの軽量化
  - 推論速度の向上（特に構造化プルーニングの場合）
- **デメリット**
  - 過度なプルーニングは精度低下を招く
  - 非構造化プルーニングは疎行列演算に対応したハードウェア／ライブラリがないと速度向上に直結しにくい

- **主な手法**
  - **非構造化プルーニング（Unstructured Pruning）**: 個々の重みを閾値やスコアに基づいて0にする。圧縮率は高いが、一般的なハードウェアでは疎行列演算の恩恵を受けにくい。
  - **構造化プルーニング（Structured Pruning）**: チャネル単位・フィルタ単位・レイヤー単位でまとめて削除する。密な行列演算のまま計算できるため、実機での高速化効果を得やすい。
  - **反復的プルーニング**: プルーニングとファインチューニングを繰り返し、精度を維持しながら段階的に圧縮率を高める。

##### 反復的プルーニングのイメージ

一度に大きく削る「one-shotプルーニング」は、モデルが失われたパラメータに適応する猶予がないため、目標圧縮率が高いほど精度が大きく落ち、ファインチューニングでも回復しきれないことが多い。反復的プルーニングは、**小さく削っては学習し直す**を繰り返すことで、モデルを少しずつ新しい構造に慣れさせながら圧縮率を積み上げていく手法である。

```text
学習済みモデル（FLOPs比 100%）
      │
      ▼
 [Round 1] 20%削減 ─▶ ファインチューニング（数epoch）─▶ FLOPs比 80%
      │
      ▼
 [Round 2] 20%削減 ─▶ ファインチューニング ─▶ FLOPs比 60%
      │
      ▼
      …（目標圧縮率に達するまで繰り返す）
      │
      ▼
 [Round N] 目標FLOPs比に到達 ─▶ 十分なエポック数で最終ファインチューニング
```

数値はあくまでイメージだが、同じ「最終的にFLOPs比40%まで削減する」という目標でも、one-shotと反復的プルーニングでは精度の推移が次のように異なりやすい。

| ラウンド | 累積FLOPs比率 | one-shot（1回で60%削減） | 反復的（毎回20%ずつ削減） |
| --- | --- | --- | --- |
| 0（元モデル） | 100% | 76.5% | 76.5% |
| 1 | 80% | ─ | 76.0%（削減直後74.5%→FT後76.0%） |
| 2 | 60% | ─ | 74.8%（削減直後71.0%→FT後74.8%） |
| 3（目標到達） | 40% | 61.5%（削減直後38%→FTでも61.5%までしか戻らない） | 71.2%（削減直後63%→FT後71.2%） |

one-shotは1回のプルーニングで多くの重要な重みも同時に失われるため、ファインチューニングだけでは構造の欠落を補いきれない。反復的プルーニングは各ラウンドの削減幅が小さく、残った重みが徐々に失われた分の役割を肩代わりできるため、同じ最終圧縮率でも精度を維持しやすい。その代わり、ラウンドごとにファインチューニングが必要になるため学習コスト（時間・GPU資源）は増える。

> 根拠: 段階的にプルーニングと再学習を繰り返す有効性は、Han et al., "Learning both Weights and Connections for Efficient Neural Networks" (NeurIPS 2015) が「学習→プルーニング→再学習」のサイクルを繰り返すことで精度を保ったまま大幅な圧縮を達成できることを示したのが端緒であり、Zhu & Gupta, "To Prune, or Not to Prune: Exploring the Efficacy of Pruning for Model Compression" (arXiv:1710.01878, 2017) は、同じ最終スパース率であっても一括（one-shot）プルーニングより段階的（gradual/iterative）にスパース率を上げていく手法の方が精度を維持しやすいことを実験的に示している。上表の数値はこれらの知見に基づく傾向を示すイメージであり、実測値ではない。

#### プルーニングの実装ガイド

modeloptの`mtp.prune`を使う場合、`constraints`のFLOPs目標を少しずつ厳しくしながらループで呼び出すことで、反復的プルーニングを実現できる。

```python
import modelopt.torch.opt as mto
import modelopt.torch.prune as mtp

model = get_model()
target_flops_schedule = [80, 60, 40]  # 段階的に絞り込むFLOPs目標(%、毎回20ポイントずつ厳しくする)

for step, flops_pct in enumerate(target_flops_schedule, start=1):
    model, _ = mtp.prune(
        model,
        mode="fastnas",
        constraints={"flops": f"{flops_pct}%"},
        dummy_input=dummy_input,
        config={
            "data_loader": train_loader,
            "score_func": lambda m: validate(m, val_loader),
            "checkpoint": f"fastnas_round{step}.pth",
        },
    )

    # 削減幅が小さいラウンドなので、フルスクラッチではなく短いエポック数で回復させる
    train(model, train_loader, optimizer, scheduler, epochs=few_epochs)

    acc = validate(model, val_loader)
    print(f"Round {step}: FLOPs {flops_pct}%, Accuracy {acc:.2f}")

mto.save(model, "iterative_pruned_model.pth")
```

`modelopt.torch.prune`（慣例的に`mtp`としてimport）は構造化プルーニング（チャネル・レイヤー単位の削除）を、`modelopt.torch.sparsity`（`mts`）は重み単位の疎化（スパース化）を提供する。

##### 構造化プルーニング（FastNAS、CVモデル向け）

```python
import torch
import modelopt.torch.prune as mtp
import modelopt.torch.opt as mto

model = get_model()  # 学習済みモデル(例: ResNet)
dummy_input = torch.randn(1, 3, 224, 224)

def score_func(model):
    return validate(model, val_loader)  # 探索時に最大化したい指標(精度など)を返す

# FLOPsを元モデルの60%まで削減する制約でチャネル構成を探索
pruned_model, pruning_scores = mtp.prune(
    model,
    mode="fastnas",
    constraints={"flops": "60%"},
    dummy_input=dummy_input,
    config={
        "data_loader": train_loader,
        "score_func": score_func,
        "checkpoint": "fastnas_search_checkpoint.pth",
    },
)

# 探索結果のsubnetは通常のstate_dictでは保存/復元できないためmto.save/mto.restoreを使う
mto.save(pruned_model, "pruned_model.pth")
model = mto.restore(get_model(), "pruned_model.pth")

# ファインチューニングして精度を回復させる
train(model, train_loader, optimizer, scheduler)

torch.onnx.export(model, dummy_input, "model_pruned.onnx", opset_version=17)
```

###### FastNASの仕組み

FastNASは、モデル各層の出力チャネル数（幅）を探索対象とする、NAS（Neural Architecture Search）の考え方をプルーニングに応用した手法である。

1. **探索空間の定義**: 各Conv/Linear層の出力チャネル数について、元のチャネル数を上限に複数の候補値（例: 64の倍数刻みで32, 64, 96, ...）を定義する。どの層を何チャネルまで削るかの組み合わせ全体が探索空間になる。
2. **候補サブネットの評価**: 学習済みモデルの重みをそのまま流用しながら、探索空間内のチャネル構成候補（サブネット）ごとに、ユーザーが指定した`score_func`（通常は検証データでの精度）でスコアを算出する。候補ごとにフルスクラッチで再学習する必要がないため、探索コストを抑えられる。
3. **制約下での最適化**: `constraints`で指定したFLOPs/パラメータ数の上限を満たしつつ、スコアを最大化するチャネル構成を選び出す。
4. **ファインチューニング**: 選ばれたサブネット（チャネル構成が確定したモデル）に対し、通常の学習パイプラインで再学習を行い、構造変更による精度低下を回復する。

FastNASによる削減はチャネル自体を取り除く構造的な変更であるため、疎行列演算のような特別なハードウェアサポートを必要とせず、通常の密な行列演算としてGPU/DLAどちらでも高速化効果を得やすい。

###### constraintsオプションの挙動（上限として扱われる点に注意）

`mtp.prune`の`constraints`（`flops`/`params`）は、**「ちょうどこの値にする」という目標値ではなく「これを超えないという上限」**として扱われる。指定した値と一致するサブネットが選ばれるとは限らない点に注意する。

- **指定方法**: 絶対値（例: `{"flops": 4.5e9}`）またはパーセンテージ文字列（例: `{"flops": "80%"}`、元モデルに対する残存比率）のいずれかで指定する。
- **上限としての扱い**: `{"flops": "80%"}`と指定した場合、探索対象になるのは「元モデルのFLOPsの80%以下」を満たすサブネットに限られ、その中から`score_func`（精度など）を最大化するものが選ばれる。したがって実際に選ばれるサブネットのFLOPsが80%ちょうどになるとは限らず、80%以下のどこか（探索空間の候補次第では75%や70%など、指定値よりかなり下回ることもある）に落ち着く。**下限ではなく上限**である点に注意（「80%は最低でも残す」という意味ではない）。
- **探索空間が離散的である点**: 各層のチャネル数の候補は連続値ではなく、`channel_divisor`等で指定した刻み幅の離散値（例: 64刻み）で構成される。そのため、指定した上限にちょうど一致するサブネットが探索空間に存在するとは限らず、「上限を超えない範囲で最もスコアが高い候補」が選ばれた結果、指定値よりも低いFLOPs比になることがある。
- **正確に近づけたい場合の進め方**: NVIDIA公式ドキュメントは、まず探索空間のセントロイド（中心値）付近を初期値として試し、得られたサブネットを実機（Jetson）でプロファイリングして実際のレイテンシを確認したうえで、必要に応じて制約値を調整していく、という反復的な進め方を推奨している。「80%ちょうど」を厳密に保証する機能ではない。

##### 重み単位の疎化（sparsity、`mts`）

```python
import modelopt.torch.sparsity as mts

sparsity_config = {
    "data_loader": calib_loader,
    "collect_func": lambda batch: batch,
}

# "sparse_magnitude"(重みの絶対値ベース、軽量)または"sparsegpt"(データに基づく高精度な手法)を選択
sparse_model = mts.sparsify(model, "sparsegpt", config=sparsity_config)

# ファインチューニングでスパース化による精度劣化を回復させる(推奨)
train(sparse_model, train_loader, optimizer, scheduler)

# マスクを重みに適用し、通常のPyTorchモデルとしてエクスポートする
mts.export(sparse_model)
torch.onnx.export(sparse_model, dummy_input, "model_sparse.onnx", opset_version=17)
```

###### 2:4スパース性の仕組み

`mts.sparsify()`のデフォルトパターンはNVIDIAの**2:4構造化スパース性**である。これは個々の重みを自由な位置で0にする「純粋な非構造化プルーニング」ではなく、GPU側のハードウェアサポートを前提とした半構造化（semi-structured）のスパースパターンである。

- **パターン**: 重み行列を4個ずつの連続したグループに分割し、各グループ内で2個を0にする（残り2個が非ゼロ）。
- **ハードウェアでの高速化の仕組み**: Ampere世代以降のGPU（**Jetson OrinのGPUは対応、Xavier系〈Volta〉は非対応**）に搭載される**Sparse Tensor Core**は、この2:4パターンに従って圧縮された重み（非ゼロ値だけを詰めた行列＋どの位置が非ゼロかを示す2bitのメタデータ）を読み込み、ゼロとの乗算を省略することで、密な行列演算に比べて理論上約2倍のスループットを実現する。
- **スパースパターンの決定方法**: modeloptは主に2つのモードを提供する。
  - `sparse_magnitude`: NVIDIAのASP（Automatic SParsity）に準拠した手法で、各4個のグループ内で絶対値が小さい2個を0にする（重みの大きさに基づく単純な基準）。計算コストが低く高速。
  - `sparsegpt`: キャリブレーションデータを用いて、重みを除去した場合の出力への影響を考慮しながら0にする重みを選ぶデータドリブンな手法。`sparse_magnitude`より計算コストは高いが、同じスパース率でも精度劣化を抑えやすい。
- **精度回復**: いずれの手法でもスパース化直後は精度が低下するため、スパースパターン（0の位置）を固定した状態でファインチューニングを行い、残った重みで精度を回復させるのが基本の流れである。

これに対し、位置に制約のない純粋な非構造化プルーニングは圧縮率（モデルサイズ削減）を高めやすい一方、TensorRT/GPU上には疎行列専用の実行カーネルが無いため、そのままでは推論速度の向上には直結しにくい。この非対称性は後述の技術別比較表「非構造化プルーニング／スパース性」の備考と対応している。

### Deep Learning Accelerator（DLA）

深層学習の推論・学習に特化したハードウェア（専用回路）。汎用CPUに比べ、行列演算・畳み込み演算を大幅に高速化・低電力化できる。

Jetsonシリーズには以下2種類の演算エンジンが搭載されており、用途に応じて使い分ける（または並行利用する）ことができる。

#### DLAの技術概要

- **GPU（CUDAコア／Tensorコア）**: 汎用性が高く、対応していない演算子（レイヤー）がほぼ無い。Ampere世代（Orin）以降はTensorコアによる低精度・混合精度演算に対応。
- **DLA（Deep Learning Accelerator／NVDLA）**: Xavier系・Orin系SoCに搭載される推論専用の固定機能アクセラレータ。**GPUとは独立したダイ上の回路**のため、DLAにレイヤーをオフロードするとGPUリソースを他の処理（別モデルの推論、CUDAカーネルによる前処理等）に空けられる。ただし対応する演算子（Conv、Pooling、活性化関数など標準的な層が中心）が限定されており、非対応レイヤーはGPUへの自動フォールバックが発生し、エンジン間のデータコピー分だけオーバーヘッドが生じる。

- **メリット**
  - 汎用プロセッサに比べ桁違いの演算スループット・電力効率（TOPS/W）を実現
  - GPUと独立に動作するため、DLAオフロードによりGPUとの並列実行（マルチストリーム推論）が可能
- **デメリット・注意点**
  - DLAは対応演算子が限定的で、非対応レイヤーの混在によりGPU⇔DLA間のフォールバック・データコピーが発生し、期待した速度向上が得られない場合がある
  - DLAの動作クロックはGPUより低く設定されていることが多く、単体性能はGPUに劣るケースがある（電力効率とのトレードオフ）
  - モデル側をハードウェアの制約（対応演算子、量子化形式など）に合わせて変換する必要がある（TensorRTエンジンのビルドが前提）

#### DLAの実装ガイド

```bash
# DLA0上で実行するエンジンをビルド(非対応レイヤーはGPUへ自動フォールバック)
trtexec --onnx=model.onnx --saveEngine=model_dla0.engine \
  --int8 --useDLACore=0 --allowGPUFallback

# GPUフォールバックを禁止し、非対応レイヤーがある場合はビルドを失敗させる
# (どの層がDLA非対応かを明確に洗い出したい場合に使う)
trtexec --onnx=model.onnx --saveEngine=model_dla0.engine \
  --int8 --useDLACore=0

# GPUとDLA0/DLA1を使い、3本の推論プロセスを並列実行する例
trtexec --loadEngine=model_gpu.engine  --avgRuns=100 &
trtexec --loadEngine=model_dla0.engine --useDLACore=0 --avgRuns=100 &
trtexec --loadEngine=model_dla1.engine --useDLACore=1 --avgRuns=100 &
wait
```

##### DLA実行状況・フォールバックの確認方法

`--useDLACore`を指定しても、非対応レイヤーが含まれていれば静かにGPUへフォールバックするため、**実際にDLAで動いているかはビルド時・実行時の両方で確認する**必要がある。

###### ビルド時: どのレイヤーがDLA/GPUどちらに割り当てられたかを確認する

```bash
# --verboseログにはDLA非対応レイヤーのフォールバック警告が出力される
trtexec --onnx=model.onnx --saveEngine=model_dla0.engine \
  --int8 --useDLACore=0 --allowGPUFallback --verbose \
  2>&1 | grep -i "fallback\|not supported by DLA"

# レイヤーごとの割り当て先(DLA/GPU)をJSONとして出力し、内容を直接確認する
trtexec --onnx=model.onnx --saveEngine=model_dla0.engine \
  --int8 --useDLACore=0 --allowGPUFallback \
  --dumpLayerInfo --exportLayerInfo=layer_info.json
```

###### 実行時: 実際にDLAが稼働しているかを確認する

```bash
# 注意: tegrastatsはCPU/GPUの使用率のみを表示し、DLAの使用率は出力されない
tegrastats

# jtopの「ENG」タブでDLA0/DLA1を含む各エンジンの使用率をリアルタイムに確認できる
sudo pip3 install -U jetson-stats
jtop   # 起動後「ENG」タブに切り替えてDLA使用率を確認する
```

> 補足: 非対応レイヤーの混在によりフォールバックが多発している場合、jtopの「ENG」タブでDLA使用率が低いままGPU使用率だけが高止まりする、という傾向として現れる。`--dumpLayerInfo`で事前にどのレイヤーがGPUに割り当てられているかを特定し、可能であれば非対応レイヤーをDLA対応の演算に置き換える（またはDLA非対応部分だけ別エンジンとして切り出す）ことで、フォールバックによるデータコピーのオーバーヘッドを減らせる。

### 演算命令最適化

モデルの構造やハードウェア自体を変えず、実行時の演算・命令列をハードウェア特性に合わせて最適化する技術。コンパイラ・ランタイムレベルでの高速化に相当する。

#### 演算命令最適化の技術概要

- **主な手法**
  - **カーネル融合（Kernel/Operator Fusion）**: 複数の演算（例: Conv→BatchNorm→ReLU）を1つの演算カーネルにまとめ、メモリアクセス回数を削減する
  - **メモリレイアウト最適化**: キャッシュ効率の良いデータ配置（例: NCHW⇔NHWC変換）によりメモリアクセスのボトルネックを減らす
  - **グラフ最適化・コンパイラ最適化**: 計算グラフ全体を解析し、冗長な演算の除去や演算順序の並び替えを行う（例: TensorRT、XLA、TVM、ONNX Runtime）
  - **演算子の低レベル実装最適化**: im2col、Winograd変換、FFTベースの畳み込みなど、アルゴリズムレベルでの計算量削減
  - **TensorRTのビルド最適化オプション**: エンジン生成時のフラグ・設定でタクティクス（カーネル実装）の選び方やメモリ計画を最適化する
  - **推論ウォームアップ（Warm-up）**: 初回推論特有のオーバーヘッドを実運用前のダミー推論で吸収する
  - **CUDA Graph**: 一連のカーネル起動をグラフとして記録し1回のAPI呼び出しで再生することで、CPU側の起動オーバーヘッドを削減する
  - **後処理のGPU実装**: NMSやデコード処理などCPU側で行われがちな後処理をGPU(CUDA)上で実装し、データ転送とCPU処理のボトルネックを解消する

- **メリット**
  - モデル精度に一切影響を与えずに高速化できる（数値誤差を除く）
  - 既存モデル・既存ハードウェアに対しても適用可能
- **デメリット**
  - ハードウェア・フレームワークごとに個別最適化が必要で汎用性・移植性が低い場合がある
  - 高度な最適化ほど実装・検証コストが高い

#### 演算命令最適化の実装ガイド

##### TensorRTビルドの仕組み

TensorRTの「ビルド」は単なるファイル形式変換ではなく、ONNXの計算グラフを対象GPU/DLA向けに実測ベースで最適化し尽くす処理である。主に以下のステップで構成される。

1. **グラフの解析・最適化**: ONNXの計算グラフを読み込み、使われていない計算の除去（dead code elimination）や定数畳み込み（constant folding）を行う。
2. **レイヤー融合（Layer Fusion）**: Conv→BatchNorm→ReLUのような特定のパターンを検出し、複数レイヤーを1つのGPUカーネルに統合する。融合によって、レイヤーごとに発生していたカーネル起動オーバーヘッド（ホスト側で1回あたり数μs〜十数μs程度）や、中間結果をいちいちDRAMに書き出す/読み出すコストを削減できる。融合は自動的に行われ、実行時に有効/無効を切り替えるAPIは存在しない。
3. **タクティクス選択（実測ベースのオートチューニング）**: 各レイヤー（融合後を含む）に対し、cuDNN実装・cuBLAS実装・TensorRT独自カーネルなど複数の候補（タクティクス）を対象GPU上で実際に実行して速度を計測する。デフォルトでは各タクティクスにつき4回以上実行して平均レイテンシを求め、最速のものを採用する。この実測にビルド時間の大半が費やされる。
4. **メモリプランニング**: 各テンソルの生存期間を解析し、GPUメモリを使い回すことでピークメモリ使用量を抑えるようにメモリ割り当てを計画する。
5. **エンジンのシリアライズ**: 上記すべての最適化結果を、対象ハードウェア・TensorRTバージョン・入力形状に固有の実行可能形式（TensorRTエンジン）としてファイルに書き出す。

前述の「TensorRTのビルド最適化オプション」（`--builderOptimizationLevel`、`--timingCacheFile`等）は、この一連の処理の**タクティクス選択のやり方**を調整するものである。例えば`--builderOptimizationLevel`を上げるとタクティクス候補をより広く探索するためビルド時間は延びるが、より速いタクティクスが見つかる可能性が上がる。`--timingCacheFile`は、同一設定のレイヤーに対する実測結果（タイミング）を再利用することでステップ3の実測を省略し、ビルド時間を短縮する。

##### エンジンビルド・推論に使えるインターフェース

TensorRTでのエンジンビルドおよび推論の実行には、公式に3種類のインターフェースが用意されている。ビルド済みのエンジン（`.engine`ファイル）自体はインターフェース間で共通の成果物なので、「ビルドはPython APIで行い、推論はC++ APIで実行する」といった使い分けも可能である。

| インターフェース | 主な用途 | 特徴 |
| --- | --- | --- |
| **trtexec（CLI）** | プロトタイピング、ベンチマーク、CI等でのエンジンビルド自動化 | コマンド一つでビルド・推論・プロファイリングができる。これまでのコマンド例の多くがこれに該当する。アプリケーションへの組み込みではなく、検証・計測用のツール。 |
| **Python API** | 研究・開発、キャリブレーションスクリプトの作成、PyTorch/ONNXツールチェーンとの連携 | NumPy/PyTorch等のPythonエコシステムと連携しやすく、プロトタイピングやデバッグが容易。反面、Pythonインタプリタのオーバーヘッドがある。 |
| **C++ API** | 実運用（プロダクション）でのエンジンビルド・推論 | オーバーヘッドが小さく、リアルタイム性・省メモリが求められる用途に向く。既存のC++ベースの認識/制御パイプライン（ロボティクス等）にそのまま組み込みやすい。 |

TensorRTの実体（`libnvinfer`等のライブラリ）はC++で実装されており、**Python API（`tensorrt`パッケージ）はpybind11によるバインディングでこのC++ APIをラップしたもの**である（PyTorchやTensorFlowと同様の「C++コア＋Pythonフロントエンド」という構成）。Pythonから呼び出す際は、Pythonオブジェクト⇔C++間のデータ変換（バッファ変換）やインタプリタのオーバーヘッドが、C++ APIを直接使う場合に比べて余分に発生する。

**実運用ではC++ APIが基本的に推奨される。** TensorRT公式ドキュメントでも、C++ APIはオーバーヘッドが小さく、性能が重要なアプリケーションや自動車分野など一部のコンプライアンス要件を満たす上で有利とされる一方、Python APIはプロトタイピング・デバッグ・テストに向くと位置付けられている。Python APIがC++コアのラッパーである以上、C++ APIを直接使う方がラッピングに伴う変換コストを回避でき、本質的にオーバーヘッドが小さくなる。Jetson上で継続稼働する推論サービスやロボティクスのような、レイテンシとメモリ使用量がシビアに問われる用途では、C++ API（またはそれをラップした自作の推論ランタイム）で最終実装を行うのが基本方針になる。

> 補足: エンジンファイル自体はPython APIとC++ APIの間で相互に利用可能（Python APIでビルドしたエンジンをC++ APIでロードして推論する、といった構成も可能）である。そのため「開発・検証はPython、実運用はC++」という役割分担が現実的な進め方になる。

##### TensorRTのビルド最適化オプション

```bash
# ビルド最適化レベル(0〜5、大きいほど探索に時間をかけ最適なタクティクスを選ぶ。デフォルトは3)
trtexec --onnx=model.onnx --saveEngine=model.engine --fp16 --builderOptimizationLevel=5

# ビルド時のworkspaceサイズを増やす(MiB単位。大きいアルゴリズム候補も選択可能になる)
trtexec --onnx=model.onnx --saveEngine=model.engine --fp16 --memPoolSize=workspace:2048

# タイミングキャッシュを保存/再利用し、同一GPUでの再ビルド時間を短縮する
trtexec --onnx=model.onnx --saveEngine=model.engine --fp16 --timingCacheFile=timing.cache

# レイヤー融合の結果や採用されたタクティクスを詳細ログで確認する
trtexec --onnx=model.onnx --saveEngine=model.engine --fp16 --verbose --profilingVerbosity=detailed
```

```python
# Python API(tensorrt)での同等の設定
import tensorrt as trt

logger = trt.Logger(trt.Logger.INFO)
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
trt.OnnxParser(network, logger).parse_from_file("model.onnx")

config = builder.create_builder_config()
config.set_flag(trt.BuilderFlag.FP16)
config.builder_optimization_level = 5
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)  # 2GB

engine_bytes = builder.build_serialized_network(network, config)
with open("model.engine", "wb") as f:
    f.write(engine_bytes)
```

##### 推論ウォームアップ（Warm-up）

初回推論はCUDAコンテキストの初期化、cuDNN/cuBLAS側のアルゴリズム選択、クロック（DVFS）の立ち上がり待ちなどにより明らかに遅くなる。実測やサービス提供の前に必ずダミー入力でのウォームアップ推論を挟むことで、この一過性のオーバーヘッドを実測から除外する。

```bash
# trtexecは--warmUp(ms)で計測前に自動的にウォームアップを行う
trtexec --loadEngine=model.engine --warmUp=500 --avgRuns=100 --duration=30
```

```python
import time

context = engine.create_execution_context()
stream = cuda.Stream()  # 例: cuda-python等で作成したCUDAストリーム

# TensorRT 10以降はexecute_v2/execute_async_v2が廃止され、execute_async_v3のみが提供される。
# バインディングをリストで渡す方式ではなく、テンソル名ごとにアドレスを設定する。
for i in range(engine.num_io_tensors):
    name = engine.get_tensor_name(i)
    context.set_tensor_address(name, bindings[i])  # bindings[i]は入出力バッファのデバイスポインタ

# ウォームアップ: 実測に影響しないダミー入力で数十回流す
for _ in range(50):
    context.execute_async_v3(stream_handle=stream.handle)
stream.synchronize()

# ここから実測
start = time.perf_counter()
for _ in range(100):
    context.execute_async_v3(stream_handle=stream.handle)
stream.synchronize()
elapsed_ms = (time.perf_counter() - start) / 100 * 1000
print(f"avg latency: {elapsed_ms:.2f} ms")
```

> 補足: Jetsonでは`jetson_clocks`でクロックを固定していても、cuDNN/cuBLASのアルゴリズム選択やメモリアロケータのキャッシュ生成は初回呼び出し時に発生する。ウォームアップなしで計測すると、実運用時よりも悲観的な（遅い）レイテンシを測ってしまう点に注意する。

##### CUDA Graph

CUDA Graphは、一連のカーネル起動・メモリ操作を事前に記録（capture）し、1回のグラフ再生（replay）にまとめる仕組み。カーネル自体は速くならないが、小バッチ・軽量モデルのように多数の小さなカーネルを起動する推論ではCPU側の起動オーバーヘッドの割合が大きくなりやすく、それを削減できる。入力形状が固定（または単一の最適化プロファイルに限定）されている場合に利用できる。

```bash
# trtexecでCUDA Graphを有効化してベンチマーク
trtexec --loadEngine=model.engine --useCudaGraph --avgRuns=100 --duration=30
```

```python
# TensorRT公式ドキュメントに準拠したキャプチャ手順(cuda-pythonのcudart経由)
from cuda import cudart

err, stream = cudart.cudaStreamCreate()

# ウォームアップ: 入力形状に対するTensorRT内部状態を確定させてからキャプチャする
context.execute_async_v3(stream)
cudart.cudaStreamSynchronize(stream)

# キャプチャ
cudart.cudaStreamBeginCapture(stream, cudart.cudaStreamCaptureMode.cudaStreamCaptureModeGlobal)
context.execute_async_v3(stream)
err, graph = cudart.cudaStreamEndCapture(stream)
err, graph_exec = cudart.cudaGraphInstantiate(graph, 0)

# 再生: 以降は1回のcudaGraphLaunchで一連のカーネルがまとめて発行される
for _ in range(num_iters):
    cudart.cudaGraphLaunch(graph_exec, stream)
    cudart.cudaStreamSynchronize(stream)
```

> 補足: 入力形状が推論ごとに変わる（動的形状）場合や、条件分岐でカーネル起動の有無が変わる場合はキャプチャできない、または形状ごとにグラフを作り直す必要がある。

##### CVモデル後処理のCUDA実装

物体検出のNMS（Non-Maximum Suppression）やボックスデコード、セグメンテーションのargmaxなどの後処理をCPU（NumPy）で行うと、GPU→CPUへのメモリ転送とCPU側の処理時間がボトルネックになりやすい。これらをGPU上で完結させることで、転送コストと処理時間の両方を削減できる。

```python
# 例1: TensorRTのEfficientNMSプラグインをONNXグラフに組み込み、
# NMSまで含めてTensorRTエンジン内(GPU上)で完結させる(YOLO系モデルでよく使われる構成)
import onnx
import onnx_graphsurgeon as gs

graph = gs.import_onnx(onnx.load("yolo_raw.onnx"))

nms_node = gs.Node(
    op="EfficientNMS_TRT",
    name="efficient_nms",
    inputs=[boxes_tensor, scores_tensor],
    outputs=[num_dets, det_boxes, det_scores, det_classes],
    attrs={
        "score_threshold": 0.25,
        "iou_threshold": 0.45,
        "max_output_boxes": 100,
        "background_class": -1,
    },
)
graph.nodes.append(nms_node)
graph.outputs = [num_dets, det_boxes, det_scores, det_classes]
graph.cleanup().toposort()
onnx.save(gs.export_onnx(graph), "yolo_with_nms.onnx")
```

```bash
# NMSまで組み込んだONNXをそのままTensorRTエンジン化する
trtexec --onnx=yolo_with_nms.onnx --saveEngine=yolo_with_nms.engine --fp16
```

```python
# 例2: 後処理だけを別途GPU上のCUDA実装で行う場合(torchvision.opsはCUDAカーネルで実装されている)
from torchvision.ops import nms

boxes = boxes_gpu_tensor    # GPU上のTensor (N, 4)
scores = scores_gpu_tensor  # GPU上のTensor (N,)

# CPUへ転送せずGPU上でNMSを実行する
keep_idx = nms(boxes, scores, iou_threshold=0.45)
```

> 補足: 後処理まで含めてGPU上で完結させると、CPU⇔GPU間の転送は最終的な検出結果（数十〜数百件程度の小さいデータ）だけになる。生の特徴マップ・スコアマップをまるごとCPUへ転送してからNumPyで後処理する場合に比べ、転送コストを大幅に削減できる。

---

## 技術評価

Jetson上での評価は理論値（演算量削減率、圧縮率）だけでは実効果を判断できない。共有メモリアーキテクチャ（CPU/GPUでメモリ帯域を共有）、DVFS（動的電圧周波数制御）によるクロック変動、熱設計（筐体・ファンの有無）による**サーマルスロットリング**の影響を強く受けるため、**実機プロファイリングによる実測値**を評価軸に含める必要がある。

### 評価対象モデル

技術評価は、Jetson上でよく使われる2種類のCVタスクを代表するモデルで検証する。検出系とセグメンテーション系はボトルネックになる箇所が異なるため、両方で確認することで各技術の効果がタスク特性にどう依存するかを把握できる。

| モデル | タスク | 特徴 | 技術評価上の観点 |
| --- | --- | --- | --- |
| **YOLOX** | 物体検出（Object Detection） | Anchor-freeな1段階検出器。出力後にNMS（Non-Maximum Suppression）による後処理が必須 | 量子化・プルーニングによる検出ヘッド部への精度影響、および「演算命令最適化」で紹介した後処理のGPU実装（EfficientNMS_TRT等）の効果を検証しやすい |
| **UNet** | セマンティックセグメンテーション（Semantic Segmentation） | Encoder-Decoder構造で、Skip Connectionにより高解像度特徴を保持する。ピクセル単位の密な出力を行うためNMSは不要だが、特徴マップが大きくメモリ帯域への依存度が高い | 量子化・プルーニングによるメモリ帯域削減効果や、DLAオフロード時の非対応レイヤー（Skip Connectionの結合処理等）によるフォールバックの影響を検証しやすい |

### 評価軸

| 評価軸 | 内容 | 主な計測指標 |
| --- | --- | --- |
| 推論性能 | 単位時間あたりの処理能力・応答速度 | レイテンシ（ms）、スループット（FPS/inferences per sec） |
| 精度 | 高速化前後でのタスク精度の変化量 | Top-1/mAP/IoU等の劣化幅 |
| 電力・発熱 | バッテリー駆動や連続稼働時の制約 | 消費電力（W）、SoC温度、サーマルスロットリング発生有無 |
| メモリ | Jetsonは統合メモリのためCPU/GPUで競合しうる | メモリ使用量、メモリ帯域使用率 |
| エンジン使用率 | GPUとDLAの負荷配分・並列度 | GPU使用率、DLA使用率、エンジン間フォールバック回数 |
| 実装コスト | 移植・検証にかかる工数 | モデル変換の要否、再学習の要否 |

### 技術別比較（Jetson実装手段ベース）

| 技術 | Jetsonでの実装手段 | 速度向上 | 精度への影響 | 電力・発熱 | 備考 |
| --- | --- | --- | --- | --- | --- |
| 量子化（PTQ, INT8） | TensorRTのINT8キャリブレーション、または`modelopt`（`mtq.quantize`）でPTQ後にONNXエクスポート | 大 | 中（キャリブレーション精度に依存） | 低下 | 最も手軽に着手しやすい |
| 量子化（QAT） | `modelopt`（`mtq`）や`pytorch-quantization`でQAT後にTensorRT変換 | 大 | 小 | 低下 | 再学習コストあり |
| 構造化プルーニング | `modelopt`（`mtp`、FastNAS等）でチャネル・レイヤーを削減後にTensorRT再ビルド | 中 | 小〜中 | 低下 | GPU/DLA双方で効果 |
| 非構造化プルーニング／スパース性 | `modelopt`（`mts.sparsify`）またはTensorRTの2:4スパース（Ampere世代＝Orin系のみ対応、Xavier系は非対応） | 小（純粋な非構造化）〜大（2:4スパース時） | 小〜中 | 低下 | 純粋な非構造化はTensorRT上で速度向上に直結しにくい |
| DLAオフロード | `trtexec --useDLACore` 等、TensorRTのDLAターゲット指定 | 中（単体）〜大（GPUと並列時） | なし（非対応層はGPUフォールバック） | 大幅に低下 | 非対応層混在時はフォールバックのオーバーヘッドに注意 |
| 演算命令最適化 | TensorRTのレイヤー融合、cuDNN/cuBLASの活用 | 小〜中 | なし | 低下 | 既存モデルに追加コストなく適用可能 |

### プロファイリング手法

Jetsonでの評価では、**測定環境を固定する前提条件**と、**実際に性能データを観測する計測ツール**を分けて考える必要がある。前者を怠ると、後者でどれだけ精密に計測してもDVFSやサーマルスロットリングによって測定のたびに結果がばらつき、技術間の比較が成立しない。

#### 前提条件（測定環境の固定）

`nvpmodel`と`jetson_clocks`は、それ自体は性能を計測する道具ではなく、**計測結果を再現可能にするために測定前に一度だけ行う準備作業**である。DVFSが有効なままではクロックが変動するため、パワーモードを固定し、クロックを最大値に固定してから初めて他のツールでの計測が意味を持つ。

```bash
# 現在のパワーモードを確認
sudo nvpmodel -q

# 最大パフォーマンスモード(MAXN、モード番号は機種により異なる)に切り替え
sudo nvpmodel -m 0

# CPU/GPU/EMCクロックを最大値に固定(ベンチマーク前に必ず実行)
sudo jetson_clocks

# 現在のクロック固定状況を確認
sudo jetson_clocks --show
```

#### 計測・プロファイリングツール

測定環境を固定した後、以下のツールで実際に性能データを収集する。

| ツール | 用途 |
| --- | --- |
| `tegrastats` / `jtop` | GPU・CPU・DLA・メモリの使用率、消費電力（レール別）、温度をリアルタイムに監視する |
| `trtexec` | TensorRTエンジンのベンチマークCLI。レイテンシ・スループットの計測、`--useDLACore`でDLA単体性能の計測、`--dumpProfile`でレイヤー単位の所要時間を取得できる |
| Nsight Systems（`nsys`） | CPU/GPU/メモリ転送を含むタイムライン全体を可視化し、前処理・後処理・データ転送がボトルネックになっていないかを特定する |
| Nsight Compute（`ncu`） | 個別CUDAカーネルレベルのプロファイリング（occupancy、メモリスループット等）。カーネル単位で更なる最適化余地を探る場合に使用 |

##### tegrastats / jtop

```bash
# 1秒間隔でGPU/CPU/メモリ/電力/温度を表示
tegrastats

# 500ms間隔でログファイルに出力(ベンチマークと並行して実行し後で突き合わせる)
tegrastats --interval 500 --logfile bench_tegrastats.log

# jetson-statsのインストールとjtopの起動(GUI風TUIでレール別電力等も確認可能)
sudo pip3 install -U jetson-stats
jtop
```

##### trtexec

```bash
# 既存エンジンのレイテンシ・スループットを計測(100回平均、30秒間)
trtexec --loadEngine=model_int8.engine --avgRuns=100 --duration=30

# レイヤー単位の所要時間をJSONで出力(ボトルネック層の特定)
trtexec --loadEngine=model_int8.engine --dumpProfile --exportProfile=profile.json

# 構造化スパース性(2:4)を有効化してビルド(Orin/Ampere世代のGPUのみ対応)
trtexec --onnx=model.onnx --saveEngine=model_sparse.engine --int8 --sparsity=enable

# 複数バッチサイズでのスループットを比較
trtexec --loadEngine=model_int8.engine --shapes=input:8x3x224x224
```

##### Nsight Systems

アプリ全体（CPU処理・データ転送・CUDAカーネル・TensorRT実行）のタイムラインを俯瞰し、「どこがボトルネックか」の当たりをつける段階で使う。

```bash
# trtexec自体を包んでTensorRTエンジンの推論タイムラインを取得
nsys profile -o trtexec_report --trace=cuda,nvtx,osrt,cudnn,cublas \
  trtexec --loadEngine=model_int8.engine --avgRuns=100 --duration=10

# 自作の推論アプリをプロファイル(起動直後の初期化を除外するため5秒後から30秒間だけ計測)
nsys profile -o report --trace=cuda,nvtx,osrt --delay=5 --duration=30 ./my_inference_app

# サンプリング/バックトレースを止めてオーバーヘッドを抑えた軽量プロファイル(組込み環境向け)
nsys profile -o report --trace=cuda,nvtx --sample=none --backtrace=none ./my_inference_app

# 取得済みレポートのサマリをCLIで確認(GUIなしでもボトルネックの当たりをつけられる)
nsys stats report.nsys-rep

# 集計軸を絞って確認: CUDA API呼び出し / GPUカーネル / メモリ転送時間
nsys stats --report cuda_api_sum report.nsys-rep
nsys stats --report cuda_gpu_kern_sum report.nsys-rep
nsys stats --report cuda_gpu_mem_time_sum report.nsys-rep

# SQLiteへエクスポートし、独自SQLクエリで詳細分析する
nsys export --type sqlite -o report.sqlite report.nsys-rep
```

> 補足: Jetson上ではパフォーマンスカウンタへのアクセスに`sudo`が必要になる場合がある。`.nsys-rep`ファイルはホストPCへ転送し、Nsight Systems GUIで開けばタイムラインを視覚的に確認できる。

##### Nsight Compute

Nsight Systemsで重いと分かった区間・カーネルをさらに深掘りし、occupancyやメモリスループットなどカーネル内部の要因を特定する段階で使う。全メトリクス取得はオーバーヘッドが大きいため、対象を絞り込むのが基本。

```bash
# 全メトリクスセットでプロファイル(最も詳細だがオーバーヘッド大・対象は絞ること推奨)
sudo ncu -o kernel_report --set full ./my_inference_app

# 軽量なメトリクスセット(basic)で高速に取得(カーネル数が多いモデルではまずこちらで当たりをつける)
sudo ncu -o kernel_report_basic --set basic ./my_inference_app

# カーネル名を正規表現で絞り込む(例: 畳み込み関連カーネルのみ)
sudo ncu --kernel-name-base demangled --kernel-name "regex:.*conv.*" -o conv_kernels ./my_inference_app

# 呼び出し回数が多いカーネルを間引いて計測(先頭100回をスキップし、以降10回だけ計測)
sudo ncu --launch-skip 100 --launch-count 10 -o kernel_report ./my_inference_app

# コード側にNVTXで区間を仕込んでいる場合、その区間内のカーネルだけに絞って深掘り
sudo ncu --nvtx --nvtx-include "inference_loop/" -o kernel_report ./my_inference_app

# 個別メトリクスのみをCLI表示(GUIなしでSM使用率・メモリ帯域使用率を素早く確認)
sudo ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed ./my_inference_app

# 取得済みレポートをCSVでエクスポートし、スクリプトで集計・比較する
ncu --import kernel_report.ncu-rep --csv --page raw > kernel_report.csv
```

> 補足: `ncu`はパフォーマンスカウンタの読み取りに管理者権限が必要なため`sudo`を付ける。権限エラーが出る場合は`/proc/sys/kernel/perf_event_paranoid`の設定を確認する。

### 評価のポイント

1. **理論値と実測値の乖離を前提とする**: 量子化・プルーニングによる理論上の演算量削減が、メモリ帯域や非対応演算子のフォールバックにより実速度に反映されないことがある。必ず`trtexec`等での実測に基づいて評価する。
2. **測定条件を固定する**: `nvpmodel`/`jetson_clocks`でクロックを固定しないと、DVFSやサーマルスロットリングにより測定のたびに結果がばらつき、技術間の比較が成立しない。
3. **GPU/DLAの負荷分散を評価に含める**: DLAオフロードは単体の速度向上よりも「GPUを他タスクに空けられる」効果が本質的な場合が多い。複数モデル・複数ストリームを同時稼働させる構成では、GPU使用率とDLA使用率を両方計測し、全体スループットで評価する。
4. **精度検証を高速化技術の適用ごとに実施する**: 量子化・プルーニング適用後は必ずタスク精度（YOLOXならmAP、UNetならIoU等）を再評価し、許容できる劣化幅内に収まっているかを確認する。
5. **タスク特性による効果の違いを踏まえる**: YOLOX（検出・NMS後処理あり）とUNet（セグメンテーション・メモリ帯域律速）ではボトルネックの性質が異なるため、同じ技術でも効果の出方が変わりうる。両モデルでの検証結果を個別に確認する。
6. **組み合わせ効果**: 「INT8量子化＋DLAオフロード＋TensorRTレイヤー融合」のように組み合わせることで、単一技術よりも大きな高速化・省電力効果を得られるケースが多い。

### まとめ

Jetson向けの技術評価は、理論上のスペック比較で終えず、**パワーモード・クロックを固定した上での`trtexec`/`tegrastats`等による実機プロファイリング**を必須の工程として組み込むべきである。特にDLAは「速いから使う」のではなく「GPUの負荷を下げつつ全体スループットを上げる」設計要素として評価し、フォールバックによるオーバーヘッドも含めて判断する必要がある。また、検出系（YOLOX）とセグメンテーション系（UNet）のように性質の異なるモデルで検証することで、技術ごとの効果がタスク特性にどう依存するかを見誤らないようにする。
