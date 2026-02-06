# 提案手法 (Proposed Method) スライド詳細案

各スライドの構成案と、記載すべき数式・ポイントをまとめました。

---

## 1. Complex Multi-Head Attention (CMHA)

### スライドのタイトル: Complex Multi-Head Attention (CMHA)

**現状の問題点 (Baseline: Split Attention)**
*   実部と虚部を独立して計算するため、**「位相差情報」が完全に欠落**する。
*   $$ \text{Attention} \approx \text{MHA}(\Re(Q), \Re(K)) + j \cdot \text{MHA}(\Im(Q), \Im(K)) $$
*   本来、干渉波の位相差（クロス項）にこそ重要な散乱情報が含まれる。

**提案手法 (Proposed)**
*   **複素エルミート内積 ($Q K^H$)** を導入し、位相の回転量を考慮した相関計算を実現。
*   Attentionスコア計算式:
    $$ \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{\Re(Q K^H)}{\sqrt{d_k}}\right) V $$
    *   $K^H$: 共役転置 (Conjugate Transpose)。$(A+jB)(C-jD)$ の計算により、実部・虚部のクロス項が含まれる。
    *   バイアス項を無効化: 回転・スケーリング等価性を保証するため、線形射影時のバイアスを除去。

**効果**
*   PolSARデータの物理的な位相構造（散乱メカニズム）を正しく学習可能に。

---

## 2. Complex Layer Normalization

### スライドのタイトル: Complex Layer Normalization

**現状の問題点 (Baseline: Split LayerNorm)**
*   実部と虚部を別々に正規化（平均0, 分散1化）する。
*   $$ \text{SplitLN}(z) = \text{LN}(\Re(z)) + j \cdot \text{LN}(\Im(z)) $$
*   **問題**: 実部と虚部の間に相関がある場合（例: 偏波が特定の楕円率を持つ場合）、その相関関係（共分散）を破壊し、円形の分布に無理やり歪めてしまう。

**提案手法 (Proposed)**
*   **合計分散 (Total Variance)** に基づいた正規化。
*   計算プロセス:
    1.  平均 ($\mu$) と合計分散 ($\sigma^2$) の計算:
        $$ \mu = \mathbb{E}[\Re(z)] + j \mathbb{E}[\Im(z)] $$
        $$ \sigma^2 = \text{Var}(\Re(z)) + \text{Var}(\Im(z)) $$
    2.  正規化:
        $$ \hat{z} = \frac{z - \mu}{\sqrt{\sigma^2 + \epsilon}} $$
*   スケーリングパラメータ $\beta, \gamma$ も複素数として学習。

**効果**
*   実部と虚部の相関（＝位相の偏り）を保存または適切に制御しながら、学習を安定化させる。

---

## 3. 位相保存型活性化関数 (ModReLU / ModTanh)

### スライドのタイトル: Phase-Preserving Activation Functions

**現状の問題点 (Baseline: Cartesian Sigmoid)**
*   Coordinate Attentionの重み生成に採用。実部・虚部独立にSigmoidを適用。
*   $$ \sigma_{cart}(z) = \text{sigmoid}(\Re(z)) + j \cdot \text{sigmoid}(\Im(z)) $$
*   **問題**: 振幅だけでなく、Attention重みの**位相（回転）までもが入力に依存して勝手に変わってしまう**ため、地物の物理的な位相差情報を破壊する。

**提案手法 (Proposed)**
*   **振幅 (Amplitude) のみに作用**し、位相 (Phase) はそのまま保持する関数を採用。

1.  **Modified ReLU (for Feature Extraction)**:
    $$ \text{ModReLU}(z) = \text{ReLU}(|z| + b) \cdot \frac{z}{|z| + \epsilon} $$
    *   $|z| + b > 0$ の時だけ値を残し、位相 $z/|z|$ は変更しない。

2.  **Modified Tanh (for Coordinate Attention)**:
    $$ \text{ModTanh}(z) = \frac{1.0 + \text{tanh}(|z| + b)}{2} \cdot \frac{z}{|z| + \epsilon} $$
    *   Attentionの重み係数生成において、位相を回転させずに振幅のみを (0, 1) の範囲でゲーティング。

**効果**
*   「特徴の強さ」だけを非線形変換し、「物理的な種類（位相）」情報は損なわない。

---

## 4. 出力層の改良 (Output Layer)

### スライドのタイトル: Output Probabilities based on Real Part

**現状の問題点 (Baseline: Softmax with Abs)**
*   複素ロジットの**絶対値**を使って確率を計算。
*   $$ P(y=k) = \text{softmax}(|z_k|) $$
*   **問題**: コヒーレンシ行列 $T_3$ の対角成分（各偏波のパワー）は実数で定義されるべきだが、絶対値を取ると位相情報が完全に「大きさ」に丸められてしまう。

**提案手法 (Proposed)**
*   複素ロジットの**実部**を使って確率を計算。
*   $$ P(y=k) = \text{softmax}(\Re(z_k)) $$

**物理的解釈**
*   PolSARのコヒーレンシ行列 $T \approx z z^H$ を考えた時、対角成分 $T_{ii}$（観測されるパワー）は実数である。
*   ニューラルネットワークの出力を「クラスへの帰属度（パワー）」と見なすならば、実部によって評価するのが物理的に自然である。
*   これにより、過度な位相回転（虚部への逃げ）を抑制し、対角成分としての整合性を保つ。
