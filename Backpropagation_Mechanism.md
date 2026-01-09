# CV-MsAtViTにおける誤差逆伝播 (Backpropagation) のメカニズム

本ドキュメントでは、このリポジトリ（TensorFlowベース）における誤差逆伝播の詳細なメカニズムを解説します。
結論から言うと、**「実数値のLoss関数」を出発点とし、「実部」と「虚部」の2本のルートで勾配流（Gradient Flow）が発生し、それらが `ComplexDense` / `ComplexConv` 層でのみ交差（クロス）しながら逆流していく** というプロセスになります。

---

## 1. 出発点：Lossは「実数」
まず大前提として、**Loss（損失）は必ず実数**です。複素数でLossを定義することは（順序関係がないため）できません。

*   **Final Activation**: `softmax_real_with_abs`
    $$ P_k = \frac{e^{|z_k|}}{\sum e^{|z_j|}} $$
    ここで複素数 $z_k = x_k + j y_k$ の絶対値 $|z_k| = \sqrt{x_k^2 + y_k^2}$ を取ることで、情報は実数になります。
*   **Loss Function**: `categorical_crossentropy`
    $$ L = - \sum_{k} t_k \log(P_k) $$
    これも実数計算です。

## 2. 勾配の発生（実数界から複素界への入り口）
逆伝播が始まると、まずLoss $L$ を $|z_k|$ で微分します（ここは通常の実数微分）。
次に、その勾配 $\frac{\partial L}{\partial |z_k|}$ を、元の複素数成分 $x_k$ (実部) と $y_k$ (虚部) に分配します。これには合成関数の微分（Chain Rule）が使われます。

*   **実部への勾配**:
    $$ \frac{\partial L}{\partial x_k} = \frac{\partial L}{\partial |z_k|} \cdot \frac{\partial |z_k|}{\partial x_k} = \frac{\partial L}{\partial |z_k|} \cdot \frac{x_k}{|z_k|} $$
*   **虚部への勾配**:
    $$ \frac{\partial L}{\partial y_k} = \frac{\partial L}{\partial |z_k|} \cdot \frac{\partial |z_k|}{\partial y_k} = \frac{\partial L}{\partial |z_k|} \cdot \frac{y_k}{|z_k|} $$

ここで、**勾配が「実部用」と「虚部用」の2つに分裂**します。

## 3. ネットワーク内の逆流 (Backward Pass)

この2つの勾配は、順伝播と逆の道を辿ります。

### A: 分離処理レイヤー (Activation, Norm, Attention)
以下の層は、実部と虚部を独立に計算していました。
*   `cart_gelu`, `cart_relu`, `cart_sigmoid`
*   `LayerNormalization` (Split-process)
*   `MultiHeadAttention` (Real/Imag separate)

式で表すと $z_{out} = f(x) + j f(y)$ です。
そのため、勾配も**独立して**通過します。
*   実部の勾配 $\frac{\partial L}{\partial x}$ は、前の層の実部にのみ流れます。
*   虚部の勾配 $\frac{\partial L}{\partial y}$ は、前の層の虚部にのみ流れます。

**★ここでは情報は混ざりません。**

### B: 混合処理レイヤー (ComplexDense, ComplexConv)
ここが最も重要です。順伝播では以下の計算が行われていました。
$$
\begin{aligned}
Re(Out) &= Re(In)Re(W) - Im(In)Im(W) \\
Im(Out) &= Re(In)Im(W) + Im(In)Re(W)
\end{aligned}
$$

逆伝播では、この式の各項を通じて勾配が**クロス**します。
例えば、「実部入力 $Re(In)$」への勾配を考えてみます。
$Re(In)$ は $Re(Out)$ にも $Im(Out)$ にも寄与しているので、両方からの勾配を受け取ります。

$$
\frac{\partial L}{\partial Re(In)} = \frac{\partial L}{\partial Re(Out)} \cdot Re(W) + \frac{\partial L}{\partial Im(Out)} \cdot Im(W)
$$

このようにして、**「実部の勾配」と「虚部の勾配」が、行列の重みを介して互いに影響を与え合い、混ざり合います。**

## 4. 重みの更新 (Optimization)
最終的に、重みパラメータ $W = W_r + j W_i$ に対する勾配も計算されます。
$$
\nabla W = \frac{\partial L}{\partial W_r} + j \frac{\partial L}{\partial W_i}
$$
Optimizer (Adam) は、この複素勾配を使って複素パラメータ $W$ を更新します。

---

## まとめ：このモデル特有の挙動
このモデルでは、逆伝播の道のりの**大部分（Activation, Norm, Attention）で勾配が分離したまま**流れます。
勾配同士が会話（情報交換）できるのは、数少ない **`ComplexDense` / `ComplexConv` 層を通過する瞬間だけ** です。

そのため、「位相をどう修正すべきか」という情報の伝達効率が悪く、学習が「振幅（大きさ）」の最適化に偏りやすいという構造的な特性（欠点）があります。
