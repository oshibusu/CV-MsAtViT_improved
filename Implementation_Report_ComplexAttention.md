# 実装レポート: 真の複素数Attention層 (ComplexMultiHeadAttention)

## 1. 背景と目的
従来の `CV-MsAtViT` 実装では、Attention層において複素数データを実部と虚部に分離し、それぞれ独立した実数 `MultiHeadAttention` に入力していました。
これは「位相情報」と「振幅情報」の相関を無視する（クロス項を計算しない）ことになり、物理的な位相情報を正しく扱えていないという問題がありました。

本実装の目的は、この分離処理を廃止し、**入出力および内部計算をすべて複素数のまま行うAttentionメカニズム**を導入することです。

## 2. 実装計画 (Algorithm Design)

### 要件
1.  入力 $X$ (complex64) に対し、Q, K, V, Output すべてを複素数として計算する。
2.  Attention Score は複素共役転置を用いた内積 $Q K^\dagger$ で計算する。
3.  Softmax は実数空間で行う（振幅ベースの確率分布）。

### 数学的設計
$$
\begin{aligned}
Q &= X W_Q, \quad K = X W_K, \quad V = X W_V \quad (W \in \mathbb{C}) \\
\text{Score} &= Q K^\dagger \quad (\dagger: \text{Conjugate Transpose}) \\
\text{Logits} &= \frac{Re(\text{Score})}{\sqrt{d_k}} \quad (\in \mathbb{R}) \\
\text{Weights} &= \text{Softmax}(\text{Logits}) \quad (\in \mathbb{R}, \sum=1) \\
\text{Context} &= \text{Weights} \cdot V \quad (\in \mathbb{C}) \\
\text{Output} &= \text{Context} W_O \quad (\in \mathbb{C})
\end{aligned}
$$

この設計により、**「どのValueに注目するか（重み）」は実数値で決まりますが、取り出される「Value（値）」は複素数そのものであるため、位相情報は保持されます。**

## 3. 実装詳細

### ファイル構成
*   **[NEW] `ComplexAttention.py`**: 新しいレイヤークラス `ComplexMultiHeadAttention` を定義。
*   **[MOD] `model_factory.py`**: 既存の `MultiHeadAttention` を置き換え。

### `ComplexMultiHeadAttention` クラスの実装ポイント
1.  **プロジェクション**: `cvnn.layers.ComplexDense` を使用。
    ```python
    self.query_dense = ComplexDense(units=self.num_heads * self.key_dim, ...)
    ```
2.  **スコア計算**: `adjoint_b=True` で共役転置積を実行。
    ```python
    scores_c = tf.matmul(query, key, adjoint_b=True)
    ```
3.  **実数Softmax**: 実部を取り出してスケーリング。
    ```python
    logits = tf.math.real(scores_c) / scale
    attn_weights = tf.nn.softmax(logits, axis=-1)
    ```
4.  **位相保存**: 実数の重みを複素数型にキャストして掛け算。
    ```python
    complex_attn_weights = tf.cast(attn_weights, value.dtype)
    context = tf.matmul(complex_attn_weights, value)
    ```

### 統合 (`model_factory.py`)
従来の分離ロジックを削除し、正規化後の実部・虚部を直ちに結合して入力するように変更しました。

```python
# Before (Separated)
# attention_r = MultiHeadAttention(x_r)
# attention_i = MultiHeadAttention(x_i)

# After (Unified)
x1 = tf.complex(x1_r, x1_i) # Normalizeは独立のままだが、即結合
attention_output = ComplexMultiHeadAttention(...)(x1, x1)
```

## 4. 結論
これにより、モデルの核となるAttention機構において、偏波データの位相情報（複素相関）が正しく考慮されるようになりました。
学習時の逆伝播においても、`tf.matmul` (complex) や `ComplexDense` を通じて、実部・虚部間の勾配が適切にクロス・混合して伝わるようになります。
