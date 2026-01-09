import tensorflow as tf
from tensorflow.keras import layers
from cvnn.layers import ComplexDense

class ComplexMultiHeadAttention(layers.Layer):
    def __init__(self, num_heads, key_dim, dropout=0.0, **kwargs):
        super(ComplexMultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.dropout = dropout

    def build(self, input_shape):
        # input_shape is assumed to be (batch, seq_len, embed_dim)
        embed_dim = input_shape[-1]
        
        # Q, K, V projections
        # Output dim = num_heads * key_dim
        self.query_dense = ComplexDense(units=self.num_heads * self.key_dim, use_bias=False)
        self.key_dense = ComplexDense(units=self.num_heads * self.key_dim, use_bias=False)
        self.value_dense = ComplexDense(units=self.num_heads * self.key_dim, use_bias=False)
        
        # Output projection
        self.output_dense = ComplexDense(units=embed_dim)
        
        self.dropout_layer = layers.Dropout(self.dropout)
        super(ComplexMultiHeadAttention, self).build(input_shape)

    def split_heads(self, x, batch_size):
        # x shape: (batch_size, seq_len, num_heads * key_dim)
        # Reshape to (batch_size, seq_len, num_heads, key_dim)
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.key_dim))
        # Transpose to (batch_size, num_heads, seq_len, key_dim) for matmul
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, query, value, key=None, attention_mask=None, return_attention_scores=False):
        if key is None:
            key = value

        batch_size = tf.shape(query)[0]

        # 1. Complex Projections
        # Q, K, V are complex64/128
        query = self.query_dense(query)
        key = self.key_dense(key)
        value = self.value_dense(value)

        # 2. Split Heads
        query = self.split_heads(query, batch_size) # (B, H, T, D)
        key = self.split_heads(key, batch_size)     # (B, H, T, D)
        value = self.split_heads(value, batch_size) # (B, H, T, D)

        # 3. Attention Score Calculation (Scaled Dot-Product)
        # scores = Q @ K^H (Conjugate Transpose)
        # adjoint_b=True performs conjugate transpose on the second matrix
        scores_c = tf.matmul(query, key, adjoint_b=True) # (B, H, T, T)

        # 4. Real-valued Logits
        # logits = Re(scores_c) / sqrt(d_head)
        # d_head is key_dim
        scale = tf.cast(tf.sqrt(tf.cast(self.key_dim, tf.float32)), scores_c.dtype.real_dtype)
        logits = tf.math.real(scores_c) / scale

        # Apply mask if exists
        if attention_mask is not None:
            # Assumes mask is compatible with (B, 1, 1, T) or matching shape
            # Add -1e9 to masked positions
            logits += (attention_mask * -1e9)

        # 5. Softmax (Real-valued)
        attn_weights = tf.nn.softmax(logits, axis=-1)
        
        # Apply dropout to weights
        attn_weights = self.dropout_layer(attn_weights)

        # 6. Weighted Sum
        # Cast weights back to complex to multiply with complex V
        # V is complex, weights is real. 
        # complex * real = complex (amplitude scaling, phase preserved)
        complex_attn_weights = tf.cast(attn_weights, value.dtype)
        
        # context = weights @ V
        context = tf.matmul(complex_attn_weights, value) # (B, H, T, D)

        # 7. Merge Heads
        # Transpose back: (B, T, H, D)
        context = tf.transpose(context, perm=[0, 2, 1, 3])
        # Reshape to (B, T, H*D)
        # Note: We must handle dynamic sequence length
        seq_len = tf.shape(context)[1]
        context = tf.reshape(context, (batch_size, seq_len, self.num_heads * self.key_dim))

        # 8. Output Projection
        output = self.output_dense(context)

        if return_attention_scores:
            return output, attn_weights
        return output
