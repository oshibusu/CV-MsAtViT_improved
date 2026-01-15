
import tensorflow as tf
from SAR_utils import softmax_real_with_real

print("Testing softmax_real_with_real...")
# Re(3.0+4.0j) = 3.0, Re(1.0+2.0j) = 1.0
logits = tf.constant([[3.0+4.0j, 1.0+2.0j]], dtype=tf.complex64)
output = softmax_real_with_real(logits)
print("Logits:", logits.numpy())
print("Output:", output.numpy())

expected = tf.nn.softmax(tf.constant([[3.0, 1.0]]))
print("Expected:", expected.numpy())

assert tf.reduce_all(tf.abs(output - expected) < 1e-6)
print("SUCCESS: Function behaves as expected.")
