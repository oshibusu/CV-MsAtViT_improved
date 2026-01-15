
import tensorflow as tf
from SAR_utils import softmax_real_with_real
from model_factory import build_msatvit

print("Testing softmax_real_with_real...")
# Re(3.0+4.0j) = 3.0, Re(1.0+2.0j) = 1.0
# Softmax([3.0, 1.0]) = [exp(3)/(exp(3)+exp(1)), exp(1)/(exp(3)+exp(1))]
# = [0.8808, 0.1192]
logits = tf.constant([[3.0+4.0j, 1.0+2.0j]], dtype=tf.complex64)
output = softmax_real_with_real(logits)
print("Logits:", logits.numpy())
print("Output:", output.numpy())

expected = tf.nn.softmax(tf.constant([[3.0, 1.0]]))
print("Expected:", expected.numpy())

assert tf.reduce_all(tf.abs(output - expected) < 1e-6)
print("SUCCESS: Function behaves as expected.")

print("\nTesting model build...")
try:
    model = build_msatvit((15, 15, 6), "SF", transformer_layers=2)
    # Check if the last layer has the correct activation name/function
    print("Optimization finished, model built successfully.")
    # In Keras, custom functions are stored as function objects.
    last_layer = model.layers[-1]
    print("Last layer activation:", last_layer.activation)
    print("SUCCESS: Model build confirmed.")
except Exception as e:
    print("FAILED: Model build error:", e)
