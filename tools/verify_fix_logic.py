import tensorflow as tf
from cvnn.layers import ComplexConv3D
import numpy as np
import os

def verify_logic():
    print("Verifying Complex Weight Save/Load Logic...")
    
    # 1. Create a dummy ComplexConv3D layer
    # Shape: (3, 3, 1, 1, 2) -> 3x3 kernel, 1 depth, 1 input channel, 2 filters
    layer = ComplexConv3D(
        filters=2,
        kernel_size=(3, 3, 1),
        input_shape=(10, 10, 10, 1),
        dtype=np.complex64
    )
    layer.build((None, 10, 10, 10, 1))
    
    # 2. Set known complex weights
    # Create random complex weights
    r = np.random.rand(3, 3, 1, 1, 2).astype(np.float32)
    i = np.random.rand(3, 3, 1, 1, 2).astype(np.float32)
    complex_kernel = r + 1j * i
    
    # Set weights (cvnn expects [kernel_r, kernel_i, bias_r, bias_i] usually, or similar)
    # We set them using the layer's expected format.
    # Let's see what set_weights expects by checking get_weights first
    initial_weights = layer.get_weights()
    print(f"Initial weights count: {len(initial_weights)}")
    
    # We expect 4 weights: kernel_r, kernel_i, bias_r, bias_i
    # Or maybe 2 if it handles complex automatically? 
    # The user's issue implies it splits them.
    
    # Let's try to set the weights explicitly assuming the split format
    # We need to match the shapes of initial_weights
    new_weights = []
    if len(initial_weights) >= 2:
        # Assuming 0 is kernel_r, 1 is kernel_i
        new_weights.append(r)
        new_weights.append(i)
        # Add biases (zeros)
        for w in initial_weights[2:]:
            new_weights.append(np.zeros_like(w))
            
        layer.set_weights(new_weights)
        print("Set weights manually using split real/imag parts.")
    else:
        print("Unexpected: Layer does not have split weights?")
        return

    # Wrap in a model to enable save_weights
    model = tf.keras.Sequential([layer])
    model.build((None, 10, 10, 10, 1))

    # 3. Save weights to H5
    h5_path = "temp_test_weights.h5"
    model.save_weights(h5_path)
    print(f"Saved weights to {h5_path}")
    
    # 4. Load weights into a NEW layer/model
    layer2 = ComplexConv3D(
        filters=2,
        kernel_size=(3, 3, 1),
        input_shape=(10, 10, 10, 1),
        dtype=np.complex64
    )
    model2 = tf.keras.Sequential([layer2])
    model2.build((None, 10, 10, 10, 1))
    model2.load_weights(h5_path)
    print("Loaded weights into new layer.")
    
    # 5. Get weights from new layer
    loaded_weights = layer2.get_weights()
    
    # 6. Verify reconstruction logic
    # Logic used in visualize_branch_heatmap.py:
    if len(loaded_weights) >= 2 and loaded_weights[0].shape == loaded_weights[1].shape:
        reconstructed_kernel = loaded_weights[0] + 1j * loaded_weights[1]
        print("Reconstructed complex kernel from loaded weights.")
    else:
        reconstructed_kernel = loaded_weights[0]
        print("Could not reconstruct (structure mismatch).")

    # 7. Compare
    diff = np.abs(reconstructed_kernel - complex_kernel)
    max_diff = np.max(diff)
    print(f"Max difference between original and reconstructed: {max_diff}")
    
    if max_diff < 1e-6:
        print("SUCCESS: Logic is correct. The reconstructed weights match the original.")
    else:
        print("FAILURE: Weights do not match.")

    # Clean up
    if os.path.exists(h5_path):
        os.remove(h5_path)

if __name__ == "__main__":
    verify_logic()
