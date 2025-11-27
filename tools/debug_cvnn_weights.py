import tensorflow as tf
from cvnn.layers import ComplexConv3D
import numpy as np

def debug_weights():
    # Create a dummy layer
    layer = ComplexConv3D(
        filters=2,
        kernel_size=(3, 3, 1),
        activation="cart_relu",
        padding="same",
        input_shape=(10, 10, 10, 1)
    )
    
    # Build layer
    layer.build((None, 10, 10, 10, 1))
    
    # Get weights
    weights = layer.get_weights()
    print(f"Number of weight arrays returned: {len(weights)}")
    
    for i, w in enumerate(weights):
        print(f"Weight {i}: shape={w.shape}, dtype={w.dtype}, is_complex={np.iscomplexobj(w)}")
        # Check if it corresponds to real or imag
        # Note: We can't easily check variable names from get_weights() values alone, 
        # but we can check the layer.variables list
        
    print("\nLayer Variables:")
    for var in layer.variables:
        print(f"Name: {var.name}, Shape: {var.shape}, Dtype: {var.dtype}")

if __name__ == "__main__":
    debug_weights()
