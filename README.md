# CV-MsAtViT
Source code for "PolSAR Image Classification Using Complex-Valued Multiscale Attention Vision Transformer (CV-MsAtViT)" Accepted for publication in **International Journal of Applied Earth Observation and Geoinformation**

The paper can be accessed through:
https://www.sciencedirect.com/science/article/pii/S1569843225000597

![image](https://github.com/user-attachments/assets/4566cc35-294c-4e9f-a0b3-8bc645a538a0)

![image](https://github.com/user-attachments/assets/44c30c1c-62b7-4cd4-ad39-b2eb1bc096c0)

# Datasets:
Three PolSAR datasets were utilized to assess the performance of the CV-MsAtViT method in this study. Flevoland, San Francisco, and Oberpfaffenhofen.
link to the datasets along their class maps is available at:
https://mega.nz/folder/WhgT1L4S#PnMttCUpjtwkD8qTEdwZsw

# Requirement
Python 3.9.18, Tensorflow (and Keras) 2.10.0, cvnn 2.0, Tensorflow Probability 0.18.0

# Results
To quantitatively measure the proposed CV-MsAtViT model, three evaluation metrics are employed to verify the effectiveness of the algorithm, Overall Accuracy (OA), Average Accuracy (AA) and Cohen's Kappa (k). Also, Each class accuracy has been reported.
![image](https://github.com/user-attachments/assets/d6a66081-9277-4d52-b1d9-edef812f9b59)
![image](https://github.com/user-attachments/assets/831df4be-b532-4fd7-9a26-0c3edc6e963a)
![image](https://github.com/user-attachments/assets/78321e6e-4f29-4121-914a-c0503bdf281f)

# Citation
@article{alkhatib2025polsar,
  title={PolSAR image classification using complex-valued multiscale attention vision transformer (CV-MsAtViT)},
  author={Alkhatib, Mohammed Q},
  journal={International Journal of Applied Earth Observation and Geoinformation},
  volume={137},
  pages={104412},
  year={2025},
  publisher={Elsevier}
}

Feel free to contact me on: mqalkhatib@ieee.org

# Troubleshooting & Fixes

During the deployment and testing of this model, several issues were encountered and resolved. Below is a summary of the errors and their fixes.

## 1. `TypeError` in `ComplexBatchNormalization`
**Error:**
When loading the model, `cvnn.layers.ComplexBatchNormalization` threw a `TypeError: The real and imag components have incorrect types: complex64 complex64. They must be consistent, and one of [tf.float32, tf.float64]`.
This was caused by `tf.complex` expecting float arguments, but the initializers returning complex values.

**Fix:**
- Created `cvnn_fix.py` containing `ForceFloatInitializer` and `FixedComplexBatchNormalization`.
- `ForceFloatInitializer` wraps Keras initializers to ensure they return `float32` (or `float64`) even if a complex dtype is requested.
- `FixedComplexBatchNormalization` applies this wrapper to its internal initializers.
- **Important:** This fix is applied **ONLY** during `load_saved_msatvit` (in `model_factory.py`) to bypass the loading error. The model is still built and trained using the standard `ComplexBatchNormalization` to ensure correct complex weight handling.

## 2. `TypeError: 'str' object is not callable` (Lambda Layer Serialization)
**Error:**
The `CoordAtt_cmplx` function used a `Lambda` layer with `tf.split`. `Lambda` layers often fail to serialize/deserialize correctly in `SavedModel` format, leading to errors when loading the model.

**Fix:**
- Replaced the `Lambda` layer with a custom Keras layer `ComplexSplit` defined in `CoordAttention.py`.
- This custom layer implements `get_config` and `from_config`, allowing it to be correctly saved and loaded.

## 3. Zero Imaginary Parts in Visualization
**Error:**
When running `tools/visualize_branch_heatmap.py`, the logs showed `max imaginary: 0.0`, suggesting that the loaded weights had lost their complex nature.

**Fix:**
- Investigation revealed that `cvnn` layers store real and imaginary parts as separate variables (e.g., `kernel_r`, `kernel_i`).
- `layer.get_weights()` returns these as separate arrays in the list (e.g., `[kernel_r, kernel_i, bias_r, bias_i]`).
- The visualization script was only reading `weights[0]` (real part).
- Updated `tools/visualize_branch_heatmap.py` to detect this split format and reconstruct the complex kernel: `kernel = weights[0] + 1j * weights[1]`.
- Verified the correctness of this logic using `tools/verify_fix_logic.py`.
