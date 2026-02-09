# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 09:21:37 2022

@author: malkhatib
"""
import scipy.io as sio
import os
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
import numpy as np
from operator import truediv
import random 
from sklearn.utils import shuffle
import matplotlib.pyplot as plt




def splitTrainTestSet(X, y, testRatio, randomState=42, coords=None):
    if X is None and coords is not None:
        # Special case: Split coords only based on y (labels)
        # Using a placeholder for X to satisfy train_test_split signature, or just split indices of y
        n_samples = len(y)
        indices = np.arange(n_samples)
        # We need to stratify by y. 
        # train_test_split can take multiple arrays. We pass indices and coords.
        # But wait, y matches the coords one-to-one if we called get_gt_coords?
        # Let's assume input y here is a 1D array of labels corresponding to coords, NOT the 2D map.
        # If y is the 2D map, this function signature is confusing. 
        # In original usage: X is (N, W, W, C), y is (N,).
        # In new usage: X is None, coords is (N, 2), y should be (N,) labels corresponding to coords.
        
        X_train, X_test, y_train, y_test, coords_train, coords_test = train_test_split(indices, y, coords, test_size=testRatio, random_state=randomState, stratify=y)
        
        # X_train/test here are just indices, which we don't really need if we have coords_train/test
        # Return None for X_train/X_test to maintain signature
        return None, None, y_train, y_test, coords_train, coords_test

    if coords is None:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testRatio, random_state=randomState,
                                                            stratify=y)
        return X_train, X_test, y_train, y_test
    else:
        X_train, X_test, y_train, y_test, coords_train, coords_test = train_test_split(X, y, coords, test_size=testRatio, random_state=randomState,
                                                            stratify=y)
        return X_train, X_test, y_train, y_test, coords_train, coords_test


def padWithZeros(X, margin=2):
    newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2* margin, X.shape[2]),dtype=('complex64'))
    x_offset = margin
    y_offset = margin
    newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X
    return newX

def createImageCubes(X, y, windowSize=5, removeZeroLabels = True, max_samples=200000, random_state=42):
    # Use float32 to save memory if original is not critical, but complex64 is requested.
    # Count valid pixels first to avoid massive pre-allocation
    margin = int((windowSize - 1) / 2)
    zeroPaddedX = padWithZeros(X, margin=margin)
    
    if removeZeroLabels:
        # Get coordinates of valid labels
        r_idx, c_idx = np.nonzero(y > 0)
    else:
        # All pixels
        r_idx, c_idx = np.indices((X.shape[0], X.shape[1]))
        r_idx = r_idx.flatten()
        c_idx = c_idx.flatten()
        
    num_samples = len(r_idx)
    
    # Subsampling to avoid OOM
    if max_samples is not None and max_samples > 0 and num_samples > max_samples:
        print(f"Subsampling data: reducing {num_samples} samples to {max_samples}...")
        if random_state is not None:
             np.random.seed(random_state)
        indices = np.random.choice(num_samples, max_samples, replace=False)
        r_idx = r_idx[indices]
        c_idx = c_idx[indices]
        num_samples = max_samples

    print(f"Generating patches for {num_samples} samples...")
    
    patchesData = np.zeros((num_samples, windowSize, windowSize, X.shape[2]), dtype=('complex64'))
    patchesLabels = np.zeros((num_samples))
    
    for i in range(num_samples):
        # r, c are original coordinates. In padded image, we shift by margin.
        r = r_idx[i]
        c = c_idx[i]
        
        patch = zeroPaddedX[r : r + windowSize, c : c + windowSize]
        patchesData[i, :, :, :] = patch
        patchesLabels[i] = y[r, c]
        
    if removeZeroLabels:
        patchesLabels -= 1
        
    # Stack coordinates (r, c)
    coords = np.column_stack((r_idx, c_idx))
        
    return patchesData, patchesLabels, coords

def get_gt_coords(y, removeZeroLabels=True):
    """
    Get coordinates (r, c) of valid pixels without generating patches.
    """
    if removeZeroLabels:
        r_idx, c_idx = np.nonzero(y > 0)
    else:
        r_idx, c_idx = np.indices((y.shape[0], y.shape[1]))
        r_idx = r_idx.flatten()
        c_idx = c_idx.flatten()
        
    coords = np.column_stack((r_idx, c_idx))
    return coords

def extract_patches_from_coords(data, coords, windowSize=5):
    """
    Generate patches for specific coordinates.
    """
    num_samples = len(coords)
    margin = int((windowSize - 1) / 2)
    zeroPaddedX = padWithZeros(data, margin=margin)
    
    patchesData = np.zeros((num_samples, windowSize, windowSize, data.shape[2]), dtype='complex64')
    
    for i in range(num_samples):
        r, c = coords[i]
        patch = zeroPaddedX[r : r + windowSize, c : c + windowSize]
        patchesData[i, :, :, :] = patch
        
    return patchesData

def AA_andEachClassAccuracy(confusion_matrix):
    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc


def target(name):
    if name == 'FL_T' or name == 'FL_C':
        target_names = ['Unassigned', 'Water', 'Forest', 'Lucerne', 'Grass', 'Rapeseed',
                        'Beet', 'Potatoes', 'Peas', 'Stem Beans', 'Bare Soil', 'Wheat', 'Wheat 2', 
                        'Wheat 3', 'Barley', 'Buildings']
    elif name == 'SF':
        target_names = ['Unassigned', 'Bare Soil', 'Mountain', 'Water', 'Urban', 'Vegetation']
    elif 'Baltrum' in name:
        target_names = ['Unassigned', 'Tidal flat', 'Water', 'Coastal shrub', 'Dense, high vegetation', 'White dune', 
                        'Peat bog', 'Grey dune', 'Couch grass', 'Upper saltmarsh', 'Lower saltmarsh', 'Sand', 'Settlement']
        
    return target_names 
    
def num_classes(dataset):
    if dataset == 'FL_T' or dataset == 'FL_T_real':
        output_units = 15
    elif dataset == 'SF' or dataset == 'SF_real':
        output_units = 5
    elif dataset == 'ober' or dataset == 'ober_real':
        output_units = 3
    elif 'Baltrum' in dataset:
        output_units = 12
    
    return output_units




def Patch(data,height_index,width_index, PATCH_SIZE):
    height_slice = slice(height_index, height_index+PATCH_SIZE)
    width_slice = slice(width_index, width_index+PATCH_SIZE)
    patch = data[height_slice, width_slice, :]
    
    return patch

def getTrainTestSplit(X_cmplx, X_rgb, y, pxls_num):
    if type(pxls_num) != list:
        pxls_num = [pxls_num]*len(np.unique(y))
        
    if len(np.unique(y)) != len(pxls_num):
        print("length of pixels list doen't match the number of classes in the dataset")
        return
    else:
        xTrain_cmplx = []
        xTrain_rgb = []
        yTrain = []
        
        xTest_cmplx  = []
        xTest_rgb  = []
        yTest  = []
        for i in range(len(np.unique(y))):
            if pxls_num[i] > len(y[y==i]):
                print("Number of training pixles is larger than total class pixels")
                return
            else:
                random.seed(42) #optional to reproduce the data
                samples = random.sample(range(len(y[y==i])), pxls_num[i])
                xTrain_cmplx.extend(X_cmplx[y==i][samples])
                xTrain_rgb.extend(X_rgb[y==i][samples])
                yTrain.extend(y[y==i][samples])
                
                tmp1 = list(X_cmplx[y==i])
                tmp2 = list(X_rgb[y==i])
                tmp3 = list(y[y==i])
                for ele in sorted(samples, reverse = True):
                    del tmp1[ele]
                    del tmp2[ele]
                    del tmp3[ele]

                xTest_cmplx.extend(tmp1)
                xTest_rgb.extend(tmp2)
                yTest.extend(tmp3)
     
  
    xTrain_cmplx, xTrain_rgb, yTrain = shuffle(xTrain_cmplx, xTrain_rgb, yTrain, random_state=42)  
    xTest_cmplx, xTest_rgb, yTest = shuffle(xTest_cmplx, xTest_rgb, yTest, random_state=42)
    
    #xTrain_rgb, yTrain = shuffle(xTrain_rgb, yTrain, random_state=321)  
    #xTest_rgb, yTest = shuffle(xTest_rgb, yTest, random_state=345)
    
    
    
    xTrain_cmplx = np.array(xTrain_cmplx)
    xTrain_rgb = np.array(xTrain_rgb)
    yTrain = np.array(yTrain)
    
    xTest_cmplx = np.array(xTest_cmplx)
    xTest_rgb = np.array(xTest_rgb)
    yTest = np.array(yTest)
    
      
    return xTrain_cmplx, xTrain_rgb, yTrain, xTest_cmplx, xTest_rgb, yTest
        
        
    
import cvnn.layers as complex_layers
def cmplx_SE_Block(xin, se_ratio = 8):
    # Squeeze Path
    xin_gap =  GlobalCmplxAveragePooling2D(xin)
    sqz = complex_layers.ComplexDense(xin.shape[-1]//se_ratio, activation='cart_relu')(xin_gap)
    
    # Excitation Path
    excite1 = complex_layers.ComplexDense(xin.shape[-1], activation='cart_sigmoid')(sqz)
    
    out = tf.keras.layers.multiply([xin, excite1])
    
    return out
    
   

import tensorflow as tf
def GlobalCmplxAveragePooling2D(inputs):
    inputs_r = tf.math.real(inputs)
    inputs_i = tf.math.imag(inputs)
    
    output_r = tf.keras.layers.GlobalAveragePooling2D()(inputs_r)
    output_i = tf.keras.layers.GlobalAveragePooling2D()(inputs_i)
    
    if inputs.dtype == 'complex' or inputs.dtype == 'complex64':
           output = tf.complex(output_r, output_i)
    else:
           output = output_r
    
    return output




def Standardize_data(X):
    new_X = np.zeros(X.shape, dtype=(X.dtype))
    _,_,c = X.shape
    for i in range(c):
        new_X[:,:,i] = (X[:,:,i] - np.mean(X[:,:,i])) / np.std(X[:,:,i])
        
    return new_X
        
        



from numpy.fft import fft2, fftshift
def getFFT(X):
    X_fft = np.zeros(X.shape, dtype='complex64')
    for ii in range(len(X)):
        for jj in range(X.shape[3]):
            X_fft[ii,:,:,jj] = fftshift(fft2(X[ii,:,:,jj])) 
            #X_fft[ii,:,:,jj] = fftshift(fft2(X[ii,:,:,jj])) 
            
            
    return X_fft


import keras
def cart_gelu(x):
    x_r = tf.math.real(x)
    x_i = tf.math.imag(x)
    
    gelu_r = keras.activations.gelu(x_r, approximate=False)
    gelu_i = keras.activations.gelu(x_i, approximate=False)
    
    if x.dtype == 'complex' or x.dtype == 'complex64':
           output = tf.complex(gelu_r, gelu_i)
    else:
           output = gelu_r
    
    return output


def cart_sigmoid(x):
    """
    Applies Sigmoid separately to real and imaginary parts.
    """
    x_r = tf.math.real(x)
    x_i = tf.math.imag(x)

    sig_r = tf.math.sigmoid(x_r)
    sig_i = tf.math.sigmoid(x_i)

    if x.dtype == 'complex' or x.dtype == 'complex64':
           output = tf.complex(sig_r, sig_i)
    else:
           output = sig_r

    return output



def softmax_real_with_real(x):
    return tf.nn.softmax(tf.math.real(x))


def save_classification_map(prediction_map, gt_map, dataset_name, save_path):
    """
    Save the classification map as a PNG image.
    prediction_map: 2D numpy array of predicted labels
    gt_map: 2D numpy array of ground truth (used for masking background if needed, but here we plot prediction)
    dataset_name: Name of the dataset for the title
    save_path: Path to save the image (e.g., results/plots/map.png)
    """
    plt.figure(figsize=(10, 8))
    # Plot the prediction map. We presume prediction_map generally matches GT spatial dims.
    # Use 'jet' or 'nipy_spectral' which are common for land cover
    plt.imshow(prediction_map, cmap='jet')
    plt.colorbar(label='Class ID')
    plt.title(f'Classification Map: {dataset_name}')
    plt.axis('off')
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()


class ComplexLayerNormalization(tf.keras.layers.Layer):
    """
    Complex Layer Normalization.
    Normalizes by centering the complex data (subtracting complex mean)
    and scaling by total variance (sum of real and imaginary variances).
    """
    def __init__(self, epsilon=1e-6, **kwargs):
        super(ComplexLayerNormalization, self).__init__(**kwargs)
        self.epsilon = epsilon
        self.gamma = None
        self.beta_real = None
        self.beta_imag = None

    def build(self, input_shape):
        # input_shape is usually (Batch, ..., Channels)
        params_shape = input_shape[-1:]
        
        self.gamma = self.add_weight(
            name="gamma",
            shape=params_shape,
            initializer="ones",
            trainable=True
        )
        # Shift (beta) is complex, so we learn real and imaginary parts separately
        self.beta_real = self.add_weight(
            name="beta_real",
            shape=params_shape,
            initializer="zeros",
            trainable=True
        )
        self.beta_imag = self.add_weight(
            name="beta_imag",
            shape=params_shape,
            initializer="zeros",
            trainable=True
        )
        super(ComplexLayerNormalization, self).build(input_shape)

    def call(self, inputs):
        # inputs: Complex tensor
        
        # 1. Compute Complex Mean (along feature axis)
        # mu = E[x] + j*E[y]
        real_mean, real_var = tf.nn.moments(tf.math.real(inputs), axes=[-1], keepdims=True)
        imag_mean, imag_var = tf.nn.moments(tf.math.imag(inputs), axes=[-1], keepdims=True)
        
        # 2. Compute Total Variance
        # sigma^2 = Var(x) + Var(y)
        total_variance = real_var + imag_var
        
        # 3. Normalize
        # z_hat = (z - mu) / sqrt(sigma^2 + epsilon)
        denom = tf.complex(tf.sqrt(total_variance + self.epsilon), 0.0)
        
        # Center the data
        centered_inputs = inputs - tf.complex(real_mean, imag_mean)
        normalized_inputs = centered_inputs / denom
        
        # 4. Affine Transformation
        # z_out = gamma * z_hat + beta
        # gamma is real scalar, beta is complex (beta_real + j*beta_imag)
        
        scaled_inputs = normalized_inputs * tf.complex(self.gamma, 0.0)
        output = scaled_inputs + tf.complex(self.beta_real, self.beta_imag)
        
        return output

    def get_config(self):
        config = super(ComplexLayerNormalization, self).get_config()
        config.update({'epsilon': self.epsilon})
        return config


class ModReLU(tf.keras.layers.Layer):
    """
    Modified ReLU (ModReLU) activation for complex-valued signals.
    Applies ReLU to the amplitude with a learnable threshold b.
    Formula: ModReLU(z) = ReLU(|z| + b) * (z / |z|)
    """
    def __init__(self, epsilon=1e-6, b_init=-0.1, **kwargs):
        super(ModReLU, self).__init__(**kwargs)
        self.epsilon = epsilon
        self.b_init = b_init

    def build(self, input_shape):
        # Create a learnable bias param 'b'
        # One b for each channel (feature map)
        channel_axis = -1
        self.b = self.add_weight(
            shape=(input_shape[channel_axis],),
            initializer=tf.constant_initializer(self.b_init), 
            trainable=True,
            name="b"
        )
        super(ModReLU, self).build(input_shape)

    def call(self, inputs):
        # inputs: Complex tensor z
        
        # 1. Compute amplitude |z|
        amplitude = tf.abs(inputs)
        
        # 2. Compute adjusted amplitude: |z| + b
        # Broadcast b across spatial dims
        adjusted_amplitude = amplitude + self.b
        
        # 3. Apply ReLU: max(0, |z| + b)
        activated_amplitude = tf.nn.relu(adjusted_amplitude)
        
        # 4. Rescale original input: z * (relu(|z|+b) / |z|) = activated_amplitude * (z / |z|)
        # Add epsilon to denominator to avoid div by zero
        scale = activated_amplitude / (amplitude + self.epsilon)
        
        # Use simple complex multiplication (scale is real)
        output = tf.complex(
            tf.math.real(inputs) * scale,
            tf.math.imag(inputs) * scale
        )
        return output

    def get_config(self):
        config = super(ModReLU, self).get_config()
        config.update({'epsilon': self.epsilon, 'b_init': self.b_init})
        return config























class ModSigmoid(tf.keras.layers.Layer):
    """
    Modified Sigmoid activation for complex-valued signals.
    Applies Sigmoid to the amplitude with a learnable threshold b.
    Formula: ModSigmoid(z) = Sigmoid(|z| + b) * (z / |z|)
    This preserves phase and scales amplitude to (0, 1).
    """
    def __init__(self, epsilon=1e-6, b_init=0.0, **kwargs):
        super(ModSigmoid, self).__init__(**kwargs)
        self.epsilon = epsilon
        self.b_init = b_init
        self.b = None

    def build(self, input_shape):
        # Create a learnable bias param 'b'
        # One b for each channel (feature map)
        channel_axis = -1
        self.b = self.add_weight(
            shape=(input_shape[channel_axis],),
            initializer=tf.constant_initializer(self.b_init), 
            trainable=True,
            name="b"
        )
        super(ModSigmoid, self).build(input_shape)

    def call(self, inputs):
        # inputs: Complex tensor z
        
        # 1. Compute amplitude |z|
        amplitude = tf.abs(inputs)
        
        # 2. Compute adjusted amplitude: |z| + b
        adjusted_amplitude = amplitude + self.b
        
        # 3. Apply Sigmoid: 1 / (1 + exp(-(|z| + b)))
        activated_amplitude = tf.math.sigmoid(adjusted_amplitude)
        
        # 4. Rescale original input: z * (sigmoid(|z|+b) / |z|)
        scale = activated_amplitude / (amplitude + self.epsilon)
        
        # Use simple complex multiplication (scale is real)
        output = tf.complex(
            tf.math.real(inputs) * scale,
            tf.math.imag(inputs) * scale
        )
        return output

    def get_config(self):
        config = super(ModSigmoid, self).get_config()
        config.update({'epsilon': self.epsilon, 'b_init': self.b_init})
        return config
class ModTanhScaled(tf.keras.layers.Layer):
    """
    Modified Tanh activation for complex-valued signals, scaled to (0, 1).
    Applies Tanh to the amplitude.
    Formula: Gate = (1 + tanh(|z|)) / 2
    Output = Gate * (z / |z|)
    This preserves phase and scales amplitude using a 0-1 normalized tanh.
    """
    def __init__(self, epsilon=1e-6, **kwargs):
        super(ModTanhScaled, self).__init__(**kwargs)
        self.epsilon = epsilon

    def build(self, input_shape):
        # No trainable weights
        super(ModTanhScaled, self).build(input_shape)

    def call(self, inputs):
        # inputs: Complex tensor z
        
        # 1. Compute amplitude |z|
        amplitude = tf.abs(inputs)
        
        # 2. Apply Scaled Tanh: (1 + tanh(|z|)) / 2
        # No bias b added
        gate = (1.0 + tf.math.tanh(amplitude)) * 0.5
        
        # 3. Rescale original input: z * (gate / |z|)
        # Add epsilon to denominator to avoid div by zero
        scale = gate / (amplitude + self.epsilon)
        
        # Use simple complex multiplication (scale is real)
        output = tf.complex(
            tf.math.real(inputs) * scale,
            tf.math.imag(inputs) * scale
        )
        return output

    def get_config(self):
        config = super(ModTanhScaled, self).get_config()
        config.update({'epsilon': self.epsilon})
        return config


class ModGated(tf.keras.layers.Layer):
    """
    Gated Identity activation for complex-valued signals.
    Applies a hard threshold (step function) to the amplitude with a learnable bias b.
    Formula: Output = z if (|z| + b >= 0) else 0
    Note: Gradients for 'b' may be zero due to the step function nature.
    """
    def __init__(self, b_init=-0.1, **kwargs):
        super(ModGated, self).__init__(**kwargs)
        self.b_init = b_init

    def build(self, input_shape):
        # Create a learnable bias param 'b'
        # One b for each channel (feature map)
        channel_axis = -1
        self.b = self.add_weight(
            shape=(input_shape[channel_axis],),
            initializer=tf.constant_initializer(self.b_init), 
            trainable=True,
            name="b"
        )
        super(ModGated, self).build(input_shape)

    def call(self, inputs):
        # inputs: Complex tensor z
        
        # 1. Compute amplitude |z|
        amplitude = tf.abs(inputs)
        
        # 2. Check condition: |z| + b >= 0
        condition = tf.math.greater_equal(amplitude + self.b, 0)
        
        # 3. Create Gate (1 if True, 0 if False)
        # Cast to same dtype as inputs (complex)
        gate = tf.cast(condition, inputs.dtype)
        
        # 4. Apply Gate
        return inputs * gate

    def get_config(self):
        config = super(ModGated, self).get_config()
        config.update({'b_init': self.b_init})
        return config


class ModSigmoidGated(tf.keras.layers.Layer):
    """
    Soft Gated Identity activation for complex-valued signals.
    Applies a soft threshold (sigmoid) to the amplitude with a learnable bias b.
    Formula: Output = z * sigmoid(|z| + b)
    Differentiable approximation of ModGated.
    """
    def __init__(self, b_init=-0.1, **kwargs):
        super(ModSigmoidGated, self).__init__(**kwargs)
        self.b_init = b_init

    def build(self, input_shape):
        # Create a learnable bias param 'b'
        # One b for each channel (feature map)
        channel_axis = -1
        self.b = self.add_weight(
            shape=(input_shape[channel_axis],),
            initializer=tf.constant_initializer(self.b_init),
            trainable=True,
            name="b"
        )
        super(ModSigmoidGated, self).build(input_shape)

    def call(self, inputs):
        # inputs: Complex tensor z
        
        # 1. Compute amplitude |z|
        amplitude = tf.abs(inputs)
        
        # 2. Compute Soft Gate: sigmoid(|z| + b)
        gate_val = tf.math.sigmoid(amplitude + self.b)
        
        # 3. Apply Gate
        # Cast to same dtype as inputs (complex)
        gate = tf.cast(gate_val, inputs.dtype)
        
        # 4. Output: z * Gate
        return inputs * gate

    def get_config(self):
        config = super(ModSigmoidGated, self).get_config()
        config.update({'b_init': self.b_init})
        return config
