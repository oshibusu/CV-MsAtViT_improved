import scipy.io as sio
import argparse
import os
import gc
import numpy as np
from SAR_utils import *
from sklearn.metrics import confusion_matrix, accuracy_score, cohen_kappa_score
from Load_Data import load_data
from cvnn.layers import complex_input, ComplexConv2D, ComplexConv3D, ComplexDense, ComplexDropout, ComplexFlatten
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from net_flops import net_flops
from CoordAttention import CoordAtt_cmplx

def predict_by_batching(model, input_tensor, batch_size):
    '''
    Function to to perform predictions by dividing large tensor into small ones 
    to reduce load on GPU
    
    Parameters
    ----------
    model: The model itself with pre-trained weights.
    input_tensor: Tensor of diemnsion batches x windowSize x windowSize x channels x 1.
    batch_size: integer value smaller than batches .

    Returns
    -------
    Predicetd labels
    '''
    
    num_samples = input_tensor.shape[0]
    Y_pred_test = None
    
    k = 0
    total_batches = (num_samples + batch_size - 1) // batch_size
    
    for i in range(0, num_samples, batch_size):
        print("batch", k, " out of", total_batches)
        print(i, "out of", num_samples)
        k += 1
        
        batch = input_tensor[i : i + batch_size]
        batch_predictions = model.predict(batch, verbose=1)
        
        if Y_pred_test is None:
            # Pre-allocate output array based on first batch result
            # shape: (num_samples, num_classes)
            output_dim = batch_predictions.shape[1]
            Y_pred_test = np.zeros((num_samples, output_dim), dtype=batch_predictions.dtype)
            
        Y_pred_test[i : i + batch_predictions.shape[0]] = batch_predictions

    return Y_pred_test
          
        
        
# Get the data
# Get the data
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='FL_T', help='Dataset to train on (e.g. FL_T, SF, ober, Baltrum_S_FP1)')
parser.add_argument('--max-samples', type=int, default=200000, help='Max samples to use for training to avoid OOM (default: 200000). Set to -1 for all.')
parser.add_argument('--epochs', type=int, default=300, help='Number of epochs to train')
parser.add_argument('--only-gt', action='store_true', help='If True, restrict processing to GT pixels only and skip full map inference.')
args = parser.parse_args()

dataset = args.dataset
max_samples = args.max_samples if args.max_samples > 0 else None
epochs = args.epochs
windowSize = 15
test_ratio = 0.99
data, gt = load_data(dataset)
if dataset =='ober':
    lr = 0.0001
else:
    lr = 0.001
    

data = Standardize_data(data)



if args.max_samples == -1 and args.only_gt:
    # --- Chunked Processing Strategy ---
    # 1. Get all valid coordinates
    print("Getting valid coordinates (no patching yet)...")
    coords = get_gt_coords(gt, removeZeroLabels=True)
    y_all_valid = gt[coords[:, 0], coords[:, 1]]
    
    # 2. Split coordinates
    print(f"Splitting {len(coords)} samples...")
    # We pass None for X, so we get None back for X_train/X_test
    _, _, y_train, y_test, coords_train, coords_test = splitTrainTestSet(None, y_all_valid, test_ratio, coords=coords, randomState=42)
    
    # 3. Load Train patches
    print(f"Loading Train patches ({len(coords_train)} samples)...")
    X_train = extract_patches_from_coords(data, coords_train, windowSize=windowSize)
    X_train = np.expand_dims(X_train, axis=4)
    y_train = y_train - 1 # 0-indexed for training
    y_test = y_test - 1   # 0-indexed for evaluation
    
    # 4. X_test is NOT loaded yet, will be processed in chunks
    X_test = None 
    
else:
    # --- Original Logic ---
    X_coh, y, coords = createImageCubes(data, gt, windowSize, max_samples=max_samples, random_state=42)
    X_coh = np.expand_dims(X_coh, axis=4)
    
    X_train, X_test, y_train, y_test, coords_train, coords_test = splitTrainTestSet(X_coh, y, test_ratio, coords=coords, randomState=42)
    del X_coh  # save RAM


total = 0
numm = []
for i in range(int(np.max(y_test)+1)):
    tmp = np.sum(y_test==i)
    total = total + tmp
    numm.append(tmp)
    print("Class #"+str(i) +": " + str(tmp))
   
    
y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)


image_size = windowSize  # Final Image Size
patch_size = 3  # Patch Dimension
num_patches = (image_size // patch_size) ** 2
projection_dim = 32
num_heads = 4
transformer_units = [
    projection_dim * 2,
    projection_dim,
]  # Size of the transformer layers
transformer_layers = 4 #8
mlp_head_units = [1024, 512] #[2048, 1024]  # Size of the dense layers


"""## Implementing Multilayer Perceptron"""
def cmplx_multilayer_perceptron(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = ComplexDense(units, activation=cart_gelu)(x)
        x = ComplexDropout(dropout_rate)(x)
    return x

"""## Implementing patch creation as a layer"""
class Patches(layers.Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches


"""## Implement the Patch Encoding Layer"""
class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = ComplexDense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + tf.cast(self.position_embedding(positions), tf.complex64)
        return encoded


def MultiScaleFeatureExtractor(inputs):    
    # Spatial Path
    x1 = ComplexConv3D(filters=8, kernel_size=(3, 3, 1), activation='cart_relu', padding='same')(inputs)
    x1 = ComplexConv3D(filters=8, kernel_size=(3, 3, 1), activation='cart_relu',padding='same')(x1)
    
    # Polarimtric Path
    x2 = ComplexConv3D(filters=8, kernel_size=(1, 1, 3), activation='cart_relu',padding='same')(inputs)
    x2 = ComplexConv3D(filters=8, kernel_size=(1, 1, 3), activation='cart_relu',padding='same')(x2)

    # Spatial-Polarimtric Path
    x3 = ComplexConv3D(filters=8, kernel_size=(3, 3, 3), activation='cart_relu',padding='same')(inputs)
    x3 = ComplexConv3D(filters=8, kernel_size=(3, 3, 3), activation='cart_relu',padding='same')(x3)

    concatenated_features = tf.concat([x1,x2,x3],axis=4);
    
    return concatenated_features


def cmplx_ViT(x):
   
    patches = Patches(patch_size)(x)

    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)


    for _ in range(transformer_layers):

        x1_r = layers.LayerNormalization(epsilon=1e-6)(tf.math.real(encoded_patches))
        x1_i = layers.LayerNormalization(epsilon=1e-6)(tf.math.imag(encoded_patches))
        
        attention_output_r = layers.MultiHeadAttention(num_heads=num_heads, key_dim=projection_dim, dropout=0.1)(x1_r, x1_r)
        attention_output_i = layers.MultiHeadAttention(num_heads=num_heads, key_dim=projection_dim, dropout=0.1)(x1_i, x1_i)
        attention_output = tf.complex(attention_output_r, attention_output_i)

        x2 = layers.Add()([attention_output, encoded_patches])

        x3_r = layers.LayerNormalization(epsilon=1e-6)(tf.math.real(x2))
        x3_i = layers.LayerNormalization(epsilon=1e-6)(tf.math.imag(x2))
        x3 = tf.complex(x3_r, x3_i)
        
        
        x3 = cmplx_multilayer_perceptron(x3, hidden_units=transformer_units, dropout_rate=0.1)

        encoded_patches = layers.Add()([x3, x2])

    representation_r = layers.LayerNormalization(epsilon=1e-6)(tf.math.real(encoded_patches))
    representation_i = layers.LayerNormalization(epsilon=1e-6)(tf.math.imag(encoded_patches))
    representation = tf.complex(representation_r, representation_i)
    
    representation = ComplexFlatten()(representation)
    representation = ComplexDropout(0.5)(representation)

    features = cmplx_multilayer_perceptron(representation, hidden_units=mlp_head_units, dropout_rate=0.3)

    
    return features

def MsAtViT(img_list, num_class):
    inputs = complex_input(shape=img_list.shape[1:])
    
    x = MultiScaleFeatureExtractor(inputs)
    x_shape = x.shape
    x = keras.layers.Reshape((x_shape[1], x_shape[2], x_shape[3]*x_shape[4]))(x)
    x = ComplexConv2D(filters=24, kernel_size=(3,3), activation='cart_relu',padding='same')(x)
   
    
    x = CoordAtt_cmplx(x, 4)
    
    
    x = cmplx_ViT(x)
    
    z = ComplexFlatten()(x)
    logits = ComplexDense(num_classes(dataset), activation="softmax_real_with_abs")(z)
    
    model = tf.keras.Model(inputs=[inputs], outputs=logits)
    model.compile(optimizer = Adam(learning_rate=lr), 
                  loss='categorical_crossentropy',
                  metrics=['accuracy']
                  )

    return model

def save_training_curve(history, dataset_tag):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    loss = history.history.get("loss", [])
    acc = history.history.get("accuracy", [])
    epochs = range(1, len(loss) + 1)
    plot_dir = os.path.join("results", "plots")
    os.makedirs(plot_dir, exist_ok=True)
    plt.figure(figsize=(8, 5))
    if loss:
        plt.plot(epochs, loss, label="loss")
    if acc:
        plt.plot(epochs, acc, label="accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.ylim(0, 1.1)  # Fix y-axis range
    plt.title(f"Training Curve ({dataset_tag})")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    out_path = os.path.join(plot_dir, f"training_curve_{dataset_tag}.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print("Saved training curve to", out_path)

"""## Compile, Train, and Evaluate the model"""
model = MsAtViT(X_train, num_classes(dataset))
model.summary()

net_flops(model)

# Perform Training
from tensorflow.keras.callbacks import EarlyStopping
early_stopper = EarlyStopping(monitor='accuracy', 
                              patience=10,
                              restore_best_weights=True
                              )

#model = MsAtViT(X_train, num_classes(dataset))
    
history = model.fit(X_train, y_train,
                            batch_size = 128, 
 
                            verbose = 1, 
                            epochs = epochs, 
                            shuffle = True,
                            callbacks = [early_stopper] )
    
save_training_curve(history, dataset)
    


if X_test is not None:
    Y_pred_test = model.predict(X_test)
    y_pred_test = np.argmax(Y_pred_test, axis=1)
    kappa = cohen_kappa_score(np.argmax(y_test, axis=1), y_pred_test)
    oa = accuracy_score(np.argmax(y_test, axis=1), y_pred_test)
    confusion = confusion_matrix(np.argmax(y_test, axis=1), y_pred_test)
    each_acc, aa = AA_andEachClassAccuracy(confusion)
    report = classification_report(np.argmax(y_test, axis=1), y_pred_test, digits=4)
    print("OA = ", oa)
    print("AA = ", aa)
    print("Kappa = ", kappa)
    print('Classification Report: \n', report)
elif args.max_samples == -1 and args.only_gt:
    # Calculate metrics for Chunked Test Set
    print("Calculating metrics for Chunked Test Set...")
    # Validation/Test prediction loop
    chunk_size = 500000
    y_pred_test_all = []
    
    num_test = len(coords_test)
    for i in range(0, num_test, chunk_size):
        chunk_coords = coords_test[i : i + chunk_size]
        chunk_patches = extract_patches_from_coords(data, chunk_coords, windowSize=windowSize)
        chunk_patches = np.expand_dims(chunk_patches, axis=4)
        
        preds = predict_by_batching(model, chunk_patches, 128)
        y_pred_chunk = np.argmax(preds, axis=1)
        y_pred_test_all.append(y_pred_chunk)
        
        del chunk_patches
        import gc
        gc.collect()

    y_pred_test = np.concatenate(y_pred_test_all)
    
    kappa = cohen_kappa_score(np.argmax(y_test, axis=1), y_pred_test)
    oa = accuracy_score(np.argmax(y_test, axis=1), y_pred_test)
    confusion = confusion_matrix(np.argmax(y_test, axis=1), y_pred_test)
    each_acc, aa = AA_andEachClassAccuracy(confusion)
    report = classification_report(np.argmax(y_test, axis=1), y_pred_test, digits=4)
    print("OA = ", oa)
    print("AA = ", aa)
    print("Kappa = ", kappa)
    print('Classification Report: \n', report)

print("--- Class-wise Accuracy ---")
for i, acc in enumerate(each_acc):
    print(f"Class {i}: {format(acc * 100, '.2f')}")



###############################################################################
# Create the predicted class map
# Create the predicted class map
# del X_train, X_test
# import gc
# gc.collect()
# keras.backend.clear_session()

pred_map = np.zeros((gt.shape[0], gt.shape[1]), dtype=np.uint8)

if args.only_gt:
    print("Using selective GT mapping (skipping full inference)...")
    
    # Predict on Train set (if it still exists in memory, or we need to be careful if we deleted it)
    # We haven't deleted X_train yet in the new flow, so we can use it.
    if X_train is not None:
        print(f"Predicting on Train set ({X_train.shape[0]} samples)...")
        Y_pred_train = predict_by_batching(model, X_train, 128)
        y_pred_train = np.argmax(Y_pred_train, axis=1) + 1 
        
        # Test set prediction (Chunked or Standard)
        if args.max_samples == -1:
            print("Predicting on Test set in chunks (500k)...")
            chunk_size = 500000
            y_pred_test_all = []
            
            num_test = len(coords_test)
            for i in range(0, num_test, chunk_size):
                print(f"  Processing chunk {i}-{min(i+chunk_size, num_test)} / {num_test}...")
                chunk_coords = coords_test[i : i + chunk_size]
                
                # Extract patches for chunk
                chunk_patches = extract_patches_from_coords(data, chunk_coords, windowSize=windowSize)
                chunk_patches = np.expand_dims(chunk_patches, axis=4)
                
                # Predict
                preds = predict_by_batching(model, chunk_patches, 128)
                y_pred_chunk = np.argmax(preds, axis=1)
                
                y_pred_test_all.append(y_pred_chunk)
                
                # Free memory
                del chunk_patches
                del preds
                gc.collect()
                
            y_pred_test = np.concatenate(y_pred_test_all)
            
        else:
            # Standard flow where X_test is already in memory
             # Test set is already predicted as y_pred_test (indices) from model.evaluate/predict block? 
             # Wait, previous block only did model.evaluate. We need labels.
             Y_pred_test = predict_by_batching(model, X_test, 128)
             y_pred_test = np.argmax(Y_pred_test, axis=1)
             
    # Prepare labels (1-based for map)
    y_pred_test_labels = y_pred_test + 1
    
    # Fill map using coords
    pred_map[coords_train[:, 0], coords_train[:, 1]] = y_pred_train
    pred_map[coords_test[:, 0], coords_test[:, 1]] = y_pred_test_labels

else:
    print("Generating full map prediction (memory efficient)...")
    
    del X_train, X_test
    gc.collect()
    keras.backend.clear_session()

    margin = int((windowSize - 1) / 2)
    zeroPaddedX = padWithZeros(data, margin=margin)

    # Process in chunks to avoid OOM
    # Total pixels to predict
    h, w = gt.shape
    total_pixels = h * w
    # Reduced batch size further to 10,000 to prevent 'Dst tensor is not initialized' / OOM errors
    batch_size = 10000 
    patch_batch = np.zeros((batch_size, windowSize, windowSize, data.shape[2]), dtype='complex64')
    coords_batch = []

    count = 0
    for r in range(h):
        for c in range(w):
            # Extract patch
            patch = zeroPaddedX[r:r+windowSize, c:c+windowSize]
            patch_batch[count] = patch
            coords_batch.append((r, c))
            count += 1
            
            if count == batch_size:
                # Predict batch
                patch_batch_input = np.expand_dims(patch_batch, axis=4)
                preds = model.predict(patch_batch_input, verbose=0)
                labels = np.argmax(preds, axis=1)
                
                # Fill map
                for idx, (rr, cc) in enumerate(coords_batch):
                    pred_map[rr, cc] = labels[idx] + 1 # Class labels usually 1-indexed in output mat
                
                # Reset
                count = 0
                coords_batch = []
                if (r * w + c) % 100000 == 0:
                    print(f"Processed {r * w + c}/{total_pixels} pixels...")

    # Process remaining
    if count > 0:
        patch_batch_input = np.expand_dims(patch_batch[:count], axis=4)
        preds = model.predict(patch_batch_input, verbose=0)
        labels = np.argmax(preds, axis=1)
        for idx, (rr, cc) in enumerate(coords_batch):
            pred_map[rr, cc] = labels[idx] + 1

name = 'CV_MsAtViT_Full'
sio.savemat(name+'.mat', {name: pred_map})

gt_binary = gt.copy()
gt_binary[gt_binary>0]=1
new_map = pred_map * gt_binary

name = 'CV_MsAtViT'
sio.savemat(name+'.mat', {name: new_map})


