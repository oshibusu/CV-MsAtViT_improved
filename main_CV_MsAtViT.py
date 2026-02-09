import argparse
import csv
import json
import math
import os
import gc
import scipy.io as sio
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import confusion_matrix, accuracy_score, cohen_kappa_score, classification_report
from Load_Data import load_data
from SAR_utils import cart_gelu, num_classes, softmax_real_with_real, save_classification_map, Standardize_data, createImageCubes, splitTrainTestSet, AA_andEachClassAccuracy, padWithZeros, get_gt_coords, extract_patches_from_coords, ModReLU, ModSigmoid, ModTanhScaled, ModGated, ModSigmoidGated
from net_flops import net_flops
from model_factory import build_msatvit


def _dataset_tag(name: str) -> str:
    """Make dataset name filesystem-friendly for checkpoints."""
    return name.replace("/", "_").replace("\\", "_")


def _combine_complex_kernel(weights):
    if not weights:
        raise ValueError("Layer has no weights")
    kernel = weights[0]
    if len(weights) >= 2 and weights[0].shape == weights[1].shape and not np.iscomplexobj(weights[0]):
        kernel = weights[0] + 1j * weights[1]
    if not np.iscomplexobj(kernel):
        kernel = kernel.astype(np.complex64)
    return kernel


def _parse_index_spec(spec: str, max_len: int):
    if not spec:
        return list(range(max_len))
    indices = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            try:
                start = int(start_s)
                end = int(end_s)
            except ValueError:
                continue
            if start > end:
                start, end = end, start
            for i in range(start, end + 1):
                if 0 <= i < max_len:
                    indices.add(i)
        else:
            try:
                val = int(part)
            except ValueError:
                continue
            if 0 <= val < max_len:
                indices.add(val)
    return sorted(indices)


class BatchTraceCallback(keras.callbacks.Callback):
    def __init__(
        self,
        dataset_tag,
        branch_names,
        filter_spec,
        in_spec,
        depth_spec,
        batch_size,
        total_samples,
        max_epochs,
    ):
        super().__init__()
        self.dataset_tag = dataset_tag
        self.branch_names = branch_names
        self.filter_spec = filter_spec
        self.in_spec = in_spec
        self.depth_spec = depth_spec
        self.batch_size = batch_size
        self.total_samples = total_samples
        self.max_epochs = max_epochs
        self.base_dir = os.path.join("ckpt", "batch_traces", dataset_tag)
        self.recording = True
        self.branch_configs = {}
        self.csv_path = os.path.join(self.base_dir, "batch_metrics.csv")
        self.csv_file = None
        self.current_epoch = 0

    def on_train_begin(self, logs=None):
        os.makedirs(self.base_dir, exist_ok=True)
        self._initialize_branch_configs()
        os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
        self.csv_file = open(self.csv_path, "w", newline="")
        writer = csv.writer(self.csv_file)
        writer.writerow(["epoch", "snapshot", "batch_index", "start", "end", "loss", "accuracy"])
        self.csv_file.flush()
        self._save_snapshot(
            epoch_idx=0,
            batch_index="pre",
            start_idx=0,
            end_idx=0,
            metrics=None,
        )

    def on_epoch_begin(self, epoch, logs=None):
        self.current_epoch = epoch
        if epoch >= self.max_epochs:
            self.recording = False
            return
        self.recording = True
        if epoch > 0:
            self._save_snapshot(
                epoch_idx=epoch,
                batch_index="pre",
                start_idx=0,
                end_idx=0,
                metrics=None,
            )

    def on_train_batch_end(self, batch, logs=None):
        if not self.recording:
            return
        start_idx = batch * self.batch_size
        end_idx = min(start_idx + self.batch_size, self.total_samples)
        metrics = {
            "loss": float(logs.get("loss")) if logs and "loss" in logs else None,
            "accuracy": float(logs.get("accuracy")) if logs and "accuracy" in logs else None,
        }
        self._save_snapshot(
            epoch_idx=self.current_epoch,
            batch_index=batch,
            start_idx=start_idx,
            end_idx=end_idx,
            metrics=metrics,
        )

    def on_train_end(self, logs=None):
        if self.csv_file:
            self.csv_file.close()

    def _save_snapshot(self, epoch_idx, batch_index, start_idx, end_idx, metrics=None):
        epoch_name = f"epoch{epoch_idx + 1:02d}"
        if batch_index == "pre":
            suffix = "batch0000_pre"
        else:
            suffix = f"batch{int(batch_index):04d}"
        batch_dir = os.path.join(self.base_dir, f"{epoch_name}_{suffix}")
        os.makedirs(batch_dir, exist_ok=True)
        with open(os.path.join(batch_dir, "progress.txt"), "w") as f:
            f.write(f"start={start_idx},end={end_idx}")
        weights_path = os.path.join(batch_dir, "weights.h5")
        self.model.save_weights(weights_path)
        metrics_path = os.path.join(batch_dir, "metrics.json")
        with open(metrics_path, "w") as mf:
            json.dump(metrics if metrics else {}, mf)
        if self.csv_file:
            writer = csv.writer(self.csv_file)
            writer.writerow(
                [
                    epoch_idx + 1,
                    os.path.basename(batch_dir),
                    batch_index,
                    start_idx,
                    end_idx,
                    metrics.get("loss") if metrics else "",
                    metrics.get("accuracy") if metrics else "",
                ]
            )
            self.csv_file.flush()
        for branch, cfg in self.branch_configs.items():
            layer = self.model.get_layer(branch)
            kernel = _combine_complex_kernel(layer.get_weights())
            branch_path = os.path.join(batch_dir, branch)
            os.makedirs(branch_path, exist_ok=True)
            for filt_idx in cfg["filters"]:
                if filt_idx >= kernel.shape[-1]:
                    continue
                filt_dir = os.path.join(branch_path, f"filter{filt_idx:02d}")
                os.makedirs(filt_dir, exist_ok=True)
                for in_idx in cfg["inputs"]:
                    if in_idx >= kernel.shape[-2]:
                        continue
                    vol = kernel[:, :, :, in_idx, filt_idx]
                    depth_len = vol.shape[2]
                    for depth_idx in cfg["depths"]:
                        d = depth_idx % depth_len
                        slice_2d = vol[:, :, d]
                        out_path = os.path.join(
                            filt_dir,
                            f"in{in_idx:02d}_depth{depth_idx:02d}.npy",
                        )
                        np.save(out_path, slice_2d)

    def _initialize_branch_configs(self):
        if self.branch_configs:
            return
        for branch in self.branch_names:
            layer = self.model.get_layer(branch)
            kernel = _combine_complex_kernel(layer.get_weights())
            kD, kH, kW, in_ch, out_ch = kernel.shape
            filters = _parse_index_spec(self.filter_spec, out_ch)
            inputs = _parse_index_spec(self.in_spec, in_ch)
            depths = _parse_index_spec(self.depth_spec, kW)
            self.branch_configs[branch] = {
                "filters": filters,
                "inputs": inputs,
                "depths": depths,
            }

class BiasMonitorCallback(keras.callbacks.Callback):
    """
    Callback to record learnable bias 'b' of ModReLU, ModSigmoid, and ModTanhScaled layers.
    Saves collected values to CSV at the end of training.
    """
    def __init__(self, dataset_tag):
        super().__init__()
        self.dataset_tag = dataset_tag
        self.base_dir = os.path.join("results", "bias_monitors", dataset_tag)
        self.csv_path = os.path.join(self.base_dir, "bias_values.csv")
        self.bias_history = []

    def on_train_begin(self, logs=None):
        os.makedirs(self.base_dir, exist_ok=True)
        self.bias_history = []

    def on_epoch_end(self, epoch, logs=None):
        target_layers = []
        for layer in self.model.layers:
            if isinstance(layer, (ModReLU, ModSigmoid, ModTanhScaled)):
                target_layers.append(layer)
            elif hasattr(layer, 'layers'):
                for sub_layer in layer.layers:
                    if isinstance(sub_layer, (ModReLU, ModSigmoid, ModTanhScaled)):
                        target_layers.append(sub_layer)

        if not target_layers:
            return

        for layer in target_layers:
            b_values = layer.get_weights()[0]
            for i, val in enumerate(b_values):
                self.bias_history.append([epoch + 1, layer.name, i, float(val)])
        print(f"Captured bias values for {len(target_layers)} layers at epoch {epoch + 1}")

    def on_train_end(self, logs=None):
        if not self.bias_history:
            print("[info] No bias values were captured; skipping CSV save.")
            return
            
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "layer_name", "channel_index", "value"])
            writer.writerows(self.bias_history)
        print(f"Successfully saved all captured bias values to {self.csv_path}")


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


def save_batch_curve(csv_path, dataset_tag):
    if not csv_path or not os.path.exists(csv_path):
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    batch_indices = []
    losses = []
    accuracies = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epoch_val = row.get("epoch")
            try:
                epoch_idx = int(epoch_val)
            except (TypeError, ValueError):
                epoch_idx = 1
            idx = row.get("batch_index")
            if idx == "pre" or idx == "" or idx is None:
                continue
            try:
                batch_idx = int(idx)
            except ValueError:
                continue
            global_idx = len(batch_indices)
            batch_indices.append(global_idx)
            loss = row.get("loss")
            acc = row.get("accuracy")
            losses.append(float(loss) if loss not in (None, "") else math.nan)
            accuracies.append(float(acc) if acc not in (None, "") else math.nan)
    if not batch_indices:
        return
    x_ticks = list(range(len(batch_indices)))
    if not batch_indices:
        return
    plot_dir = os.path.join("results", "plots")
    os.makedirs(plot_dir, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(x_ticks, losses, label="loss")
    if any(not math.isnan(a) for a in accuracies):
        plt.plot(x_ticks, accuracies, label="accuracy")
    plt.xlabel("Batch step (recorded epochs)")
    plt.ylabel("Value")
    plt.title(f"Batch Curve (epoch 1, {dataset_tag})")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    out_path = os.path.join(plot_dir, f"batch_curve_epoch1_{dataset_tag}.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print("Saved batch curve to", out_path)


def parse_args():
    parser = argparse.ArgumentParser(description="Train and export CV-MsAtViT")
    parser.add_argument("--dataset", default="SF", help="Dataset identifier (e.g., FL_T, SF, ober)")
    parser.add_argument(
        "--layer-norm-type",
        default="complex",
        choices=["amplitude", "split", "complex"],
        help="Type of Layer Normalization: 'complex' (Default, centers to complex mean), 'amplitude' (Legacy), or 'split'",
    )
    parser.add_argument(
        "--activation-type",
        default="modrelu",
        choices=["modrelu", "cart_relu", "mod_gated", "mod_sigmoid_gated"],
        help="Type of Activation for Conv layers: 'modrelu', 'cart_relu', 'mod_gated', or 'mod_sigmoid_gated'",
    )
    parser.add_argument(
        "--b-init",
        type=float,
        default=-0.1,
        help="Initial value for learnable bias 'b' in Mod activations (default: -0.1)",
    )
    parser.add_argument(
        "--coord-activation",
        default="modtanh",
        choices=["modtanh", "cart_sigmoid", "modsigmoid"],
        help="Type of Activation for Coordinate Attention: 'modtanh', 'cart_sigmoid', or 'modsigmoid'",
    )
    parser.add_argument("--window-size", type=int, default=15, help="Patch window size")
    parser.add_argument("--test-ratio", type=float, default=0.99, help="Test split ratio")
    parser.add_argument("--batch-size", type=int, default=128, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=300, help="Maximum training epochs")
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Override learning rate (default: 1e-4 for ober else 1e-3)",
    )
    parser.add_argument(
        "--record-first-epoch",
        action="store_true",
        help="Record per-batch kernels during the first epoch",
    )
    parser.add_argument(
        "--record-epochs",
        type=int,
        default=1,
        help="Number of initial epochs to record when --record-first-epoch is enabled",
    )
    parser.add_argument(
        "--record-branches",
        default="",
        help="Comma-separated branch names to record (default: all 3D branches)",
    )
    parser.add_argument(
        "--record-filters",
        default="",
        help="Filter indices to record (e.g., '0-3,5'; default all)",
    )
    parser.add_argument(
        "--record-in",
        default="",
        help="Input channel indices to record (default all)",
    )
    parser.add_argument(
        "--record-depth",
        default="",
        help="Depth indices to record (default all)",
    )
    parser.add_argument(
        "--baltrum-band",
        default="S",
        choices=["L", "S"],
        help="Band for Baltrum dataset (default: S)",
    )
    parser.add_argument(
        "--baltrum-fp",
        default="FP1",
        choices=["FP1", "FP2"],
        help="Flight path for Baltrum dataset (default: FP1)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=200000,
        help="Max samples to use for training to avoid OOM (default: 200000). Set to -1 for all.",
    )
    parser.add_argument(
        "--only-gt",
        action="store_true",
        help="If True, restrict processing to GT pixels only and skip full map inference.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--transformer-layers",
        type=int,
        default=4,
        help="Number of transformer layers (default: 4)",
    )
    return parser.parse_args()


def predict_by_batching(model, input_tensor, batch_size):
    """Run inference by chunking large tensors into smaller batches."""
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


def main():
    args = parse_args()
    # Set global seed for reproducibility
    tf.keras.utils.set_random_seed(args.seed)
    
    dataset = args.dataset
    if dataset == 'Baltrum':
        dataset = f"Baltrum_{args.baltrum_band}_{args.baltrum_fp}"
        
    dataset_tag = _dataset_tag(dataset)
    window_size = args.window_size
    test_ratio = args.test_ratio
    record_epochs = max(1, args.record_epochs)

    data, gt = load_data(dataset)
    lr = args.learning_rate if args.learning_rate is not None else (0.0001 if dataset == "ober" else 0.001)

    data = Standardize_data(data)

    # Handle max_samples logic
    max_samples = args.max_samples if args.max_samples > 0 else None
    
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
        X_train = extract_patches_from_coords(data, coords_train, windowSize=window_size)
        X_train = np.expand_dims(X_train, axis=4)
        y_train = y_train - 1 # 0-indexed for training
        y_test = y_test - 1   # 0-indexed for evaluation
        
        # 4. X_test is NOT loaded yet, will be processed in chunks
        X_test = None 
        
    else:
        # --- Original Logic ---
        X_coh, y, coords = createImageCubes(data, gt, window_size, max_samples=max_samples, random_state=42)
        X_coh = np.expand_dims(X_coh, axis=4)
        
        X_train, X_test, y_train, y_test, coords_train, coords_test = splitTrainTestSet(X_coh, y, test_ratio, coords=coords, randomState=42)
        del X_coh  # save RAM

    for i in range(int(np.max(y_test) + 1)):
        count = np.sum(y_test == i)
        print("Class #" + str(i) + ": " + str(count))

    y_train = keras.utils.to_categorical(y_train)
    y_test = keras.utils.to_categorical(y_test)

    model = build_msatvit(
        input_shape=X_train.shape[1:],
        dataset=dataset,
        window_size=window_size,
        lr=lr,
        transformer_layers=args.transformer_layers,
        layer_norm_type=args.layer_norm_type,
        activation_type=args.activation_type,
        coord_activation=args.coord_activation,
        b_init=args.b_init,
    )
    model.summary()
    net_flops(model)

    ckpt_dir = os.path.join("ckpt")
    os.makedirs(ckpt_dir, exist_ok=True)

    epoch_ckpt_path = os.path.join(
        ckpt_dir, f"CV_MsAtViT_{dataset_tag}_epoch{{epoch:03d}}.weights.h5"
    )
    epoch_checkpoint = keras.callbacks.ModelCheckpoint(
        filepath=epoch_ckpt_path,
        save_weights_only=True,
        save_freq="epoch",
    )

    early_stopper = keras.callbacks.EarlyStopping(
        monitor="accuracy", patience=10, restore_best_weights=True
    )

    bias_monitor = BiasMonitorCallback(dataset_tag)
    callbacks = [early_stopper, epoch_checkpoint, bias_monitor]


    record_callback = None
    if args.record_first_epoch:
        record_epochs = max(1, args.record_epochs)
        branch_names = [b.strip() for b in args.record_branches.split(",") if b.strip()]
        if not branch_names:
            default_branches = [
                "spatial_conv3d_block1",
                "polar_conv3d_block1",
                "joint_conv3d_block1",
            ]
            branch_names = [
                name for name in default_branches if any(layer.name == name for layer in model.layers)
            ]
        if branch_names:
            record_callback = BatchTraceCallback(
                dataset_tag,
                branch_names,
                args.record_filters,
                args.record_in,
                args.record_depth,
                args.batch_size,
                X_train.shape[0],
                record_epochs,
            )
            callbacks.append(record_callback)
        else:
            print("[warn] No matching branches found for recording; skipping batch trace")

    training_epochs = record_epochs if args.record_first_epoch else args.epochs
    if args.record_first_epoch and args.epochs > record_epochs:
        print(
            f"[info] record-first-epoch enabled: training will stop after {record_epochs} epoch(s)"
        )

    history = model.fit(
        X_train,
        y_train,
        batch_size=args.batch_size,
        verbose=1,
        epochs=training_epochs,
        shuffle=True,
        callbacks=callbacks,
    )
    save_training_curve(history, dataset_tag)
    if record_callback and args.record_first_epoch:
        save_batch_curve(record_callback.csv_path, dataset_tag)

    if X_test is not None:
        Y_pred_test = predict_by_batching(model, X_test, max(1, X_test.shape[0] // 16))
        y_pred_test = np.argmax(Y_pred_test, axis=1)
        kappa = cohen_kappa_score(np.argmax(y_test, axis=1), y_pred_test)
        oa = accuracy_score(np.argmax(y_test, axis=1), y_pred_test)
        confusion = confusion_matrix(np.argmax(y_test, axis=1), y_pred_test)
        each_acc, aa = AA_andEachClassAccuracy(confusion)
        print("oa = ", format((oa) * 100, ".2f"))
        print("aa = ", format((aa) * 100, ".2f"))
        print("Kappa = ", format((kappa) * 100, ".2f"))
        
        print("--- Class-wise Accuracy ---")
        for i, acc in enumerate(each_acc):
            print(f"Class {i}: {format(acc * 100, '.2f')}")
        print("---------------------------")
    elif args.max_samples == -1 and args.only_gt:
        # Calculate metrics for Chunked Test Set
        print("Calculating metrics for Chunked Test Set...")
        # Validation/Test prediction loop
        chunk_size = 500000
        y_pred_test_all = []
        
        num_test = len(coords_test)
        for i in range(0, num_test, chunk_size):
            chunk_coords = coords_test[i : i + chunk_size]
            chunk_patches = extract_patches_from_coords(data, chunk_coords, window_size)
            chunk_patches = np.expand_dims(chunk_patches, axis=4)
            
            preds = predict_by_batching(model, chunk_patches, 128)
            y_pred_chunk = np.argmax(preds, axis=1)
            y_pred_test_all.append(y_pred_chunk)
            
            del chunk_patches
            gc.collect()

        y_pred_test = np.concatenate(y_pred_test_all)
        
        # Convert one-hot encoded y_test back to class indices for metrics
        y_test = np.argmax(y_test, axis=1)
        
        kappa = cohen_kappa_score(y_test, y_pred_test)
        oa = accuracy_score(y_test, y_pred_test)
        confusion = confusion_matrix(y_test, y_pred_test)
        each_acc, aa = AA_andEachClassAccuracy(confusion)
        report = classification_report(y_test, y_pred_test, digits=4)
        print("OA = ", oa)
        print("AA = ", aa)
        print("Kappa = ", kappa)
        print('Classification Report: \n', report)
        
        print("--- Class-wise Accuracy ---")
        for i, acc in enumerate(each_acc):
            print(f"Class {i}: {format(acc * 100, '.2f')}")
        print("---------------------------")

        # Create the predicted class map
        # Create the predicted class map
        # Commented out per user request to skip full map inference
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
            
            # Test set prediction is already done in metrics calculation phase.
            # We reuse y_pred_test (indices) from there.
                 
        # Prepare labels (1-based for map)
        y_pred_test_labels = y_pred_test + 1
        
        # Fill map using coords
        pred_map[coords_train[:, 0], coords_train[:, 1]] = y_pred_train
        pred_map[coords_test[:, 0], coords_test[:, 1]] = y_pred_test_labels
        
    else:
        print("Generating full map prediction (memory efficient)...")
        
        # Release memory before full inference
        del X_train, X_test
        gc.collect()
        # Note: clearing session might invalidate model if we needed it again, 
        # but here we use 'model' object. Standard Keras model object survives clear_session? 
        # Actually clear_session destroys the graph. If I use model.predict after this, it might fail 
        # if the model is not re-loaded or if it's not eager execution.
        # Safe to remove clear_session here or move it after.
        # Given we have the 'model' object loaded in memory, let's keep it safe.
        
        margin = int((window_size - 1) / 2)
        zeroPaddedX = padWithZeros(data, margin=margin)

        # Process in chunks to avoid OOM
        # Total pixels to predict
        h, w = gt.shape
        total_pixels = h * w
        # Reduced batch size further to 10,000 to prevent 'Dst tensor is not initialized' / OOM errors
        batch_size = 10000 
        patch_batch = np.zeros((batch_size, window_size, window_size, data.shape[2]), dtype='complex64')
        coords_batch = []

        count = 0
        for r in range(h):
            for c in range(w):
                # Extract patch
                patch = zeroPaddedX[r:r+window_size, c:c+window_size]
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

    Y_pred = pred_map
    name = f"CV_MsAtViT_Full_{dataset_tag}"
    mat_save_path = os.path.join("results", f"{name}.mat")
    sio.savemat(mat_save_path, {name: Y_pred})

    gt_binary = gt.copy()
    gt_binary[gt_binary > 0] = 1
    new_map = Y_pred * gt_binary

    name = f"CV_MsAtViT_{dataset_tag}"
    mat_save_path_2 = os.path.join("results", f"{name}.mat")
    sio.savemat(mat_save_path_2, {name: new_map})

    # Save classification map image
    map_save_path = os.path.join("results", "plots", f"{dataset_tag}_classification_map.png")
    print(f"Saving classification map image to {map_save_path}...")
    save_classification_map(new_map, gt, dataset, map_save_path)

    # Save weights for downstream visualization
    weights_path = os.path.join(ckpt_dir, f"CV_MsAtViT_{dataset_tag}_weights.h5")
    model.save_weights(weights_path)
    print("Weights saved to", weights_path)

    saved_model_dir = os.path.join(ckpt_dir, f"CV_MsAtViT_{dataset_tag}_saved_model")
    model.save(saved_model_dir, include_optimizer=False)
    print("SavedModel exported to", saved_model_dir)


if __name__ == "__main__":
    main()
