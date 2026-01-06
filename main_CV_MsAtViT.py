import argparse
import csv
import json
import math
import os
import scipy.io as sio
import numpy as np
from tensorflow import keras
from sklearn.metrics import confusion_matrix, accuracy_score, cohen_kappa_score
from Load_Data import load_data
from SAR_utils import *  # noqa: F401,F403 retains helper utilities
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
    parser.add_argument("--dataset", default="FL_T", help="Dataset identifier (e.g., FL_T, SF, ober)")
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
    return parser.parse_args()


def predict_by_batching(model, input_tensor, batch_size):
    """Run inference by chunking large tensors into smaller batches."""
    num_samples = input_tensor.shape[0]
    k = 0
    predictions = []
    for i in range(0, num_samples, batch_size):
        print("batch", k, " out of", max(1, num_samples // batch_size))
        print(k * batch_size, "out of", num_samples)
        k += 1
        batch = input_tensor[i : i + batch_size]
        batch_predictions = model.predict(batch, verbose=1)
        predictions.append(batch_predictions)

    Y_pred_test = np.concatenate(predictions, axis=0)
    return Y_pred_test


def main():
    args = parse_args()
    args = parse_args()
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
    
    X_coh, y = createImageCubes(data, gt, window_size, max_samples=max_samples)
    X_coh = np.expand_dims(X_coh, axis=4)
    
    X_train, X_test, y_train, y_test = splitTrainTestSet(X_coh, y, test_ratio)
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

    callbacks = [early_stopper, epoch_checkpoint]

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

    Y_pred_test = predict_by_batching(model, X_test, max(1, X_test.shape[0] // 16))
    y_pred_test = np.argmax(Y_pred_test, axis=1)

    kappa = cohen_kappa_score(np.argmax(y_test, axis=1), y_pred_test)
    oa = accuracy_score(np.argmax(y_test, axis=1), y_pred_test)
    confusion = confusion_matrix(np.argmax(y_test, axis=1), y_pred_test)
    each_acc, aa = AA_andEachClassAccuracy(confusion)

    print("oa = ", format((oa) * 100, ".2f"))
    print("aa = ", format((aa) * 100, ".2f"))
    print("Kappa = ", format((kappa) * 100, ".2f"))

    # Create the predicted class map
    # Create the predicted class map
    # Note: If max_samples was used in SAR_utils (default 200k), X_coh_full will be smaller than full image.
    # We must check if we can reshape.
    del X_train, X_test
    X_coh_full, _ = createImageCubes(data, gt, window_size, removeZeroLabels=False, max_samples=200000) 
    # Calling with consistent max_samples to prevent OOM, though this means full map is impossible this way.
    
    if X_coh_full.shape[0] != gt.size:
        print(f"[WARN] Skipping full map generation: Subsampled data size ({X_coh_full.shape[0]}) does not match full image size ({gt.size}).")
        print("To generate a full map, OOM-safe sliding window inference is required (not enabled).")
    else:
        X_coh_full = np.expand_dims(X_coh_full, axis=4)

        Y_pred_full = predict_by_batching(
            model, X_coh_full, max(1, X_coh_full.shape[0] // 16)
        )
        y_pred_full = (np.argmax(Y_pred_full, axis=1)).astype(np.uint8)

        Y_pred_map = np.reshape(y_pred_full, gt.shape) + 1

        name = "CV_MsAtViT_Full"
        sio.savemat(name + ".mat", {name: Y_pred_map})

        gt_binary = gt.copy()
        gt_binary[gt_binary > 0] = 1
        new_map = Y_pred_map * gt_binary

        name = "CV_MsAtViT"
        sio.savemat(name + ".mat", {name: new_map})

    # Save weights for downstream visualization
    weights_path = os.path.join(ckpt_dir, f"CV_MsAtViT_{dataset_tag}_weights.h5")
    model.save_weights(weights_path)
    print("Weights saved to", weights_path)

    saved_model_dir = os.path.join(ckpt_dir, f"CV_MsAtViT_{dataset_tag}_saved_model")
    model.save(saved_model_dir, include_optimizer=False)
    print("SavedModel exported to", saved_model_dir)


if __name__ == "__main__":
    main()
