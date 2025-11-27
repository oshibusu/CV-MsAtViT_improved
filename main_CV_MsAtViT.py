import os
import scipy.io as sio
import numpy as np
from tensorflow import keras
from sklearn.metrics import confusion_matrix, accuracy_score, cohen_kappa_score
from Load_Data import load_data
from SAR_utils import *  # noqa: F401,F403 retains helper utilities
from net_flops import net_flops
from model_factory import build_msatvit


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
    # Configuration (edit as needed)
    dataset = "FL_T"
    window_size = 15
    test_ratio = 0.99

    data, gt = load_data(dataset)
    lr = 0.0001 if dataset == "ober" else 0.001

    data = Standardize_data(data)

    X_coh, y = createImageCubes(data, gt, window_size)
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

    early_stopper = keras.callbacks.EarlyStopping(
        monitor="accuracy", patience=10, restore_best_weights=True
    )

    model.fit(
        X_train,
        y_train,
        batch_size=128,
        verbose=1,
        epochs=100,
        shuffle=True,
        callbacks=[early_stopper],
    )

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
    del X_train, X_test
    X_coh_full, _ = createImageCubes(data, gt, window_size, removeZeroLabels=False)
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
    ckpt_dir = os.path.join("ckpt")
    os.makedirs(ckpt_dir, exist_ok=True)
    weights_path = os.path.join(ckpt_dir, "CV_MsAtViT_weights.h5")
    model.save_weights(weights_path)
    print("Weights saved to", weights_path)

    saved_model_dir = os.path.join(ckpt_dir, "CV_MsAtViT_saved_model")
    model.save(saved_model_dir, include_optimizer=False)
    print("SavedModel exported to", saved_model_dir)


if __name__ == "__main__":
    main()
