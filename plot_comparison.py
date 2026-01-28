import re
import matplotlib.pyplot as plt
import os

# Data extracted from report_20260126.md
prior_log = """
Epoch 1/100
63/63 [==============================] - 20s 84ms/step - loss: 1.3351 - accuracy: 0.7725
Epoch 2/100
63/63 [==============================] - 5s 79ms/step - loss: 0.3165 - accuracy: 0.9033
Epoch 3/100
63/63 [==============================] - 5s 80ms/step - loss: 0.2334 - accuracy: 0.9251
Epoch 4/100
63/63 [==============================] - 5s 80ms/step - loss: 0.1775 - accuracy: 0.9429
Epoch 5/100
63/63 [==============================] - 5s 81ms/step - loss: 0.1629 - accuracy: 0.9468
Epoch 6/100
63/63 [==============================] - 5s 83ms/step - loss: 0.1235 - accuracy: 0.9591
Epoch 7/100
63/63 [==============================] - 5s 83ms/step - loss: 0.1171 - accuracy: 0.9615
Epoch 8/100
63/63 [==============================] - 5s 82ms/step - loss: 0.1064 - accuracy: 0.9670
Epoch 9/100
63/63 [==============================] - 5s 84ms/step - loss: 0.1359 - accuracy: 0.9570
Epoch 10/100
63/63 [==============================] - 5s 82ms/step - loss: 0.0948 - accuracy: 0.9680
Epoch 11/100
63/63 [==============================] - 5s 81ms/step - loss: 0.0813 - accuracy: 0.9728
Epoch 12/100
63/63 [==============================] - 5s 80ms/step - loss: 0.1017 - accuracy: 0.9653
Epoch 13/100
63/63 [==============================] - 5s 82ms/step - loss: 0.0881 - accuracy: 0.9705
Epoch 14/100
63/63 [==============================] - 5s 82ms/step - loss: 0.1358 - accuracy: 0.9620
Epoch 15/100
63/63 [==============================] - 5s 83ms/step - loss: 0.0909 - accuracy: 0.9700
Epoch 16/100
63/63 [==============================] - 5s 85ms/step - loss: 0.0670 - accuracy: 0.9764
Epoch 17/100
63/63 [==============================] - 5s 84ms/step - loss: 0.0810 - accuracy: 0.9687
Epoch 18/100
63/63 [==============================] - 6s 87ms/step - loss: 0.0628 - accuracy: 0.9762
Epoch 19/100
63/63 [==============================] - 5s 86ms/step - loss: 0.0766 - accuracy: 0.9768
Epoch 20/100
63/63 [==============================] - 5s 83ms/step - loss: 0.0598 - accuracy: 0.9794
Epoch 21/100
63/63 [==============================] - 5s 86ms/step - loss: 0.0431 - accuracy: 0.9857
Epoch 22/100
63/63 [==============================] - 5s 86ms/step - loss: 0.0614 - accuracy: 0.9794
Epoch 23/100
63/63 [==============================] - 5s 83ms/step - loss: 0.0532 - accuracy: 0.9850
Epoch 24/100
63/63 [==============================] - 5s 85ms/step - loss: 0.0505 - accuracy: 0.9840
Epoch 25/100
63/63 [==============================] - 6s 88ms/step - loss: 0.0497 - accuracy: 0.9832
Epoch 26/100
63/63 [==============================] - 5s 84ms/step - loss: 0.0389 - accuracy: 0.9863
Epoch 27/100
63/63 [==============================] - 5s 84ms/step - loss: 0.0549 - accuracy: 0.9801
Epoch 28/100
63/63 [==============================] - 5s 84ms/step - loss: 0.0288 - accuracy: 0.9900
Epoch 29/100
63/63 [==============================] - 5s 82ms/step - loss: 0.0457 - accuracy: 0.9868
Epoch 30/100
63/63 [==============================] - 5s 86ms/step - loss: 0.0713 - accuracy: 0.9792
Epoch 31/100
63/63 [==============================] - 5s 84ms/step - loss: 0.0323 - accuracy: 0.9897
Epoch 32/100
63/63 [==============================] - 5s 86ms/step - loss: 0.0373 - accuracy: 0.9893
Epoch 33/100
63/63 [==============================] - 5s 86ms/step - loss: 0.0585 - accuracy: 0.9829
Epoch 34/100
63/63 [==============================] - 5s 86ms/step - loss: 0.0472 - accuracy: 0.9848
Epoch 35/100
63/63 [==============================] - 5s 84ms/step - loss: 0.0459 - accuracy: 0.9863
Epoch 36/100
63/63 [==============================] - 5s 85ms/step - loss: 0.0394 - accuracy: 0.9874
Epoch 37/100
63/63 [==============================] - 5s 83ms/step - loss: 0.0291 - accuracy: 0.9897
Epoch 38/100
63/63 [==============================] - 5s 86ms/step - loss: 0.0233 - accuracy: 0.9918
Epoch 39/100
63/63 [==============================] - 5s 83ms/step - loss: 0.0423 - accuracy: 0.9882
Epoch 40/100
63/63 [==============================] - 5s 86ms/step - loss: 0.0380 - accuracy: 0.9893
Epoch 41/100
63/63 [==============================] - 5s 85ms/step - loss: 0.0247 - accuracy: 0.9925
Epoch 42/100
63/63 [==============================] - 5s 85ms/step - loss: 0.0225 - accuracy: 0.9933
Epoch 43/100
63/63 [==============================] - 5s 83ms/step - loss: 0.0233 - accuracy: 0.9924
Epoch 44/100
63/63 [==============================] - 5s 85ms/step - loss: 0.0267 - accuracy: 0.9925
Epoch 45/100
63/63 [==============================] - 5s 84ms/step - loss: 0.0275 - accuracy: 0.9918
Epoch 46/100
63/63 [==============================] - 5s 84ms/step - loss: 0.0345 - accuracy: 0.9893
Epoch 47/100
63/63 [==============================] - 5s 87ms/step - loss: 0.0253 - accuracy: 0.9935
Epoch 48/100
63/63 [==============================] - 5s 85ms/step - loss: 0.0277 - accuracy: 0.9924
Epoch 49/100
63/63 [==============================] - 5s 86ms/step - loss: 0.0210 - accuracy: 0.9945
Epoch 50/100
63/63 [==============================] - 5s 85ms/step - loss: 0.0274 - accuracy: 0.9924
Epoch 51/100
63/63 [==============================] - 5s 83ms/step - loss: 0.0299 - accuracy: 0.9918
Epoch 52/100
63/63 [==============================] - 5s 84ms/step - loss: 0.0364 - accuracy: 0.9888
Epoch 53/100
63/63 [==============================] - 5s 86ms/step - loss: 0.0166 - accuracy: 0.9954
Epoch 54/100
63/63 [==============================] - 5s 84ms/step - loss: 0.0192 - accuracy: 0.9943
Epoch 55/100
63/63 [==============================] - 5s 84ms/step - loss: 0.0259 - accuracy: 0.9936
Epoch 56/100
63/63 [==============================] - 5s 83ms/step - loss: 0.0196 - accuracy: 0.9945
Epoch 57/100
63/63 [==============================] - 5s 87ms/step - loss: 0.0277 - accuracy: 0.9928
Epoch 58/100
63/63 [==============================] - 5s 84ms/step - loss: 0.0642 - accuracy: 0.9834
Epoch 59/100
63/63 [==============================] - 5s 83ms/step - loss: 0.0367 - accuracy: 0.9900
Epoch 60/100
63/63 [==============================] - 5s 84ms/step - loss: 0.0265 - accuracy: 0.9913
Epoch 61/100
63/63 [==============================] - 5s 87ms/step - loss: 0.0324 - accuracy: 0.9923
Epoch 62/100
63/63 [==============================] - 5s 86ms/step - loss: 0.0232 - accuracy: 0.9934
Epoch 63/100
63/63 [==============================] - 5s 83ms/step - loss: 0.0122 - accuracy: 0.9953
"""

ours_log = """
Epoch 1/300
63/63 [==============================] - 25s 102ms/step - loss: 4.5450 - accuracy: 0.8108
Epoch 2/300
63/63 [==============================] - 6s 89ms/step - loss: 1.1821 - accuracy: 0.9056
Epoch 3/300
63/63 [==============================] - 5s 85ms/step - loss: 0.8305 - accuracy: 0.9230
Epoch 4/300
63/63 [==============================] - 5s 87ms/step - loss: 0.6913 - accuracy: 0.9353
Epoch 5/300
63/63 [==============================] - 6s 89ms/step - loss: 0.7239 - accuracy: 0.9399
Epoch 6/300
63/63 [==============================] - 5s 85ms/step - loss: 0.7217 - accuracy: 0.9379
Epoch 7/300
63/63 [==============================] - 6s 88ms/step - loss: 0.6636 - accuracy: 0.9400
Epoch 8/300
63/63 [==============================] - 6s 89ms/step - loss: 0.4754 - accuracy: 0.9505
Epoch 9/300
63/63 [==============================] - 5s 87ms/step - loss: 0.4673 - accuracy: 0.9463
Epoch 10/300
63/63 [==============================] - 5s 86ms/step - loss: 0.2895 - accuracy: 0.9619
Epoch 11/300
63/63 [==============================] - 5s 86ms/step - loss: 0.2267 - accuracy: 0.9646
Epoch 12/300
63/63 [==============================] - 5s 87ms/step - loss: 0.2626 - accuracy: 0.9630
Epoch 13/300
63/63 [==============================] - 5s 86ms/step - loss: 0.2610 - accuracy: 0.9612
Epoch 14/300
63/63 [==============================] - 6s 90ms/step - loss: 0.1845 - accuracy: 0.9708
Epoch 15/300
63/63 [==============================] - 5s 86ms/step - loss: 0.2002 - accuracy: 0.9683
Epoch 16/300
63/63 [==============================] - 5s 86ms/step - loss: 0.2102 - accuracy: 0.9651
Epoch 17/300
63/63 [==============================] - 6s 92ms/step - loss: 0.1702 - accuracy: 0.9718
Epoch 18/300
63/63 [==============================] - 5s 86ms/step - loss: 0.1759 - accuracy: 0.9697
Epoch 19/300
63/63 [==============================] - 5s 85ms/step - loss: 0.1211 - accuracy: 0.9756
Epoch 20/300
63/63 [==============================] - 6s 98ms/step - loss: 0.1352 - accuracy: 0.9751
Epoch 21/300
63/63 [==============================] - 5s 87ms/step - loss: 0.0746 - accuracy: 0.9829
Epoch 22/300
63/63 [==============================] - 5s 87ms/step - loss: 0.0841 - accuracy: 0.9809
Epoch 23/300
63/63 [==============================] - 6s 91ms/step - loss: 0.0866 - accuracy: 0.9792
Epoch 24/300
63/63 [==============================] - 6s 88ms/step - loss: 0.0897 - accuracy: 0.9789
Epoch 25/300
63/63 [==============================] - 5s 86ms/step - loss: 0.0737 - accuracy: 0.9823
Epoch 26/300
63/63 [==============================] - 5s 86ms/step - loss: 0.0573 - accuracy: 0.9850
Epoch 27/300
63/63 [==============================] - 5s 85ms/step - loss: 0.1033 - accuracy: 0.9778
Epoch 28/300
63/63 [==============================] - 5s 87ms/step - loss: 0.0539 - accuracy: 0.9858
Epoch 29/300
63/63 [==============================] - 5s 86ms/step - loss: 0.0524 - accuracy: 0.9874
Epoch 30/300
63/63 [==============================] - 6s 94ms/step - loss: 0.0716 - accuracy: 0.9843
Epoch 31/300
63/63 [==============================] - 5s 86ms/step - loss: 0.0675 - accuracy: 0.9845
Epoch 32/300
63/63 [==============================] - 6s 98ms/step - loss: 0.0484 - accuracy: 0.9879
Epoch 33/300
63/63 [==============================] - 5s 86ms/step - loss: 0.0430 - accuracy: 0.9872
Epoch 34/300
63/63 [==============================] - 6s 89ms/step - loss: 0.0536 - accuracy: 0.9865
Epoch 35/300
63/63 [==============================] - 5s 87ms/step - loss: 0.0764 - accuracy: 0.9811
Epoch 36/300
63/63 [==============================] - 5s 87ms/step - loss: 0.0792 - accuracy: 0.9792
Epoch 37/300
63/63 [==============================] - 6s 87ms/step - loss: 0.0499 - accuracy: 0.9882
Epoch 38/300
63/63 [==============================] - 6s 94ms/step - loss: 0.0209 - accuracy: 0.9933
Epoch 39/300
63/63 [==============================] - 6s 88ms/step - loss: 0.0242 - accuracy: 0.9923
Epoch 40/300
63/63 [==============================] - 5s 87ms/step - loss: 0.0236 - accuracy: 0.9934
Epoch 41/300
63/63 [==============================] - 6s 93ms/step - loss: 0.0401 - accuracy: 0.9889
Epoch 42/300
63/63 [==============================] - 5s 87ms/step - loss: 0.0473 - accuracy: 0.9879
Epoch 43/300
63/63 [==============================] - 6s 87ms/step - loss: 0.0642 - accuracy: 0.9854
Epoch 44/300
63/63 [==============================] - 5s 86ms/step - loss: 0.0515 - accuracy: 0.9864
Epoch 45/300
63/63 [==============================] - 5s 86ms/step - loss: 0.0396 - accuracy: 0.9884
Epoch 46/300
63/63 [==============================] - 6s 88ms/step - loss: 0.0382 - accuracy: 0.9889
Epoch 47/300
63/63 [==============================] - 5s 85ms/step - loss: 0.0366 - accuracy: 0.9902
Epoch 48/300
63/63 [==============================] - 5s 84ms/step - loss: 0.0546 - accuracy: 0.9877
Epoch 49/300
63/63 [==============================] - 6s 90ms/step - loss: 0.0296 - accuracy: 0.9916
Epoch 50/300
63/63 [==============================] - 5s 85ms/step - loss: 0.0231 - accuracy: 0.9925
"""

def parse_log(log_text):
    loss_pattern = re.compile(r"loss: ([\d.]+) - accuracy: ([\d.]+)")
    losses = []
    accuracies = []
    lines = log_text.strip().split('\n')
    for line in lines:
        match = loss_pattern.search(line)
        if match:
            losses.append(float(match.group(1)))
            accuracies.append(float(match.group(2)))
    return losses, accuracies

def plot_curves(losses, accuracies, title, filename, max_epochs=None):
    if max_epochs:
        losses = losses[:max_epochs]
        accuracies = accuracies[:max_epochs]
        
    epochs = range(1, len(losses) + 1)
    
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, losses, label="loss")
    plt.plot(epochs, accuracies, label="accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title(title)
    plt.legend()
    plt.ylim(0, 1.1)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    os.makedirs("figs", exist_ok=True)
    plt.savefig(filename, dpi=200)
    plt.close()
    print(f"Saved {filename}")

# Parse Data
prior_loss, prior_acc = parse_log(prior_log)
ours_loss, ours_acc = parse_log(ours_log)

# Plot Prior (Limit to matching length or full? Log seems to go to 63)
plot_curves(prior_loss, prior_acc, "Training Curve (Default/Prior)", "figs/training_curve_prior.png")

# Plot Ours (Log seems to go to 50)
plot_curves(ours_loss, ours_acc, "Training Curve (Ours/ComplexLN)", "figs/training_curve_ours_complex.png")
