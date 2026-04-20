import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import numpy as np
import csv
import os
import pandas as pd

def load_data(*, dtype=torch.float32):
    """Load forest cover type dataset, normalize, and split into train/test."""
    csv_path = os.path.join(os.path.dirname(__file__), "clinvar_parsed.csv")
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    encoder = OneHotEncoder(sparse_output=False)

    x1 = encoder.fit_transform([[r["CHROM"]] for r in rows])
    x2 = np.array([int(r["POS"]) for r in rows], dtype=np.float32).reshape(-1,1)
    x2 = StandardScaler().fit_transform(x2) # normalize raw genomic postions (very large)
    x3 = encoder.fit_transform([[r["REF"]] for r in rows])
    x4 = encoder.fit_transform([[r["ALT"]] for r in rows])

    print(x1.shape, x2.shape, x3.shape, x4.shape)
    X = torch.tensor(np.column_stack([x1, x2, x3, x4]), dtype=dtype)
    label_map = {
        "Pathogenic": 1,
        "Likely_pathogenic": 1,
        "Benign": 0,
        "Likely_benign": 0
    }

    clnsig = [r['CLNSIG'] for r in rows]
    y_raw = pd.Series(clnsig).map(label_map)
    valid = y_raw.notna()

    y = torch.tensor(y_raw)
    num_classes = len(y.unique())  # number of unique CLNSIG values
    print(num_classes)  # check this
    return X, y, X.shape, num_classes

if __name__ == "__main__":
    X, y, shape, num_classes = load_data()
    n_features = shape[1]

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    print(X_train[:,5])
    # =========================================================================
    # Checkpoint A: Build and train a classifier
    # =========================================================================
    # Build a one hidden layer MLP that outputs 7 class scores.
    # Input: 54 features, Hidden: 128 neurons, Output: 7 classes
    # Use nn.CrossEntropyLoss and torch.optim.Adam (lr=0.001).
    # Train for 200 epochs. Print accuracy every 50 epochs.
    #
    # To compute accuracy:
    #   preds = logits.argmax(dim=1)
    #   acc = (preds == labels).float().mean()

    # TODO: define model, loss function, and optimizer
    # model = nn.Sequential(
    #     nn.Linear(n_features, 280),
    #     nn.ReLU(),
    #     nn.Linear(280, 640),
    #     nn.ReLU(),
    #     nn.Linear(640, num_classes)
    # )
    # loss_fn = nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # print("Begin training")
    # print("="*32)
    # # TODO: training loop
    # for epoch in range(100):
    #     logits = model(X_train)
    #     loss = loss_fn(logits, y_train)
    #     optimizer.zero_grad()
    #     loss.backward()
    #     optimizer.step()

    #     logits_test = model(X_test)
    #     loss_test = loss_fn(logits, y_test)

    #     if epoch % 10 == 0:
    #         print("Epoch:  ", epoch, "   -   Train Loss:  ", loss, "   -   Test Loss:  ", loss)

    # print("Checkpoint A:")
    # with torch.no_grad():
    #     train_acc = (model(X_train).argmax(dim=1) == y_train).float().mean()
    #     test_acc = (model(X_test).argmax(dim=1) == y_test).float().mean()
    #     print(f"  Train accuracy: {train_acc:.2%}")
    #     print(f"  Test accuracy:  {test_acc:.2%}")
    # print()

    # =========================================================================
    # Checkpoint B: Overfit on purpose
    # =========================================================================
    # Build a much larger model (3 hidden layers, 256 neurons each).
    # Architecture: 54 -> 256 -> 256 -> 256 -> 7
    # Train for 300 epochs. Track train and test loss each epoch.
    # Print final train vs. test accuracy. You should see overfitting.

    # TODO: define oversized model, loss function, and optimizer
    # model = nn.Sequential(
    #     nn.Linear(54,256),
    #     nn.ReLU(),
    #     nn.Linear(256,256),
    #     nn.ReLU(),
    #     nn.Linear(256,256),
    #     nn.ReLU(),
    #     nn.Linear(256,7)
    # )
    # loss_fn = nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # # TODO: training loop (store train_losses and test_losses lists)
    # for epoch in range(300):
    #     logits = model(X_train)
    #     loss = loss_fn(logits, y_train)
    #     optimizer.zero_grad()
    #     loss.backward()
    #     optimizer.step()

    # print("Checkpoint B:")
    # with torch.no_grad():
    #     train_acc = (model(X_train).argmax(dim=1) == y_train).float().mean()
    #     test_acc = (model(X_test).argmax(dim=1) == y_test).float().mean()
    #     print(f"  Train accuracy: {train_acc:.2%}")
    #     print(f"  Test accuracy:  {test_acc:.2%}")
    # print()

    # # =========================================================================
    # # Checkpoint C: Add dropout
    # # =========================================================================
    # # Rebuild the oversized model with nn.Dropout(0.3) after each ReLU.
    # # Train the same way (300 epochs). Compare test accuracy to Checkpoint B.
    # # Dropout should reduce overfitting and improve generalization.
    # #
    # # Remember: model.train() before training, model.eval() before evaluation.

    # # TODO: define model with dropout, loss function, and optimizer
    model = nn.Sequential(
        nn.Linear(n_features,128),
        nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(128,64),
        nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(64,num_classes)
    )

    # Weighted loss to account for Benign : Pathogenic imbalance
    class_counts = torch.bincount(y_train)
    weights = 1.0 / class_counts.float()
    weights = weights / weights.sum()
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # TODO: training loop (store train_losses and test_losses lists)
    for epoch in range(200):
        logits = model(X_train)
        loss = loss_fn(logits, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        logits_test = model(X_test)
        loss_test = loss_fn(logits_test, y_test)

        if epoch % 50 == 0:
            print(epoch, "   -   Train Loss:  ", loss, "   -   Test Loss:  ", loss_test)

    print("Checkpoint C:")
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    import matplotlib.pyplot as plt

    with torch.no_grad():
        model.eval()
        train_acc = (model(X_train).argmax(dim=1) == y_train).float().mean()
        test_acc = (model(X_test).argmax(dim=1) == y_test).float().mean()

        print(f"  Train accuracy: {train_acc:.2%}")
        print(f"  Test accuracy:  {test_acc:.2%}")

        preds = model(X_test).argmax(dim=1).numpy()
        y_true = y_test.numpy()

        cm = confusion_matrix(y_true, preds)
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["Benign", "Pathogenic"]
        )
        disp.plot(cmap="Blues")
        plt.title("Confusion Matrix - Pathogenicity Classifier")
        plt.savefig("confusion_matrix.png", dpi=150, bbox_inches="tight")
        plt.show()

        # tp = ((preds == 1) & (y_true == 1)).sum().item() 
        # tn = ((preds == 0) & (y_true == 0)).sum().item()      
        # fp = ((preds == 1) & (y_true == 0)).sum().item() 
        # fn = ((preds == 0) & (y_true == 1)).sum().item() 
        
        # precision = tp / (tp + fp + 1e-8)
        # recall    = tp / (tp + fn + 1e-8)
        # f1        = 2 * precision * recall / (precision + recall + 1e-8)
        
        # print(f"Precision: {precision:.2%}")
        # print(f"Recall:    {recall:.2%}")   # sensitivity for pathogenic
        # print(f"F1 Score:  {f1:.2%}")
        # print(f"TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}")
