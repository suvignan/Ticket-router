import keras
from sklearn.metrics import f1_score


class MacroF1(keras.callbacks.Callback):
    """Predict all of val each epoch and log val_macro_f1 (put FIRST in the callbacks list)."""

    def __init__(self, val_ds, y_val):
        super().__init__()
        self.val_ds = val_ds
        self.y_val = y_val

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        preds = self.model.predict(self.val_ds, verbose=0).argmax(axis=1)
        assert len(preds) == len(self.y_val), "Val rows dropped!"
        f1 = f1_score(self.y_val, preds, average="macro")
        logs["val_macro_f1"] = f1
        print(f"Epoch {epoch + 1}: val macro-F1 = {f1:.4f}")
