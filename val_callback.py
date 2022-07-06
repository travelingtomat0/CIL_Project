class TrueValidationScore(keras.callbacks.Callback):
  def __init__(self, mat_total, mat_validation):
    super().__init__()
    self.mat_total = mat_total
    self.mat_validation = mat_validation
    self.ids = np.asarray([np.arange(10000)]).T

  def on_epoch_end(self, epoch, logs=None):
    pred = self.model.predict([self.mat_total, self.ids])
    mask = mat_validation > 0
    rmse = np.sqrt(np.mean(np.square(pred[mask] - mat_validation[mask])))
    print(f"epoch {epoch} - val_score: {rmse:.4f}")
  
