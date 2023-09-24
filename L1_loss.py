import tensorflow as tf
# Compatible with tensorflow backend

def l1_loss(rate=0.1):
    def l1_loss_inner(y_true, y_pred):
        return rate * tf.reduce_mean(tf.abs(y_pred))
    return l1_loss_inner
