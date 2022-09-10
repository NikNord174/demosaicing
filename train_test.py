import tensorflow as tf

from losses import MSE

tf.config.run_functions_eagerly(True)


train_loss = MSE
val_loss = MSE


@tf.function
def train_step(x, y, model, optimizer) -> float:
    with tf.GradientTape() as tape:
        outputs = model(x, training=True)
        loss = MSE(y, outputs)
    grads = tape.gradient(loss, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    train_loss.update_state(y, outputs)
    return loss


@tf.function
def test_step(x, y, model):
    outputs = model(x, training=False)
    val_loss.update_state(y, outputs)


'''def train_step(x, y, model, optimizer):
    with tf.GradientTape() as tape:
        outputs = model(x, training=True)
        loss = (tf.reduce_mean(SSIM(y, outputs, max_val=1)) + MSE(y, outputs))
    grads = tape.gradient(loss, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    return loss


def test_step(x, y, model):
    outputs = model(x, training=False)
    return MSE(y, outputs) + SSIM(y, outputs)'''
