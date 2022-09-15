import tensorflow as tf

# tf.config.run_functions_eagerly(True)


def train_step(
    real: tf.Tensor,
    ground: tf.Tensor,
    model: tf.keras.Model,
    loss_function: tf.keras.losses,
    optimizer: tf.keras.optimizer,
    train_loss: tf.keras.metrics
) -> None:
    """Train step for training"""
    with tf.GradientTape() as tape:
        predictions = model(real)
        loss = loss_function(ground, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss(loss)


def val_step(
    real: tf.Tensor,
    ground: tf.Tensor,
    model: tf.keras.Model,
    loss_function: tf.keras.losses,
    val_loss: tf.keras.metrics
) -> None:
    "Test step for training"""
    predictions = model(real)
    v_loss = loss_function(ground, predictions)
    val_loss(v_loss)
