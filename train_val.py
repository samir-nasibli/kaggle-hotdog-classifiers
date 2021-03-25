import tensorflow as tf
from keras.callbacks import TensorBoard
from tf.keras.callbacks import LearningRateScheduler


def tensorboard_callback(logs_dir="logs"):
    """
    # Create TensorBoard logs
    """
    tensorboard = TensorBoard(log_dir=logs_dir, histogram_freq=0, write_graph=True)
    return tensorboard


def scheduler(epoch):
  if epoch < 10:
    return 0.001
  else:
    return 0.001 * tf.math.exp(0.1 * (10 - epoch))


def train_CNN(model, logs_dir, train_generator, validation_generator,
              training_samples_number, testing_samples_number, batch_size, epochs):
    """
    For training simple CNN
    """
    lr_callback = LearningRateScheduler(scheduler)
    tensorboard = tensorboard_callback(logs_dir)

    model.compile(
    loss='binary_crossentropy',
    optimizer='rmsprop',
    metrics=['accuracy']
    )
    # Train the model with data from our generators
    model.fit_generator(
        train_generator,
        steps_per_epoch=training_samples_number // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=testing_samples_number // batch_size,
        verbose=1,
        callbacks=[tensorboard, lr_callback]
    )


def evaluate_CNN(model, validation_generator, testing_samples_number):
    """
    For evaluation simple CNN
    """
    # Print loss rate and accuracy
    error_rate = model.evaluate_generator(validation_generator,
                                          testing_samples_number)
    print("The model's loss rate is {0:0.2} (binary crossentropy) and accuracy is {0:.2%}"
          .format(error_rate[0], error_rate[1]))
