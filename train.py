import tensorflow as tf
import dataset, config

input_tensors = tf.data.Dataset.from_generator(dataset.load(config.COCO_FILES), (tf.uint8, tf.uint8, tf.int32))
                               .batch(BATCH_SIZE)
                               .prefetch(PREFETCH_SIZE)
                               .make_one_shot_iterator()
                               .get_next()


model = get_personlab(train=True, input_tensors=input_tensors, with_preprocess_lambda=True)
callbacks = [LambdaCallback(on_epoch_end=save_model)]

# The paper uses SGD optimizer with lr=0.0001
model.compile(target_tensors=None, loss=None, optimizer=Adam(), metrics=[identity_metric])
model.fit(steps_per_epoch=64115//batch_size,
                                epochs=config.NUM_EPOCHS, callbacks=callbacks)
