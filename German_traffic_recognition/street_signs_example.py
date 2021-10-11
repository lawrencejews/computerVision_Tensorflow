from my_utils import split_data, order_test_set, create_generators
from deeplearning_models import streetsigns_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import tensorflow as tf


if __name__=="__main__":

    # False deactivated the path -> remove to generate data.
    if False:
        path_to_data = "C:\\Users\\Owner\\Desktop\\ComputerVision_Tensorflow\\German_traffic_recognition\\Train"
        path_to_save_train = "C:\\Users\\Owner\\Desktop\\ComputerVision_Tensorflow\\German_traffic_recognition\\training_data\\train"
        path_to_save_val = "C:\\Users\\Owner\\Desktop\\ComputerVision_Tensorflow\\German_traffic_recognition\\training_data\\val"
    
        split_data(path_to_data, path_to_save_train=path_to_save_train, path_to_save_val=path_to_save_val)

    # Moving test data to its corresponding folders 
    if False:
        path_to_images = "C:\\Users\\Owner\\Desktop\\ComputerVision_Tensorflow\\German_traffic_recognition\\Test"
        path_to_csv = "C:\\Users\\Owner\\Desktop\\ComputerVision_Tensorflow\\German_traffic_recognition\\Test.csv"
        order_test_set(path_to_images, path_to_csv)

    
    path_to_train = "C:\\Users\\Owner\\Desktop\\ComputerVision_Tensorflow\\German_traffic_recognition\\training_data\\train"
    path_to_val = "C:\\Users\\Owner\\Desktop\\ComputerVision_Tensorflow\\German_traffic_recognition\\training_data\\val"
    path_to_test = "C:\\Users\\Owner\\Desktop\\ComputerVision_Tensorflow\\German_traffic_recognition\\Test"

    batch_size = 64
    epochs = 12
    lr = 0.0001

    train_generator, val_generator, test_generator = create_generators(batch_size, path_to_train, path_to_val, path_to_test)
    nb_classes = train_generator.num_classes

    TRAIN = False
    TEST = True

    if TRAIN:
        path_to_ckpt_model = "./Models"
        ckpt_saver = ModelCheckpoint(
            path_to_ckpt_model,
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            save_freq="epoch",
            verbose=1
        )

        early_stop = EarlyStopping(
            monitor="val_accuracy",
            patience=10,
        )

        model = streetsigns_model(nb_classes)

        # amsgrad can help adam converge.
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr, amsgrad=False)
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        model.fit(
            train_generator,
            epochs=epochs, 
            batch_size=batch_size,
            validation_data=val_generator,
            callbacks=[ckpt_saver, early_stop]
        )

    if TEST:
        model = tf.keras.models.load_model('./Models')
        model.summary()

        print("Evaluating Validation Set:")
        model.evaluate(val_generator)

        print("Evaluating Test Set: ")
        model.evaluate(test_generator)