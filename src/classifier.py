import cv2
import numpy as np
import tensorflow as tf

class VehicleClassifier:
    def __init__(self, model_path, idx_to_class):
        self.model = tf.keras.models.load_model(model_path)
        self.idx_to_class = idx_to_class

    def predict(self, crop):
        if crop is None or crop.size == 0:
            return "unknown", 0.0
        
        img = cv2.cvtColor(
            crop,
            cv2.COLOR_BGR2RGB
        )

        img = cv2.resize(img, (224, 224))
        img = img.astype("float32") / 255.0
        img = np.expand_dims(img, axis=0)

        preds = self.model.predict(
            img,
            verbose=0
        )[0]

        class_id = np.argmax(preds)
        confidence = float(np.max(preds))

        return(
            self.idx_to_class[class_id],
            confidence
        )
    
