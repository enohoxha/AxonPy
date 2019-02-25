from playgrounds.opencv import face_detection
from playgrounds.keras_models.features.dog_cat import detect_dog_cat
from playgrounds.keras_models.features.multi_dector import featureDetector
from playgrounds.keras_models.features.face_recognition import face_recognition_detector
from playgrounds.keras_models.features.girl_boy import girl_boy_feature


env = {

    "keras": {
        "features": {
            "cat_dog": detect_dog_cat.DogCatFeature(),
            "person": featureDetector.FeatureDetector(),
            "facerec": face_recognition_detector.FaceRecognition(),
            "boy_grirl": girl_boy_feature.GenderClassifier()
        }
    },

    "tensorflow": {},

    "opencv": {

        "features": {

            "face-detect": {
                "w1": face_detection
            }

        }

    }

}


