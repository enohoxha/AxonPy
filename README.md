# Axonpy


Axonpy is a python app that through terminal commands can perform some deeplearning tasks.

* ### Commands and format

  Each of arguments below are required and have a specific role, lets try yo explain them on detail.

    ```python
    parser.add_argument('-env', help='Environments to work with '
                                             'Options: keras, tensorflow, opencv', required=True)
    parser.add_argument('-feature', help="Select Features to execute", required=True)
    parser.add_argument('-worker', help="Select Algorithm to execute", required=True)
    parser.add_argument('-input', help="Select Input to feed to network")
    parser.add_argument('-type', help="Select Input to feed to network")
    parser.add_argument('-ftype', help="Select Input type(video, image) to feed to network")
    
    ```

  So what are env, feature, worker etc lets take a closer look at the config file now
  
  ```python
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

    ```
    
    So we can have command like 
    
    ```
    python axonpy -env keras -feature facerec -worker ?  -input ? -type ? -ftype ?
    ```
    
    I think that env and features are clear to you env is the playground tensorfllow or keras and features are tasks
    you want to do. But we have some question mark on our command lets discuss abouth them .
    
    ```python
    self.workers = {
      "c1": workers.DogCatWorkers().buildModel(),
      "VGG16": workers.DogCatVGG16().buildModel()
    }
    ```
    
    as we can see worker is algorithm type each feature have its own algorithm, you have to check the code for
    that.
    
    `Note: FaceNet weights are removed because too large upload for git.`
    
* ### Outputs from axonpy

  Here we have some simple output from axonpy
  
  ![Image of Yaktocat](https://octodex.github.com/images/yaktocat.png)
  
  
  
    
  
