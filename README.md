# Axonpy


Axonpy is a python app that through terminal commands can perform some deeplearning tasks.

* ### Getting started
   
  Because size of datasets and pretrained models are to large to upload on git, i have upload them on
  google drive so if you want to run axonpy in your pc first download models from:
  
  [Pre Trained models](https://drive.google.com/open?id=1qaFHtU_m_3XmQv3fOW-GEl-OIbSvNAc9)
  
  <img src="https://lh3.googleusercontent.com/pIWjo26PxE-X7il6OFAOeToDNEHWNFjdSxxq5aM6XlgXJXVgiiMofhMggQu_Foi59Eb5HHTsjkcjfg=w1853-h950" alt="drawing" width="400" height="200"/>  
  
  After you have download the weights please put them in the same directory hierarchy as env and playgrounds 
  
  I used pre trained models only for facenet network and yolo v3.
  
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

  Here we have some simple output from axonpy you can find all on output folder
  
  <img src="https://raw.githubusercontent.com/enohoxha/Axonpy/master/outputs/tiny_yolo/car1.jpg" alt="drawing" width="300" height="300"/>
  <img src="https://raw.githubusercontent.com/enohoxha/Axonpy/master/outputs/tiny_yolo/d1.jpg" alt="drawing" width="300" height="300"/>
  <img src="https://raw.githubusercontent.com/enohoxha/Axonpy/master/outputs/cat_dog/DSC_0645.JPG" alt="drawing" width="300" height="300"/>

  
  
  
    
  
