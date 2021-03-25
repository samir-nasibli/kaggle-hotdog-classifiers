# hotdog_classifiers
Hot dog image classifiers

This repo is created for using on colab notebook for university project hot_dog/not_hot_dog: https://colab.research.google.com/drive/16aS-kiqgxyGxw3jPp5H9lj6eLRHgHMGa#scrollTo=BlJxzLkbALkp

Kaggle competition: https://www.kaggle.com/c/hotdogornot/

## Prerequisites
* Python3.5+
* CUDA 10.1
* tensorflow  2.2.0
* keras 2.3.0+
* numpy 1.18+

## Experiment results

The provided table shows comparative analysys. Test accuracy (0.3 of full train data).

| Name      | Accuracy | Estimated time, mins| AP |
|-----------|---------:|--------:|:-----------------:|
|[svc](https://drive.google.com/file/d/1-QqsA26RcR62whaTMwzqQ-Zy3aYcs65V/view?usp=sharing)   |    76.53%    | 13   |0.64|
|[xgboost](https://drive.google.com/file/d/1-G5apqTLPWQYp_Y0HiFrh--QRaGUl_q2/view?usp=sharing)   |    79.53%    | 20   |0.69|
|[cnn - 20 epoc 256 batch 224x224 input](https://drive.google.com/file/d/1-9vCzsB6e7agh4uzH_G3as-cGVoOlN2u/view?usp=sharing)   |    80.91%    | 7   |-|
|[cnn - 100 epoc 32 batch 224x224 input](https://drive.google.com/file/d/1--UprpUwXQPptJuXOXO2OI0NcOO4cF_x/view?usp=sharing)  |    71.34%    | 33   |-|
|[cnn - 20 epoc 256 batch 256x256 input](https://drive.google.com/file/d/1-4kp9TlJCX1r2lU4knhbaRs8RVozZahR/view?usp=sharing)   |    81.91%    | 10  |-|
|[cnn - 40 epoc 256 batch 256x256 input](https://drive.google.com/file/d/1-9vdJ1ktbvUE2c7IdepxvwbIJOZ29JdZ/view?usp=sharing)   |    82.66%    | 21   |-|
|[__svm on vgg16 features**__](https://drive.google.com/file/d/1-582EMGqDKj1Y1ygRUGLGORiJV-EXH7-/view?usp=sharing)   |    __93.88%__    | <1   |__0.97__|
|[__xgboost_on_vgg16_features**__](~~)   |    __93.71%__    | <1   |0.95|
|[VGG16 *](~~)   |    62.03%    | ~53   |-|
|[__Fast AI ResNet50__]()   |    __94.11%__    | ~9   |-|

\* VGG16 training (full; not pretrained on ImageNet) takes long time (12 epochs taken), 64 batch. Accuracy raises very slow and error rate raises on validation after 9 epochs

\** All VGG16 was pre-trained on ImageNet.

## see submissions:

| Name      | Private score | Public score|
|-----------|---------:|--------:|
|[svc_model_vgg16](test_labels/prediction_svc_model_vgg16/submission.csv)   |    0.91760    | 0.94230   |
|[xgb_model_vgg16](test_labels/prediction_xgb_model_vgg16/submission.csv)   |    0.90151    | 0.92822   |
|[fast_ai_resnet50](test_labels/prediction_fast_ai_resnet50/submission.csv)   |    0.95716    | 0.97115   |

