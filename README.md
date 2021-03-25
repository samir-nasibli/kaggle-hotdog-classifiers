# hotdog_classifiers
Hot dog image classifiers

This repo is created for using on colab notebook for university project hot_dog/not_hot_dog: https://colab.research.google.com/drive/16aS-kiqgxyGxw3jPp5H9lj6eLRHgHMGa#scrollTo=BlJxzLkbALkp

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
|[svc](](~~)   |    73.53%    | 20   |0.69|
|[xgboost](](~~)   |    79.53%    | 20   |0.69|
|[cnn - 20 epoc 256 batch 224x224 input](](~~)   |    80.91%    | 7   |-|
|[cnn - 100 epoc 32 batch 224x224 input](](~~)   |    71.34%    | 33   |-|
|[cnn - 20 epoc 256 batch 256x256 input](](~~)   |    81.91%    | 10  |-|
|[cnn - 40 epoc 256 batch 256x256 input](](~~)   |    82.66%    | 21   |-|
|[__svm on vgg16 features__](](~~)   |    __98.18%__    | <1   |__0.97__|
|[xgboost_on_vgg16_features](~~)   |    97.28%    | <1   |0.95|
|[VGG16 *](~~)   |    62.03%    | ~53   |-|
|[Fast AI ResNet50 *](~~)   |    62.03%    | ~53   |-|

\* VGG16 training (full; not pretrained on ImageNet) takes long time (12 epochs taken), 64 batch. Accuracy raises very slow and error rate raises on validation after 9 epochs
\* All VGG16 was pre-trained ImageNet.

#### see submissions:
* `submission.scv` for [svc_model_vgg16](test_labels/prediction_svc_model_vgg16/submission.scv) -> []
* `submission.scv` for [xgb_model_vgg16](test_labels/prediction_xgb_model_vgg16/submission.scv) -> []
* `submission.scv` for [fast_ai_resnet50](test_labels/prediction_fast_ai_resnet50/submission.scv) -> []

| Name      | Public score | Private score|
|-----------|---------:|--------:|
|[svc](](~~)   |    45.53%    | 36   |
