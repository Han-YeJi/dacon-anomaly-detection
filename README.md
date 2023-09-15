# Computer Vision Anomaly Detection Algorithm
## Introduction
This repository is the 3rd place solution for [DACON Anomaly Detection Algorithm contest](https://dacon.io/competitions/official/235894/overview/description). We developed an algorithm that can classify the type and state of objects by learning an imbalanced dataset. The key strategies used are as follows:
1. We consider the type and state of an object as a pair of classes and perform class-wise augmentation.
2. To solve problems caused by class imbalance, the model additionally learned about specific objects is used for hard voting to make the final prediction.

This is the overall model process.

![model process](https://github.com/Han-YeJi/dacon-anomaly-detection/assets/84916071/f49a66ea-6c9e-420d-a6ec-70a57c7ea93f)

## Dataset description
MVTec's Anomaly Detection dataset consists of 15 object types and 49 object states, with a total of 88 labels.

## Main strategy
1. To reduce confusion between state within an object, different augmentation is applied to each object to learn the model. (class-wise augmentation)
2. In addition to the main model, we additionally learn class-specific model for post-processing.
   - To prevent overfitting due to class imbalance, change the state of the object to normal/anomaly and learn ① binary classification model. During inference ①, in the case of a sample classified as normal with low confidence, the predictions of main model is also adjusted to the second highest prediction when it is normal. (Based on F1 score, 0.8548 -> 0.8729)
   - To resolve low confidence in specific objects, we learned class-specific model. ② toothbrush model and ③ zipper model were used as post-processing. (Based on F1 score, 0.8729 -> 0.9087)

## How to Use

1. Install Library
    ```
    pip install -r requirements.txt
    pip install jupyter
    ```
2. Download data.zip from[ https://dacon.io/competitions/official/235870/data](https://dacon.io/competitions/official/235894/data) to data path.
    ```bash
    #./workspace
    mkdir data
    cd data
    (Download data to ./workspace/data/)
    unzip data.zip
    unzip train.zip
    unzip test.zip
    ```
3. Prepare data : You can run this section in this [ipynb](make_df.ipynb)

4. Training
   ```bash
   sh multi_train.sh
   ```
5. Test : You can run this section in [test.py](test.py)

6. Inference & Post-processing : You can run this section in [solution.ipynb](solution.ipynb)

