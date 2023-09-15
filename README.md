# Computer Vision Anomaly Detection Algorithm
## Introduction
This repository is the 3rd place solution for [DACON Anomaly Detection Algorithm contest](https://dacon.io/competitions/official/235894/overview/description). We developed an algorithm that can classify the type and state of objects by learning an imbalanced dataset. The key strategies used are as follows:
1. We consider the type and state of an object as a pair of classes and perform class-wise augmentation.
2. To solve problems caused by class imbalance, the model additionally learned about specific objects is used for hard voting to make the final prediction.

This is the overall model process.

![model process](https://github.com/Han-YeJi/dacon-anomaly-detection/assets/84916071/f49a66ea-6c9e-420d-a6ec-70a57c7ea93f)

## Dataset description


### Directory Structure
```
/workspace
├── data
│   ├── train
│   │    ├── 10001.png
│   │    ├── ...
│   │    └── 14276.png
│   ├── test
│   │    ├── 20001.png
│   │    ├── ...
│   │    └── 22153.png
│   │    
│   ├── train_df.csv
│   ├── test_df.csv
│   └── sample_submission.csv
│
├── dacon-anomaly
│   ├── config.py
│   ├── dataloader.py
│   ├── hardvoting.py
│   ├── main.py
│   ├── make_df.ipynb
│   ├── multi_train.sh
│   ├── network.py
│   ├── prediction_ensemble.ipynb
│   ├── test.py
│   ├── trainer.py
│   ├── files
│   │    ├── effb4_bad_5fold.npy
│   │    ├── softmax_142.npy
│   │    ├── softmax_156.npy
│   │    ├── softmax_pillzip_266.npy
│   │    ├── softmax_pillzip_274.npy
│   │    ├── softmax_sy_123.npy
│   │    ├── softmax_sy_133.npy
│   │    ├── softmax_sy_266.npy
│   │    ├── softmax_sy_272.npy
│   ├── utils
│   │    ├── __init__.py
│   │    ├── image_utils.py
│   │    ├── logger_utils.py
│   │    ├── scheduler_utils.py
```
<br>

## Usage
- `make_df.ipynb`를 통해 state를 good과 bad로만 구분하는 `train_df_bad.csv`, one class만을 저장하는 `{class}_df.csv` 를 생성할 수 있습니다.

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

