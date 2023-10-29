## 해야할 것

### AI 모델

0. 모델 아웃풋 사이즈를 512x512에 맞게 재조정.

1. 통상적인 이미지를 잘라서 모델에 넣고 나온 마스크를 다시 이어붙이는 루틴(3시간이면 가능하다고 함).

2. 도로 탐지 평가 메트릭을 찾기(보통 semantic segmentation에 사용하는 IoU는 적합하지 않다).

3. Augmentation을 하여 성능 올리기(위에서 찾은 성능 지표를 사용해 평가).

4. 모델 학습 파라미터를 수정 또는 여러 모델을 앙상블 또는 ... 하여 성능 올리기.

5. postprocessing(우선순위 낮음)

---------------------

## 명령어

training

```bash
$ python train_eval.py resnet34_512_02_02.json --training
```

evaluation

```bash
$ python train_eval.py resnet34_512_02_02.json
```

--------------------

## 폴더

<h5>weight</h5>

/results/weights/2m_4fold_512_30e_d0.2_g0.2

<h5>train data set</h5>

/opt/datasets/train

