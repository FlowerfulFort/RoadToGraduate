## 모델 구조

Resnet34를 기반으로 하는 Unet

Resnet34: X -> 64 -> 128 -> 256 -> 512

pytorch_zoo/resnet.py, pytorch_zoo/unet.py 참조.

## 모델 학습

1. other_tools/gen_folds.py를 이용하여 현재 dataset에 맞는 folds.csv를 생성.
2. dataset의 경로를 설정.
    - /path/to/dataset/images: 학습에 필요한 이미지셋
    - /path/to/dataset/masks2m: 학습에 필요한 도로 마스크 이미지셋
3. resnet34_412_02_02.json에서 경로, batch_size 등을 수정.
4. train_eval.py 에서 num_workers를 조정.(메모리 오류가 나지 않는 선에서)
5. train_eval.py resnet34_412_02_02.json --training

## Prediction

```python
# src/pytorch_utils/eval.py 참조.
def predict(model, batch, flips=flip.FLIP_NONE):
    # predict with tta on gpu
    pred1 = F.sigmoid(model(batch))
    if flips > flip.FLIP_NONE:
        pred2 = flip_tensor_lr(model(flip_tensor_lr(batch)))
        masks = [pred1, pred2]
        if flips > flip.FLIP_LR:
            pred3 = flip_tensor_ud(model(flip_tensor_ud(batch)))
            pred4 = flip_tensor_ud(flip_tensor_lr(model(flip_tensor_ud(flip_tensor_lr(batch)))))
            masks.extend([pred3, pred4])
        masks = list(map(F.sigmoid, masks))
        new_mask = torch.mean(torch.stack(masks, 0), 0)
        return to_numpy(new_mask)
    return to_numpy(pred1)
```

flip_tensor_XX 메소드로 상하좌우 뒤집어 가며 prediction, 이들의 평균치로 새로운 마스크를 만들어 내어 저장함.

```python
# src/train_eval.py
keval = FullImageEvaluator(config, ds, test=test, flips=3, num_workers=num_workers, border=22)
```

flip 기본값은 3이기 때문에 항상 if를 통과하여 상하좌우 플립을 실행함.

```python
class RawImageTypePad(RawImageType):
    def finalyze(self, data):
        return self.reflect_border(data, 22)

def eval_roads():
    global config
    rows, cols = 1344, 1344
    config = update_config(config, target_rows=rows, target_cols=cols)
    ds = ReadingImageProvider(RawImageTypePad, paths, fn_mapping, image_suffix=image_suffix)

# ...
```

Evaluation 과정에서 상하좌우 22 만큼 reflect border를 만듬.

![image](https://github.com/FlowerfulFort/RoadToGraduate/assets/42996160/7d9188c3-0f7b-4a77-8d42-864fa57dde28)

모델의 예측값의 크기가 44만큼 깎여있는 것과 관련되어 있다고 생각됨.

## 최신 python, torch과 현재 학습하는 데이터셋에 맞춰 코드 변경

> https://github.com/FlowerfulFort/RoadToGraduate/commit/a72919b275d1c74411cb82b0bd9facea6270c6a2

1. scipy.misc.imread -> Deprecated

    imageio.imread로 대체.

2. cuda(async=True) -> SyntaxError: Invalid syntax

    python3.7 이후 생기는 문제. cuda(non_blocking=True)로 대체.

3. tensor.cpu().numpy()[0] -> scalar 오류

    tensor.item() 으로 대체.

4. train_eval.py 내의 image_suffix를 None으로 변경.

## Origin source from albu

> https://github.com/SpaceNetChallenge/RoadDetector/tree/master/albu-solution
