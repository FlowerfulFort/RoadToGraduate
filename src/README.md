## 모델 구조

Resnet34를 기반으로 하는 Unet

512 -> 256 -> 128 -> 64 -> 64 -> 128 -> 256 -> 512

pytorch_zoo/resnet.py, pytorch_zoo/unet.py 참조.

## 모델 학습

1. other_tools/gen_folds.py를 이용하여 현재 dataset에 맞는 folds.csv를 생성.
2. dataset의 경로를 설정.
   - /path/to/dataset/images: 학습에 필요한 이미지셋
   - /path/to/dataset/masks2m: 학습에 필요한 도로 마스크 이미지셋
3. resnet34_412_02_02.json에서 경로, batch_size 등을 수정.
4. train_eval.py 에서 num_workers를 조정.(메모리 오류가 나지 않는 선에서)
5. train_eval.py resnet34_412_02_02.json --training

## 최신 python, torch과 현재 학습하는 데이터셋에 맞춰 코드 변경

> https://github.com/FlowerfulFort/RoadToGraduate/commit/a72919b275d1c74411cb82b0bd9facea6270c6a2

1. scipy.misc.imread -> Deprecated

    imageio.imread로 대체.

2. cuda(async=True) -> SyntaxError: Invalid syntax
    
    python3.7 이후 생기는 문제. cuda(non_blocking=True)로 대체.

3. tensor.cpu().numpy()[0] -> scalar 오류

    tensor.item() 으로 대체.

4. train_eval.py 내의 image_suffix를 None으로 변경.
