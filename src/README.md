# 알려진 의존성 오류

1. scipy.misc.imread -> Deprecated

    imageio.imread로 대체.

2. cuda(async=True) -> SyntaxError: Invalid syntax
    
    python3.7 이후 생기는 문제. cuda(non_blocking=True)로 대체.
