# common.py
def _orig_key(uid: int, img_id: int) -> str:
    """
    Redis에서 사용되는 고유한 키 값을 생성
    사용자마다 고유한 key를 사용하도록 설계
    """
    return f"user:{uid}:img:{img_id}"
