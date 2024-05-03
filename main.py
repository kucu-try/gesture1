import cv2
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode

def print_result(result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
    # 결과가 None이 아닐 때 결과를 출력합니다.
    if result is not None:
        print('제스처 인식 결과: {}'.format(result.gestures))
    else:
        # 제스처가 인식되지 않으면 기본 메시지를 출력합니다.
        print('인식된 제스처가 없습니다.')

# 제스처 인식기 옵션을 생성합니다.
base_options = python.BaseOptions(model_asset_path='gesture_recognizer.task')
options = vision.GestureRecognizerOptions(
    base_options=base_options,
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result
)

# 제스처 인식기 인스턴스를 생성합니다.
with vision.GestureRecognizer.create_from_options(options) as recognizer:
    # 웹캠을 초기화합니다.
    cap = cv2.VideoCapture(0)
    timestamp = 0

    while cap.isOpened():
        # 프레임을 한 프레임씩 캡처합니다.
        ret, frame = cap.read()
        if not ret:
            print("빈 프레임은 무시됩니다.")
            break
        timestamp += 1

        # 프레임을 RGB 포맷으로 변환합니다.
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=frame_rgb
        )

        # 실시간 이미지 데이터를 보내 제스처 인식을 수행합니다.
        recognizer.recognize_async(mp_image, timestamp)

        # 이미지를 디스플레이합니다.
        cv2.imshow('제스처 인식', frame)
        
        # 'q' 키를 눌러 루프를 종료합니다.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# 웹캠을 해제하고 모든 OpenCV 창을 닫습니다.
cap.release()
cv2.destroyAllWindows()
