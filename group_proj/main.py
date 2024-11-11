import cv2
import streamlit as st
from ultralytics import YOLO

# Streamlit 设置
st.set_page_config(page_title="Group Assignment", page_icon="🧠", layout="wide")
st.title("Group Assignment")

# 加载 YOLOv8 模型
model = YOLO('./fruit3.pt').to('cuda')  # 可以根据需要加载不同的模型

st.session_state["camera_id"] = 0

# 获取可用摄像头数量
camera_count = 0

for i in range(4):
    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
    if cap.isOpened():
        camera_count += 1
    cap.release()

# 选择摄像头
st.session_state["camera_id"] = st.selectbox("选择摄像头", list(range(camera_count)))

cap = cv2.VideoCapture(st.session_state["camera_id"], cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)


# 设置框架布局
frame_window = st.image([])

# 主循环
if not cap.isOpened():
    st.error("无法读取摄像头")
else:
    while True:
        # ret, frame = cap.read()
        ret, frame = cap.read()
        frame = frame[:, 0:1280, :]
        if not ret:
            st.error("无法从摄像头获取帧")
            break

        # 应用 YOLOv8 进行检测
        results = model.predict(frame, conf=0.6, iou=0.5)

        # 绘制检测结果
        for result in results:
            if result.boxes is not None:
                frame = result.plot(img=frame)
                # for i, xyxy in enumerate(result.boxes.xyxy):
                #     x1, y1, x2, y2 = map(int, xyxy)
                #     frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                #     frame = cv2.putText(
                #         frame,
                #         f"{result.names[int(result.boxes.cls[i])]}: {result.boxes.conf[i]:.2f}",
                #         (x1, y1 - 10),
                #         cv2.FONT_HERSHEY_SIMPLEX,
                #         0.5,
                #         (0, 255, 0),
                #         2
                #     )

        # OpenCV 是 BGR，Streamlit 需要 RGB，因此需要转换颜色格式
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 显示帧
        frame_window.image(frame, channels='RGB')

# 释放资源
cap.release()
