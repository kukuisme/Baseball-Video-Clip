
## 🚀 Steps to Run Code

- **Go to the cloned folder**:
  ```bash
  cd Baseball-Video-Clip
  ```
- **Create a virtual environment** (recommended):
  ```bash
  # Mac 確定local端有python3.10
  python3.10 -m venv psestenv
  source psestenv/bin/activate
  ```
- **Upgrade pip**:
  ```bash
  pip install --upgrade pip
  ```
- **Install requirements**:
  ```bash
  pip install -r requirements.txt
  ```
- **Download YOLOv7 weights** 下載完放/yolov7-pose-estimation 底下 :
  [yolov7-w6-pose.pt](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-w6-pose.pt)

- **Run the code**:
  ```bash
  python pose-estimate.py

  # Options:
  python pose-estimate.py --source "your-video.mp4" --device cpu  # For CPU
  python pose-estimate.py --source 0 --view-img  # For Webcam
  python pose-estimate.py --source "rtsp://your-ip" --device 0 --view-img  # For LiveStream
  ```