# Linux 流媒体服务器搭建与 RTSP/RTMP 推拉流实战

在物联网 IoT、智慧工业视觉与 AI 算法落地中，经常需要搭建流媒体服务器以接收 IPC（网络摄像机）视频流或向算法管线分发 RTSP/RTMP 视频流。

---

## 1. 方案 A：MediaMTX 流媒体服务器一键部署 (推荐)

[MediaMTX](https://github.com/bluenviron/mediamtx)（原名 `rtsp-simple-server`）是目前最流行、轻量且零外部依赖的实时流媒体服务器，原生支持 **RTSP、RTMP、HLS、WebRTC、SRT** 多协议无缝转推与分发。

### 1.1 使用 Docker 容器化一键部署
```bash
# 启动 MediaMTX 流媒体服务 (映射 RTSP 8554, RTMP 1935, HLS 8888, WebRTC 8889 端口)
docker run -d --name mediamtx \
  --restart always \
  -p 8554:8554 \
  -p 1935:1935 \
  -p 8888:8888 \
  -p 8889:8889 \
  bluenviron/mediamtx:latest
```

### 1.2 使用独立二进制文件运行
```bash
# 下载预编译可执行文件 (以 Linux amd64 为例)
wget https://github.com/bluenviron/mediamtx/releases/download/v1.9.0/mediamtx_v1.9.0_linux_amd64.tar.gz
tar -zxvf mediamtx_v1.9.0_linux_amd64.tar.gz

# 启动服务 (默认自动读取 mediamtx.yml 配置文件)
./mediamtx
```

---

## 2. 方案 B：Ubuntu 源码编译构建 x264 与 FFmpeg

适用于需要深度定制编译参数或在特定 Linux 发行版上构建静态二进制的场景：

### 2.1 安装基础编译工具链
```bash
sudo apt update
sudo apt install -y build-essential cmake git pkg-config yasm nasm libtool
```

### 2.2 编译安装 x264 (H.264 视频编码库)
```bash
git clone https://code.videolan.org/videolan/x264.git
cd x264
./configure --prefix=/usr/local/x264 --enable-shared --enable-static --disable-asm
make -j$(nproc)
sudo make install
```

### 2.3 编译安装 FFmpeg (集成 libx264)
```bash
wget https://www.ffmpeg.org/releases/ffmpeg-5.1.4.tar.gz
tar -zxvf ffmpeg-5.1.4.tar.gz
cd ffmpeg-5.1.4

./configure --prefix=/usr/local/ffmpeg \
  --enable-shared \
  --enable-gpl \
  --enable-libx264 \
  --enable-pthreads \
  --extra-cflags="-I/usr/local/x264/include" \
  --extra-ldflags="-L/usr/local/x264/lib"

make -j$(nproc)
sudo make install
```

### 2.4 配置动态库路径与环境变量
```bash
# 添加动态链接库路径
echo "/usr/local/x264/lib" | sudo tee -a /etc/ld.so.conf.d/custom_libs.conf
echo "/usr/local/ffmpeg/lib" | sudo tee -a /etc/ld.so.conf.d/custom_libs.conf
sudo ldconfig

# 添加 PATH 环境变量
echo 'export PATH=/usr/local/ffmpeg/bin:$PATH' >> ~/.bashrc
source ~/.bashrc

# 验证安装
ffmpeg -version
```

---

## 3. 端到端推流与拉流实战验证

### 3.1 FFmpeg 循环推流到 RTSP 服务
```bash
# -re 模拟实时速率，-stream_loop -1 无限循环，-rtsp_transport tcp 保证传输不丢包
ffmpeg -re -stream_loop -1 -i test.mp4 -c:v copy -an -f rtsp -rtsp_transport tcp rtsp://localhost:8554/live/camera1
```

### 3.2 客户端拉流验证

#### 方式 1：使用 FFplay / VLC 播放器测试
```bash
ffplay -rtsp_transport tcp rtsp://127.0.0.1:8554/live/camera1
```

#### 方式 2：使用 Python OpenCV 读取实时视频流（AI 算法推理）
```python
import cv2

# 连接 RTSP 视频流 (指定 TCP 传输防止 UDP 丢包花屏)
rtsp_url = "rtsp://127.0.0.1:8554/live/camera1"
cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("无法接收视频帧 (流已中断)")
        break

    # 在此处接入目标检测/时序动作识别模型 (如 YOLOv8 / RTMPose)
    # results = model(frame)

    cv2.imshow("RTSP Stream", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```
