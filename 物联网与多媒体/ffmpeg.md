# FFmpeg 多媒体音视频处理实战手册

[FFmpeg](https://ffmpeg.org/) 是业界最强大的开源音视频处理与流媒体工具套件。本文档汇总生产环境中高频使用的推拉流、转码压制、硬件加速、滤镜剪辑与探测命令。

---

## 1. 流媒体推流与拉流 (RTSP / RTMP / HLS)

### 1.1 本地视频循环推流到 RTSP 流媒体服务器
```bash
# 1. 循环推流本地 mp4 到 RTSP 服务器 (-re 保持原始帧率, -stream_loop -1 无限循环)
ffmpeg -re -stream_loop -1 -i test.mp4 -vcodec copy -acodec copy -f rtsp -rtsp_transport tcp rtsp://127.0.0.1:8554/live/stream1

# 2. 重新编码为 H.264 推流 (适配非标准源视频格式)
ffmpeg -re -stream_loop -1 -i test.mp4 -c:v libx264 -preset veryfast -tune zerolatency -b:v 2000k -c:a aac -f rtsp -rtsp_transport tcp rtsp://127.0.0.1:8554/live/camera1
```

### 1.2 推流到 RTMP / 直播平台
```bash
ffmpeg -re -stream_loop -1 -i input.mp4 -c:v libx264 -preset medium -b:v 3000k -maxrate 3000k -bufsize 6000k -pix_fmt yuv420p -g 50 -c:a aac -b:a 128k -ar 44100 -f flv rtmp://live.server.com/app/stream_key
```

### 1.3 拉流并转存为本地文件
```bash
# 拉取 RTSP 流并保存为 mp4 (无损复制流，每小时切片或直接录制)
ffmpeg -rtsp_transport tcp -i rtsp://192.168.1.100:554/h264 -c copy -f mp4 output_record.mp4

# 拉取流并切片为 HLS (.m3u8 + .ts)
ffmpeg -i rtsp://127.0.0.1:8554/live/stream1 -c copy -f hls -hls_time 4 -hls_list_size 5 -hls_flags delete_segments /var/www/html/live/index.m3u8
```

---

## 2. 视频转码、压缩与硬件加速

### 2.1 极佳画质/体积比 (CRF 恒定质量模式)
```bash
# H.264 编码 (CRF: 18~28, 推荐 23)
ffmpeg -i input.mp4 -c:v libx264 -crf 23 -preset medium -c:a aac -b:a 128k output_264.mp4

# H.265 (HEVC) 编码 (体积比 H.264 减小约 30%~50%, CRF 推荐 28)
ffmpeg -i input.mp4 -c:v libx265 -crf 28 -preset medium -c:a copy output_265.mp4
```

### 2.2 GPU 硬件加速编解码
```bash
# NVIDIA NVENC 硬件加速 (GPU 极速转码)
ffmpeg -hwaccel cuda -i input.mp4 -c:v h264_nvenc -preset p4 -cq 23 -c:a copy output_gpu.mp4

# Apple Silicon (macOS VideoToolbox) 硬件加速
ffmpeg -i input.mp4 -c:v h264_videotoolbox -b:v 3000k -c:a aac output_mac.mp4
```

---

## 3. 视频剪辑、提取与画面处理

### 3.1 极速无损剪切片段
```bash
# 快速剪切 (将 -ss 放在 -i 之前利用关键帧秒级寻址，-t 指定持续时间)
ffmpeg -ss 00:01:30 -to 00:03:00 -i input.mp4 -c copy cut_output.mp4
```

### 3.2 抽帧与生成缩略图
```bash
# 1. 提取视频单帧作为封面截图 (在第 10 秒处)
ffmpeg -ss 00:00:10 -i input.mp4 -frames:v 1 thumbnail.jpg

# 2. 按固定频率抽帧 (每秒抽取 1 帧用于目标检测算法数据集)
ffmpeg -i input.mp4 -vf "fps=1" frame_%04d.jpg

# 3. 每 10 秒抽取 1 帧
ffmpeg -i input.mp4 -vf "fps=1/10" keyframe_%03d.png
```

### 3.3 分辨率缩放与滤镜 (Scale & Overlay)
```bash
# 1. 缩放到 1080P / 720P (保持宽高比自动计算高度: -1 或 -2)
ffmpeg -i input.mp4 -vf "scale=1280:-2" -c:a copy output_720p.mp4

# 2. 视频叠加水印 (右上角 10 像素边距)
ffmpeg -i input.mp4 -i logo.png -filter_complex "overlay=W-w-10:10" -codec:a copy output_watermarked.mp4
```

### 3.4 视频无损拼接 (Concat Demuxer)
创建 `filelist.txt`：
```text
file 'part1.mp4'
file 'part2.mp4'
file 'part3.mp4'
```
执行拼接：
```bash
ffmpeg -f concat -safe 0 -i filelist.txt -c copy merged.mp4
```

---

## 4. 音频处理与提取

```bash
# 1. 静音视频 (剥离音频流)
ffmpeg -i input.mp4 -an -vcodec copy output_mute.mp4

# 2. 从视频中无损提取音频流
ffmpeg -i input.mp4 -vn -c:a copy output_audio.aac

# 3. 转换为标准 MP3
ffmpeg -i input.mp4 -vn -c:a libmp3lame -q:a 2 output.mp3
```

---

## 5. 媒体流探测工具：`ffprobe`

```bash
# 1. 查看音视频详细封装与流信息
ffprobe -v quiet -print_format json -show_format -show_streams input.mp4

# 2. 快速获取视频分辨率、时长与帧数
ffprobe -v error -select_streams v:0 -show_entries stream=width,height,r_frame_rate,nb_frames -of default=noprint_wrappers=1 input.mp4
```
