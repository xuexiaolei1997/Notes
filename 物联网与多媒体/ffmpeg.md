# ffmpeg命令

## ffmpeg 推流

`ffmpeg -re -stream_loop -1 -i input_video_name -vcodec copy -codec copy -f rtsp rtsp://ip/name`

## ffmpeg截取视频

`ffmpeg -ss 00:00:00 -i input_video_name -t 00:05:00 -c copy -acodec pcm_alaw output.avi`

-ss后面为开始时间， -t后面为持续时间
