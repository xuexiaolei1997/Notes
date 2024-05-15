# Linux推流

## 1.准备ubuntu服务器

## 2.安装x264依赖包

```text
git clone https://code.videolan.org/videolan/x264.git
```

```text
cd x264
```

```text
./configure --prefix=/usr/local/x264 --enable-shared --enable-static --disable-asm
```

```text
make && make install
```

编辑环境变量

```text
vim /etc/profile
```

```text
export PATH=/usr/local/x264/bin:$PATH
export PATH=/usr/local/x264/include:$PATH
export PATH=/usr/local/x264/lib:$PATH
```

```text
source /etc/profile
```

## 3.安装FFmpeg

```text
wget http://www.ffmpeg.org/releases/ffmpeg-4.4.tar.gz
```

```text
tar -zxvf ffmpeg-4.4.tar.gz
```

```text
cd ffmpeg-4.4
```

```text
./configure --prefix=/usr/local/ffmpeg --enable-shared --enable-libx264 --enable-gpl --enable-pthreads --extra-cflags=-I/usr/local/x264/include --extra-ldflags=-L/usr/local/x264/lib
```

```text
make && make install
```

```text
vi /etc/profile
```

```text
export PATH=$PATH:/usr/local/ffmpeg/bin
```

```text
ffmpeg -version
```

## 注意

* 报错 yasm/nasm not found or too old. Use –disable-yasm for a crippled build.

```text
wget http://www.tortall.net/projects/yasm/releases/yasm-1.3.0.tar.gz
tar -zxvf yasm-1.3.0.tar.gz
cd yasm-1.3.0
./configure
make && make install
```

* 报错 error while loading shared libraries: libx264.so: cannot open shared object file: No such file

```text
vim /etc/ld.so.conf //增加以下内容
/usr/local/x264/lib //添加x264库路径，添加完保存退出
ldconfig //使配置生效
```

* 报错：ffmpeg: error while loading shared libraries: libavdevice.so.57: cannot open shared object file: No such file or directory

```text
vim /etc/ld.so.conf //增加以下内容
/usr/local/ffmpeg/lib  //添加ffmpeg库路径，添加完保存退出
ldconfig //使配置生效
```

## 4.下载并启动rtsp流媒体服务

```text
wget https://github.com/aler9/rtsp-simple-server/releases/download/v0.17.0/rtsp-simple-server_v0.17.0_linux_amd64.tar.gz
```

```text
tar -zxvf rtsp-simple-server_v0.17.0_linux_amd64.tar.gz
```

```text
vim rtsp-simple-server.yml
```

```text
将rtspAddress: :8554改为rtspAddress: :4000并保存
```

```text
./rtsp-simple-server 
```
