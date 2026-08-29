# Linux 常用运维与系统排查手册

---

## 1. 软件源与包管理 (APT)

### Ubuntu 换源 (以 Ubuntu 20.04/22.04 阿里云源为例)
```bash
# 1. 备份原始镜像源
sudo cp /etc/apt/sources.list /etc/apt/sources.list.bak

# 2. 修改镜像源
sudo vim /etc/apt/sources.list
```
*Ubuntu 20.04 (Focal) 阿里源配置参考：*
```text
deb http://mirrors.aliyun.com/ubuntu/ focal main restricted universe multiverse  
deb http://mirrors.aliyun.com/ubuntu/ focal-security main restricted universe multiverse  
deb http://mirrors.aliyun.com/ubuntu/ focal-updates main restricted universe multiverse  
deb http://mirrors.aliyun.com/ubuntu/ focal-backports main restricted universe multiverse
```
```bash
# 3. 更新索引与升级
sudo apt update && sudo apt upgrade -y
```

### 解决 APT 锁占用 (Could not get lock) 问题
当 apt 操作被异常中断（如 Ctrl+C）或有后台自动更新时，会产生锁冲突：

**情况一：存在正在运行的 apt 进程**
```bash
# 检查正在运行的 apt 进程
ps aux | grep -i apt

# 杀掉残留进程
sudo kill -9 <PID>
# 或直接终止所有 apt 进程
sudo killall apt apt-get
```

**情况二：无进程运行但锁文件未释放**
```bash
# 检查并解除锁文件占用
sudo rm -f /var/lib/apt/lists/lock
sudo rm -f /var/cache/apt/archives/lock
sudo rm -f /var/lib/dpkg/lock
sudo rm -f /var/lib/dpkg/lock-frontend

# 重新配置中断的包
sudo dpkg --configure -a
sudo apt update
```

---

## 2. 硬件与系统信息排查

### CPU 与内存
```bash
# 查看 CPU 概括信息
lscpu

# 查看详细 CPU 核与缓存
cat /proc/cpuinfo

# 查看内存使用 (易读格式)
free -h

# 查看内存硬件规格
cat /proc/meminfo
```

### 磁盘与存储排查
```bash
# 查看挂载点与磁盘空间占用
df -h

# 查看磁盘分区与扇区
sudo fdisk -l

# 查看当前目录下各子目录/文件大小并降序排列 (排查磁盘占满利器)
du -sh * | sort -hr | head -n 20
```

### PCI 设备与硬件日志
```bash
# 查看 PCI 设备列表 (显卡、网卡、声卡等)
lspci
lspci -v | grep -i vga

# 查看 USB 设备
lsusb
cat /proc/bus/usb/devices

# 查看内核硬件自检日志
dmesg -T | tail -n 50
```

### 系统版本与运行状态
```bash
# 查看内核版本与系统架构
uname -a

# 查看发行版详细信息 (Ubuntu / Debian / CentOS / Kylin)
lsb_release -a
cat /etc/os-release

# 查看系统运行时间与负载
uptime
```

---

## 3. GPU 与 CUDA 监控

```bash
# 实时监控 GPU 状态 (每秒刷新一次)
watch -n 1 nvidia-smi

# 查看 CUDA 驱动编译器版本
nvcc -V

# 查看 CUDA Toolkit 安装版本
cat /usr/local/cuda/version.txt 2>/dev/null || cat /usr/local/cuda/version.json 2>/dev/null
```

---

## 4. 网络排查与远程传输

### 端口与连接排查
```bash
# 查看指定端口被哪个进程占用
sudo lsof -i :8080

# 查看系统所有正在监听的 TCP/UDP 端口
sudo ss -tulpn
# 或
sudo netstat -tulpn
```

### 远程文件传输
```bash
# SCP 本地文件上传到远程
scp /path/to/local/file username@remote_ip:/path/to/remote/

# SCP 递归传输文件夹
scp -r /path/to/local/dir username@remote_ip:/path/to/remote/

# Rsync 增量同步 (支持断点续传与进度展示)
rsync -avzP /path/to/local/dir username@remote_ip:/path/to/remote/
```

---

## 5. 终端会话复用 (Screen / Tmux)

### Screen 常用操作
```bash
# 创建并进入名为 my_session 的会话
screen -S my_session

# 从当前会话分离 (后台运行)
# 快捷键: Ctrl + A 然后按 D

# 查看所有正在运行的会话
screen -ls

# 恢复/重新连接会话
screen -r my_session

# 强行踢出其他连接并恢复
screen -d -r my_session

# 销毁/结束当前会话
exit
```
