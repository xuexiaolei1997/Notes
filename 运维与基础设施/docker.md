# Docker 常用命令与容器化运维手册

---

## 1. 镜像管理

### 镜像构建与标记
```bash
# 构建镜像
docker build -t my_image:v1.0 .

# 使用指定 Dockerfile 构建并传递参数
docker build -f Dockerfile.prod -t my_image:v1.0 --build-arg ENV=production .

# 重命名/打标签
docker tag <image_id_or_name> my_registry.com/repo/image:tag
```

### 镜像离线保存与加载 (跨机器部署)
```bash
# 保存镜像为 tar 归档文件
docker save -o docker_image_name.tar <image_name>:<tag>

# 批量保存多个镜像
docker save -o all_images.tar image1:v1 image2:v2

# 从 tar 文件加载镜像
docker load -i docker_image_name.tar

# 导出容器快照 (与 save 的区别：export 丢弃历史层，仅导出当前文件系统)
docker export -o container_snapshot.tar <container_id>
cat container_snapshot.tar | docker import - <image_name>:<tag>
```

### 镜像查看与清理
```bash
# 查看所有镜像
docker images

# 删除单个/未使用的虚悬镜像 (dangling)
docker rmi <image_id>
docker image prune
```

---

## 2. 容器生命周期管理

### 启动与运行
```bash
# 后台启动并映射端口与目录
docker run -d \
  --name my_service \
  -p 8080:80 \
  -v /host/path:/container/path:rw \
  --restart always \
  <image_name>:<tag>

# 交互式启动并进入终端 (退出时容器自动销毁)
docker run -it --rm <image_name> /bin/bash
```

### 资源限制与 GPU 分配 (算法/AI场景)
```bash
# 限制 CPU 和内存使用
docker run -d \
  --name ai_task \
  --cpus=4 \
  --memory=8g \
  --memory-swap=12g \
  <image_name>

# 挂载全部 GPU (需要宿主机安装 nvidia-container-toolkit)
docker run -d --gpus all <image_name>

# 挂载指定 GPU
docker run -d --gpus '"device=0,1"' <image_name>
```

### 容器运维与调试
```bash
# 查看正在运行的容器 (加上 -a 查看所有容器)
docker ps
docker ps -a

# 进入正在运行的容器
docker exec -it <container_id_or_name> /bin/bash

# 查看容器实时日志
docker logs -f --tail 100 <container_id_or_name>

# 查看容器资源消耗 (CPU/内存/网络/磁盘IO)
docker stats

# 宿主机与容器间双向复制文件
docker cp /local/path <container_id>:/container/path
docker cp <container_id>:/container/path /local/path

# 将运行中的容器保存为新镜像
docker commit -m "commit message" -a "author" <container_id> <new_image_name>:<tag>
```

---

## 3. 网络与存储卷管理

### 存储卷 (Volume)
```bash
# 创建命名卷
docker volume create my_data

# 查看与清理未使用的卷
docker volume ls
docker volume prune
```

### 网络模式
```bash
# 创建自定义 bridge 网络
docker network create --driver bridge my_network

# 运行容器加入网络
docker run -d --network my_network --name app_a image_a
docker run -d --network my_network --name app_b image_b
# 同网络下容器可通过容器名互相 ping 通通信
```

---

## 4. 系统维护与垃圾清理

```bash
# 查看 Docker 磁盘占用概况
docker system df

# 一键清理所有已停止容器、悬空镜像、未使用的网络 (生产谨慎使用)
docker system prune

# 深度清理 (包含所有未被容器引用的镜像和卷)
docker system prune -a --volumes
```

---

## 5. Windows / WSL2 Docker 环境配置

> **前置条件**：Docker Desktop 依赖 Windows 10 (Build 19044+) 或 Windows 11，且需要开启 **Hyper-V** 和 **WSL 2**。

### WSL 2 安装与配置
```powershell
# 查看可在线安装的分发版本
wsl --list --online

# 安装 Ubuntu 22.04
wsl --install -d Ubuntu-22.04

# 更新 WSL 内核
wsl --update

# 设置默认使用 WSL 2
wsl --set-default-version 2

# 查看已安装的子系统及其 WSL 版本
wsl -l -v
```

> **常见网络报错处理**：  
> 若 `wsl --install` 或 `wsl --list --online` 出现网络连接超时，可尝试修改 Windows 网络适配器的 DNS 服务器为：
> - 首选 DNS：`114.114.114.114`
> - 备用 DNS：`8.8.8.8`
