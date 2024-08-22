# docker命令

## 镜像保存为tar

`docker save -o docker_image_name.tar docker_image_name`

## 打包容器为镜像

`docker commit $CONTAINER_ID <image-name>`:`<tag>`

## 加载镜像

`docker load -i image_name.tar`

## 日志

`docker log -f [--tail 100] $CONTAINER_ID`

## 复制

`docker cp dir container_id:dir`

## Windows安装docker

> 注意，docker支持windows 19044版本之后的
>
> windows的docker依赖**hyper-V**和**wsl**

安装wsl：

`wsl --list -- online`

若这一步报错，将DNS修改为**114.114.114.114，子网掩码为8.8.8.8**

`wsl --install -d Ubuntu-22.04`

`wsl --update`

`wsl --set-default-version 2`

`wsl -l - v` 查看当前wsl版本

## 重命名镜像

docker tag IMAGEID REPOSITORY:TAG
