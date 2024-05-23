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
