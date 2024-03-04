# 镜像保存为tar

`docker save -o myimage.tar myimage`

# 打包容器为镜像

`docker commit $CONTAINER_ID <image-name>`:`<tag>`

# 加载镜像

`docker load -i image_name.tar`
