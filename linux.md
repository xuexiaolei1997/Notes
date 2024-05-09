# ubuntu

## 换源

1、备份原始镜像源

`cp /etc/apt/sources.list /etc/apt/sources.list.bak`

2、更新镜像源

 `vim /etc/apt/sources.list`

删除所有源，粘贴其他镜像源

```
deb http://mirrors.aliyun.com/ubuntu/ focal main restricted universe multiverse  
deb http://mirrors.aliyun.com/ubuntu/ focal-security main restricted universe multiverse  
deb http://mirrors.aliyun.com/ubuntu/ focal-updates main restricted universe multiverse  
deb http://mirrors.aliyun.com/ubuntu/ focal-backports main restricted universe multiverse
```

3、更新

`apt update`

4、升级系统

`apt upgrade`

## apt解锁

`rm /var/apt/lists/lock`

## 远程传输

本地传到远程： `scp [本地文件路径] [用户名]@[远程主机地址] : [目标路径]`

传输文件夹，scp -r 

## 查看内存、CPU、硬盘

`free -h`

`lscpu`

`df -h`
