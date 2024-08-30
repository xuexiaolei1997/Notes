# ubuntu

## 换源

1、备份原始镜像源

`cp /etc/apt/sources.list /etc/apt/sources.list.bak`

2、更新镜像源

 `vim /etc/apt/sources.list`

删除所有源，粘贴其他镜像源

```shell
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

## 关于apt get 存在lock锁的问题


第一种情况：

进程中存在与apt相关的正在运行的进程：

首先检查是否在运行apt,apt-get相关的进程

ps aux | grep -i apt

sudo kill -9 `<process id>` 或者  sudo killall apt apt-get

第二种情况：

进程列表中已经没有与apt,apt-get相关的进程在运行，但依然报错，在这种情况下，产生错误的根本原因是lock file。 lock file用于防止两个或多个进程使用相同的数据。 当运行apt或apt-commands时，它会在几个地方创建lock files。 当前一个apt命令未正确终止时，lock file未被删除，因此它们会阻止任何新的apt / apt-get命令实例，比如正在执行apt-get upgrade，在执行过程中直接ctrl+c取消了该操作，很有可能就会造成这种情况。

要解决此问题，首先要删除lock file。

lsof /var/lib/dpkg/lock

lsof /var/lib/apt/lists/lock

lsof /var/cache/apt/archives/lock

sudo rm /var/lib/apt/lists/lock

sudo rm /var/cache/apt/archives/lock

sudo rm /var/lib/dpkg/lock

sudo dpkg --configure -a

lsof /var/lib/dpkg/lock-frontend

sudo kill -9 PID

sudo rm /var/lib/dpkg/lock-frontend

sudo dpkg --configure -a
