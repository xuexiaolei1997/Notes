# ThingsBoard

## ThingsBoard

ThingsBoard是一个物联平台，需要结合tb-gateway结合使用，集成了多种协议。

可以从[官网](https://thingsboard.io/docs/user-guide/install/docker-windows/?ubuntuThingsboardQueue=inmemory)查看教程下载安装。安装完成后，通过浏览器进入页面，但是现在页面上还不能直接运行。需要添加网关、设备等。

登录用户名密码：

System Administrator: sysadmin@thingsboard.org / sysadmin
Tenant Administrator: tenant@thingsboard.org / tenant
Customer User: customer@thingsboard.org / customer

第二个为租户管理员，通常使用这个进行登录。

## thingsboard-gateway

docker pull thingsboard/tb-gateway

docker run -it -v 本地日志/扩展/配置:远程目录 --name tb-gateway --restart always thingsboard/tb-gateway

启动之后，需要在config中添加网关，这一部分建议通过源码运行，docker本身中不含初始的config文件。

源码中包含默认配置文件，通过编辑config文件夹中的tbgateway.json文件，确定thingboard的地址与ip，再填写对应设备号，在最后的connection中进行配置所需连接，即可启动。

## 联动

thingsboard与gateway，thingsboard中本身是可以编辑的，但是编辑完成保存后，gateway中会有延迟（这一部分暂时还没有解决，因为从docker构建的thingsboard），这个延迟会让数据无法生效。
