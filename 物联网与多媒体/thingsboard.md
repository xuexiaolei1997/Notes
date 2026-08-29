# ThingsBoard 物联网平台与 tb-gateway 实战指南

[ThingsBoard](https://thingsboard.io/) 是目前最流行的开源工业物联网（IIoT）平台，支持设备连接、多协议数据采集、规则链（Rule Engine）处理、三维可视化看板与告警管理。

---

## 1. ThingsBoard 核心架构与一键部署

### 1.1 使用 Docker Compose 快速部署 (带 PostgreSQL 存储)
创建 `docker-compose.yml`：
```yaml
version: '3.8'
services:
  mytb:
    image: "thingsboard/tb-postgres:latest"
    container_name: mytb
    restart: always
    environment:
      TB_QUEUE_TYPE: in-memory
    ports:
      - "8080:8080"       # Web UI / HTTP REST API
      - "1883:1883"       # MQTT 协议端口
      - "7070:7070"       # Edge 通信端口
      - "5683-5688:5683-5688/udp" # CoAP 端口
    volumes:
      - tb-data:/data
      - tb-log:/var/log/thingsboard
volumes:
  tb-data:
  tb-log:
```
启动命令：
```bash
docker compose up -d
```

### 1.2 默认系统初始账号

| 角色 | 默认邮箱 / 用户名 | 默认密码 | 用途说明 |
| :--- | :--- | :--- | :--- |
| **系统管理员 (SysAdmin)** | `sysadmin@thingsboard.org` | `sysadmin` | 管理租户、全局设置、邮件服务器配置 |
| **租户管理员 (Tenant)** | `tenant@thingsboard.org` | `tenant` | **日常最常用**：创建设备、网关、仪表盘、规则链 |
| **普通客户 (Customer)** | `customer@thingsboard.org` | `customer` | 仅查看租户分配给该客户的设备与看板 |

---

## 2. ThingsBoard IoT Gateway (`tb-gateway`) 配置

[tb-gateway](https://github.com/thingsboard/thingsboard-gateway) 是官方提供的多协议边缘网关，用于连接 Modbus、CAN、OPC-UA、BLE、MQTT 等各类传统工业协议并统一上报。

### 2.1 核心配置文件：`tb_gateway.yaml` / `tbgateway.json`
```yaml
thingsboard:
  host: "192.168.1.100"  # ThingsBoard 平台 IP 地址
  port: 1883
  remoteConfiguration: false
  security:
    accessToken: "YOUR_GATEWAY_ACCESS_TOKEN"  # 在 ThingsBoard UI 中创建的网关设备 Access Token

storage:
  type: memory
  read_records_per_iteration: 100
  max_records_count: 10000

connectors:
  - name: Modbus Connector
    type: modbus
    configuration: modbus.json
  - name: MQTT Connector
    type: mqtt
    configuration: mqtt.json
```

### 2.2 启动网关容器
```bash
docker run -d --name tb-gateway \
  --restart always \
  -v /path/to/config:/thingsboard_gateway/config \
  -v /path/to/extensions:/thingsboard_gateway/extensions \
  -v /path/to/logs:/thingsboard_gateway/logs \
  thingsboard/tb-gateway:latest
```

---

## 3. 设备数据直接上报实战 (HTTP / MQTT)

### 3.1 HTTP 协议上报遥测数据 (Telemetry)
```bash
# $ACCESS_TOKEN 替换为平台的设备凭证 Token
curl -v -X POST http://localhost:8080/api/v1/$ACCESS_TOKEN/telemetry \
  --header "Content-Type:application/json" \
  --data '{"temperature": 26.5, "humidity": 68.2, "pressure": 101.3}'
```

### 3.2 Python MQTT 异步上报遥测
```python
import paho.mqtt.client as mqtt
import json
import time

THINGSBOARD_HOST = "localhost"
ACCESS_TOKEN = "YOUR_DEVICE_ACCESS_TOKEN"

client = mqtt.Client()
client.username_pw_set(ACCESS_TOKEN)
client.connect(THINGSBOARD_HOST, 1883, 60)
client.loop_start()

try:
    while True:
        payload = {
            "temperature": 25.4,
            "vibration_rms": 0.035,
            "status": "RUNNING"
        }
        client.publish("v1/devices/me/telemetry", json.dumps(payload), qos=1)
        time.sleep(2)
except KeyboardInterrupt:
    client.loop_stop()
    client.disconnect()
```

---

## 4. 常见问题排查与优化 (网关数据同步与延迟)

1. **网关与平台配置同步延迟**：
   - **原因**：网关默认通过 MQTT 订阅 `v1/gateway/attributes` 主题接收云端下发。如果网络抖动或 Docker 容器内时间未与宿主机 NTP 同步，会导致属性更新感知滞后。
   - **解决**：在网关配置中将 `storage.type` 设置为可靠的文件存储（`file`），并确保 Docker 挂载宿主机时区 `-v /etc/localtime:/etc/localtime:ro`。
2. **遥测入库与规则链卡顿**：
   - ThingsBoard 采用异步队列处理遥测入库，若大量设备同时并发上报导致入库延迟，可在平台 `thingsboard.yml` 中调大 `TB_QUEUE_SUB_TOTAL_THREADS` 线程池或将内存队列升级为 **Kafka / RabbitMQ** 消息中间件。
