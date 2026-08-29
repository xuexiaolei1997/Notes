# Windows PE 系统维护与装机救砖指南

Windows PE (Preinstallation Environment) 是轻量级的 Windows 预安装环境，广泛用于系统安装、灾难备份、数据恢复、离线驱动注入与引导修复。

---

## 1. 启动 U 盘制作方案

### 方案 A：Ventoy 多合一启动盘 (强烈推荐)
- **原理**：只需格式化一次 U 盘安装 Ventoy，之后可直接把 Windows ISO、Ubuntu ISO、WePE ISO 文件直接拷入 U 盘根目录，开机启动时自动生成启动菜单。
- **官网**：[ventoy.net](https://www.ventoy.net/)

### 方案 B：微 PE 工具箱 (WePE)
- **特点**：纯净无捆绑、体积小、兼容 UEFI 与 Legacy BIOS 双启动。
- **制作步骤**：运行 WePE 安装程序 -> 选择“安装 PE 到 U 盘” -> 格式化方案建议选 `ExFAT`（支持单文件 >4GB 的 ISO 镜像）。

---

## 2. 高频系统维护与救砖场景

### 2.1 系统备份与还原 (DISM 命令行)
在 PE 环境下，直接使用 Windows 自带的 DISM 工具进行全盘增量捕获与还原（比第三方工具更稳定且无驱动兼容问题）：

```cmd
:: 1. 将 C 盘系统捕获备份为 WIM 镜像到 D 盘
dism /Capture-Image /ImageFile:D:\Backup_Win10.wim /CaptureDir:C:\ /Name:"Win10_Backup"

:: 2. 从 WIM 镜像还原恢复系统到 C 盘
dism /Apply-Image /ImageFile:D:\Backup_Win10.wim /Index:1 /ApplyDir:C:\
```

### 2.2 UEFI / GPT 引导修复
当系统无法开机、提示 `No bootable device` 或 BCD 损坏时，在 PE 的 CMD 中执行：

```cmd
:: 自动修复 C 盘系统的 UEFI 引导到 ESP 分区 (假设 ESP 盘符为 S:)
bcdboot C:\Windows /s S: /f UEFI /l zh-cn
```

### 2.3 离线注入 RAID / NVMe / 网卡驱动
在给工控机或服务器装系统时，若找不到硬盘或网卡，可在 PE 中通过 DISM 直接向目标系统离线打入驱动：

```cmd
dism /Image:C:\ /Add-Driver /Driver:D:\Drivers /Recurse
```

---

## 3. 微 PE 工具箱 (WePE) 资源下载备忘

### 微PE工具箱 V2.2
- **64位镜像站**：[山东大学镜像](https://mirrors.sdu.edu.cn/software/Windows/WePE/WePE64_V2.2.exe)
- **32位镜像站**：[山东大学镜像](https://mirrors.sdu.edu.cn/software/Windows/WePE/WePE32_V2.2.exe)
- **百度网盘**：`https://pan.baidu.com/share/init?surl=dbZMps3gk2v_XQ5TmCccPA` (提取码: `wepe`)

### 微PE工具箱 V2.1
- **64位**：[镜像站下载](https://mirrors.sdu.edu.cn/software/Windows/WePE/WePE_64_V2.1.exe) ｜ [微云](https://share.weiyun.com/iV7Hy54v) ｜ 百度云提取码: `hmts`
- **32位**：[镜像站下载](https://mirrors.sdu.edu.cn/software/Windows/WePE/WePE_32_V2.1.exe) ｜ [微云](https://share.weiyun.com/WqdJX5Mn) ｜ 百度云提取码: `9mup`
