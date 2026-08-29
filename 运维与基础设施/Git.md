# Git 高效工作流与核心命令手册

---

## 1. 远程仓库与网络代理配置

### 远程仓库管理
```bash
# 查看所有关联的远程仓库及 URL
git remote -v

# 修改现有远程仓库 URL (如修改 origin)
git remote set-url origin <new_remote_repository_url>

# 添加额外的远程上游仓库 (如 upstream)
git remote add upstream <upstream_repo_url>
```

### Git 代理配置 (加速 GitHub 克隆/推送)
```bash
# 设置 HTTP/HTTPS 代理 (以本地 7890 端口为例)
git config --global http.proxy http://localhost:7890
git config --global https.proxy http://localhost:7890

# 设置 SOCKS5 代理
git config --global http.proxy socks5://127.0.0.1:7890
git config --global https.proxy socks5://127.0.0.1:7890

# 仅对 github.com 域名生效代理
git config --global http.https://github.com.proxy http://localhost:7890

# 取消全局代理
git config --global --unset http.proxy
git config --global --unset https.proxy
```

---

## 2. 分支管理与选择性合并 (Cherry-Pick)

### 分支基础
```bash
# 查看本地分支 / 全部远程分支
git branch
git branch -a

# 创建并切换到新分支
git checkout -b feature/new-algorithm
# 或使用新版 switch 命令
git switch -c feature/new-algorithm
```

### Cherry-Pick 精确合并提交
适用于从某个分支挑出特定的几个 Commit 合并到当前分支：

```bash
# 1. 切换到目标分支
git checkout branch-target

# 2. 挑选单个提交
git cherry-pick <commit_hash>

# 3. 挑选一系列连续提交 (包含 commit_A 到 commit_B)
git cherry-pick <commit_A>^..<commit_B>

# 4. 挑选多个不连续提交
git cherry-pick <commit_hash_1> <commit_hash_2> <commit_hash_3>

# 5. 若有冲突：解决冲突后暂存并继续
git add .
git cherry-pick --continue
# 若想放弃此次 cherry-pick
git cherry-pick --abort
```

---

## 3. Fork 开源仓库与上游同步工作流

```bash
# 1. 克隆自己 Fork 的仓库
git clone https://github.com/<your-username>/<repo>.git
cd <repo>

# 2. 关联官方上游仓库
git remote add upstream https://github.com/<upstream-owner>/<repo>.git

# 3. 获取上游最新提交
git fetch upstream

# 4. 合并上游 master/main 到本地 master
git checkout master
git merge upstream/master

# 5. 推送最新代码同步至自己的 Fork 仓库
git push origin master
```

---

## 4. 撤销、回滚与暂存 (防丢代码与版本回退)

### 提交撤销与版本重置 (Reset vs Revert)
```bash
# 软重置：保留工作区与暂存区修改，仅撤销 Commit (适合想重新写 commit message)
git reset --soft HEAD~1

# 混合重置 (默认)：保留工作区修改，清空暂存区
git reset HEAD~1

# 硬重置：彻底丢弃工作区和暂存区的所有修改 (谨慎操作！)
git reset --hard <commit_hash_or_HEAD~1>

# 安全反转：生成一个反向的新 Commit 来抵消旧 Commit (公共分支推荐)
git revert <commit_hash>
```

### 工作区临时暂存 (Stash)
```bash
# 临时保存未提交修改 (含未跟踪文件)
git stash -u -m "wip: 保存临时实验代码"

# 查看暂存列表
git stash list

# 恢复最近一次暂存并从暂存列表删除
git stash pop

# 恢复指定暂存记录 (保留记录)
git stash apply stash@{0}
```

### 误操作救命神器 (Reflog)
```bash
# 查看所有 HEAD 的移动记录 (即使被 reset --hard 丢弃的 commit 也能找回)
git reflog

# 找到误删前的 Commit Hash 并恢复
git reset --hard <reflog_hash>
```

---

## 5. 交互式变基 (Rebase) 与 Commit 整理

```bash
# 合并最近 3 次 Commit 为 1 个干净的 Commit
git rebase -i HEAD~3

# 在弹出的编辑器中：
# 将第 2、3 行前面的 pick 改为 s (squash) 或 f (fixup)
# 保存退出后编辑最终的统一提交信息即可
```

