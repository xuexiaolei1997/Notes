# Git命令

## 修改远程仓库

`git remote set-url origin <new_remote_repository_url>`

## 修改代理

`git config --global http.proxy http://localhost:7890`

## 取消代理

`git config --global --unset http.proxy	`

## fork如何同步github

1.fork原始仓库

2.git clone <仓库地址>

3.git remote add upstream <上游仓库地址>

4.git fetch upstream

5.git merge upstream/master

6.git push origin master

## 显示分支

git branch

## 分支选择性合并

1.切换到源分支

`git checkout branch-origin`

2.查看提交历史以找到需要的提交哈希值：

`git log --oneline`

3.切换到目标分支

`git checkout branch-target`

4.执行 git cherry-pick 命令

1) 挑选单个提交  `git cherry-pick <commit_hash>`

2) 挑选一系列连续的提交  `git cherry-pick <commit_A>^..<commit_B>`

3) 挑选多个不连续的提交  `git cherry-pick <commit_hash_1> <commit_hash_2> <commit_hash_3>`

5.解决冲突、合并代码、推送

