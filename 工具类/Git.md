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
