# 仿生机器人



## 依赖

```shell
apt-get install swig cmake ffmpeg
pip install -r requirements.txt
```



## 训练

```shell
# algo: [sac, tqc, trpo, ...]
bash train_humanoid.sh [<alog>]
```



## 验证

```shell
# algo: [sac, tqc, trpo, ...]
bash eval_humanoid.sh [<alog>]
```





