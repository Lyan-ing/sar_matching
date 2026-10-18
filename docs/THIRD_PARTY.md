# 外部代码与权重

|组件|仓库中的处理|来源与许可|
|---|---|---|
|LightGlue|保留独立推理需要的Apache源码与许可；其余历史阶段使用官方安装入口|https://github.com/cvg/LightGlue ，固定commit `eb42fee2d71449efb0aa5c10549752b5d75384d8`，Apache-2.0|
|SuperPoint|不上传限制性源码、预训练权重或本项目派生权重；由有权使用者在本地准备|https://github.com/magicleap/SuperPointPretrainedNetwork ，单独的非商业研究限制许可|
|RoMa及其他调研仓库|不上传原服务器vendor、sources、依赖缓存或下载权重；仅保留本项目调用/诊断代码|按原项目官方仓库及原许可独立获取|
|SAR影像及人工标注|完全不进入代码仓库|各数据发布方的许可与已有项目记录|

完整第三方许可文件保留于`sar_match_portable/licenses/`。本仓库不是对受限权重的公开下载服务。若无法合法取得研究模型，应停止相应推理，不以随机权重或公开SuperPoint冒充最终模型。

`prepare_superpoint.py`只在用户明确确认许可后获取固定官方源码，校验源文件SHA并移除构造函数的自动下载；生成文件由.gitignore排除。此方式不替代使用者取得合法授权的义务。
