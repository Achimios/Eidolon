# 踩过的坑（带翻车现场）

## 原曲 mp4 的音频比视频晚 42 ms（2026-10-21）
- 坑：`Assets/...Can't Kill Us...mp4` 的音频第一个样本 pts = 0.042 s。从头整段解码（感官.json 就是这样算的）数的是样本时间；`-ss` 截取 / 播放器听到的是容器时间 —— 同一个鼓点，感官.json 写 40.31，剪辑表 / 配乐版里在 40.35。
- 为什么要紧：差 1 帧。按感官.json 卡点的画面会比声音早 1 帧。
- 怎么避：~~拿感官.json 的时间 +0.042~~ → 2026-10-21 已在源头修：`sensorium.decode / load_audio` 探测音频第一个包的 pts 并在前面补零，所有分析器（感官 / beat_this / Demucs / motif_find / stills.audio）都走它，`push_remap --sense-offset` 默认 0。⚠️从旧数据**推出来的**东西（如锚点.json 的 phrase_grid.start）也带着这 42 ms，要跟着重推。新写解码别再自己拼 ffmpeg 命令。查法：`ffmpeg -i x.mp4 -map 0:a -af ashowinfo -t 0.1 -f null -` 看第一个 pts_time。
