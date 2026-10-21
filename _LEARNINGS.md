# 踩过的坑（带翻车现场）

## 原曲 mp4 的音频比视频晚 42 ms（2026-10-21）
- 坑：`Assets/...Can't Kill Us...mp4` 的音频第一个样本 pts = 0.042 s。从头整段解码（感官.json 就是这样算的）数的是样本时间；`-ss` 截取 / 播放器听到的是容器时间 —— 同一个鼓点，感官.json 写 40.31，剪辑表 / 配乐版里在 40.35。
- 为什么要紧：差 1 帧。按感官.json 卡点的画面会比声音早 1 帧。
- 怎么避：拿感官.json 的时间去对齐「听到的」时间时 +0.042（`tools/push_remap.py --sense-offset`）。查法：`ffmpeg -i x.mp4 -map 0:a -af ashowinfo -t 0.1 -f null -` 看第一个 pts_time。
