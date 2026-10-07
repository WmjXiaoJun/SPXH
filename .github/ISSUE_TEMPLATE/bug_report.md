---
name: Bug 报告
about: 跑不通、结果不对、报错
title: "[Bug] "
labels: bug
assignees: ''
---

## 现象

发生了什么？预期是什么？

## 复现

```bash
# 贴出完整命令，最好用 data/demo 里的案例或带 seed 的 generate 命令
python -m spxh ...
```

```text
# 完整输出（输出很长时请先重定向到文件，再贴关键部分： > out.log 2>&1）
```

## 环境

- 操作系统：
- Python 版本（`python -V`）：
- 安装方式（`pip install -e ".[dev,ml]"` 还是别的）：
- SPXH 版本（`python -m spxh --version`）：

## 数据

- 用的是 `data/demo` 里的案例，还是自己生成 / 自己的录波？
- 自己生成的话，贴出 `python -m spxh generate ...` 的完整命令与 seed。

## 已确认

- [ ] 我贴的命令可以直接复制运行（不依赖本地私有文件）
- [ ] 我已看过该里程碑验收报告（`docs/MX-验收报告.md`）里的“已知局限”
- [ ] 我已经用 `python -m pytest -q > pytest.log 2>&1` 跑过测试
