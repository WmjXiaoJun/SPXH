# 贡献指南（CONTRIBUTING）

感谢你愿意花时间。SPXH 是一个纯 Python 的信号分析平台，**没有隐藏的数值**：
每一级估计都带"值 / 单位 / 置信度 / 方法 / 证据"，每个里程碑都有一份可复跑的验收报告。
贡献请沿用这套习惯。

- 面向使用者的问题、跑不通、报错 → 开 Issue（模板见 `.github/ISSUE_TEMPLATE/`）
- 想改代码 → 先开 Issue 说明动机，再提 PR，避免大改被拒

## 1. 环境

仓库用 `pyproject.toml`（PEP 621）声明依赖，**没有 requirements.txt**。

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -e ".[dev,ml,plot]"   # dev=pytest；ml=scikit-learn；plot=matplotlib（出图）
```

测试：

```bash
python -m pytest -q
```

> 如果测试输出特别大导致管道被打断，请重定向到文件后再读：
> `python -m pytest -q > pytest.log 2>&1`（本项目的验收脚本同理）。

## 2. 代码结构约定

| 位置 | 放什么 |
|---|---|
| `spxh/core/<领域>/` | 算法本体，一个文件一件事（如 `demod/sync.py` 只放定时/载波环） |
| `spxh/core/<领域>/types.py` | 该领域的数据结构（可 JSON 化的 dataclass / TypedDict） |
| `spxh/cli/main.py` | 每个能力对应一个子命令；子命令只做"读参数 → 调 core → 打印/落盘" |
| `spxh/web/server.py` + `web/static/` | 本地工作台；接口契约见 `docs/前端接口约定-v1.md` |
| `spxh/api.py` | 给 Python 调用者的门面（避免使用者直接摸 core 内部） |
| `tests/test_*.py` | 单元测试；每个新模块至少配一个文件 |
| `scripts/check_mX.py` | 里程碑验收自检（见第 4 节） |
| `docs/MX-验收报告.md` | 里程碑验收报告：范围、方法、**实测数字**、已知局限 |

硬性要求：

- **不黑箱**：新增估计量必须带置信度/证据；新增判决必须能说出依据。
- **不编数字**：报告、README、Issue 里出现的数字都要能追溯到某次可复跑的脚本输出。
- **如实报边界**：做不到就写"不可用/不可纠/低置信度"，不要给一个自信的错值。
- 不引入重量级依赖；numpy + scipy 是核心，其余（scikit-learn / matplotlib / PySide6 / h5py）一律可选。

## 3. 加一个里程碑（例如 M12）

1. **写方案**：在 `docs/` 新增或更新设计说明，明确输入、输出、出口指标与判据。
2. **写 core**：在 `spxh/core/` 下新增子包；对外 API 走 `spxh/core/<领域>/__init__.py`。
3. **接 CLI**：在 `spxh/cli/main.py` 加子命令，`--json` 输出结构化结果，方便脚本与前端复用。
4. **接接口（可选）**：按 `docs/前端接口约定-v1.md` 的格式新增 HTTP 路由，并在 `spxh/web/static/mock/` 补一份离线 mock。
5. **写验收脚本**：`scripts/check_m12.py`（见第 4 节）。
6. **写报告**：`docs/M12-验收报告.md`，包含：范围、方法与关键设计决策、实测表格、期间修掉的问题、**已知局限**。
7. **测试**：`tests/test_*` 覆盖正常路径 + 边界（不可纠、门控拒绝、非法参数、JSON 里不出现 Infinity/NaN）。
8. **回归**：`python -m pytest -q` 全绿，且不降低已有里程碑的验收判据。

## 4. 怎么写一个验收脚本

参考 `scripts/check_m0.py`（最小）与 `scripts/check_m7.py`（多 part）。

约定：

- 固定随机种子（`np.random.default_rng(seed)`），保证同一机器上可复现；
- 先打印一张人可读的表（终端里直接能看懂结论）；
- 判据**显式列出**：每条 check 带 `name / value / target / pass`，最后打印 `[PASS]/[FAIL]` 清单；
- 把结构化结果写进 `models/check_m12_report.json`（`ensure_ascii=False`），前端与报告都从这里取数；
- 需要图就写 `docs/figures/m12_*.png`；
- 返回值：全过 `0`，任一不过 `1`（CI 直接看退出码）；
- **低误码率区间要跑够样本**：零误码点用 95% 置信上界 3/N 表示并标记，真实误码数另给字段，判据用误码数。

## 5. 跑自检

```bash
python scripts/check_m0.py                # 合成链（秒级）
python scripts/check_m1.py                # 参数估计 + 因果曲线
python scripts/check_m2.py                # 分类 / 校准 / 门控 / OOD（需先有分类器）
python scripts/check_m3.py                # 解调 BER + 同步环价值 + LLR 校准
python scripts/check_m4.py                # 编码增益 / 软判决 / RS / 交织（约 3~5 分钟）
python scripts/check_m5.py                # 帧同步 + 相位模糊 + 载荷
python scripts/check_m6.py                # 编码链端到端门限
python scripts/check_m7.py --part curves   # 平滑端到端曲线（约 3.5 分钟）
python scripts/check_m7.py --part blind    # 盲编码识别（秒级）
python scripts/check_m7.py --part erasure  # RS 擦除译码（秒级）
```

> 已知状态：在本仓库当前版本上，`check_m7.py --part erasure` 的"16 字节擦除"一行会返回失败
> （退出码 1），与 `docs/M7-验收报告.md` 记录的通过不一致。细节见
> [docs/开源检查清单.md](docs/开源检查清单.md)。CI 暂不跑这一项。

## 6. 提交与 PR

- 一次 PR 只做一件事；标题写"里程碑/模块 + 做了什么"。
- PR 描述里贴：命令、退出码、关键输出（表格或截图）。
- 改了数值行为，就必须同步更新对应的 `docs/MX-验收报告.md`（不允许"代码变了、报告没变"）。
- 提交信息用祈使句，写清"为什么"。

## 7. 许可

提交即表示你同意以 [MIT License](LICENSE) 授权你的贡献（版权人记为 `WmjXiaoJun`）。
