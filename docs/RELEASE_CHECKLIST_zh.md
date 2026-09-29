# RVSAuto GitHub 发布前检查清单

在 `git push` / 创建 Release 前逐项确认。

## 1. 是否必须放进仓库？

| 路径 | 是否提交 GitHub | 说明 |
|------|-----------------|------|
| `rvsauto/` | **是** | 核心 Python 包 |
| `pyproject.toml` `[project.scripts]` | **是** | `rvsauto`（`screen` / `redock` 子命令）及别名 `rvsauto-unidock` / `rvsauto-redock` |
| `tests/`, `conda/`, `scripts/` | **是** | 测试与环境 |
| `README.md`, `LICENSE`, `pyproject.toml` | **是** | 元数据 |
| `Uni-Dock-main/` | **否** | 见下文；已在 `.gitignore` |
| `p2rank_*/` | **否** | 用户自行下载 P2Rank 二进制 |
| `AF2BIND_out/` | **否** | 本地 AF2BIND 结果 |
| `workdir/`, `results/`, `logs/` | **否** | 流水线输出 |
### Uni-Dock-main 是否必要？

**不必纳入 GitHub 仓库。**

- RVSAuto **不依赖** vendored 源码；对接通过系统/conda 里的 `unidock` 可执行文件。
- 推荐安装：`conda env create -f conda/environment-unidock.yml`（`conda-forge` 的 `unidock` 包）。
- `rvsauto/common.py` 里 `find_unidock()` 仅在本地**已编译**时才会探测 `Uni-Dock-main/.../build/unidock`，属于开发机可选路径，不是发布物。
- 该目录体积大、含独立许可证与 CI，与 RVSAuto 重复；**保留在本地即可，不要 `git add`**。

若历史上曾 `git add` 过，发布前执行：

```bash
git rm -r --cached Uni-Dock-main p2rank_* AF2BIND_out 2>/dev/null
```

## 2. 元数据与占位符

- [x] `pyproject.toml` / `README.md` → `Xin-DongXu/RVSAuto`
- [ ] `authors` 改为真实姓名/单位（可选）
- [ ] 打 tag：`git tag v11.0.0`（与 `rvsauto.__version__` 一致）

## 3. 安全与隐私

- [ ] 无 `.env`、密钥、HPC 账号路径（当前代码库内无 `/share/home/...` 硬编码）
- [ ] 不提交 `*.pdbqt`、大规模 `workdir/`、个人 `xdxu_workdir/`
- [ ] `AF2BIND_out/` 仅含公开蛋白 ID 时可考虑示例化；否则保持 ignore

## 4. 功能与测试

```bash
pip install -e ".[dev]"
pytest -v
rvsauto screen --help
rvsauto redock --help
python -m build   # 检查 sdist 能否打包
```

- [ ] CI（`.github/workflows/ci.yml`）在 push 后通过
- [ ] 新参数已文档化：`--p2rank_min_probability`、`--ligand_pdbqt`、`--p2rank_min_probability` 等见 README

## 5. 用户文档（README 必备信息）

- [ ] UniDock：Linux + NVIDIA GPU，conda `unidock_env`
- [ ] ADT：`adt_env`（MGLTools + obabel）
- [ ] P2Rank：单独下载，放仓库旁或 `--p2rank_path`
- [ ] AF2BIND：本仓库**不运行** AF2BIND，只消费 CSV 目录
- [ ] 引用：UniDock / P2Rank / AF2BIND 论文（README 已有）

## 6. 首次推送 GitHub 建议流程

```bash
cd RVSAuto   # 仓库根目录；确保未跟踪 p2rank_*、Uni-Dock-main、工作目录
git init   # 若尚未初始化
git add rvsauto tests scripts conda .github README.md LICENSE CHANGELOG.md pyproject.toml MANIFEST.in requirements.txt docs .gitignore
git status   # 确认无 Uni-Dock-main / p2rank / AF2BIND_out
git commit -m "Initial public release of RVSAuto"
git remote add origin https://github.com/Xin-DongXu/RVSAuto.git
git branch -M main
git push -u origin main
git tag v11.0.0
git push origin v11.0.0
```

## 7. Release 说明模板（GitHub Releases）

- 支持 P2Rank / AF2BIND 双口袋引擎
- UniDock 批量虚拟筛选 + 自对接 RMSD
- AlphaFold 受体 PDBQT 多级重试、对接盒体积上限
- P2Rank `probability` 过滤：默认 `>= 0.05`，`--p2rank_min_probability 0` 关闭
- pip 安装：`pip install .` → `rvsauto screen` / `rvsauto redock`
