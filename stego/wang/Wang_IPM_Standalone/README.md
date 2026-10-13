# Wang（2015）HEVC IPM 角度差隐写：独立复现工程

> **这是 Wang 自己的项目，不需要放在 Tar5Stego 下，也不会运行/覆盖 Tar5 的 STC 脚本。**
>
> 改造思路：复用 Tar5 的自定义 x265 与 HM 编解码工具的命令行接口，完全替换 IPM 修改算法。

## 1. 工程目录

```text
Wang_IPM_Standalone/
├── algorithm/
│   ├── wang_ipm.py                 # Wang 角度差规则、嵌入、提取
│   ├── batch_wang_ipm.py           # 可独立调用的生成器
│   └── test_wang_ipm.py            # 映射/嵌入/提取自检
├── bin/
│   ├── C_Coder/x265.exe           # 定制版 x265，通过 setup_tools.ps1 自动获取
│   └── Decoder/TAppDecoder.exe    # 定制版 HM decoder，同上
├── tools/
│   ├── setup_tools.ps1             # 从本机 Tar5 或原始 GitHub 仓库提取两个程序
│   ├── make_full_package.ps1       # 重新打包成含两个 EXE 的真正完整 ZIP
│   └── quick_test.ps1
├── input/
│   ├── cover_ipm/                 # 每视频段的 Cover IPM TXT，用户放入
│   └── videos/                    # 对应原始 YUV，用户放入
├── examples/cover_ipm/            # 小型合成样例，可直接运行，不代表论文实验
├── outputs/                       # 自动生成 Wang TXT、265 码流、解码 IPM 和校验结果
├── run_pipeline.py                # 统一跑 Wang 生成、编码、解码、提取校验
└── README.md
```

**关于 ZIP 中的二进制：** 当前提供的基础压缩包已包含独立 Python 代码与封装接口，但由于当前 GitHub 文件读取渠道不支持直接下载二进制文件，`x265.exe`、`TAppDecoder.exe` 不能在此基础 ZIP 中直接放入。它们在你的 `Tar5Stego` 项目中已经存在，使用下方**一条命令**即可自动复制到 **Wang 项目内部**。然后可直接生成最终的含 EXE 完整 ZIP。完成该步骤后，这套工程不再引用 Tar5 的目录。

此外，**原始 YUV 视频和 Cover IPM 数据**没有公开包含在你的 GitHub 代码仓库中，需使用你已有的数据集，无法随压缩包一起附上。

## 2. 首次安装（只需要一次）

Windows PowerShell 进入**解压后的 Wang 项目目录**（无需进入 Tar5）：

```powershell
cd E:\xhy\Wang_IPM_Standalone
.\tools\setup_tools.ps1
```

默认自动从你之前的 `E:\xhy\Tar5Stego` 取出以下文件并复制到 Wang 独立目录：

```text
Tar5Stego/C_Coder/x265.exe      => Wang_IPM_Standalone/bin/C_Coder/x265.exe
Tar5Stego/Decoder/TAppDecoder.exe => Wang_IPM_Standalone/bin/Decoder/TAppDecoder.exe
```

如果你的 Tar5 不在默认路径：

```powershell
.\tools\setup_tools.ps1 -Tar5Path 'D:\Project\Tar5Stego'
```

如果本地源目录不存在，脚本会尝试直接从 [XHY-five/Tar5Stego](https://github.com/XHY-five/Tar5Stego) 下载同一份二进制。网络无法访问 GitHub raw 时请使用本地目录方式。若 PowerShell 禁止执行脚本，可以临时在**当前终端**执行 `Set-ExecutionPolicy -Scope Process Bypass`（非永久修改），再运行脚本。

完成工具提取后，如果你想要将 Wang 交给另一台电脑，并且**压缩包本身带 EXE**，请在上述 Wang 目录执行：

```powershell
.\tools\make_full_package.ps1
```

将生成同级文件 `Wang_IPM_Standalone_FULL.zip`，里面包括两个定制 EXE 和所有 Wang 源码。**不要把大型输入 YUV 与输出结果混进完整 ZIP**；打包脚本已排除它们。

## 3. 先运行不需要外部工具的单元测试

```powershell
python .\algorithm\test_wang_ipm.py
```

预期：`ALL TESTS PASSED`。

然后用项目自带合成样例尝试 Wang TXT 生成：

```powershell
python .\run_pipeline.py --steps embed `
  --cover-root .\examples\cover_ipm `
  --only-segment BQMallsegment000 --payloads 0.3
```

生成位置：

```text
outputs/stego_ipm/payload_0.3/BQMallsegment000/
   BQMallsegment000_stego_IPM.txt
   BQMallsegment000_stego_IPM.txt.json
   BQMallsegment000_stego_IPM.txt.bits
```

说明：这个样例没有真实视频，不能用来跑 x265；它只是检查 Python 生成结果。

## 4. 放入真实数据（不再依赖 Tar5 项目）

**Cover IPM**：把原本 Tar3/Tar5 输入所用的 `_cover_ipm.txt` 放进 Wang 的 `input/cover_ipm/<segment>/`。

**YUV**：把对应视频段放入 `input/videos/`（子目录不限）。原 Tar5 约定文件名结构：

```text
input/cover_ipm/BQMallsegment000/BQMallsegment000_cover_ipm.txt
input/videos/某个文件夹/BQMall_832x480_50_segment000.yuv
```

注意：YUV 名称必须让程序识别出 `原片段名`、`宽x高`、`fps`、`segmentNNN`。如果你实际 YUV 的文件名格式不同，请改 `run_pipeline.py:get_videos` 的解析逻辑，或重命名数据。Cover TXT 必须保留原始帧、块顺序与 `CTU Address/坐标/paritionSize/depth/size/IPM` 字段。

## 5. 一条命令完成 Wang → x265 → HM → 提取验证

在 Wang 工程目录内运行：

```powershell
python .\run_pipeline.py --steps embed encode decode verify `
  --cover-root .\input\cover_ipm `
  --video-root .\input\videos `
  --only-segment BQMallsegment000 `
  --payloads 0.3 --qp 35 --frames 60 --keyint 4
```

确认单段成功后去掉 `--only-segment` 并调整 payloads 批量：

```powershell
python .\run_pipeline.py --steps embed encode decode verify `
  --payloads 0.1 0.3 0.5 --qp 35 --frames 60 --keyint 4
```

可以单独指定阶段，例如：

```powershell
python .\run_pipeline.py --steps embed --payloads 0.3
python .\run_pipeline.py --steps encode decode verify --payloads 0.3
```

需要覆盖已有 Wang 输出时加 `--overwrite`。

`run_pipeline.py` 直接调用 `bin/C_Coder/x265.exe` 的定制 `--InputStego=<Wang TXT>`；码流保存至 `outputs/bitstreams/`；`bin/Decoder/TAppDecoder.exe` 的 `--OutputIPMFile`、`--OutputCoeffFile` 保存至 `outputs/decoded_ipm/`；最后对照嵌入 metadata 检查从解码码流提取的比特是否一致。完整流程无需 Tar5 项目及 Tar5 的 STC/DDG 生成器。

## 6. Wang 论文规则及目前的近似

Wang et al. (2015) 将两块连续 4×4 亮度 PU 配对：

- 两个模式均在 2..34 时，将角度差映射至 00/01/10/11 两比特；不匹配时**只改第二块**。
- 两个模式均为 0 或 1 时，按秘密比特设置各块 0/1；一个方向、一个 0/1 的组合跳过。
- 输出与 Tar5 定制 x265 接口一致的 `Frame begin/end` 和完整 `oriIPM/stegIPM` 块表；针对 4×4 PU 仍采用 `paritionSize=3 depth=3 size=8x8` 这一历史格式，拼写不修改。

**严格性界限：** 原论文在满足目标角度差的候选 IPM 中以 RD Cost 选最优；独立生成器在缺少完整 RD Cost 时采用**最近模式近似**，并记录 `nearest_fallback`。若已另有 Tar5 生成的 DDG（不需要整个 Tar5 项目），可以指定 `--ddg-root` 路径，但通常只有部分候选，不代表完整 RD 最优。**这不是已严格复现论文中的码率失真最优性。**

论文还要求重新编码后的 4×4 PU 在最终编码决策中保留。自定义 x265 会如何处理要求修改的预测模式、解码后结构是否变化，必须根据 `verify` 的真实码流输出确认。仅在计划 TXT 上提取成功不能证明码流级复现。

注意：`payload=0.3` 在此是“活动帧每个 4×4 PU 的目标秘密比特数”，**不是论文规定的 2 bit/有效块对的全部理论容量**。为了同时跑 0.1/0.3/0.5 进行对照而设置了这个接口；生成的 JSON 保存了实际容量。

## 7. 核心接口示例

```text
Frame 1 begin
CTU Address: 4 x=272,y=24 paritionSize=3 depth=3 size=8x8 oriIPM=26 stegIPM=25
CTU Address: 4 x=276,y=24 paritionSize=3 depth=3 size=8x8 oriIPM=20 stegIPM=21
Frame 1 end
```

这个格式与 [Tar5 原版 stc_embed_ipm.py](https://github.com/XHY-five/Tar5Stego/blob/main/stc_embed_ipm.py) 的输出相同；实际编码逻辑借鉴 [Cencoder.py](https://github.com/XHY-five/Tar5Stego/blob/main/Cencoder.py)，实际解码逻辑借鉴 [StegoDecoder.py](https://github.com/XHY-five/Tar5Stego/blob/main/StegoDecoder.py)。没有复制 Tar5 原本的 STC 层级处理代码，因为 Wang 无需这些模块。

## 8. 本次自检与限制

- Wang 映射 + 合成 IPM 测试：通过，84 bit 比对一致。
- Wang 自带多帧样例：生成 TXT 后再本地提取，36 bit 比对一致。
- 定制 Windows x265 + HM：当前工具执行环境无法运行 Windows EXE，且远程 GitHub 二进制未直接获取，因此**尚未验证真实码流**。请先用单段完成完整编码/解码与码流提取测试。
- 可执行程序来源及分发许可请以源码的版权说明为准。修改版 x265 源码并未出现在该 GitHub 仓库中，后续公开分发其 EXE 前应核对开源许可与对应源码可得性。
