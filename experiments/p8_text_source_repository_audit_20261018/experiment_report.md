# P8 公开 DNS 清单交叉核验

## 目的

检查公开 DNS-2020 复现实验仓库是否为 P8 候选音频提供原句文本。

## 方法

只读获取 `github.com/IMYBo/SDDNet` 的浅层仓库，检查 `data/clean_cv.lst`。其中能找到与 P8 候选相同的 book/chapter/reader 文件名，例如：

- `book_02324_chp_0015_reader_10658_12.wav`
- `book_09391_chp_0005_reader_10921_18.wav`
- `book_07500_chp_0005_reader_10927_70.wav`
- `book_04733_chp_0031_reader_10929_13.wav`

## 结果

清单每行只有 clean WAV 路径和 `31.0` 秒时长，没有 transcript、sentence、chapter text 或逐句标签字段。仓库中也没有与这些候选对应的原句文件。

## 判定

该公开清单只能交叉确认文件命名和 DNS-2020 数据来源，不能恢复 P8 原句。它不改变现有证据边界：Whisper clean 转写仍只能作听音提示；正式内容判断仍需人工从 clean 核实或获得合法原始文本。
