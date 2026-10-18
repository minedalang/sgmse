# P8 听感对照包

每个编号有四段同一时间片段：`clean（参考）`、`Y_pre（前端）`、`Router-off`、`Router`。
所有片段都取原始 30 秒语音的中央 12 秒，采样率 16 kHz。

建议先听 `clean（参考）`，再依次听三个增强结果。clean 只用于听感参考，不能作为推理时输入。

## 只需要听这些现象

- 人声是否突然中断、吞字或少一截；
- 是否出现突然变调、金属声、机械声或不自然的重复；
- 哪个版本整体更容易听清人声；
- 如果三段听起来差不多，直接记为“差不多”。

不需要听懂英文，不需要猜原句，也不需要填写转写。
反馈时只需写类似：`样本03：Router 有明显断续，Y_pre 和 off 差不多`。

`key.csv` 是条件映射和文件校验记录；音频文件名已经标注条件，便于直接播放。

## Clean 自动转写参考

以下文字对应每个 `clean.wav` 的中央 12 秒。它们由 Whisper `tiny.en` 自动识别，仅作参考，不是人工标注；部分词可能识别错误。

### 样本 00
`out, in whom she loved with an all-absorbing passion, had caught the war fever, gone to Canada and enlisted. Mrs. McGuire herself was Canadian by birth, and all-`

### 样本 01
`Then Buster went off, sinking what a kind person old Mr. Crow was. In that very afternoon, long before sunset, he curled himself up in and out of the way corner of the house and went to-`

### 样本 02
`about them, far from it. Although I was handicapped in my reconstitution of their friendship by lack of the Archbishop's letters, he had nevertheless made a note here and there, upon the communications he received from the Countess. Throughout the letters-`

### 样本 03
`But there are no sharp lines in nature. From the homicidal lunatic to the same man of violent passions, there is a continuous gradation. Even in the most perfect community, there will-`

### 样本 04
`as if other people question them any when as they may indeed certainly will but anything is better than the present condition of things God forbid that you should ever have`

### 样本 05
`The successes for which it strives, the achievements upon which it sets store, are connected with fighting and victory. The presence of this medium incites Belico's exhibitions in a boy, first in games.`

### 样本 06
`from below saw him slip his hands softly to his side and then place something between his teeth. It was his dagger. Reaching up, he clutched the window the seal above him and with a silence.`

### 样本 07
`She had a good deal of affection forever, art, who was unmistakably a clever man, as well as a good looking one. The prospect seemed fair and cloudless. Nothing pressaged the terrible storm that was about to break.`
