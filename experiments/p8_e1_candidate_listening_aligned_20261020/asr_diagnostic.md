# 修复后原候选 ASR 辅助诊断

同一 Whisper tiny.en beam-5 对 20 个修复片段解码。clean 提示来自自动识别/既有文本，仅用于试听定位；不能作为真实 WER 或内容恢复证据。

## file-id 5243
clean 自动提示：...gone to Canada and enlisted. Mrs. McGuire herself was Canadian by birth, and all-

- **clean**（伪距离 0.714）：was Canadian by birth.
- **noisy**（伪距离 1.000）：（空）
- **y_pre**（伪距离 0.857）：self as community by birth.
- **off**（伪距离 0.929）：self is to make you my birth.
- **router**（伪距离 0.929）：self is to make you my birth.

## file-id 4812
clean 自动提示：Although I was handicapped in my reconstitution of their friendship by lack of the Archbishop's letters...

- **clean**（伪距离 0.706）：to my reconstitution of their friendship.
- **noisy**（伪距离 1.000）：（空）
- **y_pre**（伪距离 0.941）：cat but I need my attention to the work, won't you?
- **off**（伪距离 1.000）：capper that I even want to cook in your love, Bensha.
- **router**（伪距离 1.000）：cap for that even while the cooking you love, Bensha.

## file-id 11999
clean 自动提示：...there is a continuous gradation. Even in the most perfect community, there will-

- **clean**（伪距离 0.538）：even in the most perfect community.
- **noisy**（伪距离 1.000）：（空）
- **y_pre**（伪距离 1.000）：of those points. You mean it?
- **off**（伪距离 1.000）：So I'm gonna have to pray it in minute
- **router**（伪距离 0.923）：Some dead preyed in minute

## file-id 10720
clean 自动提示：...they may indeed certainly will, but anything is better than the present condition of things...

- **clean**（伪距离 0.800）：indeed certainly will.
- **noisy**（伪距离 1.000）：Thank you.
- **y_pre**（伪距离 1.000）：Make it back as well with this.
- **off**（伪距离 0.933）：on there into the old Albreedus.
- **router**（伪距离 0.933）：on the old Albreedus.
