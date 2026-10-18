# 三臂中央 12 秒 ASR 辅助对照

同一个 Whisper tiny.en 对 clean/Y_pre/Router-off/Router 解码。clean 也是 ASR 伪参考，以下不能当作真实 WER，只用于发现需要重点听的样本。

## 样本 0（SNR -19 dB）

**clean 参考：** out, in whom she loved with an all-absorbing passion, had caught the war fever, gone to Canada and enlisted. Mrs. McGuire herself was Canadian by birth, and all-

**y_pre：** And whom she loved with an all-absorbing passion had caught the war fever gone to Canada and he listed. Mrs. McGuire herself was commuting by birth.
- 相对 clean ASR 的归一化编辑距离：0.241

**off：** and whom she loved with an all-absorbing passion had caught the war fever gone to Canada and in this demo. Mrs. Bivoir, for self-esteem, my birth, I'm
- 相对 clean ASR 的归一化编辑距离：0.414

**router：** and whom she loved with an all-absorbing passion had caught the war fever gone to Canada and in this dingle. Mrs. Bivoir, for self-esteemating my first.
- 相对 clean ASR 的归一化编辑距离：0.414

## 样本 1（SNR -17 dB）

**clean 参考：** Then Buster went off, sinking what a kind person old Mr. Crow was. In that very afternoon, long before sunset, he curled himself up in and out of the way corner of the house and went to-

**y_pre：** Fender, went off. That's what you're saying, Mr. Harwell. That's very out of the long before I send the car. The cars are so locked in the way we want to rip the house and once this is...
- 相对 clean ASR 的归一化编辑距离：0.865

**off：** Ben Duster went off with a dipping light fight first and would history of roles. At Barry F. Gimliou, we'll bomb the brovers handhand. Yeah, I have a self-bought to have. Look my hand's owner of the house and my s-
- 相对 clean ASR 的归一化编辑距离：1.027

**router：** Ben Duster went off, did he want fine first and old history roles? At Barry F. and Lee were wrong the brothers had turned the eye of a self-bought I look my hand's corner of the house and my...
- 相对 clean ASR 的归一化编辑距离：0.919

## 样本 2（SNR -17 dB）

**clean 参考：** about them, far from it. Although I was handicapped in my reconstitution of their friendship by lack of the Archbishop's letters, he had nevertheless made a note here and there, upon the communications he received from the Countess. Throughout the letters-

**y_pre：** far from it. Although I was handicapped by eating my teeth from the war bullshit by lack of lagerb
- 相对 clean ASR 的归一化编辑距离：0.786

**off：** Far from it, although I was at Capradale, Montecucian, you love Bencha, by Raga, Bora, Bora, Trits, and Bloodos. He never mores either no take where, have public communications he receives from the Countess. Throughout her...
- 相对 clean ASR 的归一化编辑距离：0.667

**router：** Far from it, although I was at Capradale, Montecucian, you love Bencha, by Rachaburaj, Richard Budders. He never mores either no take where, of public communications he receives from the throughout turn
- 相对 clean ASR 的归一化编辑距离：0.690

## 样本 3（SNR -17 dB）

**clean 参考：** But there are no sharp lines in nature. From the homicidal lunatic to the same man of violent passions, there is a continuous gradation. Even in the most perfect community, there will-

**y_pre：** But there are no sharp light.
- 相对 clean ASR 的归一化编辑距离：0.844

**off：** But there are no shy buying of the match and in a depth some end of the time they'll say Also, oh, oh, and I waking the hiring From there to pray the minute
- 相对 clean ASR 的归一化编辑距离：0.938

**router：** But there are no shy playing of the box and in a deck some end of the time they'll say Pause those whole hook and I working the hiring From there to pray the minute
- 相对 clean ASR 的归一化编辑距离：0.969

## 样本 4（SNR -16 dB）

**clean 参考：** as if other people question them any when as they may indeed certainly will but anything is better than the present condition of things God forbid that you should ever have

**y_pre：** you should ever have
- 相对 clean ASR 的归一化编辑距离：0.871

**off：** of all the stories had evolved to horsemen in a tase, and they were learned to go down realists. They still called this Bistips, they were in the trace of the azure, and not the way that you should ever have...
- 相对 clean ASR 的归一化编辑距离：1.097

**router：** 
- 相对 clean ASR 的归一化编辑距离：1.000

## 样本 5（SNR -17 dB）

**clean 参考：** The successes for which it strives, the achievements upon which it sets store, are connected with fighting and victory. The presence of this medium incites Belico's exhibitions in a boy, first in games.

**y_pre：** The
- 相对 clean ASR 的归一化编辑距离：0.971

**off：** to assess how gat it strives, the achievements upon which it sets the lie out. A connect-out and fighting to fix the presence of ultimatial insights, elicose exhibitions, and a void. First in games.
- 相对 clean ASR 的归一化编辑距离：0.588

**router：** to assess how the added stripes, the achievements upon which it sets the light out. A connect-out and fighting to fix the presence of ultimatial, insights, elicose, exhibitions, and a void. First in games.
- 相对 clean ASR 的归一化编辑距离：0.647

## 样本 6（SNR -19 dB）

**clean 参考：** from below saw him slip his hands softly to his side and then place something between his teeth. It was his dagger. Reaching up, he clutched the window the seal above him and with a silence.

**y_pre：** from the low saw him slip his hands softly to his side and then face something between his teeth. It was his dagger reaching up. He clutched the window seal a bucket with a
- 相对 clean ASR 的归一化编辑距离：0.222

**off：** from below saw him slip his hands softly to his side and then face something between his teeth. It was his dagger. Reaching up, he clutched the window sill above him with a
- 相对 clean ASR 的归一化编辑距离：0.139

**router：** from below saw him slip his hands softly to his side and then brace something between his teeth it was his dagger reaching up he clutch the window sill above him with a shut
- 相对 clean ASR 的归一化编辑距离：0.167

## 样本 7（SNR -17 dB）

**clean 参考：** She had a good deal of affection forever, art, who was unmistakably a clever man, as well as a good looking one. The prospect seemed fair and cloudless. Nothing pressaged the terrible storm that was about to break.

**y_pre：** Where did you know if it was actually forever out? It was unmistakably a flat land, as well as a good looking world. The prospect seemed fair and cleibreous. Nothing pressaged a terrible storm that was a back to the point.
- 相对 clean ASR 的归一化编辑距离：0.500

**off：** He had a dual of affectionate, Raphth art, who was unmistakiny at time and land, as well as a good looking them. The Prospeccy, Faren, Playbours. Nothing pressaged to terror the storm that was abact-
- 相对 clean ASR 的归一化编辑距离：0.579

**router：** He had a tool of affectionate for F-ars, who was unmistakingly at time of land, as well as a good looking them. The prospectsy, fair and pliebers. Nothing trussaged and terrible storm that was abact-
- 相对 clean ASR 的归一化编辑距离：0.553
