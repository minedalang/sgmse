# P8 辅音/高频声学代理诊断

对修复后的四个同候选目标窗（每窗 0.5 s）计算高频能量、谱平坦度、谱质心、包络变化和瞬态峰值。没有真实文本或音素标签，因此不能把这些量解释成辅音恢复率。

## noisy 相对 clean 的均值/中位数差
- rms_dbfs: mean=12.7228, median=12.6325
- low_band_db: mean=12.5917, median=11.8711
- mid_band_db: mean=13.5494, median=13.5328
- high_band_db: mean=15.6236, median=14.8121
- high_to_mid_db: mean=2.0742, median=-0.5201
- spectral_flatness_80_7600: mean=0.0531, median=0.0214
- spectral_centroid_hz: mean=184.2540, median=120.1922
- envelope_std_db: mean=3.5467, median=3.9956
- transient_peaks_per_s: mean=24.0000, median=24.0000

## y_pre 相对 clean 的均值/中位数差
- rms_dbfs: mean=-8.5131, median=-8.2240
- low_band_db: mean=-11.4112, median=-12.5316
- mid_band_db: mean=-16.2261, median=-18.9659
- high_band_db: mean=-21.5074, median=-18.5172
- high_to_mid_db: mean=-5.2813, median=1.1838
- spectral_flatness_80_7600: mean=-0.0035, median=-0.0034
- spectral_centroid_hz: mean=1.3360, median=-50.4298
- envelope_std_db: mean=-5.2122, median=-4.4190
- transient_peaks_per_s: mean=-26.0000, median=-27.0000

## off 相对 clean 的均值/中位数差
- rms_dbfs: mean=-1.8333, median=-2.3177
- low_band_db: mean=-1.7314, median=-2.3118
- mid_band_db: mean=-7.3161, median=-7.3451
- high_band_db: mean=-13.3561, median=-14.1335
- high_to_mid_db: mean=-6.0401, median=-5.2337
- spectral_flatness_80_7600: mean=-0.0019, median=-0.0014
- spectral_centroid_hz: mean=-62.1404, median=-59.9274
- envelope_std_db: mean=-0.7657, median=-0.8918
- transient_peaks_per_s: mean=-10.5000, median=-10.0000

## router 相对 clean 的均值/中位数差
- rms_dbfs: mean=-3.3847, median=-4.0962
- low_band_db: mean=-3.6849, median=-4.5605
- mid_band_db: mean=-9.0161, median=-9.6346
- high_band_db: mean=-15.1885, median=-16.5335
- high_to_mid_db: mean=-6.1724, median=-4.2089
- spectral_flatness_80_7600: mean=-0.0023, median=-0.0015
- spectral_centroid_hz: mean=-49.0788, median=-61.2849
- envelope_std_db: mean=-1.9693, median=-2.2806
- transient_peaks_per_s: mean=-11.5000, median=-11.0000

## 解释边界

- 高频能量下降可能表示辅音/擦音被抑制，也可能只是原始噪声减少；谱平坦度变化可能来自电流声或噪声，不能单独判定语音内容。
- 该 probe 只用于决定是否值得继续做带人工文本/音素标签的验证，不启动训练。
