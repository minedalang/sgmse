"""
调试脚本：单独测试冻结 SEMamba 的推理效果 (GPU版本)
"""
import torch
import soundfile as sf
import os

from sgmse.joint_model import JointSemambaSgmseModel
from sgmse.joint_data_module import JointWaveDataModule

# 设置路径 - 使用冻结 SEMamba 的模型
CKPT_PATH = "joint_train_logs/xqq5xglv-None/last.ckpt"
SEMAMBA_ROOT = "/home/xyf/cwh_project/SEMamba-main/SEMamba-main"
SEMAMBA_CONFIG = "/home/xyf/cwh_project/SEMamba-main/SEMamba-main/exp/SEMamba_trainedby_dnslowsnr-20_-15/config.yaml"
SEMAMBA_CKPT = "/home/xyf/cwh_project/SEMamba-main/SEMamba-main/exp/SEMamba_trainedby_dnslowsnr-20_-15/g_00134000.pth"

# 测试音频 - 使用 DNS 测试集
TEST_WAV = "/home/xyf/cwh_project/DNS-Challenge-1/dns1-repo/synthetic_snr-20_-15_testset/noisy/book_00198_chp_0017_reader_00957_5_8Vo48iZX8Ro_snr-17.85_tl-26_fileid_147.wav"
OUTPUT_DIR = "vb_test_enhanced_dir_by_vb_test_enhanced_dir_by_vb_test_enhanced_dir_by_vb_test_enhanced_dir_by_xqq5xglv_None_v323"

# 使用GPU
device = torch.device("cuda")

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("加载冻结 SEMamba 的 Joint 模型 (GPU)...")
print("=" * 60)

model = JointSemambaSgmseModel.load_from_checkpoint(
    CKPT_PATH,
    map_location=device,
    strict=True,
    semamba_root=SEMAMBA_ROOT,
    semamba_config=SEMAMBA_CONFIG,
    semamba_ckpt=SEMAMBA_CKPT,
    data_module_cls=JointWaveDataModule,
    base_dir=".",
    batch_size=1,
    num_workers=0,
    dummy=True,
)

model.eval()
model = model.to(device)
print("模型加载完成!")

print("\n" + "=" * 60)
print("测试 SEMamba 前端...")
print("=" * 60)

# 加载原始 noisy 音频
y, sr = sf.read(TEST_WAV)
print(f"原始音频: shape={y.shape}, sr={sr}Hz")
print(f"原始音频 stats: mean={y.mean():.6f}, std={y.std():.6f}, min={y.min():.6f}, max={y.max():.6f}")

# 转换为 tensor [1,1,T] 并移到GPU
y_t = torch.from_numpy(y).float().unsqueeze(0).unsqueeze(0).to(device)
print(f"输入 tensor shape: {y_t.shape}, device: {y_t.device}")

# SEMamba 前向
with torch.no_grad():
    y_pre = model._semamba_forward(y_t)

print(f"\nSEMamba 输出: shape={y_pre.shape}")
print(f"SEMamba 输出 stats: mean={y_pre.mean():.6f}, std={y_pre.std():.6f}, min={y_pre.min():.6f}, max={y_pre.max():.6f}")

# 检查是否有 NaN 或 Inf
if torch.isnan(y_pre).any():
    print("WARNING: SEMamba 输出包含 NaN!")
if torch.isinf(y_pre).any():
    print("WARNING: SEMamba 输出包含 Inf!")

# 保存 SEMamba 输出 (移到CPU)
semamba_out_path = os.path.join(OUTPUT_DIR, "semamba_output.wav")
sf.write(semamba_out_path, y_pre.squeeze().cpu().numpy(), sr)
print(f"\nSEMamba 输出已保存到: {semamba_out_path}")

print("\n" + "=" * 60)
print("测试完整流程 (noisy -> SEMamba -> SGMSE)...")
print("=" * 60)

# 对齐长度
from sgmse.joint_model import _match_length
y_pre_aligned = _match_length(y_pre, y_t)

# mode2: y_in = pre_enhanced, y_cond = noisy
# mode3: y_in = noisy, y_cond = pre_enhanced
# 根据模型类型选择
if model.train_type == "mode2":
    y_in = y_pre_aligned.squeeze(0)   # [1,T] - 增强后的作为输入
    y_cond = y_t.squeeze(0)            # [1,T] - noisy 作为条件
else:  # mode3
    y_in = y_t.squeeze(0)              # [1,T] - noisy 作为输入
    y_cond = y_pre_aligned.squeeze(0)  # [1,T] - 增强后的作为条件

print(f"train_type: {model.train_type}")

print(f"y_in shape: {y_in.shape}, y_cond shape: {y_cond.shape}")

# 完整 enhance
with torch.inference_mode():
    enhanced = model.enhance(
        y_in.squeeze(0).cpu(),  # [T] - 移到CPU因为enhance内部可能有问题
        y_cond=y_cond.squeeze(0).cpu(),
        sampler_type="ode",
        predictor="reverse_diffusion",
        corrector="ald",
        N=30,  # 用较少的步数加快调试
        snr=0.5,
    )

print(f"增强后音频: shape={enhanced.shape}")
print(f"增强后 stats: mean={enhanced.mean():.6f}, std={enhanced.std():.6f}, min={enhanced.min():.6f}, max={enhanced.max():.6f}")

# 检查是否有 NaN 或 Inf
if torch.isnan(torch.tensor(enhanced)).any():
    print("WARNING: 增强输出包含 NaN!")
if torch.isinf(torch.tensor(enhanced)).any():
    print("WARNING: 增强输出包含 Inf!")

# 保存增强后的音频
enhanced_path = os.path.join(OUTPUT_DIR, "enhanced_output.wav")
sf.write(enhanced_path, enhanced, sr)
print(f"\n增强后音频已保存到: {enhanced_path}")

print("\n" + "=" * 60)
print("调试完成！请听一下这两个音频：")
print(f"1. SEMamba 输出: {semamba_out_path}")
print(f"2. 完整增强输出: {enhanced_path}")
print("=" * 60)
