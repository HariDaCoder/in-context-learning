"""
Script để chạy lại evaluation/metrics cho các experiments đã train xong.
Chỉ cần thêm run_path vào danh sách RUN_PATHS và chạy script.
"""

import os
from pathlib import Path
from eval import get_run_metrics

# Đường dẫn project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

# =============================================================================
# DANH SÁCH CÁC RUN CẦN TÍNH LẠI METRICS
# =============================================================================
# Thêm tên folder của experiment vào đây (tên folder trong models/)
# Ví dụ: "fig3_noise_bernoulli_p0.3", "fig1_exp_w_rate0.5", etc.

RUN_PATHS = [
    # Thêm tên experiments vào đây:
    "fig3_noise_bernoulli_p0.3",
    "fig3_noise_gamma_k4.0_lambda1.0",
    "fig3_noise_poisson_lambda2.0",
    # "fig3_noise_poisson_lambda3.0",
    # "fig3_noise_t-student_df3.0",
]

# =============================================================================
# CẤU HÌNH
# =============================================================================
FORCE_RECOMPUTE = True  # True = luôn tính lại, False = chỉ tính nếu chưa có hoặc cũ
SKIP_BASELINES = True   # True = chỉ tính cho model chính, False = tính cả baselines


def recompute_single_run(run_name, force=True, skip_baselines=True):
    """Chạy lại metrics cho một run"""
    run_path = MODELS_DIR / run_name
    
    if not run_path.exists():
        print(f"❌ KHÔNG TÌM THẤY: {run_path}")
        return False
    
    print(f"\n{'='*70}")
    print(f"📊 Đang tính metrics cho: {run_name}")
    print(f"{'='*70}")
    
    try:
        # Xóa file metrics cũ nếu force recompute
        metrics_file = run_path / "metrics.json"
        if force and metrics_file.exists():
            print(f"🗑️  Xóa metrics cũ...")
            metrics_file.unlink()
        
        # Chạy evaluation
        metrics = get_run_metrics(
            str(run_path),
            step=-1,
            cache=True,
            skip_model_load=False,
            skip_baselines=skip_baselines
        )
        
        print(f"✅ THÀNH CÔNG: {run_name}")
        print(f"📁 Metrics saved to: {metrics_file}")
        return True
        
    except Exception as e:
        print(f"❌ LỖI khi tính metrics cho {run_name}:")
        print(f"   {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print(f"\n{'#'*70}")
    print(f"🔄 RECOMPUTE METRICS")
    print(f"📂 Models directory: {MODELS_DIR}")
    print(f"📊 Total runs to process: {len(RUN_PATHS)}")
    print(f"⚙️  Force recompute: {FORCE_RECOMPUTE}")
    print(f"⚙️  Skip baselines: {SKIP_BASELINES}")
    print(f"{'#'*70}\n")
    
    if not RUN_PATHS:
        print("⚠️  Danh sách RUN_PATHS trống!")
        print("   Vui lòng thêm tên experiments vào RUN_PATHS trong file này.")
        return
    
    success_count = 0
    failed_runs = []
    
    for i, run_name in enumerate(RUN_PATHS, 1):
        print(f"\n[{i}/{len(RUN_PATHS)}] Processing: {run_name}")
        
        success = recompute_single_run(
            run_name,
            force=FORCE_RECOMPUTE,
            skip_baselines=SKIP_BASELINES
        )
        
        if success:
            success_count += 1
        else:
            failed_runs.append(run_name)
    
    # Tổng kết
    print(f"\n{'#'*70}")
    print(f"✅ HOÀN THÀNH!")
    print(f"📊 Thành công: {success_count}/{len(RUN_PATHS)}")
    if failed_runs:
        print(f"❌ Thất bại: {len(failed_runs)}")
        print(f"   Runs failed:")
        for run in failed_runs:
            print(f"   - {run}")
    print(f"{'#'*70}\n")


if __name__ == "__main__":
    main()
