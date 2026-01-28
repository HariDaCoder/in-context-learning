# -*- coding: utf-8 -*-
"""
Script de chay lai evaluation/metrics cho cac experiments da train xong.
Chi can them run_path vao danh sach RUN_PATHS va chay script.
"""

import os
import sys
import signal
from pathlib import Path
from eval import get_run_metrics

# Timeout handler
class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("Evaluation timed out after 1 hour")

# Duong dan project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

# =============================================================================
# DANH SACH CAC RUN CAN TINH LAI METRICS
# =============================================================================
# Them ten folder cua experiment vao day (ten folder trong models/)
# Vi du: "fig3_noise_bernoulli_p0.3", "fig1_exp_w_rate0.5", etc.

RUN_PATHS = [
    # Them ten experiments vao day:
    # "fig3_noise_bernoulli_p0.3",
    # "fig3_noise_gamma_k4.0_lambda1.0",
    # "fig3_noise_poisson_lambda2.0",
    # "fig3_noise_poisson_lambda3.0",
    "fig3_noise_t-student_df3.0",
]

# =============================================================================
# CAU HINH
# =============================================================================
FORCE_RECOMPUTE = True  # True = luon tinh lai, False = chi tinh neu chua co hoac cu
SKIP_BASELINES = False  # True = chi tinh cho model chinh, False = tinh ca baselines


def recompute_single_run(run_name, force=True, skip_baselines=True):
    """Chạy lại metrics cho một run"""
    run_path = MODELS_DIR / run_name
    
    if not run_path.exists():
        print(f"[ERROR] KHONG TIM THAY: {run_path}")
        sys.stdout.flush()
        return False
    
    # Kiểm tra nếu config.yaml tồn tại trực tiếp
    if not (run_path / "config.yaml").exists():
        # Nếu không, tìm subfolder (wandb run_id) chứa config.yaml
        subfolders = [d for d in run_path.iterdir() if d.is_dir()]
        if subfolders:
            # Lấy subfolder đầu tiên (thường chỉ có 1)
            run_path = subfolders[0]
            print(f"   [FOUND] Run subfolder: {run_path.name}")
            sys.stdout.flush()
        else:
            print(f"[ERROR] KHONG TIM THAY config.yaml trong: {run_path}")
            sys.stdout.flush()
            return False
    
    print(f"\n{'='*70}")
    print(f"[EVAL] Dang tinh metrics cho: {run_name}")
    print(f"{'='*70}")
    sys.stdout.flush()
    
    try:
        # Xóa file metrics cũ nếu force recompute
        metrics_file = run_path / "metrics.json"
        if force and metrics_file.exists():
            print(f"[DELETE] Xoa metrics cu...")
            sys.stdout.flush()
            metrics_file.unlink()
        
        # Chạy evaluation
        print(f"[LOADING] Tai model va config...")
        sys.stdout.flush()
        
        # Set timeout to 3600 seconds (1 hour)
        if hasattr(signal, 'SIGALRM'):  # Unix only
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(3600)
        
        metrics = get_run_metrics(
            str(run_path),
            step=-1,
            cache=True,
            skip_model_load=False,
            skip_baselines=skip_baselines
        )
        
        if hasattr(signal, 'SIGALRM'):
            signal.alarm(0)  # Cancel alarm
        
        print(f"[SUCCESS] THANH CONG: {run_name}")
        print(f"[SAVED] Metrics saved to: {metrics_file}")
        sys.stdout.flush()
        return True
        
    except TimeoutException as e:
        print(f"[TIMEOUT] {e}")
        sys.stdout.flush()
        return False
    except KeyboardInterrupt:
        print(f"[INTERRUPTED] Bi ngat boi user")
        sys.stdout.flush()
        return False
    except Exception as e:
        print(f"[ERROR] LOI khi tinh metrics cho {run_name}:")
        print(f"   {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        return False


def main():
    print(f"\n{'#'*70}")
    print(f"[RECOMPUTE] RECOMPUTE METRICS")
    print(f"[DIR] Models directory: {MODELS_DIR}")
    print(f"[COUNT] Total runs to process: {len(RUN_PATHS)}")
    print(f"[FORCE] Force recompute: {FORCE_RECOMPUTE}")
    print(f"[SKIP] Skip baselines: {SKIP_BASELINES}")
    print(f"{'#'*70}\n")
    sys.stdout.flush()
    
    if not RUN_PATHS:
        print("[WARNING] Danh sach RUN_PATHS trong!")
        print("   Vui long them ten experiments vao RUN_PATHS trong file nay.")
        sys.stdout.flush()
        return
    
    success_count = 0
    failed_runs = []
    
    for i, run_name in enumerate(RUN_PATHS, 1):
        print(f"\n[{i}/{len(RUN_PATHS)}] Processing: {run_name}")
        sys.stdout.flush()
        
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
    print(f"[COMPLETE] HOAN THANH!")
    print(f"[SUCCESS] Thanh cong: {success_count}/{len(RUN_PATHS)}")
    if failed_runs:
        print(f"[FAILED] That bai: {len(failed_runs)}")
        print(f"   Runs failed:")
        for run in failed_runs:
            print(f"   - {run}")
    print(f"{'#'*70}\n")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
