# Benign/Harmful ICL - Quick Notes

## Core idea
- Muc tieu: tim nguong chuyen tu benign sang harmful theo training dynamics.
- Dau hieu harmful: train loss van giam (hoac hoi tu) nhung test loss tang len.
- De tach hieu ung ro rang, dung 2 nhom:
  - OOD-noise: giu data goc, chi thay noise scale.
  - OOD-data: doi data sang nonstation/markov.

## File map (ngan gon)
- plot: `src/plot_benign_harmful_dynamics.py`
  - Quet checkpoint theo step.
  - Eval ID + OOD moi step.
  - Ve do thi loss (y) theo training step (x).
  - Danh dau best_step va harmful_onset_step.

- process: `src/eval.py`
  - Ham `build_benign_harmful_dynamics_evals(...)` tao profile ID/OOD.
  - Ho tro sweep noise scale cho test.
  - Ho tro optional nonstation data shift.

- process train: `src/train.py`
  - Train model va luu checkpoint (`model_<step>.pt`, `state.pt`).
  - Day la nguon du lieu de file plot quet theo step.

- idea/config: `src/conf/benign_harmful_dynamics.yaml`
  - Noi set data/task/noise cho run benhign-harmful.
  - Co the set train noise = 0.0 (clean) hoac >0 (noisy train).

## Minimal run flow
1. Train 1 run:
   - `python src/train.py --config src/conf/benign_harmful_dynamics.yaml`
2. Plot dynamics theo checkpoint:
   - `python src/plot_benign_harmful_dynamics.py --run_path <RUN_PATH> --ood_noise_scales 0.5 1.0 1.5 2.0 --use_noise_multipliers`
3. Doc output:
   - PNG: do thi train-step vs loss.
   - JSON: series ID/OOD + best_step + harmful_onset_step.

## Build-up nhanh (truoc khi chay lon)
1. Quet dynamics thua de tiet kiem compute:
  - `python src/plot_benign_harmful_dynamics.py --run_path <RUN_PATH> --ood_noise_scales 0.5 1.0 1.5 2.0 --use_noise_multipliers --step_stride 10000 --num_eval_examples 128 --prefix quick_scan`
2. Fix 3 moc step va quet noise (x=noise, y=loss):
  - `python src/plot_benign_harmful_dynamics.py --run_path <RUN_PATH> --ood_noise_scales 0.25 0.5 0.75 1.0 1.25 1.5 2.0 --use_noise_multipliers --min_step 20000 --max_step 100000 --step_stride 10000 --fixed_steps_for_noise_plot 20000 60000 100000 --num_eval_examples 256 --prefix fixed_step_noise`
3. Doc them artifact moi:
  - `<prefix>_fixed_step_noise.png`: moi line la 1 checkpoint co dinh, bieu dien do nhay theo noise.
  - `<prefix>_fixed_step_noise.json`: metadata cua fixed-step slices.

## Cap nhat moi (da bo sung)
- `src/plot_benign_harmful_dynamics.py`:
  - Co mode fixed-step noise slices:
   - Them arg `--fixed_steps_for_noise_plot`.
   - Them arg `--noise_plot_include_nonstation`.
  - Tao them artifact:
   - `<prefix>_fixed_step_noise.png`
   - `<prefix>_fixed_step_noise.json`

- `src/eval.py`:
  - Da de mac dinh `skip_baselines=True` trong `get_run_metrics(...)`.
  - Khi chay batch qua `python src/eval.py <run_dir>`, baselines cung bi bo qua de tranh treo.

## Cach chay de xai ngay (ban clean train + OOD noise rong)
1. Chinh config train clean:
  - Trong `src/conf/benign_harmful_dynamics.yaml`, dat `training.task_kwargs.noise_std: 0.0`.

2. Train 1 run (giu checkpoint theo step):
  - `python src/train.py --config src/conf/benign_harmful_dynamics.yaml`

3. Quet OOD Gaussian noise std tu 0 den 10 (100 muc), x=step, y=loss:
  - PowerShell:
    - `$scales = 0..99 | ForEach-Object { [Math]::Round($_ * 10.0 / 99, 4) }`
    - `python src/plot_benign_harmful_dynamics.py --run_path <RUN_PATH> --ood_noise_scales $scales --use_absolute_noise_std --step_stride 5000 --num_eval_examples 256 --reduction last --prefix noise_0_10_100`

4. Cat theo 3 moc step (x=noise, y=loss) de thay nguong nhanh:
  - `python src/plot_benign_harmful_dynamics.py --run_path <RUN_PATH> --ood_noise_scales $scales --use_absolute_noise_std --min_step 20000 --max_step 100000 --step_stride 5000 --fixed_steps_for_noise_plot 20000 60000 100000 --num_eval_examples 256 --reduction last --prefix fixed_step_noise_0_10_100`

5. Neu muon bo nhieu checkpoint hon luc scan dau:
  - Tang `--step_stride` (vi du 10000) de giam thoi gian.
  - Sau khi thay vung chuyen pha, giam stride de zoom lai.

## Luu y curriculum
- Neu uu tien train on dinh va nhanh hoi tu: giu curriculum.
- Neu uu tien dien giai truong x=training step that sach: chay them 1 ban no-curriculum de doi chieu.

## Practical tip
- Neu muon bam sat baseline paper:
  - train clean (noise_std=0), test noise tang dan.
- Neu muon thay harmful dynamics ro hon:
  - train co noise (noise_std > 0), sau do van sweep test noise.
