# Reproduction guide — In-Context Learning và Benign Overfitting

Tài liệu này mô tả repo theo code hiện tại, để cài môi trường, hiểu pipeline, chạy smoke test và tái tạo các thí nghiệm ICL/benign-harmful/Markov.

## 1. Mục tiêu

Repo hiện thực bài What Can Transformers Learn In-Context? Mỗi episode sinh các cặp (x_i,y_i) từ một hàm ẩn, sau đó Transformer dự đoán y_query từ context. Các mở rộng nghiên cứu benign/harmful overfitting, label noise/SNR, input Markov/AR, train-test dynamics, bias-variance và generalization gap.

## 2. Cấu trúc

- src/train.py: entrypoint training chuẩn.
- src/models.py: GPT-2 Transformer và baseline.
- src/base_models.py: MLP/ParallelNetworks cho GD baseline.
- src/tasks.py: sinh task và label y.
- src/samplers.py: sinh input x.
- src/curriculum.py: schedule số chiều/số point.
- src/schema.py: schema Quinine.
- src/eval.py: load checkpoint, evaluation, aggregate metrics.
- src/conf: YAML; gồm tiny/small/standard, task configs và Markov/benign configs.
- src/run_benign_harmful_icl.py: pipeline unified grid Markov scale x SNR x seed.
- src/run_benign_overfitting_snr_experiment.py: SNR sweep.
- src/markov_benign_overfitting_experiments.py: solver/dynamics Markov.
- src/diagnose_s_shape.py, plot_*.py: diagnostic và plotting.
- models, ket_qua_tiny: output/checkpoint.

## 3. Cài đặt và cách chạy

Repo gốc dùng Python 3.8.12, PyTorch 1.11.0, Transformers 4.17.0. Cài:

    conda env create -f environment.yml
    conda activate in-context-learning
    python -c "import torch, transformers, quinine; print(torch.__version__); print(torch.cuda.is_available())"

Import của training là theo thư mục src, nên chạy chuẩn từ src:

    cd src
    python train.py --config conf/toy.yaml --test_run true

Runner unified có thể chạy từ repo root:

    python src/run_benign_harmful_icl.py --mode all --pilot

## 4. Config quan trọng

model.family hiện build_model chỉ implement gpt2 dù schema cho phép lstm. n_dims là chiều input; n_positions là số point, GPT-2 nội bộ cần 2*n_positions token; n_embd/n_layer/n_head là kích thước Transformer.

training.task chọn task; training.data chọn sampler; task_kwargs/data_kwargs là tham số chi tiết; batch_size là số episode/step; learning_rate thường 1e-4; train_steps là số step; save_every_steps lưu state; keep_every_steps lưu model_<step>.pt; num_tasks là finite task pool; num_training_examples là finite input corpus; seed là seed toàn cục; resume_id là UUID resume.

Curriculum gồm dims và points với start/end/inc/interval. Sau mỗi interval step, giá trị tăng inc nhưng không vượt end. n_dims_truncated làm các tọa độ còn lại bằng zero.

## 5. Data flow

samplers.get_data_sampler trả xs shape [B,T,d]. Có Gaussian, sparse Gaussian, AR1, AR2, VAR1, VR2, nonstationary, Markov và các distribution sampler uniform/exponential/Laplace/Gamma/Beta/Student-t/Poisson/Rayleigh/Cauchy.

MarkovSampler sinh x_t = A_t x_(t-1) + eta_t; hỗ trợ stationary, drift, regime_switch. markov_scale là scale của A. normalize_variance=true chọn innovation noise để variance biên xấp xỉ 1/d, giúp so sánh IID scale=0 và Markov.

tasks.get_task_sampler tạo task; task.evaluate(xs) trả ys shape [B,T]. Task chính: LinearRegression, SparseLinearRegression, LinearClassification, NoisyLinearRegression, MarkovNoisyLinearRegression, NoisyContextCleanQueryRegression, UniformHypersphereRegression, Relu2nnRegression, DecisionTree và QuadraticRegression.

NoisyContextCleanQueryRegression là task chính unified benign/harmful: label context nhiễu nhưng query label sạch. NoisyLinearRegression là linear score cộng noise. Metric chuẩn squared error; training metric thường mean squared error.

TransformerModel interleave x và y thành hai token/point. _read_in: Linear(d -> n_embd), GPT-2 backbone, _read_out: Linear(n_embd -> 1). Output lấy vị trí chẵn [:,::2,0], vì đó là vị trí dự đoán. Input xs [B,T,d], ys [B,T], output [B,T].

## 6. train.py

Flow: parse YAML -> tạo UUID output dir -> ghi config.yaml -> build Transformer -> tạo Curriculum/sampler/task -> mỗi step sample xs, sample task, tính ys, forward, MSE, backward, Adam -> log WandB -> lưu checkpoint.

training.seed seed Python/NumPy/Torch/CUDA. num_training_examples chọn seed input trong [0,N), task dùng seed+1. num_tasks tạo finite pool task; hai tham số này khác nhau. Training chuẩn gọi model.cuda(), nên cần CUDA. test_run=true bỏ WandB/checkpoint, ép curriculum về end và chạy 100 step. Nếu state.pt tồn tại, model/optimizer/step được restore.

Output chuẩn: out_dir/UUID/config.yaml, state.pt, train_losses.json và model_<step>.pt nếu keep_every_steps phù hợp. state.pt bị overwrite; model_<step>.pt dùng cho dynamics. eval.get_model_from_run step=-1 load state, step>=0 load model file.

## 7. Evaluation và baselines

eval.py có get_model_from_run, eval_batch, eval_model, build_evals, aggregate_metrics, compute_evals, get_run_metrics. ID giữ distribution train; OOD noise đổi noise_std; OOD nonstation đổi input sampler. train.py/eval.py đều sanitize kwargs bằng whitelist.

models.py có NNModel, LeastSquaresModel (torch.linalg.lstsq, CPU), AveragingModel, Lasso, Ridge, GDModel với MLP, DecisionTree, XGBoost, LP và ADMM. Phần lớn baseline refit tại từng prefix nên chậm hơn Transformer.

## 8. Unified benign/harmful ICL

Chạy: python src/run_benign_harmful_icl.py --mode all --pilot; hoặc mode train/eval/plot. Default grid: SNR [0.3,1,3,10,30,100], markov_scale [0,0.3,0.6,0.8,0.9], seed [0,1,2], tức 90 runs. Pilot: SNR [1,10,100], Markov [0,0.6], seed [0].

Runner đổi SNR theo sigma_y = 1/sqrt(SNR). Base dùng d=20, T=41, n_positions=81, clean-query task, num_tasks=32, num_training_examples=1024. Mỗi combination sinh YAML riêng, train, skip/resume run cũ, rồi evaluate checkpoint.

Presets: tiny=(n_embd 32, 1 layer, 1 head), small=(64,2,2), medium=(128,4,4). Lúc đầu dùng jobs_per_gpu=1 để tránh OOM.

## 9. SNR và Markov runners

run_benign_overfitting_snr_experiment.py tạo config theo noise/seed, train, evaluate và xuất results.csv, results_aggregated.csv, transition_summary.csv, PNG và interpretation_notes.md. Phải phân biệt noise std với variance; unified pipeline dùng sigma_y=1/sqrt(SNR), runner cũ cần đọc make_generated_config/signal_variance.

markov_benign_overfitting_experiments.py sinh A, Markov data, OLS, split evaluation và dynamics. Config chuẩn: T=200, d=50, train_T=150, num_trials=64, stationary_scales [0,.3,.6,.9], sigma_x [.1,.5,1,2], sigma_y [0,.01,.1,.5,1]. markov_train_py_integration.yaml và markov_transformer_dynamics.yaml nối MarkovSampler vào train.py.

## 10. Plot/diagnostic

plot_benign_harmful_icl.py vẽ dynamics, SNR slice, heatmap OOD/gap, Markov comparison, phase boundary. plot_markov_compare.py kết hợp train loss và Markov evaluation. diagnose_s_shape.py tạo train/test loss, bias-variance, OLS diagnostics, parameter norm, generalization gap và diagnostics.json. recompute_metrics.py tính lại metric cũ.

## 11. Recipe reproduce

1. Smoke test: cd src; python train.py --config conf/toy.yaml --test_run true.
2. Copy benign_harmful_dynamics_tiny.yaml, giảm train_steps vài nghìn, đặt keep_every_steps khoảng 500, dùng out_dir mới.
3. Chạy unified pilot: python src/run_benign_harmful_icl.py --mode all --pilot.
4. Kiểm tra mỗi run có config.yaml, state.pt, model_<step>.pt và raw evaluation.
5. Chỉ sau đó chạy full grid; lưu command, commit, GPU, seed và environment.
6. Đọc cả JSON/CSV và PNG; kiểm tra step, seed, Markov scale, SNR và sample count.

## 12. Gotchas

- Training chuẩn cần CUDA; family=lstm chưa được build.
- n_positions là số point, không phải số token.
- state.pt zero-based và bị overwrite; dynamics cần keep_every_steps.
- WandB cần login/entity/project đúng; test_run bỏ WandB.
- Relative out_dir phụ thuộc working directory; config cũ có absolute path máy khác.
- Kwargs thừa bị sanitize; kiểm tra whitelist khi debug.
- CUDA/PyTorch/version/thread có thể làm kết quả không deterministic tuyệt đối.
- Prediction point đầu tiên bằng zero là expected.
- Phân biệt SNR, noise std và noise variance.

## 13. Checklist

- [ ] Ghi commit, Python, PyTorch, Transformers, CUDA/GPU.
- [ ] Lưu base và generated config.
- [ ] Ghi training/task/evaluation seed.
- [ ] Có state.pt và model_<step>.pt.
- [ ] Lưu raw metrics ngoài PNG.
- [ ] Kiểm tra ID/OOD/train loss và checkpoint count.
- [ ] Sửa absolute paths/WandB entity.
- [ ] Pilot chạy đúng trước full grid.

## 14. Sơ đồ module

YAML -> schema.py -> train.py -> curriculum.py + samplers.py + tasks.py -> models.py -> optimizer/loss -> checkpoints -> eval.py -> plot/diagnostic scripts.
