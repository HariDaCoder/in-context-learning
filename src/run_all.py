import os
import uuid
import yaml
import argparse
import sys
import tempfile
from quinine import QuinineArgumentParser

from schema import schema as quinine_schema
from train import main as train_main


def prepare_out_dir(args):
    if not args.test_run:
        run_id = args.training.resume_id
        if run_id is None:
            run_id = str(uuid.uuid4())

        out_dir = os.path.join(args.out_dir, run_id)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        args.out_dir = out_dir

        # Persist the resolved config for this run (mirrors train.py behaviour)
        with open(os.path.join(out_dir, "config.yaml"), "w") as yaml_file:
            yaml.dump(args.__dict__, yaml_file, default_flow_style=False)


def build_parser():
    parser = argparse.ArgumentParser(description="Run all noisy_linear_regression variants")
    parser.add_argument(
        "--config",
        default="src/conf/toy.yaml",
        help="Base config yaml (e.g., src/conf/toy.yaml)",
    )
    parser.add_argument(
        "--noise_types",
        nargs="*",
        default=[
            "uniform",
            "t-student",
            "rayleigh",
            "normal",
            "exponential",
            "beta",
            "poisson",
            "cauchy",
            "laplace",
        ],
        help="Which noise_type values to iterate",
    )
    parser.add_argument(
        "--base_run_name",
        default="noisy_sweep",
        help="Prefix for wandb.name; final name will be '<noise>_<base_run_name>'",
    )
    return parser


def run_one(base_config_path: str, noise_type: str, base_run_name: str):
    # --- ĐÂY LÀ CHỖ SỬA ---
    # 1. Lấy đường dẫn thư mục của file config gốc
    #    ví dụ: /content/in-context-learning/src/conf
    config_dir = os.path.dirname(base_config_path)
    # --- KẾT THÚC SỬA LỖI ---

    # 2. Đọc nội dung file config gốc
    with open(base_config_path, 'r') as f:
        base_config = yaml.safe_load(f)

    # 3. Sửa đổi dictionary config trong Python
    base_config['training']['task'] = 'noisy_linear_regression'
    base_config['training']['task_kwargs'] = {'noise_type': noise_type}
    base_config['wandb']['name'] = f"{noise_type}_{base_run_name}"
    base_config['training']['resume_id'] = noise_type
    # 4. Tạo một file config tạm thời
    temp_config_file = tempfile.NamedTemporaryFile(
        mode='w+t', 
        delete=False, 
        suffix='.yaml',
        dir=config_dir  # <-- YÊU CẦU TẠO FILE TẠM TRONG THƯ MỤC CẤU HÌNH
    )
    
    try:
        # 5. Ghi config mới vào file tạm thời
        yaml.dump(base_config, temp_config_file, default_flow_style=False)
        temp_config_file.close()

        # 6. Xây dựng danh sách đối số CHỈ chứa file config mới
        cli_args_list = ["--config", temp_config_file.name]

        # 7. Sử dụng "thủ thuật" sys.argv để chạy parser
        qparser = QuinineArgumentParser(schema=quinine_schema)
        original_argv = sys.argv
        try:
            sys.argv = ["run_one_script_placeholder"] + cli_args_list
            args = qparser.parse_quinfig()
        finally:
            sys.argv = original_argv  # Luôn khôi phục argv gốc

        # 8. Chuẩn bị thư mục output và chạy training
        prepare_out_dir(args)
        train_main(args)

    finally:
        # 9. Luôn đảm bảo xóa file tạm thời sau khi hoàn tất
        os.remove(temp_config_file.name)
def main():
    parser = build_parser()
    cli_args = parser.parse_args()

    for noise in cli_args.noise_types:
        run_one(cli_args.config, noise, cli_args.base_run_name)


if __name__ == "__main__":
    main()

