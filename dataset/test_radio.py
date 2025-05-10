import os
import json
import random
from argparse import ArgumentParser

def split_jsonl_files(input_dir, output_dir, train_ratio=0.8, shuffle=True, seed=42):
    """
    将目录中的所有JSONL文件合并后按比例划分为训练集和测试集

    参数:
        input_dir: 包含JSONL文件的输入目录路径
        output_dir: 输出目录路径
        train_ratio: 训练集比例（默认0.8）
        shuffle: 是否打乱数据（默认True）
        seed: 随机种子（默认42）
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 获取所有JSONL文件
    jsonl_files = [
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.endswith('.jsonl')
    ]

    if not jsonl_files:
        raise FileNotFoundError(f"No JSONL files found in {input_dir}")

    # 读取所有数据
    all_data = []
    for file_path in jsonl_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    all_data.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    print(f"Warning: 解析失败，跳过无效行 in {file_path}")

    # 打乱数据
    if shuffle:
        random.seed(seed)
        random.shuffle(all_data)

    # 划分数据集
    split_idx = int(len(all_data) * train_ratio)
    train_data = all_data[:split_idx]
    test_data = all_data[split_idx:]

    # 写入文件
    train_path = os.path.join(output_dir, 'train.jsonl')
    test_path = os.path.join(output_dir, 'test.jsonl')

    for data, path in [(train_data, train_path), (test_data, test_path)]:
        with open(path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"划分完成！共处理 {len(all_data)} 条数据")
    print(f"训练集: {len(train_data)} 条 -> {train_path}")
    print(f"测试集: {len(test_data)} 条 -> {test_path}")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-i', '--input_dir', required=True, help='输入目录路径')
    parser.add_argument('-o', '--output_dir', required=True, help='输出目录路径')
    parser.add_argument('-r', '--train_ratio', type=float, default=0.8,
                        help='训练集比例（默认0.9）')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子（默认42）')
    parser.add_argument('--no_shuffle', action='store_false', dest='shuffle',
                        help='禁用数据打乱')

    args = parser.parse_args()

    split_jsonl_files(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        shuffle=args.shuffle,
        seed=args.seed
    )