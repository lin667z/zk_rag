from cal_metrics import cal_gen,compute_metrics
import json


def load_ground_truth(file_path, max_items=15000):
    """
    加载真实答案数据集
    :param file_path: JSONL文件路径
    :param max_items: 最大读取条目数 (默认34706)
    :return: 包含output标签内容的列表
    """
    ground_truth = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= max_items:
                break
            try:
                data = json.loads(line.strip())
                # 处理可能的键缺失情况（返回空字符串）
                ground_truth.append(data.get('output', ''))
            except json.JSONDecodeError:
                print(f"警告：忽略无效行 {i + 1}")
                continue
    return ground_truth


def load_predictions(file_path, max_items=15000):
    """
    加载预测答案数据集
    :param file_path: JSONL文件路径
    :param max_items: 最大读取条目数 (默认35028)
    :return: 包含answer标签内容的列表
    """
    predictions = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= max_items:
                break
            try:
                data = json.loads(line.strip())
                # 处理可能的键缺失情况（返回空字符串）
                predictions.append(data.get('answer', ''))
            except json.JSONDecodeError:
                print(f"警告：忽略无效行 {i + 1}")
                continue
    return predictions


# 使用示例
if __name__ == "__main__":
    # 加载数据
    gt = load_ground_truth('../dataset/test.jsonl')
    pred = load_predictions('./output_answers.jsonl')


    # 验证数据
    assert len(gt) == len(pred), "数据集长度不一致！"
    print(f"成功加载 {len(gt)} 条数据")
    print("第一条真实数据样例:", gt[0])
    print("第一条预测数据样例:", pred[0])
    cal_gen(gt,pred)
    compute_metrics(pred, gt)

