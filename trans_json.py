import json

# 加载现有JSON数据
with open('full.json', 'r') as f:
    data = json.load(f)

# 创建新的格式
new_data = []

for item in data['data']:
    sample_id = str(item['sample_id']).zfill(12)  # 格式化为12位数字
    source_image = f"<image>"  # 源图像占位符
    target_image = f"<image>"  # 目标图像占位符
    instruction = data['metadata']['task_instruction'][item['task_instruction_id']]

    # 构造新的对话格式
    conversation = [
        {
            "from": "human",
            "value": f"context: Source Image: {source_image} Target Image: {target_image} Instruction: {instruction}."
        },
        {
            "from": "gpt",
            "value": item['response']
        }
    ]
    
    # 添加到新数据中
    new_data.append({
        "id": sample_id,
        "image": ', '.join(item['task_instance']['images_path']),
        "conversations": conversation
    })

# 保存为新的JSON格式
with open('output.json', 'w') as f:
    json.dump(new_data, f, ensure_ascii=False, indent=2)
