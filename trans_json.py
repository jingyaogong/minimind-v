import json

# 加载现有JSON数据
with open('full.json', 'r') as f:
    data = json.load(f)

# 创建新的格式
new_data = []

# 定义选择列表的翻译字典
translation_dict = {
    "True": "是的",
    "False": "不是",
    "painted": "涂鸦",
    "ripe": "成熟"
}

for item in data['data']:
    sample_id = str(item['sample_id']).zfill(12)  # 格式化为12位数字
    images = [f"<image>" for _ in item['task_instance']['images_path']]
    
    # 直接从task_instance中获取choice_list
    choice_list = item['task_instance']['choice_list']
    
    # 判断是否需要翻译
    if all(choice in translation_dict for choice in choice_list):
        choice_list = [translation_dict[choice] for choice in choice_list]
        response = translation_dict[item['response']]
    else:
        response = item['response']
    
    # 构造新的对话格式
    conversation = [
        {
            "from": "human",
            "value": f"{''.join(images)}\n{data['metadata']['task_instruction'][item['task_instruction_id']]} 选择列表:  {', '.join(choice_list)}."
        },
        {
            "from": "gpt",
            "value": response
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
