import json
from tqdm import tqdm

# 输入和输出文件路径
input_file = 'pretrain_data.jsonl'
output_file = 'pretrain_t2i_data.jsonl'

# 打开输入文件和输出文件
with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
    for line in tqdm(infile):
        # 读取每行JSON数据
        data = json.loads(line.strip())
        
        # 获取对话内容和图片字段
        conversations = data.get("conversations", [])
        image = data.get("image", "")
        
        # 修改对话内容
        if len(conversations) == 2:
            # 交换user和assistant的内容
            user_content = conversations[0]["content"]
            assistant_content = conversations[1]["content"]
            
            conversations[0]["content"] = assistant_content
            conversations[1]["content"] = "<image>"
        
        # 创建新的数据格式
        new_data = {
            "conversations": conversations,
            "image": image
        }
        
        # 将处理后的数据写入到新的jsonl文件
        outfile.write(json.dumps(new_data, ensure_ascii=False) + "\n")

print("处理完成，结果保存在", output_file)
