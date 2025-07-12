from main import *


for k, v in data_dir.items():
    data_dir[k] = 'nell' + v
dataset = dict()
#dataset['fw_dev_tasks'] = json.load(open(data_dir['few_shot_dev_tasks']))
with open(data_dir['few_shot_dev_tasks'], 'rb') as f:  # 二进制模式
    dataset['fw_dev_tasks'] = json.load(f)
with open(data_dir['few_shot_dev_tasks'], 'r', encoding='utf-8') as f:
    content = f.read()
    print("文件头部:", content[:100])  # 打印前100字符
    print("文件尾部:", content[-100:])  # 打印最后100字符
step='fw_dev'
tasks = dataset[step + '_tasks']
all_rels = list(tasks.keys())
print(all_rels)
