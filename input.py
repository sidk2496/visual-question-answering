import json

open_ended_val= '/home/siddharth/10-707/project/data/Questions_Val_mscoco/OpenEnded_mscoco_val2014_questions.json'

with open(open_ended_val, 'r') as f:
    data = json.load(f)


mcq_val = '/home/siddharth/10-707/project/data/Questions_Val_mscoco/MultipleChoice_mscoco_val2014_questions.json'
with open(mcq_val, 'r') as file:
    data = json.load(file)
print(data.keys())
for key in data.keys():
    print(data[key].__class__)

print(data.keys())
questions = data['questions']
info = data['info']
print(info.keys())
task_type = data['task_type']
print(task_type)
data_type = data['data_type']
print(data_type)
license = data['license']
print(license.keys())
data_subtype = data['data_subtype']
print(data_subtype)
num_choices = data['num_choices']
print(num_choices)


