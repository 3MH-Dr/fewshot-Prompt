import json
import random



# split random shuffle NELL continual dataset base 96 novel 96
if __name__ == "__main__":
    print('This')
    con_dev = json.load(open("NELL/continual_dev_tasks.json"))
    rels = sorted(list(con_dev.keys()))
    random.seed(42)
    random.shuffle(rels)
    con_new = {}
    for j, rel in enumerate(rels):
        con_new[rel] = []
        random.shuffle(con_dev[rel])
        if j < 30:
            for i in range(6):
                con_new[rel].append(con_dev[rel][i])
        else:
            for i in range(len(con_dev[rel])):
                con_new[rel].append(con_dev[rel][i])

    json.dump(con_new, open('NELL/con_base100_n100_dev_tasks.json', 'w'))

    sum = 0
    for i, (rel, t) in enumerate(con_new.items()):
        if (i - 30) % 3 == 0 and i >= 30:
            print(sum - 9)
            sum = 0
        if i >= 30:
            sum += len(t)
             