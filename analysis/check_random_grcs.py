import random

grcs = []
grcs_set = set()

fname = '/home/tmn7/analysis_mf_grc/completed_neurons_with_axons_201208'
with open(fname) as fin:
    for line in fin:
        line = line.strip()
        if line == '':
            continue
        grcs.append(line)
        if line in grcs_set:
            print(line)
            # assert False
        grcs_set.add(line)

print(len(set(grcs)))
print(len(grcs))
# assert len(set(grcs)) == len(grcs)

random.shuffle(grcs)

for i, grc in enumerate(grcs):
    if i % 10 == 0:
        print()
    print(grc)



