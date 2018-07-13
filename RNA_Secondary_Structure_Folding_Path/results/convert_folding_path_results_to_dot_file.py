folding_paths = []

with open('find_path.txt', 'r') as f:
    for line in f:
        line = line.strip()
        a = eval(line)
        folding_paths.append(a)

# -----------------------------------

class Node():
    def __init__(self, state):
        self.N = 0
        self.state = state
        self.childs = []
        self.pairs = []

    def __repr__(self):
        return 'Node(N:{})'.format(self.N)

node_layer = [{} for _ in range(10)]
root_node = Node('root')
join_key = '_'


for p in folding_paths:
    state_list = []

    for i in range(10):
        a, b = p[i]
        state_list.append(a)
        state_list.sort()
        state = join_key.join([str(s) for s in state_list])

        if state in node_layer[i]:
            node_layer[i][state].N += 1
            if i != 9:
                state_list2 = state_list[:]
                state_list2.append(p[i+1][0])
                state_list2.sort()
                c = join_key.join([str(s) for s in state_list2])

                if c not in node_layer[i][state].childs:
                    node_layer[i][state].childs.append(c)
        else:
            n = Node(state)
            n.N += 1
            n.pairs = p[:i+1]
            if i != 9:
                state_list2 = state_list[:]
                state_list2.append(p[i+1][0])
                state_list2.sort()
                c = join_key.join([str(s) for s in state_list2])
                n.childs.append(c)
            node_layer[i][state] = n

print('digraph G {\nnode [shape=box];')
for state, node in node_layer[0].items():
    print('"S0" -> "S1\\n' + str(node.pairs) + '" [penwidth=1];')

for i in range(10):
    nodes = node_layer[i]
    for state, node in nodes.items():

        p1 = str(sorted(node.pairs))
        try:
            i1 = p1.index('),', 22)
            p1 = p1[:i1+2] + '\\n' + p1[i1+3:]
        except:
            pass
        try:
            i1 = p1.index('),', 52)
            p1 = p1[:i1+2] + '\\n' + p1[i1+3:]
        except:
            pass

        print('"S' + str(i+1) + '\\n' + p1 + '" [penwidth='+ str(node.N / 4) +'];')

        for c in node.childs:
            p2 = str(sorted(node_layer[i+1][c].pairs))
            try:
                i2 = p2.index('),', 22)
                p2 = p2[:i2+2] + '\\n' + p2[i2+3:]
            except:
                pass
            try:
                i2 = p2.index('),', 52)
                p2 = p2[:i2+2] + '\\n' + p2[i2+3:]
            except:
                pass

            print('"S' + str(i+1) + '\\n' + p1 + '" -> "S' + str(i+2) + '\\n' + p2 + '";')
print('}')

# dot 1.dot -Tpng -o 1.png
