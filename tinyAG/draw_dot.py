from graphviz import Digraph 

def trace(r):
    nodes, edges = set(), set()
    def build(n):
        if n not in nodes:
            nodes.add(n)
            for c in n._prev:
                edges.add((c, n))
                build(c)
    build(r)

    return nodes, edges


def draw_dot(r, format='svg', rankdir='LR'):
    assert rankdir in ['LR', 'UD']
    dot = Digraph(format=format, graph_attr={'rankdir':rankdir})

    nodes, edges = trace(r)
    for n in nodes:
        nid = str(id(n))
        dot.node(name=nid, label=f'{n.label} | data {n.data} | grad {n.grad}', shape='record')
        if n._op:
            dot.node(name=nid+n._op, label=n._op)
            dot.edge(nid+n._op, nid)
    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)
    
    return dot