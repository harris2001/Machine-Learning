class node():
    def __init__(self,weights,bias):
        self.weights = weights
        self.bias = bias

class layer():
    
    def __init__(self,inputs,nodes):
        self.inputs = inputs
        self.nodes = nodes

    def output(self):
        out_layer = []
        for node_i in self.nodes:
            out = 0
            for edge_number in range(len(node_i.weights)):
                out += node_i.weights[edge_number]*self.inputs[edge_number]
            out += node_i.bias
            out_layer.append(out)
        return out_layer

inputs = [1, 2, 3, 2.5]

node1 = node(
    [0.2, 0.8, -0.5, 1.0],
    2.0
)
node2 = node(
    [0.5, -0.91, 0.26, -0.5],
    3.0
)
node3 = node(
    [-0.26, -0.27, 0.17, 0.87],
    0.5
)

layer_nodes = [node1,node2,node3]

last_layer = layer(
    inputs,
    layer_nodes
)


print(last_layer.output())
