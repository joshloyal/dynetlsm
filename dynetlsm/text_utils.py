import numpy as np
import networkx as nx


def repel_labels(X, node_names, datasize, k=1.0, textsize=10, mask=None,
                 ax=None):
    G = nx.DiGraph()

    data_nodes = []
    init_pos = {}
    data_fmt = 'data_{}'
    label_fmt = '{} ({})'
    for i, (x, y) in enumerate(X):
        if mask[i]:
            data_str = data_fmt.format(i)
            if node_names is None:
                label_str = "{}".format(i)
            else:
                label_str = label_fmt.format(node_names[i], i)
            data_nodes.append(data_str)
            G.add_node(data_str)
            G.add_node(label_str)
            G.add_edge(label_str, data_str)
            init_pos[data_str] = (x, y)
            init_pos[label_str] = (x, y)

    pos = nx.spring_layout(G, pos=init_pos, fixed=data_nodes, k=k)

    # undo re-scaling
    pos_after = np.vstack([pos[d] for d in data_nodes])
    pos_before = np.vstack([init_pos[d] for d in data_nodes])
    scale, shift_x = np.polyfit(pos_after[:, 0], pos_before[:, 0], 1)
    scale, shift_y = np.polyfit(pos_after[:, 1], pos_before[:, 1], 1)
    shift = np.array([shift_x, shift_y])
    for key, val in pos.items():
        pos[key] = (val * scale) + shift

    for label, data_str in G.edges():
        ax.annotate(label,
                    xy=pos[data_str],
                    xytext=pos[label],
                    size=textsize,
                    alpha=0.9,
                    xycoords='data',
                    textcoords='data',
                    arrowprops=dict(arrowstyle='-|>',
                                    shrinkA=0, shrinkB=np.sqrt(datasize) / 2.,
                                    connectionstyle='arc3',
                                    mutation_scale=10,
                                    color='black'))
