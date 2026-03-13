import networkx as nx

# Change path to the graphml file
INPUT_PATH = ''
OUTPUT_PATH = 'composite_risk.graphml'

def to_float(value):
    try:
        return float(value)
    except (TypeError):
        return print(f'{value} is not able to be converted to float')
    except (ValueError):
        return print(f'{value} is not able to be converted to float')

def load_graph(file_name):
    G = nx.read_graphml(file_name)
    return G

def add_total_flow(G, flow_attr='flow')
    for n in G.nodes():
        in_flow = sum(to_float(data.get(flow_attr, 0.0)) for _, _, data in G.in_edges(n, data=True))
        out_flow = sum(to_float(data.get(flow_attr, 0.0)) for _, _, data in G.out_edges(n, data=True))
        G.nodes[n]['in_flow'] = in_flow
        G.nodes[n]['out_flow'] = out_flow
        G.nodes[n]['flow_loss'] = in_flow + out_flow

        return G

def add_betweenness_centrality(G):
    btwn = nx.betweenness_centrality(G, weight='flow')
    for n, val in btwn.items():
        G.nodes[n]['betweenness'] = val
    return G

def add_pagerank(G):
    pr = nx.pagerank(G, weight='flow')
    for n, val in pr.items():
        G.nodes[n]['pagerank'] = val
    return G

def compute_composite_score(input_file, output_file):
    G = load_graph(input_file)
    add_total_flow(G)
    add_betweenness_centrality(G)
    add_pagerank(G)
    nx.write_graphml(G, output_file)

    return G

#TODO: implement call function
    input_file = INPUT_PATH
    output_file = OUTPUT_PATH
    compute_composite_score(input_file, output_file)

