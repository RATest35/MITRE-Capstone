## Version 0.1
Project organization is not fully developed. Have residual code and datasets from initial exploration.

Important and relevant content is stored in the 'examples / ip-address' folder.
IP-dataset stored in our onedrive 'Dataset' folder to not store large files in GitHub.

generate_graph.py is used to our graph model using a csv dataset. Nodes are unique IP addresses and edges used to show
the flow of information from source to destination. Edges have an attribute called 'flow' which stores
the total number of bytes transferred between a source and destination. The flow is used to calculate the amount of flow
that is lost if a node is removed. The flow-loss is stored in the node.

A corresponding grahpml file is generated as 'ip_graph_39_nodes_243_edges_with_flow.graphml'