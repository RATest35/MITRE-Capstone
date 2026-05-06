## About Project
Your GNN project focuses on analyzing network traffic data by turning it into a graph. In this graph, each node
represents an IP address or device, and each edge represents a connection or flow of traffic between devices. Before
building the model, the raw dataset has to be cleaned and processed because it contains millions of rows and many columns.
After the data is cleaned, you calculate graph-based features such as in-flow, out-flow, PageRank, and other importance
measures. These features are used to describe how active or important each node is in the network. You also create a
composite score, which acts as the value the model is trained to predict. The main machine learning part of the project
uses a Graph Neural Network to predict each node’s composite score. This allows the model to learn from both the node’s
own features and the connections it has with nearby nodes. Overall, the project combines large-scale data processing,
graph theory, and machine learning to identify important or potentially risky nodes in a network.

## Version 0.4
We have implemented 3 more GNN architectures: Principle Neighbor Aggregation (PNA), Transformer, and GAT models.
The graphSAGE model only uses node attributes, while the new architectures use node and edge attributes. We have
stored various runs for hyperparameter tuning alongside the predictions, and the resulting pytorch model.

## Version 0.3
Changed the initial GNN model to a graphSAGE architecture. We store various runs for hyperparameter tuning.
Some updates have been made to composite_score.py. We have updated the composite score to use the flow_loss,
PageRank, and weighted betweenness. The weighted betweenness is calculated using the capacities in the edges having
faster connections be a shorter distance for betweenness. We have also switched from a pure regression model to a hybrid
model using regression and classfication.

## Version 0.2
We have implemented a way for the entire starting dataset to be cleaned and not load the entire >1GB file into memory.
We take the expanded dataset with ~23k nodes and ~60k edges and use a graph model to calculate the a composite score of
each node, using flow-loss, PageRank and other centrality measures.

A GNN model is used to predict the composite score of each node.

New Python file added to clean initial dataset and calculate composite score:
dataset_processing.py, composite_score.py

Python files added for GNN model:
dataset.py, model.py, train.py

## Version 0.1
The dataset used in the project was downloaded from the Kaggle platform (https://www.kaggle.com/jsrojas/ip-network-traffic-flows-labeled-with-87-apps)


Project organization is not fully developed. Have residual code and datasets from initial exploration.

Important and relevant content is stored in the 'examples / ip-address' folder.
IP-dataset stored in our onedrive 'Dataset' folder to not store large files in GitHub.

generate_graph.py is used to our graph model using a csv dataset. Nodes are unique IP addresses and edges used to show
the flow of information from source to destination. Edges have an attribute called 'flow' which stores
the total number of bytes transferred between a source and destination. The flow is used to calculate the amount of flow
that is lost if a node is removed. The flow-loss is stored in the node.

A corresponding grahpml file is generated as 'ip_graph_39_nodes_243_edges_with_flow.graphml'
