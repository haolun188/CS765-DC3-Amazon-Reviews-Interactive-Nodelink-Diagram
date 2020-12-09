import dash
import dash_core_components as dcc
import dash_html_components as html
import networkx as nx
import plotly.graph_objs as go
import json
from colour import Color
from textwrap import dedent as d
import pickle
import matplotlib.pyplot as plt

node_dict = {}
current_node_set = set()


def common_graph(G, colors):
    nodes_pos_dict = nx.spring_layout(G)
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = nodes_pos_dict[edge[0]]
        x1, y1 = nodes_pos_dict[edge[1]]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    node_x = []
    node_y = []
    text = []
    hover_text = []
    custom_text = []
    for node in G.nodes():
        x, y = nodes_pos_dict[node]
        node_x.append(x)
        node_y.append(y)
        text.append(node_dict[node].name)
        hover_text.append("ID: " + str(node))
        custom_text.append(",".join(
            [node_dict[node].name, node_dict[node].parent.name if node != 0 else node_dict[node].name, str(len(node_dict[node].children)), str(node)]))


    node_trace = go.Scatter(
        x=node_x, y=node_y,
        text=text,
        hovertext=hover_text,
        customdata=custom_text,
        mode='markers+text',
        hoverinfo='text',
        textposition="bottom center",
        marker=dict(
            showscale=False,
            # colorscale options
            # 'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
            # 'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
            # 'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
            # colorscale='YlGnBu',
            reversescale=True,
            color=colors,
            size=10,
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            ),
            line_width=2))
    return [edge_trace, node_trace]


def network_graph(G, current_node_id, parent_node_id):
    colors = []
    for node in G.nodes():
        if node == current_node_id:
            colors.append(0)
        elif node == parent_node_id:
            colors.append(1)
        else:
            colors.append(2)
    
    return common_graph(G, colors)

def network_graph_alsos(G, node_id, also_node_id):
    colors = []
    for node in G.nodes():
        if node == node_id or node == also_node_id:
            colors.append(1)
        elif node == 0:
            colors.append(0)
        else:
            colors.append(2)
    
    return common_graph(G, colors)


def constructNetworkX(start_node):
    current_node_set.clear()

    G = nx.Graph()
    # DFS
    queue = [(start_node, 0)]
    while len(queue) > 0:
        node, depth = queue.pop()
        current_node_set.add(node.id)
        if depth >= 1 or depth <= -1:
            continue

        for _, child in node.children.items():
            G.add_edge(node.id, child.id)
            queue.append((child, depth + 1))

        if node.id != 0:
            G.add_edge(node.id, node.parent.id)
            queue.append((node.parent, depth - 1))

    return G
    # return nx.random_geometric_graph(200, 0.125)


def constructDict(root):
    G = nx.Graph()
    # DFS
    queue = [root]
    while len(queue) > 0:
        node = queue.pop()
        node_dict[node.id] = node

        for _, child in node.children.items():
            queue.append(child)
    
    pickle.dump(node_dict, open("tree-all-dict.pickle", 'wb'))


def constructTopAlsoNetwork(node):
    current_node_set.clear()

    G = nx.Graph()

    tmp_node = node
    current_node_set.add(0)
    while tmp_node.id != 0:
        G.add_edge(tmp_node.id, tmp_node.parent.id)
        current_node_set.add(tmp_node.id)
        tmp_node = tmp_node.parent

    if len(node.also) == 0:
        return G

    also_node_id = node.also.most_common(1)[0][0]
    also_node = node_dict[also_node_id]

    tmp_node = also_node
    while tmp_node.id != 0:
        G.add_edge(tmp_node.id, tmp_node.parent.id)
        current_node_set.add(tmp_node.id)
        tmp_node = tmp_node.parent

    return G

root = pickle.load(open("tree-all.pickle", 'rb'))
node_dict = pickle.load(open("tree-all-dict.pickle", 'rb'))
G = constructNetworkX(root)

fig = go.Figure(data=network_graph(G, root.id, None))
fig.update_layout(height=800, clickmode='event+select', showlegend=False)

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.title = "Transaction Network"

app.layout = html.Div([
    html.Div([html.H1("Amazon Reviews")],
             className="row",
             style={'textAlign': "center"}),

    html.Div(
        className="row",
        children=[
            html.Div(
                className="two columns",
                children=[
                    html.Div(
                        className="twelve columns",
                        children=[
                            html.Button(id='reset-btn',
                                        n_clicks=0, children='Reset')
                        ],
                        style={'height': '50px'}
                    ),
                    html.Div(
                        className="twelve columns",
                        children=[
                            dcc.Markdown(d("""
                            **Category to zoom in**
                            """)),
                            dcc.Input(id="search-input", type="text",
                                      placeholder="Category Id"),
                            html.Button(id='submit-search-btn',
                                        n_clicks=0, children='Submit')
                        ],
                        style={'height': '300px'}
                    ),
                    html.Div(
                        className="twelve columns",
                        children=[
                            dcc.Markdown(d("""
                            **Relation between two categories which share the most common products**\n
                            Input a category to visualize
                            """)),
                            dcc.Input(id="alsos-input", type="text",
                                      placeholder="Category Id"),
                            html.Button(id='submit-alsos-btn',
                                        n_clicks=0, children='Submit')
                        ],
                        style={'height': '300px'}
                    )
                ]
            ),
            html.Div(
                className="eight columns",
                children=[dcc.Graph(id="my-graph",
                                    animate=True,
                                    figure=fig)],
                style={'height': '800px'}
            ),
            html.Div(
                className="two columns",
                children=[
                    html.Div(
                        className='twelve columns',
                        children=[
                            dcc.Markdown(d("""
                            **Detailed Information**
                            """)),
                            html.Pre(id='hover-data')
                        ],
                        style={'height': '400px'})
                ]
            )
        ]
    )
])


def RepresentsInt(s):
    try:
        int(s)
        return True
    except ValueError:
        return False


def isValidInput(value):
    if value == None:
        return False
    if not RepresentsInt(value):
        return False
    return True


def zoomIn(value):
    global fig
    global current_node_set
    if not isValidInput(value):
        return fig
    if int(value) not in current_node_set:
        return fig
    current_node_id = int(value)
    new_fig = go.Figure(data=network_graph(
        constructNetworkX(node_dict[current_node_id]), current_node_id, node_dict[current_node_id].parent.id if current_node_id != 0 else None))
    fig = new_fig
    return new_fig


def alsos_update(value):
    global fig
    if not isValidInput(value):
        return fig
    current_node_id = int(value)
    also_node_id = node_dict[current_node_id].also.most_common(1)[0][0] if len(node_dict[current_node_id].also) > 0 else None
    new_fig = go.Figure(data=network_graph_alsos(
        constructTopAlsoNetwork(node_dict[current_node_id]), current_node_id, also_node_id))
    fig = new_fig
    return new_fig


def reset():
    global fig
    new_fig = go.Figure(data=network_graph_alsos(constructNetworkX(node_dict[0]), 0, None))
    fig = new_fig
    return new_fig

n_clicks_dict = {
    "reset": 0,
    "search": 0,
    "alsos": 0
}

@app.callback(
    dash.dependencies.Output('my-graph', 'figure'),
    dash.dependencies.Input('reset-btn', 'n_clicks'),
    dash.dependencies.Input('submit-search-btn', 'n_clicks'),
    dash.dependencies.Input('submit-alsos-btn', 'n_clicks'),
    dash.dependencies.State('search-input', 'value'),
    dash.dependencies.State('alsos-input', 'value'))
def update_output(reset_n_clicks, search_n_clicks, alsos_n_clicks, value, alsos_value):
    if search_n_clicks > n_clicks_dict["search"]:
        n_clicks_dict["search"] = search_n_clicks
        return zoomIn(value)
    elif alsos_n_clicks > n_clicks_dict["alsos"]:
        n_clicks_dict["alsos"] = alsos_n_clicks
        return alsos_update(alsos_value)
    else:
        return reset()


@app.callback(
    dash.dependencies.Output('hover-data', 'children'),
    [dash.dependencies.Input('my-graph', 'hoverData')])
def display_hover_data(hoverData):
    global node_dict
    if hoverData == None:
        return ""
    try:
        texts = hoverData["points"][0]["customdata"].split(",")
    except Exception:
        print(Exception)
        return ""
    node_name = texts[0]
    parent_name = texts[1]
    num_children = texts[2]
    node_id = int(texts[3])
    top1also_id = None if len(
        node_dict[node_id].also) == 0 else node_dict[node_id].also.most_common(1)[0][0]
    top1also_name = "" if top1also_id == None else node_dict[top1also_id].name
    num_share_products = 0 if len(
        node_dict[node_id].also) == 0 else node_dict[node_id].also.most_common(1)[0][1]

    return "ID: " + texts[3] + \
        "\nCategory: " + node_name + \
        "\nParent category: " + parent_name + \
        "\nNumber of subcategories: " + num_children + \
        "\nShare most products with Category: \n" + top1also_name + \
        "\nNumber of shared products: " + str(num_share_products)


if __name__ == '__main__':
    app.run_server(debug=True)
