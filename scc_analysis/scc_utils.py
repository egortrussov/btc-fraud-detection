
from collections import defaultdict
import pandas as pd

def get_scc_data(edges_raw: pd.DataFrame):
    nodes = list(set(list(edges_raw["input_address"]) + list(edges_raw["output_address"])))

    edges = defaultdict(lambda : [])
    edges_t = defaultdict(lambda : [])

    for index, edge in edges_raw.iterrows():
        edges[edge["input_address"]].append(edge["output_address"])
        edges_t[edge["output_address"]].append(edge["input_address"])

    visited = defaultdict(lambda : False)
    component = defaultdict(lambda : -1)
    cnt_components = 0
    order = []

    def dfs1(current_node):
        visited[current_node] = True
        for nei in edges[current_node]:
            if visited[nei]:
                continue
            dfs1(nei)
        order.append(current_node)

    def dfs2(current_node):
        component[current_node] = cnt_components
        for nei in edges_t[current_node]:
            if component[nei] == -1:
                dfs2(nei)

    for node in nodes:
        if not visited[node]:
            dfs1(node)

    for node in reversed(order):
        if component[node] == -1:
            dfs2(node)
            cnt_components += 1

    component_dataframe_dict = {
        "address_id": [],
        "component": [],
    }
    for node, comp in component.items():
        component_dataframe_dict["address_id"].append(node)
        component_dataframe_dict["component"].append(comp)

    components_df = pd.DataFrame(component_dataframe_dict)

    comp_to_size = components_df.groupby("component").nunique().reset_index()

    components_df = components_df[components_df["component"].isin(comp_to_size[comp_to_size["address_id"] != 1]["component"])]

    return components_df, cnt_components
