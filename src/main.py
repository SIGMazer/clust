import javalang
import os 
import glob
from typing import *
import networkx as nx
import matplotlib.pyplot as plt
import json
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


INPUT_DIR = 'input'
JAVA_DATA = []
INHERITANCE_GRAPH = {}
METHOD_CALL_GRAPH = {}
DEPTH = {}
FAN_IN = {}
FAN_OUT = {}
FEATURE_MATRIX = {}
LOC = {}
CBO = {}
LCOM = {}
WMC = {}
VISIBILITY = {}

def get_java_files() -> List[str]:
    return glob.glob(f'{INPUT_DIR}/**/*.java', recursive=True)


def compute_coupling_between_classes():

    global CBO
    for java_class in JAVA_DATA:
        class_name = java_class['name']
        coupled_classes = set()

        # Check extends and implements
        if java_class['extends']:
            coupled_classes.add(java_class['extends'])
        if java_class['implements']:
            coupled_classes.update(java_class['implements'])

        # Check method calls
        for method in java_class['methods']:
            if isinstance(method, dict):
                for call in method.get('calls', []):
                    if call in FAN_IN:  # Calls mapped to methods of other classes
                        called_class = FAN_IN[call]
                        coupled_classes.add(called_class)

        CBO[class_name] = len(coupled_classes)


def get_method_features() -> Tuple[List[Dict[str, Any]], List[str]]:
    method_features = []
    method_names = []

    for java_class in JAVA_DATA:
        class_name = java_class['name']
        methods = [m for m in java_class['methods'] if isinstance(m, dict)]
        for method in methods:
            method_name = method['name']
            fan_in = FAN_IN.get(method_name, 0)
            fan_out = FAN_OUT.get(method_name, 0)

            method_features.append({
                'fan_in': fan_in,
                'fan_out': fan_out,
                'loc': method['loc'],
                'cyclomatic_complexity': method['cyclomatic_complexity'],
                'num_parameters': method['parameters'],
            })
            method_names.append(f'{class_name}.{method_name}')

    return method_features, method_names

def cluster_methods():
    method_features, method_names = get_method_features()

    # If there are no methods, return
    if not method_features:
        print("No method features available to cluster.")
        return

    # Create a DataFrame for method features
    method_data = pd.DataFrame(method_features, columns=[
        'fan_in', 'fan_out', 'loc', 'cyclomatic_complexity', 'num_parameters'
    ], index=method_names)

    # Fill missing values (if any)
    method_data = method_data.fillna(method_data.median())

    # Standardize the method features
    scaler = StandardScaler()
    method_data_scaled = scaler.fit_transform(method_data)

    # Perform clustering (KMeans)
    kmeans = KMeans(n_clusters=5)
    method_data['cluster'] = kmeans.fit_predict(method_data_scaled)

    # Apply PCA for visualization
    pca = PCA(n_components=2)
    pca_components = pca.fit_transform(method_data_scaled)
    method_data['pca1'] = pca_components[:, 0]
    method_data['pca2'] = pca_components[:, 1]

    # Plot the clustered methods
    plt.scatter(method_data['pca1'], method_data['pca2'], c=method_data['cluster'], cmap='viridis')
    plt.xlabel('code_complexity')
    plt.ylabel('code_dependency')
    plt.title('PCA Components for Method Clusters')
    plt.colorbar()
    plt.show()

    # Print the cluster centers
    print(kmeans.cluster_centers_)

    # Export the clustered method Data 
    method_data.to_csv('method_clusters.csv')

def compute_lack_of_cohesion():

    global LCOM
    for java_class in JAVA_DATA:
        class_name = java_class['name']
        methods = [m for m in java_class['methods'] if isinstance(m, dict)]
        fields = set()
        method_field_usage = {}

        # Identify fields and method-field usage
        for method in methods:
            method_name = method['name']
            used_fields = set()

            for call in method.get('calls', []):
                if '.' in call:  # Example: 'this.fieldName' or 'obj.fieldName'
                    field_name = call.split('.')[1]
                    used_fields.add(field_name)

            method_field_usage[method_name] = used_fields
            fields.update(used_fields)

        # Calculate LCOM: Pairs of methods without shared fields
        total_pairs = 0
        disjoint_pairs = 0
        method_names = list(method_field_usage.keys())

        for i in range(len(method_names)):
            for j in range(i + 1, len(method_names)):
                total_pairs += 1
                if method_field_usage[method_names[i]].isdisjoint(method_field_usage[method_names[j]]):
                    disjoint_pairs += 1

        lcom = disjoint_pairs / total_pairs if total_pairs > 0 else 0
        LCOM[class_name] = lcom



def compute_weighted_methods_per_class():
    """
    Compute the Weighted Methods per Class (WMC) for each class.
    """

    global WMC
    for java_class in JAVA_DATA:
        class_name = java_class['name']
        wmc = 0

        for method in java_class['methods']:
            if isinstance(method, dict):
                wmc += method.get('cyclomatic_complexity', 1)  # Use CC as weight

        WMC[class_name] = wmc


def compute_private_public_methods():
    """
    Compute the number of private and public methods for each class.
    """

    global VISIBILITY
    for java_class in JAVA_DATA:
        class_name = java_class['name']
        private_count = 0
        public_count = 0

        for method in java_class['methods']:
            if isinstance(method, dict):
                # Analyze method modifiers for visibility
                modifiers = method.get('modifiers', [])
                if 'private' in modifiers:
                    private_count += 1
                elif 'public' in modifiers:
                    public_count += 1

        VISIBILITY[class_name] = {
            'private_methods': private_count,
            'public_methods': public_count
        }





def compute_cyclomatic_complexity(tree: javalang.ast.Node) -> int:
    decision_nodes = (
        javalang.tree.IfStatement,
        javalang.tree.ForStatement,
        javalang.tree.WhileStatement,
        javalang.tree.DoStatement,
        javalang.tree.SwitchStatement,
        javalang.tree.CatchClause,
    )

    complexity = 1
    for path, node in tree:
        if isinstance(node, decision_nodes):
            complexity += 1
        elif isinstance(node, javalang.tree.BinaryOperation):
            if node.operator in {"&&", "||"}:
                complexity += 1
    return complexity



def init_feature_matrix():
    global FEATURE_MATRIX
    FEATURE_MATRIX = {
            java_class['name']: {
                'depth': DEPTH.get(java_class['name'], 0),
                'num_methods': len([m for m in java_class.get('methods', []) if isinstance(m, dict)]),
                'num_interfaces': len(java_class.get('implements', [])),
                'num_subclasses': len(INHERITANCE_GRAPH.get(java_class['name'], [])),
                'cbo': CBO.get(java_class['name'], 0),
                'lcom': LCOM.get(java_class['name'], 0),
                'wmc': WMC.get(java_class['name'], 0),
                'private_methods': VISIBILITY.get(java_class['name'], {}).get('private_methods', 0),
                'public_methods': VISIBILITY.get(java_class['name'], {}).get('public_methods', 0),
                'fan_in': FAN_IN.get(java_class['name'], 0),
                'fan_out': FAN_OUT.get(java_class['name'], 0),
                'loc': java_class['loc']
                }
            for java_class in JAVA_DATA
            } 


def compute_method_features():
    global FEATURE_MATRIX
    for java_class in JAVA_DATA:
        class_name = java_class['name']
        if class_name not in FEATURE_MATRIX:
            continue

        method_features = []
        for method in java_class.get('methods', []):
            if not isinstance(method, dict):
                continue

            method_name = method['name']
            fan_in = FAN_IN.get(method_name, 0)
            fan_out = FAN_OUT.get(method_name, 0)

            method_features.append({
                'method_name': method_name,
                'fan_in': fan_in,
                'fan_out': fan_out,
                'loc': method['loc'],
                'cyclomatic_complexity': method['cyclomatic_complexity'],
                'num_parameters': method['parameters'],

            })

        FEATURE_MATRIX[class_name]['methods'] = method_features

def compute_loc(file_code: str, method_start_pos: tuple) -> int:
    lines = file_code.splitlines()
    start_line = method_start_pos[0] - 1  
    method_lines = lines[start_line:]
    
    open_braces = 0
    loc = 0
    method_started = False
    
    for line in method_lines:
        stripped_line = line.strip()
        
        open_braces += stripped_line.count('{')
        open_braces -= stripped_line.count('}')
        
        if '{' in stripped_line:
            method_started = True
        
        if method_started:
            loc += 1
        
        if open_braces == 0 and method_started:
            break
    
    return loc


def analyze_file(file_path: str):
    with open(file_path, 'r') as file:
        code = file.read()
    tree = javalang.parse.parse(code)
    for path, node in tree:
        if isinstance(node, javalang.tree.ClassDeclaration):
            class_name = node.name
            extends = node.extends.name if node.extends else None
            implements = [i.name for i in (node.implements or [])]

            methods = []
            class_loc = compute_loc(code, node.position)
            LOC[class_name] = class_loc
            for member in node.body:
                if isinstance(member, javalang.tree.MethodDeclaration):
                    methods.append(member.name)
                    calls = []
                    parameters = [p.type.name for p in member.parameters]
                    for path, node in member.filter(javalang.tree.MethodInvocation):
                        calls.append(node.member)
                    method_loc = compute_loc(code , member.position)
                    cc = compute_cyclomatic_complexity(member)
                    methods.append({
                        'name': member.name,
                        'calls': calls,
                        'loc':method_loc,
                        'cyclomatic_complexity': cc,
                        'parameters': len(parameters),
                    })
            JAVA_DATA.append({
                'name': class_name,
                'extends': extends,
                'implements': implements,
                'methods': methods,
                'loc': class_loc
            })



def build_inheritance_graph() -> None:
    for java_class in JAVA_DATA:
        if not isinstance(java_class, dict):
            continue
        if 'name' not in java_class or 'extends' not in java_class:
            continue
        parent = java_class['extends']
        children = [java_class['name']]
        if parent:
            INHERITANCE_GRAPH.setdefault(parent, []).extend(children)

def build_method_call_graph() -> None:
    for java_class in JAVA_DATA:
        if not isinstance(java_class, dict):
            continue
        if 'methods' not in java_class:
            continue
        for method in java_class['methods']:
            if not isinstance(method, dict):
                continue
            if 'name' not in method or 'calls' not in method:
                continue
            for call in method['calls']:
                METHOD_CALL_GRAPH.setdefault(method['name'], []).append(call)


def compute_inheritance_depth_helper(class_name: str) -> int:
    depth = 0
    if class_name in INHERITANCE_GRAPH:
        for child in INHERITANCE_GRAPH[class_name]:
            depth = max(depth, 1 + compute_inheritance_depth_helper(child))
    return depth

def compute_inheritance_depth():
    global DEPTH 
    depths = {}
    for class_name in INHERITANCE_GRAPH:
        depths[class_name] = compute_inheritance_depth_helper(class_name)
    DEPTH = depths



def export_feature_matrix(output_file='feature_matrix.json'):
    with open(output_file, 'w') as f:
        json.dump(FEATURE_MATRIX, f, indent=4)
    print(f"Feature matrix exported to {output_file}")

def compute_fan_in_out():
    """
    Compute fan-in and fan-out for both classes and methods.
    Updates FAN_IN and FAN_OUT for methods and classes.
    """
    global FAN_IN, FAN_OUT
    fan_in = {}
    fan_out = {}

    # Reverse map: method -> class
    method_to_class = {}
    for java_class in JAVA_DATA:
        class_name = java_class['name']
        for method in java_class.get('methods', []):
            if isinstance(method, dict) and 'name' in method:
                method_to_class[method['name']] = class_name

    # Method-Level Fan-In and Fan-Out
    method_fan_in = {}
    method_fan_out = {}

    for method, calls in METHOD_CALL_GRAPH.items():
        # Fan-Out for the current method
        method_fan_out[method] = len(set(calls))

        # Fan-In for the methods being called
        for call in calls:
            method_fan_in[call] = method_fan_in.get(call, 0) + 1

    # Class-Level Fan-In and Fan-Out
    class_fan_in = {}
    class_fan_out = {}

    for method, calls in METHOD_CALL_GRAPH.items():
        if method in method_to_class:
            calling_class = method_to_class[method]

            # Fan-Out for the calling class
            for call in calls:
                if call in method_to_class:
                    called_class = method_to_class[call]
                    if called_class != calling_class:
                        class_fan_out.setdefault(calling_class, set()).add(called_class)

                    # Fan-In for the called class
                    class_fan_in.setdefault(called_class, set()).add(calling_class)

    # Convert sets to counts for class-level fan-in/out
    for class_name in class_fan_in:
        fan_in[class_name] = len(class_fan_in[class_name])
    for class_name in class_fan_out:
        fan_out[class_name] = len(class_fan_out[class_name])

    # Merge method-level fan-in/out
    for method in method_fan_in:
        fan_in[method] = method_fan_in[method]
    for method in method_fan_out:
        fan_out[method] = method_fan_out[method]

    FAN_IN = fan_in
    FAN_OUT = fan_out



def main():
    files =get_java_files()
    for file in files:
        analyze_file(file)
    build_inheritance_graph()
    build_method_call_graph()
    compute_inheritance_depth()
    compute_fan_in_out()
    compute_coupling_between_classes()
    compute_lack_of_cohesion()
    compute_weighted_methods_per_class()
    compute_private_public_methods()

    init_feature_matrix()
    compute_method_features()
    # export_feature_matrix()

    # feature_data = pd.DataFrame.from_dict(FEATURE_MATRIX).T
    # columns_to_cluster = [
    # 'depth', 'num_methods', 'num_interfaces', 'num_subclasses', 'cbo', 'lcom', 'wmc',
    # 'private_methods', 'public_methods', 'fan_in', 'fan_out', 'loc'
    # ]
    # feature_data = feature_data[columns_to_cluster]
    #
    # feature_data = feature_data.fillna(feature_data.median())
    #
    # scaler = StandardScaler()
    # feature_data_scaled = scaler.fit_transform(feature_data)
    #
    # kmeans = KMeans(n_clusters=2, random_state=42)
    # feature_data['cluster'] = kmeans.fit_predict(feature_data_scaled)
    #
    # pca = PCA(n_components=2)
    # pca_components = pca.fit_transform(feature_data_scaled)
    # feature_data['pca1'] = pca_components[:, 0]
    # feature_data['pca2'] = pca_components[:, 1]
    #
    # plt.scatter(feature_data['pca1'], feature_data['pca2'], c=feature_data['cluster'], cmap='viridis')
    # plt.xlabel('PCA1')
    # plt.ylabel('PCA2')
    # plt.title('PCA Components')
    # plt.colorbar()
    # plt.show()
    #
    # print(kmeans.cluster_centers_)
    
    cluster_methods()




if __name__ == '__main__':
    main()

