
from collections import Counter, defaultdict

class Node:
    def __init__(self, name: str, id: int, path: List[str]):
        self.name = name
        self.id = id
        self.path = path
        self.exampleProduct = None
        self.parent = None
        self.children = dict()
        self.productCount = 0
        self.subtreeProductCount = 0
        self.also = Counter()

    def __repr__(self):
        return "<{}:{}:{}>".format(self.id, self.name, len(self.children))
