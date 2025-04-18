try:
    import pandas as pd
    print(f"✅ pandas {pd.__version__}")
except ImportError as e:
    print(f"❌ pandas: {e}")

try:
    import networkx as nx
    print(f"✅ networkx {nx.__version__}")
except ImportError as e:
    print(f"❌ networkx: {e}")

try:
    import neo4j
    print(f"✅ neo4j {neo4j.__version__}")
except ImportError as e:
    print(f"❌ neo4j: {e}")

try:
    from fastapi import FastAPI
    import fastapi
    print(f"✅ fastapi {fastapi.__version__}")
except ImportError as e:
    print(f"❌ fastapi: {e}") 