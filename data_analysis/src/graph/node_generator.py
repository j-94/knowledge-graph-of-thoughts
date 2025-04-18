import networkx as nx
import yaml
import os
from typing import Dict, List, Set, Optional, Any, Tuple
import uuid
from neo4j import GraphDatabase
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class NodeGenerator:
    """
    Class responsible for generating and managing nodes in a knowledge graph.
    
    This class handles the creation of bookmark nodes and concept nodes,
    manages the relationships between them, and provides export functionality
    to NetworkX and Neo4j.
    """
    
    def __init__(self, config_path: str = "config/parameters.yaml"):
        """
        Initialize a new NodeGenerator.
        
        Args:
            config_path (str): Path to the YAML configuration file.
        """
        # Initialize graph
        self.graph = nx.Graph()
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Set parameters from config
        graph_config = self.config.get('graph', {})
        self.concept_similarity_threshold = graph_config.get('concept_similarity_threshold', 0.75)
        self.max_concepts_per_bookmark = graph_config.get('max_concepts_per_bookmark', 5)
        self.min_concept_relevance = graph_config.get('min_concept_relevance', 0.3)
        
        # Track node IDs, concepts and their embeddings
        self.node_ids = set()
        self.concepts = {}  # Maps concept name to node ID
        self.concept_embeddings = {}  # Maps node ID to embedding
        
        # Neo4j connection
        self.driver = None
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from YAML file.
        
        Args:
            config_path (str): Path to the YAML configuration file.
            
        Returns:
            dict: The loaded configuration.
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
            
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    
    def add_bookmark_node(self, bookmark_data: Dict[str, Any]) -> str:
        """
        Add a bookmark node to the graph.
        
        Args:
            bookmark_data (dict): Dictionary containing bookmark data.
                Must include 'url' and 'title'.
                
        Returns:
            str: The node ID of the created bookmark node.
            
        Raises:
            ValueError: If required fields are missing.
        """
        # Validate required fields
        if 'url' not in bookmark_data or 'title' not in bookmark_data:
            raise ValueError("Bookmark data must include 'url' and 'title'")
        
        # Generate a unique node ID
        node_id = f"b{len(self.node_ids)}"
        
        # Add node to graph with all bookmark data
        node_data = {
            'type': 'bookmark',
            **bookmark_data
        }
        self.graph.add_node(node_id, **node_data)
        
        # Track the node ID
        self.node_ids.add(node_id)
        
        return node_id
    
    def add_concept_node(self, 
                        concept_name: str, 
                        embedding: Optional[List[float]] = None,
                        relevance_score: float = 0,
                        source_node_id: Optional[str] = None) -> Tuple[str, bool]:
        """
        Add a concept node to the graph, checking for duplicates.
        
        Args:
            concept_name (str): The name of the concept.
            embedding (list, optional): Vector embedding for the concept.
            relevance_score (float, optional): Relevance score of the concept (0-1).
            source_node_id (str, optional): Node ID that this concept is connected to.
            
        Returns:
            tuple: (node_id, is_new) where is_new indicates if a new node was created.
        """
        # Check if relevance score is above threshold
        if relevance_score < self.min_concept_relevance:
            return None, False
        
        # Convert embedding to numpy array if provided
        if embedding is not None:
            embedding = np.array(embedding).reshape(1, -1)
        
        # Check if we already have this concept or a similar one
        if concept_name in self.concepts:
            # Exact match by name
            node_id = self.concepts[concept_name]
            is_new = False
        elif embedding is not None:
            # Check for similar concepts using embeddings
            similar_node_id = self._find_similar_concept(concept_name, embedding)
            if similar_node_id:
                node_id = similar_node_id
                is_new = False
            else:
                # No similar concept found, create new one
                node_id = f"c{len(self.node_ids)}"
                self.graph.add_node(node_id, type='concept', name=concept_name, relevance=relevance_score)
                self.node_ids.add(node_id)
                self.concepts[concept_name] = node_id
                self.concept_embeddings[node_id] = embedding.flatten().tolist()
                is_new = True
        else:
            # No embedding provided, create new concept node
            node_id = f"c{len(self.node_ids)}"
            self.graph.add_node(node_id, type='concept', name=concept_name, relevance=relevance_score)
            self.node_ids.add(node_id)
            self.concepts[concept_name] = node_id
            is_new = True
        
        # If source node is provided, create relationship
        if source_node_id and source_node_id in self.graph:
            self.graph.add_edge(source_node_id, node_id, type='HAS_CONCEPT')
        
        return node_id, is_new
    
    def _find_similar_concept(self, concept_name: str, embedding: List[float]) -> Optional[str]:
        """
        Find a similar concept based on embedding similarity.
        
        Args:
            concept_name (str): Name of the concept.
            embedding (numpy.ndarray): Embedding vector for the concept.
            
        Returns:
            str or None: Node ID of the similar concept, or None if none found.
        """
        if not self.concept_embeddings:
            return None
            
        # Collect all existing embeddings
        existing_ids = list(self.concept_embeddings.keys())
        existing_embeddings = [np.array(self.concept_embeddings[node_id]) for node_id in existing_ids]
        existing_embeddings_matrix = np.array(existing_embeddings)
        
        # Calculate similarity
        similarity_scores = cosine_similarity(embedding, existing_embeddings_matrix)[0]
        
        # Find the most similar concept
        max_idx = np.argmax(similarity_scores)
        max_score = similarity_scores[max_idx]
        
        # Return the similar concept if above threshold
        if max_score >= self.concept_similarity_threshold:
            return existing_ids[max_idx]
        return None
    
    def add_task_node(self, task_data: Dict[str, Any]) -> str:
        """
        Add a task node to the graph.
        
        Args:
            task_data (dict): Dictionary containing task data.
                Must include 'id' and 'title'.
                
        Returns:
            str: The node ID of the created task node.
            
        Raises:
            ValueError: If required fields are missing.
        """
        # Validate required fields
        if 'id' not in task_data or 'title' not in task_data:
            raise ValueError("Task data must include 'id' and 'title'")
        
        # Generate a unique node ID
        node_id = f"t{len(self.node_ids)}"
        
        # Add node to graph with all task data
        node_data = {
            'type': 'task',
            **task_data
        }
        self.graph.add_node(node_id, **node_data)
        
        # Track the node ID
        self.node_ids.add(node_id)
        
        return node_id
    
    def add_task_dependency(self, task_node_id: str, depends_on_node_id: str) -> bool:
        """
        Add a dependency relationship between two task nodes.
        
        Args:
            task_node_id (str): Node ID of the task that depends on another.
            depends_on_node_id (str): Node ID of the task that is depended upon.
            
        Returns:
            bool: True if relationship was created, False otherwise.
            
        Raises:
            ValueError: If either node doesn't exist or isn't a task.
        """
        # Verify both nodes exist and are task nodes
        if task_node_id not in self.graph:
            raise ValueError(f"Task node {task_node_id} does not exist")
        if depends_on_node_id not in self.graph:
            raise ValueError(f"Task node {depends_on_node_id} does not exist")
            
        if self.graph.nodes[task_node_id].get('type') != 'task':
            raise ValueError(f"Node {task_node_id} is not a task node")
        if self.graph.nodes[depends_on_node_id].get('type') != 'task':
            raise ValueError(f"Node {depends_on_node_id} is not a task node")
        
        # Add dependency edge
        self.graph.add_edge(task_node_id, depends_on_node_id, type='DEPENDS_ON')
        return True
    
    def add_subtask_relationship(self, parent_task_id: str, subtask_id: str) -> bool:
        """
        Add a subtask relationship between two task nodes.
        
        Args:
            parent_task_id (str): Node ID of the parent task.
            subtask_id (str): Node ID of the subtask.
            
        Returns:
            bool: True if relationship was created, False otherwise.
            
        Raises:
            ValueError: If either node doesn't exist or isn't a task.
        """
        # Verify both nodes exist and are task nodes
        if parent_task_id not in self.graph:
            raise ValueError(f"Task node {parent_task_id} does not exist")
        if subtask_id not in self.graph:
            raise ValueError(f"Task node {subtask_id} does not exist")
            
        if self.graph.nodes[parent_task_id].get('type') != 'task':
            raise ValueError(f"Node {parent_task_id} is not a task node")
        if self.graph.nodes[subtask_id].get('type') != 'task':
            raise ValueError(f"Node {subtask_id} is not a task node")
        
        # Add subtask edge
        self.graph.add_edge(parent_task_id, subtask_id, type='HAS_SUBTASK')
        return True
    
    def update_task_status(self, task_node_id: str, status: str) -> bool:
        """
        Update the status of a task node.
        
        Args:
            task_node_id (str): Node ID of the task to update.
            status (str): New status value (e.g., 'pending', 'done', 'in-progress').
            
        Returns:
            bool: True if status was updated, False otherwise.
            
        Raises:
            ValueError: If the node doesn't exist or isn't a task.
        """
        # Verify node exists and is a task node
        if task_node_id not in self.graph:
            raise ValueError(f"Task node {task_node_id} does not exist")
            
        if self.graph.nodes[task_node_id].get('type') != 'task':
            raise ValueError(f"Node {task_node_id} is not a task node")
        
        # Update status
        self.graph.nodes[task_node_id]['status'] = status
        return True
    
    def export_to_networkx(self) -> nx.Graph:
        """
        Export the current graph to a NetworkX Graph object.
        
        Returns:
            networkx.Graph: The current knowledge graph.
        """
        return self.graph
    
    def export_to_neo4j(self, uri: Optional[str] = None, username: Optional[str] = None, password: Optional[str] = None) -> None:
        """
        Export the current graph to a Neo4j database.
        
        Args:
            uri (str, optional): Neo4j connection URI.
            username (str, optional): Neo4j username.
            password (str, optional): Neo4j password.
        """
        # Get Neo4j connection parameters from config if not provided
        if not uri or not username or not password:
            neo4j_config = self.config.get('graph', {}).get('neo4j', {})
            uri = uri or neo4j_config.get('uri')
            username = username or neo4j_config.get('username')
            password = password or neo4j_config.get('password')
        
        if not uri or not username or not password:
            raise ValueError("Neo4j connection parameters missing. Provide parameters or configure in YAML.")
        
        # Connect to Neo4j
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
        
        # Create indices for better performance
        with self.driver.session() as session:
            # Create constraints for unique IDs
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (b:Bookmark) REQUIRE b.id IS UNIQUE")
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (c:Concept) REQUIRE c.id IS UNIQUE")
            
            # Add all nodes
            for node_id in self.graph.nodes:
                node_data = self.graph.nodes[node_id]
                node_type = node_data.get('type')
                
                if node_type == 'bookmark':
                    # Create bookmark node
                    session.run(
                        """
                        MERGE (b:Bookmark {id: $id})
                        SET b.url = $url,
                            b.title = $title,
                            b.description = $description,
                            b.tags = $tags,
                            b.created_at = $created_at
                        """,
                        id=node_id,
                        url=node_data.get('url'),
                        title=node_data.get('title'),
                        description=node_data.get('description', ''),
                        tags=node_data.get('tags', []),
                        created_at=node_data.get('created_at', '')
                    )
                elif node_type == 'concept':
                    # Create concept node
                    session.run(
                        """
                        MERGE (c:Concept {id: $id})
                        SET c.name = $name,
                            c.relevance = $relevance
                        """,
                        id=node_id,
                        name=node_data.get('name'),
                        relevance=node_data.get('relevance', 0.0)
                    )
                elif node_type == 'task':
                    # Create task node
                    session.run(
                        """
                        MERGE (t:Task {id: $id})
                        SET t.title = $title,
                            t.description = $description,
                            t.status = $status,
                            t.priority = $priority,
                            t.task_id = $task_id
                        """,
                        id=node_id,
                        title=node_data.get('title'),
                        description=node_data.get('description', ''),
                        status=node_data.get('status', 'pending'),
                        priority=node_data.get('priority', 'medium'),
                        task_id=node_data.get('id')
                    )
            
            # Add all edges
            for source_id, target_id, edge_data in self.graph.edges(data=True):
                edge_type = edge_data.get('type', 'RELATED_TO')
                
                # Determine node types
                source_type = self.graph.nodes[source_id].get('type')
                target_type = self.graph.nodes[target_id].get('type')
                
                # Create the appropriate relationship
                if source_type == 'bookmark' and target_type == 'concept':
                    session.run(
                        """
                        MATCH (b:Bookmark {id: $source_id})
                        MATCH (c:Concept {id: $target_id})
                        MERGE (b)-[r:HAS_CONCEPT]->(c)
                        """,
                        source_id=source_id,
                        target_id=target_id
                    )
                elif source_type == 'concept' and target_type == 'concept':
                    session.run(
                        """
                        MATCH (c1:Concept {id: $source_id})
                        MATCH (c2:Concept {id: $target_id})
                        MERGE (c1)-[r:RELATED_TO]->(c2)
                        """,
                        source_id=source_id,
                        target_id=target_id
                    )
                elif source_type == 'task' and target_type == 'task':
                    # Determine the relationship type
                    if edge_type == 'HAS_SUBTASK':
                        session.run(
                            """
                            MATCH (t1:Task {id: $source_id})
                            MATCH (t2:Task {id: $target_id})
                            MERGE (t1)-[r:HAS_SUBTASK]->(t2)
                            """,
                            source_id=source_id,
                            target_id=target_id
                        )
                    elif edge_type == 'DEPENDS_ON':
                        session.run(
                            """
                            MATCH (t1:Task {id: $source_id})
                            MATCH (t2:Task {id: $target_id})
                            MERGE (t1)-[r:DEPENDS_ON]->(t2)
                            """,
                            source_id=source_id,
                            target_id=target_id
                        )
                    else:
                        # Default task relationship
                        session.run(
                            """
                            MATCH (t1:Task {id: $source_id})
                            MATCH (t2:Task {id: $target_id})
                            MERGE (t1)-[r:RELATED_TO]->(t2)
                            """,
                            source_id=source_id,
                            target_id=target_id
                        )
    
    def close(self) -> None:
        """Close any open connections."""
        if self.driver:
            self.driver.close()
            self.driver = None 