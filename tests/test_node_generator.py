import unittest
import os
import sys
import yaml
import networkx as nx
from unittest.mock import patch, MagicMock
import pytest
import numpy as np
import tempfile

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../data_analysis/src')))

from graph.node_generator import NodeGenerator

# Sample configuration for testing
TEST_CONFIG = {
    'graph': {
        'concept_similarity_threshold': 0.75,
        'max_concepts_per_bookmark': 5,
        'min_concept_relevance': 0.3,
        'neo4j': {
            'uri': 'bolt://localhost:7687',
            'username': 'neo4j',
            'password': 'password'
        }
    },
    'bookmark_processing': {
        'batch_size': 100,
        'min_description_length': 50
    }
}

class TestNodeGenerator(unittest.TestCase):
    """Test cases for the NodeGenerator class."""
    
    @pytest.fixture
    def config_path(self):
        """Create a temporary configuration file for testing."""
        with tempfile.NamedTemporaryFile(suffix='.yaml', delete=False) as temp:
            config = {
                'graph': {
                    'concept_similarity_threshold': 0.8,
                    'max_concepts_per_bookmark': 3,
                    'min_concept_relevance': 0.2,
                    'neo4j': {
                        'uri': 'bolt://localhost:7687',
                        'username': 'neo4j',
                        'password': 'password'
                    }
                }
            }
            yaml.dump(config, temp)
            temp_path = temp.name
        
        yield temp_path
        # Cleanup
        if os.path.exists(temp_path):
            os.unlink(temp_path)
    
    @pytest.fixture
    def node_generator(self, config_path):
        """Initialize a NodeGenerator instance for testing."""
        generator = NodeGenerator(config_path)
        yield generator
        generator.close()
    
    def test_init(self, node_generator, config_path):
        """Test initialization of NodeGenerator."""
        assert isinstance(node_generator.graph, nx.Graph)
        assert node_generator.concept_similarity_threshold == 0.8
        assert node_generator.max_concepts_per_bookmark == 3
        assert node_generator.min_concept_relevance == 0.2
        assert len(node_generator.node_ids) == 0
        assert len(node_generator.concepts) == 0
        assert node_generator.driver is None

    def test_init_file_not_found(self):
        """Test initialization with non-existent config file."""
        with pytest.raises(FileNotFoundError):
            NodeGenerator("nonexistent_config.yaml")

    def test_add_bookmark_node(self, node_generator):
        """Test adding a bookmark node to the graph."""
        bookmark_data = {
            'url': 'https://example.com',
            'title': 'Example Page',
            'description': 'This is an example page',
            'tags': ['example', 'test'],
            'created_at': '2023-01-01'
        }
        
        node_id = node_generator.add_bookmark_node(bookmark_data)
        
        # Check node was added
        assert node_id in node_generator.graph
        assert node_id in node_generator.node_ids
        
        # Check node properties
        node_data = node_generator.graph.nodes[node_id]
        assert node_data['type'] == 'bookmark'
        assert node_data['url'] == 'https://example.com'
        assert node_data['title'] == 'Example Page'
        assert node_data['description'] == 'This is an example page'
        assert node_data['tags'] == ['example', 'test']
        assert node_data['created_at'] == '2023-01-01'

    def test_add_bookmark_node_missing_required_fields(self, node_generator):
        """Test adding a bookmark node with missing required fields."""
        # Missing title
        with pytest.raises(ValueError):
            node_generator.add_bookmark_node({'url': 'https://example.com'})
        
        # Missing url
        with pytest.raises(ValueError):
            node_generator.add_bookmark_node({'title': 'Example Page'})
        
        # Both fields present should work
        node_id = node_generator.add_bookmark_node({
            'url': 'https://example.com',
            'title': 'Example Page'
        })
        assert node_id in node_generator.graph

    def test_add_concept_node_new(self, node_generator):
        """Test adding a new concept node."""
        concept_name = "Machine Learning"
        embedding = [0.1, 0.2, 0.3, 0.4]
        relevance_score = 0.8
        
        node_id, is_new = node_generator.add_concept_node(
            concept_name, embedding, relevance_score
        )
        
        # Check node was added
        assert node_id in node_generator.graph
        assert node_id in node_generator.node_ids
        assert is_new is True
        
        # Check node properties
        node_data = node_generator.graph.nodes[node_id]
        assert node_data['type'] == 'concept'
        assert node_data['name'] == 'Machine Learning'
        assert node_data['relevance'] == 0.8
        
        # Check concept tracking
        assert concept_name in node_generator.concepts
        assert node_generator.concepts[concept_name] == node_id
        assert node_id in node_generator.concept_embeddings

    def test_add_concept_node_with_source(self, node_generator):
        """Test adding a concept node with a source bookmark."""
        # Create a bookmark node first
        bookmark_data = {
            'url': 'https://example.com/ml',
            'title': 'Machine Learning Guide'
        }
        bookmark_id = node_generator.add_bookmark_node(bookmark_data)
        
        # Add a concept node connected to the bookmark
        concept_name = "Neural Networks"
        node_id, is_new = node_generator.add_concept_node(
            concept_name, 
            embedding=[0.5, 0.6, 0.7, 0.8],
            relevance_score=0.9,
            source_node_id=bookmark_id
        )
        
        # Check that the edge was created
        assert node_generator.graph.has_edge(bookmark_id, node_id)
        edge_data = node_generator.graph.get_edge_data(bookmark_id, node_id)
        assert edge_data['type'] == 'HAS_CONCEPT'

    def test_add_concept_node_below_relevance_threshold(self, node_generator):
        """Test adding a concept node with relevance below threshold."""
        concept_name = "Low Relevance"
        node_id, is_new = node_generator.add_concept_node(
            concept_name, 
            relevance_score=0.1  # Below the 0.2 threshold from config
        )
        
        # Should not add the node
        assert node_id is None
        assert is_new is False
        assert concept_name not in node_generator.concepts

    def test_add_duplicate_concept_by_name(self, node_generator):
        """Test adding a concept that already exists by name."""
        concept_name = "Duplicate Concept"
        
        # Add the first node
        node_id1, is_new1 = node_generator.add_concept_node(
            concept_name, 
            relevance_score=0.8
        )
        
        # Add the same concept again
        node_id2, is_new2 = node_generator.add_concept_node(
            concept_name, 
            relevance_score=0.9
        )
        
        # Should return the existing node
        assert node_id1 == node_id2
        assert is_new1 is True
        assert is_new2 is False

    def test_find_similar_concept(self, node_generator):
        """Test finding a similar concept based on embedding."""
        # Add a concept with an embedding
        concept1 = "Machine Learning"
        embedding1 = np.array([0.1, 0.2, 0.3, 0.4])
        node_id1, _ = node_generator.add_concept_node(
            concept1, 
            embedding1,
            relevance_score=0.8
        )
        
        # Create a similar embedding
        similar_embedding = np.array([0.11, 0.21, 0.31, 0.39]).reshape(1, -1)
        
        # Find similar concept
        found_node_id = node_generator._find_similar_concept("Similar", similar_embedding)
        
        # Should find the first concept
        assert found_node_id == node_id1
        
        # Create a very different embedding
        different_embedding = np.array([0.9, 0.8, 0.7, 0.6]).reshape(1, -1)
        
        # Try to find similar concept
        found_node_id = node_generator._find_similar_concept("Different", different_embedding)
        
        # Should not find any similar concept
        assert found_node_id is None

    def test_export_to_networkx(self, node_generator):
        """Test exporting to NetworkX."""
        # Add some nodes
        node_generator.add_bookmark_node({
            'url': 'https://example.com',
            'title': 'Example Page'
        })
        node_generator.add_concept_node("Test Concept", relevance_score=0.8)
        
        # Export
        graph = node_generator.export_to_networkx()
        
        # Should be the same graph
        assert graph is node_generator.graph
        assert len(graph.nodes) == 2

    @patch('neo4j.GraphDatabase.driver')
    def test_export_to_neo4j(self, mock_driver, node_generator):
        """Test exporting to Neo4j."""
        # Setup mock
        mock_session = MagicMock()
        mock_driver.return_value.session.return_value.__enter__.return_value = mock_session
        
        # Add some nodes and edges
        bookmark_id = node_generator.add_bookmark_node({
            'url': 'https://example.com',
            'title': 'Example Page'
        })
        concept_id, _ = node_generator.add_concept_node(
            "Test Concept", 
            relevance_score=0.8,
            source_node_id=bookmark_id
        )
        
        # Export to Neo4j
        node_generator.export_to_neo4j()
        
        # Check that driver was created
        mock_driver.assert_called_once()
        
        # Check that session was used
        assert mock_session.run.call_count >= 4  # At least 4 queries executed

    def test_export_to_neo4j_missing_params(self, node_generator):
        """Test exporting to Neo4j with missing parameters."""
        # Create a node generator with a config that doesn't have Neo4j params
        with tempfile.NamedTemporaryFile(suffix='.yaml', delete=False) as temp:
            yaml.dump({'graph': {}}, temp)
            temp_path = temp.name
            
        generator = NodeGenerator(temp_path)
        
        # Should raise an error
        with pytest.raises(ValueError):
            generator.export_to_neo4j()
            
        # Cleanup
        generator.close()
        if os.path.exists(temp_path):
            os.unlink(temp_path)

    def test_close(self, node_generator):
        """Test closing connections."""
        # Mock the driver
        node_generator.driver = MagicMock()
        
        # Close connections
        node_generator.close()
        
        # Driver should be closed
        node_generator.driver.close.assert_called_once()
        assert node_generator.driver is None

    def test_add_task_node(self, node_generator):
        """Test adding a task node."""
        task_data = {
            'id': '1',
            'title': 'Implement task functionality',
            'description': 'Add task node support to the knowledge graph',
            'status': 'pending',
            'priority': 'high'
        }
        
        node_id = node_generator.add_task_node(task_data)
        
        # Check node was added
        assert node_id in node_generator.graph
        assert node_id in node_generator.node_ids
        
        # Check node properties
        node_data = node_generator.graph.nodes[node_id]
        assert node_data['type'] == 'task'
        assert node_data['id'] == '1'
        assert node_data['title'] == 'Implement task functionality'
        assert node_data['status'] == 'pending'
        assert node_data['priority'] == 'high'
    
    def test_add_task_node_missing_required_fields(self, node_generator):
        """Test adding a task node with missing required fields."""
        # Missing title
        with pytest.raises(ValueError):
            node_generator.add_task_node({'id': '1'})
        
        # Missing id
        with pytest.raises(ValueError):
            node_generator.add_task_node({'title': 'Test Task'})
        
        # Both fields present should work
        node_id = node_generator.add_task_node({
            'id': '1',
            'title': 'Test Task'
        })
        assert node_id in node_generator.graph
    
    def test_add_task_dependency(self, node_generator):
        """Test adding a dependency relationship between tasks."""
        # Create two task nodes
        task1_id = node_generator.add_task_node({
            'id': '1',
            'title': 'Parent Task'
        })
        
        task2_id = node_generator.add_task_node({
            'id': '2',
            'title': 'Dependency Task'
        })
        
        # Add dependency (task1 depends on task2)
        result = node_generator.add_task_dependency(task1_id, task2_id)
        
        # Check relationship was created
        assert result is True
        assert node_generator.graph.has_edge(task1_id, task2_id)
        assert node_generator.graph.edges[task1_id, task2_id]['type'] == 'DEPENDS_ON'
    
    def test_add_subtask_relationship(self, node_generator):
        """Test adding a subtask relationship between tasks."""
        # Create parent and subtask nodes
        parent_id = node_generator.add_task_node({
            'id': '1',
            'title': 'Parent Task'
        })
        
        subtask_id = node_generator.add_task_node({
            'id': '1.1',
            'title': 'Subtask'
        })
        
        # Add subtask relationship
        result = node_generator.add_subtask_relationship(parent_id, subtask_id)
        
        # Check relationship was created
        assert result is True
        assert node_generator.graph.has_edge(parent_id, subtask_id)
        assert node_generator.graph.edges[parent_id, subtask_id]['type'] == 'HAS_SUBTASK'
    
    def test_update_task_status(self, node_generator):
        """Test updating a task's status."""
        # Create a task node
        task_id = node_generator.add_task_node({
            'id': '1',
            'title': 'Test Task',
            'status': 'pending'
        })
        
        # Update status to 'in-progress'
        result = node_generator.update_task_status(task_id, 'in-progress')
        
        # Check status was updated
        assert result is True
        assert node_generator.graph.nodes[task_id]['status'] == 'in-progress'
        
        # Update status to 'done'
        result = node_generator.update_task_status(task_id, 'done')
        
        # Check status was updated
        assert result is True
        assert node_generator.graph.nodes[task_id]['status'] == 'done'

if __name__ == '__main__':
    unittest.main() 