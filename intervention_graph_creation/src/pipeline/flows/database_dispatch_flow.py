from typing import Optional

from .base import Flow
from intervention_graph_creation.src.local_graph_extraction.db.ai_safety_graph import AISafetyGraph
from intervention_graph_creation.src.pipeline.local_graph import LocalGraph

class DatabaseDispatchFlow(Flow):
    """Flow that handles database operations and disk storage."""
    
    def __init__(self, next_flow: Optional[Flow] = None):
        super().__init__(next_flow)
        self.graph = AISafetyGraph()

    def _push_to_database(self, local_graph: LocalGraph) -> None:
        """Push the LocalGraph data to the database."""
        try:
            self.graph.ingest_local_graph(local_graph)
        except Exception as e:
            print(f"⚠️  Database push failed: {e}")

    def process(self, local_graph: LocalGraph) -> LocalGraph:
        """
        Process the LocalGraph by pushing to database and saving to disk.
        
        Args:
            local_graph: The LocalGraph instance to process
            
        Returns:
            The LocalGraph (passed through)
        """
        print("Starting database dispatch...")
        # Push to database
        self._push_to_database(local_graph)
        
        print("Database dispatch completed successfully")
        
        # Pass to next flow in chain
        return self._call_next(local_graph)
