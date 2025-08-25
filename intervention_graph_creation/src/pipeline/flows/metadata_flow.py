import json
import logging
from pathlib import Path
from typing import Optional, Union

from .base import Flow
from ..local_graph import LocalGraph
from ..metadata import Metadata
from ...local_graph_extraction.add_metadata import MetadataAdder

logger = logging.getLogger(__name__)


class MetadataFlow(Flow):
    """Flow that adds publication metadata to GraphNode and GraphEdge objects."""

    def __init__(
        self,
        next_flow: Optional[Flow] = None,
        ard_dataset_path: Optional[str] = None,
        arxiv_pdfs_path: Optional[str] = None,
        source_identifier: Optional[str] = None,
    ):
        """
        Initialize MetadataFlow.

        Args:
            next_flow: Next flow in the pipeline chain
            ard_dataset_path: Path to ARD dataset directory
            arxiv_pdfs_path: Path to directory containing ArXiv PDFs
            source_identifier: Explicit source identifier (ArXiv ID, filename, etc.)
        """
        super().__init__(next_flow)
        self.ard_dataset_path = ard_dataset_path or "./alignment-research-dataset"
        self.arxiv_pdfs_path = (
            arxiv_pdfs_path or "./intervention_graph_creation/data/raw/pdfs_local"
        )
        self.source_identifier = source_identifier
        self._metadata_adder = None
        self.save_to_file = False
        self.output_path = None

    def _get_metadata_adder(self) -> MetadataAdder:
        """Lazy initialization of MetadataAdder."""
        if self._metadata_adder is None:
            logger.info(
                f"Initializing MetadataAdder with ARD path: {self.ard_dataset_path}, ArXiv path: {self.arxiv_pdfs_path}"
            )
            self._metadata_adder = MetadataAdder(
                ard_dataset_path=self.ard_dataset_path,
                arxiv_pdfs_path=self.arxiv_pdfs_path,
            )
        return self._metadata_adder

    def _determine_source_identifier(self, local_graph: LocalGraph) -> str:
        """
        Determine the source identifier for the graph.

        This could be enhanced to extract from graph metadata, filename, etc.
        For now, uses the provided source_identifier or a default.
        """
        if self.source_identifier:
            return self.source_identifier

        # Try to extract from existing metadata if available
        if (
            local_graph.nodes
            and hasattr(local_graph.nodes[0], "metadata")
            and local_graph.nodes[0].metadata
        ):
            return local_graph.nodes[0].metadata.paper_id

        # Default fallback - this should be set by the calling code
        logger.warning("No source identifier provided, using 'unknown'")
        return "unknown"

    def _create_metadata_from_publication(
        self, source_identifier: str
    ) -> Optional[Metadata]:
        """Create Metadata object from publication data."""
        metadata_adder = self._get_metadata_adder()
        publication = metadata_adder.get_publication(source_identifier)

        if publication:
            metadata_dict = metadata_adder.create_source_metadata(
                publication, source_identifier
            )
            return Metadata.from_dict(metadata_dict)
        else:
            # Create minimal metadata with just the identifier
            logger.warning(
                f"Could not find publication for identifier: {source_identifier}"
            )
            return Metadata(
                paper_id=source_identifier,
                title=None,
                authors=None,
                date_published=None,
                url=None,
                source_type="unknown",
                abstract=None,
            )

    def _add_metadata_to_nodes(
        self, local_graph: LocalGraph, metadata: Metadata
    ) -> None:
        """Add metadata to all nodes in the local graph."""
        for node in local_graph.nodes:
            node.metadata = metadata
            logger.debug(f"Added metadata to node: {node.name}")

    def _add_metadata_to_edges(
        self, local_graph: LocalGraph, metadata: Metadata
    ) -> None:
        """Add metadata to all edges in the local graph."""
        for edge in local_graph.edges:
            edge.metadata = metadata
            logger.debug(
                f"Added metadata to edge: {edge.type} ({edge.source_node} -> {edge.target_node})"
            )

    def _convert_local_graph_to_json_dict(self, local_graph: LocalGraph) -> dict:
        """Convert LocalGraph back to JSON format compatible with PaperSchema."""
        nodes_data = []
        for node in local_graph.nodes:
            node_dict = {
                "name": node.name,
                "aliases": node.aliases,
                "type": node.type,
                "description": node.description,
                "concept_category": node.concept_category,
                "intervention_lifecycle": node.intervention_lifecycle,
                "intervention_maturity": node.intervention_maturity,
            }

            # Add metadata if present
            if node.metadata:
                node_dict["source_metadata"] = node.metadata.to_dict()

            nodes_data.append(node_dict)

        # Convert edges back to logical chains format
        # Group edges by concept_meta (logical chain title)
        chains_dict = {}
        for edge in local_graph.edges:
            chain_title = edge.concept_meta or "Default Chain"

            if chain_title not in chains_dict:
                chains_dict[chain_title] = {"title": chain_title, "edges": []}

            edge_dict = {
                "type": edge.type,
                "source_node": edge.source_node,
                "target_node": edge.target_node,
                "description": edge.description,
                "edge_confidence": edge.edge_confidence,
            }

            if edge.metadata:
                edge_dict["source_metadata"] = edge.metadata.to_dict()

            chains_dict[chain_title]["edges"].append(edge_dict)

        # Convert chains dict to list
        logical_chains = list(chains_dict.values())

        return {"nodes": nodes_data, "logical_chains": logical_chains}

    def _save_to_json_file(self, local_graph: LocalGraph, output_path: Path) -> None:
        """Save LocalGraph to JSON file in PaperSchema format."""
        json_data = self._convert_local_graph_to_json_dict(local_graph)

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save to file
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved enriched graph to: {output_path}")

    def set_output_file(
        self, output_path: Union[str, Path], save_to_file: bool = True
    ) -> "MetadataFlow":
        """Set the output file path and enable saving."""
        self.output_path = (
            Path(output_path) if isinstance(output_path, str) else output_path
        )
        self.save_to_file = save_to_file
        return self

    def process(self, local_graph: LocalGraph) -> LocalGraph:
        """
        Process the LocalGraph and add metadata to all nodes and edges.
        """
        logger.info("Starting metadata enrichment...")

        # Determine source identifier
        source_identifier = self._determine_source_identifier(local_graph)
        logger.info(f"Using source identifier: {source_identifier}")

        # Create metadata object
        metadata = self._create_metadata_from_publication(source_identifier)

        if metadata:
            logger.info(f"Created metadata: {metadata}")

            # Add metadata to nodes
            self._add_metadata_to_nodes(local_graph, metadata)
            logger.info(f"Added metadata to {len(local_graph.nodes)} nodes")

            # Add metadata to edges
            self._add_metadata_to_edges(local_graph, metadata)
            logger.info(f"Added metadata to {len(local_graph.edges)} edges")
        else:
            logger.error(f"Failed to create metadata for source: {source_identifier}")

        logger.info("Metadata enrichment completed")

        # Save to file if requested
        if self.save_to_file and self.output_path:
            self._save_to_json_file(local_graph, self.output_path)
        elif self.save_to_file and not self.output_path:
            # Auto-generate output path based on source identifier
            if source_identifier and source_identifier != "unknown":
                auto_path = Path(
                    f"./intervention_graph_creation/data/processed/{source_identifier}_with_metadata.json"
                )
                self._save_to_json_file(local_graph, auto_path)
            else:
                logger.warning(
                    "Cannot save to file: no output path specified and source identifier is unknown"
                )

        # Pass to next flow in chain
        return self._call_next(local_graph)

    def set_source_identifier(self, source_identifier: str) -> "MetadataFlow":
        """Set the source identifier for this flow."""
        self.source_identifier = source_identifier
        return self


def create_metadata_flow(
    source_identifier: Optional[str] = None,
    ard_path: str = "./alignment-research-dataset",
    arxiv_path: str = "./intervention_graph_creation/data/raw/pdfs_local",
    next_flow: Optional[Flow] = None,
    save_to_file: bool = False,
    output_path: Optional[Union[str, Path]] = None,
) -> MetadataFlow:
    flow = MetadataFlow(
        next_flow=next_flow,
        ard_dataset_path=ard_path,
        arxiv_pdfs_path=arxiv_path,
        source_identifier=source_identifier,
    )

    if save_to_file:
        flow.save_to_file = True
        if output_path:
            flow.output_path = (
                Path(output_path) if isinstance(output_path, str) else output_path
            )

    return flow
