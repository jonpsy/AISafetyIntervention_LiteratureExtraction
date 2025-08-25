import json
import logging
from pathlib import Path
from ..src.pipeline.local_graph import LocalGraph, GraphNode, GraphEdge
from ..src.pipeline.flows.embedder_flow import EmbedderFlow
from ..src.pipeline.flows.metadata_flow import create_metadata_flow

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def create_sample_graph() -> LocalGraph:
    """Create a simple test graph to avoid repeated creation."""
    nodes = [
        GraphNode(
            name="AI Safety Concept",
            type="concept",
            description="Core concept in AI safety research",
            aliases=["Safety", "AI Alignment"],
            concept_category="safety",
        ),
        GraphNode(
            name="Constitutional AI",
            type="intervention",
            description="Training approach for safe AI systems",
            intervention_lifecycle=4,
            intervention_maturity=3,
        ),
    ]

    edges = [
        GraphEdge(
            type="enables",
            source_node="AI Safety Concept",
            target_node="Constitutional AI",
            description="Safety concepts enable constitutional approaches",
            edge_confidence=4,
            concept_meta="Safety Chain",
        )
    ]

    return LocalGraph(nodes=nodes, edges=edges)


def test_all_metadata_functionality():
    """test of all MetadataFlow features with shared ARD loading."""

    logger.info("Testing: metadata addition, file saving, and pipeline integration")
    logger.info("Note: ARD dataset loaded once and reused across tests for efficiency")

    # Create shared sample graph
    sample_graph = create_sample_graph()
    logger.info(
        f"Created test graph: {len(sample_graph.nodes)} nodes, {len(sample_graph.edges)} edges"
    )

    # Test 1: Basic metadata addition (loads ARD once)
    logger.info("\n1️⃣ BASIC METADATA ADDITION")
    metadata_flow = create_metadata_flow(source_identifier="2310.01405")
    enriched_graph = metadata_flow.process(sample_graph)

    # Store the loaded MetadataAdder to reuse in other tests
    shared_metadata_adder = metadata_flow._metadata_adder

    # Verify publication was added
    sample_node = enriched_graph.nodes[0]
    sample_edge = enriched_graph.edges[0]

    if sample_node.publication and sample_edge.publication:
        logger.info("✅ Publication metadata added successfully")
        logger.info(f"   Paper: {sample_node.publication.title}")
        logger.info(f"   Authors: {len(sample_node.publication.authors)} authors")
    else:
        logger.error("❌ Publication metadata not found")
        return False

    # Test 2: File saving (reuse loaded MetadataAdder)
    logger.info("\n2️⃣ FILE SAVING")
    output_path = "intervention_graph_creation/data/processed/test_complete_output.json"

    # Create new flow but reuse the loaded MetadataAdder
    save_flow = create_metadata_flow(
        source_identifier="2310.01405", save_to_file=True, output_path=output_path
    )
    save_flow._metadata_adder = shared_metadata_adder  # Reuse loaded data
    save_flow.process(sample_graph)

    # Verify file was created and has correct content
    output_file = Path(output_path)
    if output_file.exists():
        with open(output_file, "r") as f:
            saved_data = json.load(f)

        # Check structure
        has_nodes = "nodes" in saved_data and len(saved_data["nodes"]) > 0
        has_metadata = "source_metadata" in saved_data["nodes"][0]
        has_chains = (
            "logical_chains" in saved_data and len(saved_data["logical_chains"]) > 0
        )

        if has_nodes and has_metadata and has_chains:
            logger.info(
                f"✅ File saved successfully: {output_file.name} ({output_file.stat().st_size} bytes)"
            )
            logger.info(
                f"   Contains: {len(saved_data['nodes'])} nodes, {len(saved_data['logical_chains'])} chains"
            )
        else:
            logger.error("❌ Saved file has incorrect structure")
            return False
    else:
        logger.error("❌ Output file not created")
        return False

    # Test 3: Pipeline integration (only if extraction file exists)
    logger.info("\n3️⃣ PIPELINE INTEGRATION")
    extraction_file = "intervention_graph_creation/data/processed/2307.16513v2.json"

    if Path(extraction_file).exists():
        # Create pipeline with metadata saving
        pipeline_output = (
            "intervention_graph_creation/data/processed/test_pipeline_complete.json"
        )
        metadata_flow_pipeline = create_metadata_flow(
            source_identifier="2307.16513v2",
            save_to_file=True,
            output_path=pipeline_output,
        )

        embedder = EmbedderFlow(next_flow=metadata_flow_pipeline)

        logger.info("   Processing extraction file through pipeline...")
        pipeline_result = embedder.process(extraction_file)

        # Verify pipeline result
        pipeline_file = Path(pipeline_output)
        if pipeline_file.exists() and len(pipeline_result.nodes) > 0:
            logger.info("✅ Pipeline completed successfully")
            logger.info(
                f"   Processed: {len(pipeline_result.nodes)} nodes, {len(pipeline_result.edges)} edges"
            )
            logger.info(
                f"   Saved: {pipeline_file.name} ({pipeline_file.stat().st_size} bytes)"
            )

            # Check if metadata exists in pipeline result
            if pipeline_result.nodes[0].publication:
                logger.info(f"   Paper: {pipeline_result.nodes[0].publication.title}")
            else:
                logger.warning("   No metadata in pipeline result")
        else:
            logger.error("❌ Pipeline failed")
            return False
    else:
        logger.info("   Skipping pipeline test (extraction file not found)")

    # Test 4: Error handling (reuse loaded MetadataAdder)
    logger.info("\n4️⃣ ERROR HANDLING")
    error_flow = create_metadata_flow(source_identifier="nonexistent_paper_12345")
    error_flow._metadata_adder = shared_metadata_adder  # Reuse loaded data
    error_result = error_flow.process(sample_graph)

    # Should still work but with unknown metadata
    if (
        error_result.nodes[0].publication
        and error_result.nodes[0].publication.title == "Unknown"
    ):
        logger.info(
            "✅ Error handling works correctly (unknown paper → unknown metadata)"
        )
    else:
        logger.error("❌ Error handling failed")
        return False

    return True


def cleanup_test_files():
    """Clean up test files created during testing."""
    test_files = [
        "intervention_graph_creation/data/processed/test_complete_output.json",
        "intervention_graph_creation/data/processed/test_pipeline_complete.json",
        "intervention_graph_creation/data/processed/2310.01405_with_metadata.json",
        "intervention_graph_creation/data/processed/test_metadata_output.json",
    ]

    cleaned = 0
    for file_path in test_files:
        path = Path(file_path)
        if path.exists():
            path.unlink()
            cleaned += 1

    if cleaned > 0:
        logger.info(f"🧹 Cleaned up {cleaned} test files")


if __name__ == "__main__":
    logger.info("Starting comprehensive MetadataFlow test...")

    try:
        success = test_all_metadata_functionality()

        if success:
            logger.info("\n🎉 ALL TESTS PASSED!")
            logger.info("MetadataFlow is working correctly with:")
            logger.info("  ✅ Metadata addition from ARD/ArXiv sources")
            logger.info("  ✅ File saving in correct JSON format")
            logger.info("  ✅ Pipeline integration with EmbedderFlow")
            logger.info("  ✅ Error handling for unknown papers")
        else:
            logger.error("\n❌ SOME TESTS FAILED!")

        # Optional cleanup
        cleanup_response = input("\nClean up test files? (y/n): ").lower().strip()
        if cleanup_response == "y":
            cleanup_test_files()

    except Exception as e:
        logger.error(f"Test suite failed with error: {e}")
        import traceback

        traceback.print_exc()
