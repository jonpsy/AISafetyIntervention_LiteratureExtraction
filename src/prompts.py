NODE_TYPES = [
    "Method",
    "Model",
    "Dataset",
    "Task",
    "Metric",
    "Result",
    "Claim",
    "Assumption",
    "Threat",
    "Mitigation",
    "Concept",
]

EDGE_LABELS = [
    "PROPOSES",
    "EVALUATES_ON",
    "REPORTS_METRIC",
    "IMPROVES_OVER",
    "ASSUMES",
    "IDENTIFIES_THREAT",
    "PROPOSES_MITIGATION",
]

OUTPUT_SCHEMA = {
    "type": "Object",
    "properties": {
        "relationships": {
            "type": "Array",
            "items": {
                "type": "Object",
                "properties": {
                    "source": {
                        "type": "Object",
                        "properties": {
                            "id": {"type": "string"},
                            "label": {"type": "string"},
                            "properties": {
                                "type": "Object",
                                "properties": {
                                    "canonical_name": {"type": "string"},
                                    "aliases": {
                                        "type": "Array",
                                        "items": {"type": "string"},
                                    },
                                    "confidence": {
                                        "type": "Number",
                                        "minimum": 0.0,
                                        "maximum": 1.0,
                                    },
                                    "notes": {"type": "string"},
                                },
                            },
                        },
                    },
                    "target": {
                        "type": "Object",
                        "properties": {
                            "id": {"type": "string"},
                            "label": {"type": "string"},
                            "properties": {
                                "type": "Object",
                                "properties": {
                                    "canonical_name": {"type": "string"},
                                    "aliases": {
                                        "type": "Array",
                                        "items": {"type": "string"},
                                    },
                                    "confidence": {
                                        "type": "Number",
                                        "minimum": 0.0,
                                        "maximum": 1.0,
                                    },
                                    "notes": {"type": "string"},
                                },
                            },
                        },
                    },
                    "label": {"type": "string"},
                    "properties": {
                        "type": "Object",
                        "properties": {
                            "rationale": {"type": "string"},
                            "confidence": {
                                "type": "Number",
                                "minimum": 0.0,
                                "maximum": 1.0,
                            },
                            "value": {"type": "string"},
                            "split": {"type": "string"},
                            "baseline": {"type": "string"},
                            "delta": {"type": "string"},
                        },
                    },
                },
            },
        },
        "suggested_new_node_types": {
            "type": "Array",
            "items": {
                "type": "Object",
                "properties": {
                    "label": {"type": "string"},
                    "definition": {"type": "string"},
                    "example_instance": {"type": "string"},
                },
            },
        },
        "suggested_new_edge_types": {
            "type": "Array",
            "items": {
                "type": "Object",
                "properties": {
                    "label": {"type": "string"},
                    "definition": {"type": "string"},
                    "example_instance": {"type": "string"},
                },
            },
        },
    },
}


EXTRACTION_PROMPT_TEMPLATE = f"""
You are an expert AI safety researcher tasked with extracting structured knowledge from academic papers to build a comprehensive knowledge graph. Your goal is to identify key concepts (nodes) and their relationships (edges) that will help researchers understand the landscape of AI safety research.

You will be provided with the text of a research paper. Extract nodes and relationships from this paper, adhering strictly to the following guidelines and the required JSON output schema.

---
**Extraction Guidelines**

**1. Core Ontology (Primary Schema):**
You MUST use the following predefined node types and edge labels for your primary extractions.
*   **Node Types:** {", ".join(NODE_TYPES)}
*   **Edge Labels:** {", ".join(EDGE_LABELS)}

**2. Creative & Causal Extraction (Suggesting New Types):**
While adhering to the core ontology, you should also identify important causal or domain-specific relationships and node types not listed above (e.g., `CAUSES`, `PREVENTS`, `RISK_TYPE`). If you find a recurring, important type that is not in the core list, you MUST add it to the `suggested_new_node_types` or `suggested_new_edge_types` field in your output.

**3. Strict Requirements (Mandatory for all extractions):**
*   **Conservative Policy:** Extract only what the paper explicitly states or strongly implies. Do not invent connections or attributes.
*   **Confidence:** Every extraction **must** have a `confidence` score (0.0–1.0). Use lower confidence (<0.6) for extractions that rely on inference rather than explicit text.
*   **Normalization:** Canonicalize method and dataset names (e.g., "RLHF" -> "Reinforcement Learning from Human Feedback"). The `canonical_name` should be the full name. List original text in `aliases`. If uncertain, copy the surface form into `canonical_name` and explain your reasoning in the `notes` field.
*   **Node Naming:** Create concise but descriptive node `id`s (use snake_case for multi-word concepts, e.g., "strategic_deception").
*   **Exclusions:** Do NOT extract authors, organizations, or bibliographic citations. The source of all extractions is implicitly the current paper.

---

**Your Task:**
Based on the paper content provided to you, generate a single JSON object with the final knowledge graph. The JSON object must conform to the schema provided.
"""
