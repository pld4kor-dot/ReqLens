"""Unit tests for the graph store."""

from reqinone2.storage.graph_store import GraphStore


class TestGraphStore:
    def test_add_and_get_node(self):
        gs = GraphStore()
        gs.upsert_requirement_node("REQ-001", {"text": "Login requirement"})
        node = gs.get_node("REQ-001")
        assert node is not None
        assert node["node_type"] == "requirement"
        assert node["text"] == "Login requirement"

    def test_add_edge(self):
        gs = GraphStore()
        gs.upsert_node("REQ-001", "requirement")
        gs.upsert_node("SPN-001", "source_span")
        gs.upsert_edge("REQ-001", "SPN-001", "derived_from", {"confidence": 0.9})

        edges = gs.get_edges_from("REQ-001")
        assert len(edges) == 1
        assert edges[0][1] == "SPN-001"
        assert edges[0][2]["edge_type"] == "derived_from"

    def test_neighborhood(self):
        gs = GraphStore()
        gs.upsert_node("REQ-001", "requirement")
        gs.upsert_node("REQ-002", "requirement")
        gs.upsert_node("SPN-001", "source_span")
        gs.upsert_edge("REQ-001", "SPN-001", "derived_from")
        gs.upsert_edge("REQ-001", "REQ-002", "requires")

        result = gs.get_requirement_neighborhood("REQ-001", depth=1)
        assert len(result["nodes"]) == 3
        assert len(result["edges"]) == 2

    def test_conflict_detection(self):
        gs = GraphStore()
        gs.upsert_node("REQ-001", "requirement")
        gs.upsert_node("REQ-002", "requirement")
        gs.upsert_edge("REQ-001", "REQ-002", "conflicts_with")

        pairs = gs.find_conflict_candidates("project-1")
        assert len(pairs) == 1
        assert pairs[0] == ("REQ-001", "REQ-002")

    def test_export_dict(self):
        gs = GraphStore()
        gs.upsert_node("REQ-001", "requirement", {"text": "Hello"})
        data = gs.to_dict()
        assert len(data["nodes"]) == 1
        assert data["nodes"][0]["id"] == "REQ-001"

    def test_empty_graph(self):
        gs = GraphStore()
        assert gs.node_count == 0
        assert gs.edge_count == 0
        assert gs.to_dict() == {"nodes": [], "edges": []}
