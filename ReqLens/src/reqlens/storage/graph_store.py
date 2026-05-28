"""
Knowledge graph store abstraction.

This implementation uses a hybrid approach:
1. On initialization, it loads all nodes and edges for a specific project 
   from the database into an in-memory NetworkX graph.
2. Read operations (e.g., get_neighborhood, to_dict) use this fast in-memory graph.
3. Write operations (e.g., upsert_edge) update both the in-memory graph AND add
   the new edge to the database session to be persisted.
"""

from __future__ import annotations

from typing import Any
import networkx as nx
import structlog
from sqlalchemy.orm import Session

# Import the ORM models needed to build the graph from the database
from reqlens.storage.db import RequirementRow, SourceSpanRow, GraphEdgeRow
from reqlens.domain.enums import DependencyEdgeType
from reqlens.domain.ids import generate_id

logger = structlog.get_logger(__name__)

class GraphStore:
    """
    A project-specific graph store that loads from and persists to a database.
    """

    def __init__(self, db_session: Session, project_id: str):
        self.db = db_session
        self.project_id = project_id
        self.graph = nx.DiGraph()
        self._load_graph_from_db()

    def _load_graph_from_db(self):
        """Initializes the in-memory graph with data from the database for the current project."""
        logger.debug("graph.load_from_db", project_id=self.project_id)
        
        # 1. Load Nodes
        req_nodes = self.db.query(RequirementRow).filter_by(project_id=self.project_id).all()
        for node in req_nodes:
            self.graph.add_node(
                node.id, 
                node_type="requirement", 
                text=node.text, 
                kind=node.kind, 
                status=node.status
            )

        span_nodes = self.db.query(SourceSpanRow).filter_by(project_id=self.project_id).all()
        for node in span_nodes:
            self.graph.add_node(
                node.id,
                node_type="source_span",
                text=node.text,
                document_id=node.document_id
            )

        # 2. Load Edges
        edges = self.db.query(GraphEdgeRow).filter_by(project_id=self.project_id).all()
        for edge in edges:
            self.graph.add_edge(
                edge.source_node_id,
                edge.target_node_id,
                edge_type=edge.edge_type,
                confidence=edge.confidence
            )
        
        logger.info(
            "graph.loaded", 
            project_id=self.project_id, 
            nodes=self.graph.number_of_nodes(), 
            edges=self.graph.number_of_edges()
        )

    # -- Node operations ---------------------------------------------
    # These now primarily update the in-memory graph; persistence is handled by other Repositories.

    def upsert_node(self, node_id: str, node_type: str, properties: dict[str, Any] | None = None) -> None:
        props = dict(properties or {})
        props["node_type"] = node_type
        self.graph.add_node(node_id, **props)

    def upsert_requirement_node(self, requirement_id: str, properties: dict[str, Any]) -> None:
        self.upsert_node(requirement_id, "requirement", properties)

    def upsert_source_span_node(self, span_id: str, properties: dict[str, Any]) -> None:
        self.upsert_node(span_id, "source_span", properties)

    def update_node_status(self, node_id: str, status: str, review_status: str) -> None:
        """Update the status and review_status attributes of an existing in-memory node.

        This is a lightweight operation — it only mutates the in-memory graph.
        The authoritative status is stored on the RequirementRow; the graph node
        mirrors it so that graph-based queries reflect the latest decision.
        """
        if node_id in self.graph:
            self.graph.nodes[node_id]["status"] = status
            self.graph.nodes[node_id]["review_status"] = review_status
            logger.debug(
                "graph.node_status_updated",
                node_id=node_id,
                status=status,
                review_status=review_status,
            )
        else:
            logger.warning("graph.node_not_found_for_status_update", node_id=node_id)

    def remove_node(self, node_id: str) -> None:
        """Remove a node (and all its incident edges) from both the in-memory graph
        and the persisted graph_edges table.

        Called when a requirement is *rejected* so that the knowledge graph only
        contains accepted (or still-pending) requirements.
        """
        if node_id not in self.graph:
            logger.warning("graph.node_not_found_for_removal", node_id=node_id)
            return

        # 1. Remove from in-memory graph (NetworkX handles incident edges automatically)
        self.graph.remove_node(node_id)
        logger.debug("graph.node_removed_in_memory", node_id=node_id)

        # 2. Delete persisted edges that reference this node from the DB session
        edges_to_delete = (
            self.db.query(GraphEdgeRow)
            .filter(
                GraphEdgeRow.project_id == self.project_id,
                (GraphEdgeRow.source_node_id == node_id)
                | (GraphEdgeRow.target_node_id == node_id),
            )
            .all()
        )
        for edge in edges_to_delete:
            self.db.delete(edge)

        logger.info(
            "graph.node_and_edges_removed",
            node_id=node_id,
            edges_deleted=len(edges_to_delete),
        )

    def sync_requirement_decision(
        self,
        requirement_id: str,
        decision: str,  # "accepted" | "rejected" | "needs_revision" | "deferred" | …
        requirement_text: str | None = None,
        kind: str | None = None,
    ) -> None:
        """Synchronise the knowledge graph with a human review decision.

        Rules
        -----
        * **accepted**       → ensure the node exists with up-to-date attributes;
                               update its status to ``accepted`` in both the
                               in-memory graph and the RequirementRow in the DB.
        * **rejected**       → remove the node (and its edges) entirely so the graph
                               only contains non-rejected requirements; persist the
                               status change to RequirementRow.
        * **anything else**  → update the node's ``review_status`` attribute in-place
                               in both the in-memory graph and RequirementRow.

        All DB writes are staged on ``self.db`` (the SQLAlchemy session) but NOT
        committed here — the caller (the route handler) is responsible for calling
        ``session.commit()`` so that the review decision record and the requirement
        status change are always committed atomically.
        """
        # Persist the status change to RequirementRow so it survives across
        # requests. GraphStore is instantiated per-request and rebuilds the
        # in-memory graph from the DB each time, so without this write the
        # in-memory update is lost the moment the request ends.
        req_row = self.db.get(RequirementRow, requirement_id)

        if decision == "rejected":
            # Remove from graph entirely – rejected requirements should not appear
            # in the knowledge graph.
            self.remove_node(requirement_id)
            if req_row is not None:
                req_row.status = "rejected"
                req_row.review_status = "rejected"
                logger.debug(
                    "graph.sync.row_status_updated",
                    node_id=requirement_id,
                    status="rejected",
                )

        elif decision == "accepted":
            # Ensure the node is present and marked accepted.
            if requirement_id not in self.graph:
                # Node was absent (edge case: graph loaded before requirement existed)
                props: dict[str, Any] = {
                    "status": "accepted",
                    "review_status": "accepted",
                }
                if requirement_text is not None:
                    props["text"] = requirement_text
                if kind is not None:
                    props["kind"] = kind
                self.upsert_requirement_node(requirement_id, props)
                logger.info(
                    "graph.node_added_on_accept",
                    node_id=requirement_id,
                )
            else:
                self.update_node_status(requirement_id, "accepted", "accepted")

            if req_row is not None:
                req_row.status = "accepted"
                req_row.review_status = "accepted"
                logger.debug(
                    "graph.sync.row_status_updated",
                    node_id=requirement_id,
                    status="accepted",
                )

        else:
            # needs_revision / deferred / pending – keep node, just refresh status
            self.update_node_status(requirement_id, decision, decision)
            if req_row is not None:
                req_row.review_status = decision
                logger.debug(
                    "graph.sync.row_review_status_updated",
                    node_id=requirement_id,
                    review_status=decision,
                )

    # -- Edge operations ---------------------------------------------
    # THIS IS THE CRITICAL CHANGE FOR PERSISTENCE

    def upsert_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: str,
        properties: dict[str, Any] | None = None,
    ) -> None:
        """Upserts an edge to both the in-memory graph and the database session."""
        props = dict(properties or {})
        props["edge_type"] = edge_type

        # 1. Update in-memory graph for immediate use
        self.graph.add_edge(source_id, target_id, **props)

        # 2. Create and add DB row to the session for persistence
        # Note: We do not commit here. The caller who manages the session is responsible for the commit.
        edge_row = GraphEdgeRow(
            id=generate_id("GED"),
            project_id=self.project_id,
            source_node_id=source_id,
            target_node_id=target_id,
            edge_type=edge_type,
            confidence=props.get("confidence"),
            # Add other properties from props if they exist in your GraphEdgeRow model
        )
        self.db.add(edge_row)
        logger.debug("graph.edge_added_to_session", source=source_id, target=target_id)


    # =================================================================
    # NOTE: The following read-methods can remain largely unchanged
    # because they operate on `self.graph`, which is now a pre-loaded,
    # project-specific graph.
    # =================================================================

    def get_node(self, node_id: str) -> dict[str, Any] | None:
        if node_id not in self.graph:
            return None
        return dict(self.graph.nodes[node_id])

    def get_nodes_by_type(self, node_type: str) -> list[tuple[str, dict[str, Any]]]:
        return [
            (n, dict(d)) for n, d in self.graph.nodes(data=True)
            if d.get("node_type") == node_type
        ]

    def get_edges_from(self, node_id: str) -> list[tuple[str, str, dict[str, Any]]]:
        if node_id not in self.graph:
            return []
        return [
            (node_id, target, dict(data))
            for target, data in self.graph[node_id].items()
        ]

    def get_edges_to(self, node_id: str) -> list[tuple[str, str, dict[str, Any]]]:
        if node_id not in self.graph:
            return []
        return [
            (source, node_id, dict(data))
            for source, data in self.graph.pred[node_id].items()
        ]

    # ... (get_requirement_neighborhood, find_conflict_candidates, to_dict, etc. remain the same) ...
    # ... (properties node_count and edge_count also remain the same) ...
    def get_requirement_neighborhood(
        self,
        requirement_id: str,
        depth: int = 2,
    ) -> dict[str, Any]:
        """Return subgraph around a requirement node."""
        if requirement_id not in self.graph:
            return {"nodes": [], "edges": []}
        # This logic correctly uses the in-memory graph, which is already scoped to the project
        visited_nodes = set(nx.bfs_tree(self.graph.to_undirected(), source=requirement_id, depth_limit=depth))
        subgraph = self.graph.subgraph(visited_nodes)
        
        nodes = [{"id": n, **dict(d)} for n, d in subgraph.nodes(data=True)]
        edges = [{"source": u, "target": v, **dict(d)} for u, v, d in subgraph.edges(data=True)]
        return {"nodes": nodes, "edges": edges}


    def find_conflict_candidates(self) -> list[tuple[str, str]]:
        """Find requirement pairs connected by conflict edges."""
        # This logic is also fine as it operates on the project-specific in-memory graph
        pairs: list[tuple[str, str]] = []
        for u, v, data in self.graph.edges(data=True):
            if data.get("edge_type") == DependencyEdgeType.conflicts_with.value:
                pairs.append((u, v))
        return pairs

    def find_duplicate_candidates(self) -> list[tuple[str, str]]:
        """Find requirement pairs connected by duplicate edges."""
        pairs: list[tuple[str, str]] = []
        for u, v, data in self.graph.edges(data=True):
            if data.get("edge_type") == DependencyEdgeType.duplicates.value:
                pairs.append((u, v))
        return pairs

    def to_dict(self) -> dict[str, Any]:
        """Export full graph as serializable dict."""
        return {
            "nodes": [{"id": n, **dict(d)} for n, d in self.graph.nodes(data=True)],
            "edges": [
                {"source": u, "target": v, **dict(d)}
                for u, v, d in self.graph.edges(data=True)
            ],
        }

    def to_graphml(self) -> str:
        """Export graph as GraphML string."""
        import io
        buf = io.BytesIO()
        nx.write_graphml(self.graph, buf)
        return buf.getvalue().decode("utf-8")

    @property
    def node_count(self) -> int:
        return self.graph.number_of_nodes()

    @property
    def edge_count(self) -> int:
        return self.graph.number_of_edges()

