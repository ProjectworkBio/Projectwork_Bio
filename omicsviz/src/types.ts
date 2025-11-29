export type NodeType = 'protein' | 'disease';

// Data payload stored in each Cytoscape node's `data` field. 
export interface NodeData {
  // Unique identifier (also used as the Cytoscape node id). 
  id: string;
  // Short display label shown on the node. 
  label: string;
  // Domain-specific type for styling/filtering. 
  type: NodeType;
  // Optional UniProt id for proteins. 
  uniprot?: string;
}

// Data payload stored in each Cytoscape edge's `data` field. 
export interface EdgeData {
  // Unique identifier for the edge. 
  id: string;
  // Source node id. 
  source: string;
  // Target node id. 
  target: string;
  /*
   * Free-form relationship type coming from the relations in
   * ppi_relations_grouped.json (e.g. "Binds to / Interacts with", "Activates").
   */
    type?: string;
  // Optional evidence identifiers (e.g., pmids). 
  evidence?: string[];
}

// Concrete Cytoscape element wrappers we store/receive from JSON. 
export interface CyNode {
  data: NodeData;
  position?: { x: number; y: number };
}
export interface CyEdge {
  data: EdgeData;
}

// Top-level graph structure consumed by NetworkView. 
export interface GraphData {
  nodes: CyNode[];
  edges: CyEdge[];
}

// Paper-related types for the right-hand table. 
export interface PaperRelation {
  source: string;
  target: string;
  // Same free-form relation label as on the edges.
  type: string;
}
export interface Paper {
  pmid: string;
  title: string;
  relations: PaperRelation[];
}