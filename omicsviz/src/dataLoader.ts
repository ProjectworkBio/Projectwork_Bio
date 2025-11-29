/**
 * Build a Cytoscape-ready graph from a list of papers.
 *
 * Each paper has:
 *  - a PMID
 *  - a list of relations, where each relation connects a "source" to a "target" with a given "type".
 *
 * This function:
 *  - creates one node for each unique source/target across all papers
 *  - creates one edge for each unique (source, target, type) combination
 *  - accumulates all PMIDs that support the same edge as "evidence"
 */
import { CyEdge, CyNode, GraphData, NodeData, Paper } from './types';

export function buildGraphFromPapers(papers: Paper[]): GraphData {

  // nodeMap: keeps track of nodes we've already seen, keyed by node id (e.g. protein name)
  const nodeMap = new Map<string, NodeData>();

  // edgeMap: keeps track of edges we've already created, keyed by "source||target||type"
  // so multiple papers supporting the same relation can reuse the same edge
  const edgeMap = new Map<string, CyEdge>();

  // Go through each paper one by one
  for (const paper of papers) {
    // For each relation in the current paper, we will:
    // 1. Ensure both source and target nodes exist
    // 2. Create or update the edge between them
    for (const rel of paper.relations) {
      const { source, target, type } = rel;

      // If we haven't seen this source node before, create it
      if (!nodeMap.has(source)) {
        nodeMap.set(source, {
          id: source,
          label: source,   // label shown in the graph; for now just reuse the id
          type: 'protein', // hardcoded type; could be made dynamic later
        });
      }

      // Same for the target node
      if (!nodeMap.has(target)) {
        nodeMap.set(target, {
          id: target,
          label: target,
          type: 'protein',
        });
      }

      // Build a unique key for this edge based on source, target, and relation type
      // This lets us merge evidence from multiple papers into the same edge
      const key = `${source}||${target}||${type}`;

      // The evidence we add for this paper is just "pmid:<pmid>"
      const pmidEvidence = `pmid:${paper.pmid}`;

      // Check if we've already created an edge for this (source, target, type)
      let edge = edgeMap.get(key);

      if (!edge) {
        // No edge yet: create a brand new edge with this paper as the first piece of evidence
        edge = {
          data: {
            id: `e${edgeMap.size}`, // give the edge a unique id based on how many edges we have so far
            source,
            target,
            type,
            evidence: [pmidEvidence],
          },
        };
        edgeMap.set(key, edge);
      } else {
        // Edge already exists: just add this paper's PMID as additional evidence (if not already present)
        const ev = edge.data.evidence ?? (edge.data.evidence = []);
        if (!ev.includes(pmidEvidence)) {
          ev.push(pmidEvidence);
        }
      }
    }
  }

  // Convert all nodeMap entries into Cytoscape-compliant node objects: { data: NodeData }
  const nodes: CyNode[] = Array.from(nodeMap.values()).map((data) => ({ data }));

  // Edges are already stored in Cytoscape format, just turn the map values into an array
  const edges: CyEdge[] = Array.from(edgeMap.values());

  // Return the final graph structure
  return { nodes, edges };
}