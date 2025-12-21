/* Responsibilities :
 *  - Hold all state (graph, papers, filters, selection, etc.)
 *  - Load papers (default JSON or uploaded)
 *  - Build derived data (visibleGraph, relatedToSelection)
 *  - Wire callbacks for the UI layout component
 */

import { useEffect, useMemo, useState, type ChangeEvent } from 'react';

import { GraphData, Paper as PaperT } from './types';
import { buildGraphFromPapers } from './dataLoader';
import OmicsVizLayout from './OmicsVizLayout';

// Default number of papers to load when the app starts
const BASELINE_PAPER_LIMIT = 100;

interface ActiveRelation {
  source: string;
  target: string;
  type?: string;
}

export default function App() {
  // -----------------------------
  // Core data state
  // -----------------------------

  // Graph rendered in Cytoscape (nodes + edges)
  const [graph, setGraph] = useState<GraphData | null>(null);
  // Raw papers array; the graph is derived from this
  const [papers, setPapers] = useState<PaperT[]>([]);

  // -----------------------------
  // Selection, hovering & filters
  // -----------------------------

  // Free-text search query for proteins/diseases
  const [query, setQuery] = useState('');

  // IDs of nodes currently selected in the network
  const [selectedIds, setSelectedIds] = useState<string[]>([]);
  // Temporary highlight when hovering (e.g. over relations in the list)
  const [hoveredIds, setHoveredIds] = useState<string[] | null>(null);

  // When the user clicks a specific relation in the frequency list,
  // we store that relation here to filter papers more precisely.
  const [activeRelation, setActiveRelation] = useState<ActiveRelation | null>(null);

  // Simple type filters: show/hide proteins and diseases in the graph
  const [filterProteins, setFilterProteins] = useState(true);
  const [filterDiseases, setFilterDiseases] = useState(true);

  // -----------------------------
  // Data source & limits
  // -----------------------------

  // Max number of papers to load (both for default and uploaded data)
  const [paperLimit, setPaperLimit] = useState<number>(BASELINE_PAPER_LIMIT);

  // Track where the data came from:
  // 'default' = bundled JSON, 'uploaded' = user-provided file
  const [dataSource, setDataSource] = useState<'default' | 'uploaded'>('default');

  // Keep a copy of *all* uploaded papers so we can slice them by paperLimit
  const [uploadedPapers, setUploadedPapers] = useState<PaperT[] | null>(null);

  // -----------------------------
  // Display & layout knobs
  // -----------------------------

  // Whether to show evidence labels / counts on edges
  const [showEdgeEvidence, setShowEdgeEvidence] = useState(true);

  // Controls how spread out nodes are in the network layout (in pixels)
  const [nodeGapPx, setNodeGapPx] = useState<number>(80);

  // -----------------------------
  // Initial data load + reacting to dataSource / paperLimit changes
  // -----------------------------

  useEffect(() => {
    // When dataSource or paperLimit changes, we either:
    //  - reload from the default bundled JSON, or
    //  - slice the uploaded set and rebuild the graph.
    if (dataSource === 'default') {
      // Load default JSON from /proteindata/ppi_v3.json
      loadDefaultData(paperLimit).then(({ graph, papers }) => {
        setGraph(graph);
        setPapers(papers);
      });
    } else if (dataSource === 'uploaded' && uploadedPapers) {
      // Use the uploaded data, but respect the current paperLimit
      const limited = uploadedPapers.slice(0, paperLimit);
      setPapers(limited);
      setGraph(buildGraphFromPapers(limited));
    }
  }, [paperLimit, dataSource, uploadedPapers]);

  /**
   * visibleGraph:
   *  - Starts from the full graph
   *  - Applies type filters (proteins / diseases)
   *  - Then applies the search query (if any)
   */
  const visibleGraph = useMemo<GraphData | null>(() => {
    if (!graph) return null;

    const q = query.trim().toLowerCase();

    // Step 1: Apply type filters (proteins / diseases)
    const baseNodes = graph.nodes.filter((n) => {
      const type = n.data.type;
      if (type === 'protein' && !filterProteins) return false;
      if (type === 'disease' && !filterDiseases) return false;
      return true;
    });

    const baseNodeIds = new Set(baseNodes.map((n) => n.data.id));
    // Keep only edges whose endpoints remain visible after type filtering
    const baseEdges = graph.edges.filter(
      (e) => baseNodeIds.has(e.data.source) && baseNodeIds.has(e.data.target),
    );

    // No search query: just return the type-filtered graph
    if (q.length === 0) {
      return { nodes: baseNodes, edges: baseEdges };
    }

    // Step 2: Find nodes that match the search query
    const matchingNodes = baseNodes.filter((n) => {
      const label = String(n.data.label ?? '').toLowerCase();
      const id = String(n.data.id ?? '').toLowerCase();
      return label.includes(q) || id.includes(q);
    });

    // If nothing matches, return an empty graph (no nodes, no edges)
    if (matchingNodes.length === 0) {
      return { nodes: [], edges: [] };
    }

    const matchingIds = new Set(matchingNodes.map((n) => n.data.id));

    // Step 3: Keep edges that touch at least one matching node
    const edges = baseEdges.filter(
      (e) => matchingIds.has(e.data.source) || matchingIds.has(e.data.target),
    );

    // Step 4: From those edges, collect all endpoints
    // (matching nodes + neighbours of matches)
    const neighbourIds = new Set<string>();
    edges.forEach((e) => {
      neighbourIds.add(e.data.source);
      neighbourIds.add(e.data.target);
    });

    // Final node set: every node that lies on the kept edges
    const nodes = baseNodes.filter((n) => neighbourIds.has(n.data.id));

    return { nodes, edges };
  }, [graph, query, filterProteins, filterDiseases]);

  /**
   * relatedToSelection:
   *  - If a specific relation is active (from RelationFrequencyList),
   *    we show *only* papers that contain that exact relation (optionally matching type).
   *  - Otherwise, if there are selected node IDs, we show papers that touch
   *    at least one of those nodes.
   *  - If nothing is selected, we show all papers.
   */
  const relatedToSelection = useMemo(() => {
    // 1) If the user clicked on a specific relation in the frequency list,
    //    filter papers to those that contain that relation.
    if (activeRelation) {
      const { source, target, type } = activeRelation;

      return papers.filter((p) =>
        p.relations.some((r) => {
          const samePair =
            (r.source === source && r.target === target) ||
            (r.source === target && r.target === source); // treat edges as undirected

          if (!samePair) return false;

          // If we know the relation type, try to match it too;
          // otherwise matching the node pair is enough.
          if (!type || !r.type) return samePair;
          return r.type === type;
        }),
      );
    }

    // 2) No active relation: fallback to generic node selection
    if (selectedIds.length === 0) return papers;

    const ids = new Set(selectedIds);
    return papers.filter((p) =>
      p.relations.some((r) => ids.has(r.source) || ids.has(r.target)),
    );
  }, [papers, selectedIds, activeRelation]);

  // If we're hovering something (e.g. a relation row), use that as the selection
  // for highlighting in the network; otherwise use the normal selectedIds.
  const selectionToDisplay = hoveredIds && hoveredIds.length > 0 ? hoveredIds : selectedIds;

  // -----------------------------
  // File upload handlers
  // -----------------------------

  /** Safely parse JSON and return null instead of throwing if parsing fails. */
  function parseJsonSafe<T>(raw: string): T | null {
    try {
      return JSON.parse(raw) as T;
    } catch {
      return null;
    }
  }

  /**
   * Handle user uploading a papers.json file.
   * Expected format: an array of PaperT objects.
   */
  function onUploadPapers(evt: ChangeEvent<HTMLInputElement>) {
    const file = evt.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = () => {
      const parsed = parseJsonSafe<PaperT[]>(String(reader.result));

      if (parsed && Array.isArray(parsed)) {
        // Switch into "uploaded" mode and keep the full set around
        setDataSource('uploaded');
        setUploadedPapers(parsed);

        const limit = Math.max(1, paperLimit);
        const limited = parsed.slice(0, limit);

        setPapers(limited);
        setGraph(buildGraphFromPapers(limited));
      } else {
        // Simple user-facing error if the JSON structure isn't what we expect
        alert('Invalid papers.json (expected an array).');
      }
    };
    reader.readAsText(file);
  }

  /**
   * Reset UI-level state back to defaults.
   * (Does not reload data source; it just resets filters and visual options.)
   */
  function reset() {
    setQuery('');
    setSelectedIds([]);
    setHoveredIds(null);
    setFilterProteins(true);
    setFilterDiseases(true);
    setShowEdgeEvidence(true);
    setNodeGapPx(80);
    setActiveRelation(null);
  }

  // -----------------------------
  // Render: delegate to layout component
  // -----------------------------
  return (
    <OmicsVizLayout
      graph={visibleGraph}
      papers={papers}
      relatedToSelection={relatedToSelection}
      selectionToDisplay={selectionToDisplay}
      query={query}
      onQueryChange={setQuery}
      filterProteins={filterProteins}
      onToggleProteins={() => setFilterProteins((v) => !v)}
      filterDiseases={filterDiseases}
      onToggleDiseases={() => setFilterDiseases((v) => !v)}
      showEdgeEvidence={showEdgeEvidence}
      onToggleShowEdgeEvidence={() => setShowEdgeEvidence((v) => !v)}
      nodeGapPx={nodeGapPx}
      onNodeGapPxChange={setNodeGapPx}
      paperLimit={paperLimit}
      onPaperLimitChange={setPaperLimit}
      onReset={reset}
      onUploadPapers={onUploadPapers}
      onNetworkSelectionChange={(ids) => {
        // When the user directly selects nodes in the network,
        // clear any active relation filter and update the selection.
        setActiveRelation(null);
        setSelectedIds(ids);
      }}
      onRelationClick={(relation) => {
        // Clicking a relation:
        //  - sets it as the active relation (for paper filtering)
        //  - selects its two endpoint nodes in the network
        setActiveRelation(relation);
        setSelectedIds([relation.source, relation.target]);
      }}
      onSelectPaper={(paper) => {
        // When a paper is clicked:
        //  - clear any active relation filter
        //  - select all nodes that participate in its relations
        setActiveRelation(null);
        const ids = Array.from(
          new Set(paper.relations.flatMap((r) => [r.source, r.target])),
        );
        setSelectedIds(ids);
      }}
    />
  );
}

/**
 * Helper to load the default bundled dataset from /proteindata/ppi_v3.json,
 * slice it to the requested limit, and build the corresponding graph.
 */
export async function loadDefaultData(
  limit: number = BASELINE_PAPER_LIMIT,
): Promise<{ graph: GraphData; papers: PaperT[] }> {
  const allPapers: PaperT[] = await fetch('/proteindata/ppi_v3.json').then((r) => r.json());
  const papers = allPapers.slice(0, limit);
  const graph = buildGraphFromPapers(papers);
  return { graph, papers };
}