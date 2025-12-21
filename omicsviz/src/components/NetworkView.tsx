import { useEffect, useMemo, useRef } from 'react';
import CytoscapeComponent from 'react-cytoscapejs';
import cytoscape from 'cytoscape';
import { GraphData } from '../types';

/**
 * === Layout controls =========================================================
 * Single knob (nodeGapPx) to spread/pack nodes.
 */
const DEFAULT_NODE_GAP_PX = 80;
const VIEW_PADDING = 10; // padding for cy.fit()
const LAYOUT_RESTARTS = 3;

/**
 * === Edge color mapping ======================================================
 */
const EDGE_COLORS = [
  '#90caf9',
  '#ffcc80',
  '#a5d6a7',
  '#ce93d8',
  '#ffab91',
  '#80cbc4',
  '#e6ee9c',
  '#f48fb1',
  '#b39ddb',
  '#bcaaa4',
];

const edgeTypeColorMap: Record<string, string> = {};

export function getEdgeColor(type?: string): string {
  if (!type) return '#bdbdbd'; // fallback grey
  const existing = edgeTypeColorMap[type];
  if (existing) return existing;

  const index = Object.keys(edgeTypeColorMap).length % EDGE_COLORS.length;
  const color = EDGE_COLORS[index];
  edgeTypeColorMap[type] = color;
  return color;
}

/**
 * === Geometry helpers ========================================================
 */
type XY = cytoscape.Position;
type CyCore = cytoscape.Core;
type NodePositionMap = cytoscape.NodePositionMap;
type LayoutOptions = cytoscape.LayoutOptions;

/** Non self-loop edges only, used by several metrics. */
function getNonSelfLoopEdges(cy: CyCore) {
  return (cy.edges() as cytoscape.EdgeCollection)
    .toArray()
    .filter((e) => e.source().id() !== e.target().id());
}

/**
 * Count approximate straight-line edge crossings.
 */
function countEdgeCrossings(cy: CyCore): number {
  const edges = getNonSelfLoopEdges(cy);
  let crossings = 0;

  for (let i = 0; i < edges.length; i++) {
    const e1 = edges[i];
    const p1a = e1.source().position();
    const p1b = e1.target().position();

    for (let j = i + 1; j < edges.length; j++) {
      const e2 = edges[j];

      // skip if they share a node
      const s1 = e1.source().id();
      const t1 = e1.target().id();
      const s2 = e2.source().id();
      const t2 = e2.target().id();
      if (s1 === s2 || s1 === t2 || t1 === s2 || t1 === t2) continue;

      const p2a = e2.source().position();
      const p2b = e2.target().position();

      if (segmentsIntersect(p1a, p1b, p2a, p2b)) crossings++;
    }
  }

  return crossings;
}

/** Simple orientation-test segment intersection. */
function segmentsIntersect(a1: XY, a2: XY, b1: XY, b2: XY): boolean {
  const cross = (p: XY, q: XY, r: XY) =>
    (q.x - p.x) * (r.y - p.y) - (q.y - p.y) * (r.x - p.x);

  const onDiffSides = (p: XY, q: XY, r: XY, s: XY) => {
    const d1 = cross(p, q, r);
    const d2 = cross(p, q, s);
    return (d1 > 0 && d2 < 0) || (d1 < 0 && d2 > 0);
  };

  return onDiffSides(a1, a2, b1, b2) && onDiffSides(b1, b2, a1, a2);
}

/**
 * Mean length of non-self-loop edges, used as a "spacing" metric.
 * Higher = more spread out.
 */
function meanEdgeLength(cy: CyCore): number {
  const edges = getNonSelfLoopEdges(cy);
  if (edges.length === 0) return 0;

  let total = 0;
  for (const e of edges) {
    const p1 = e.source().position();
    const p2 = e.target().position();
    const dx = p1.x - p2.x;
    const dy = p1.y - p2.y;
    total += Math.hypot(dx, dy);
  }

  return total / edges.length;
}

/**
 * Treat nodes as "balls" with radius ~ minDist/2.
 * For every pair closer than minDist, push them apart symmetrically.
 */
function enforceMinNodeDistance(
  cy: CyCore,
  minDist: number,
  iterations = 10,
) {
  const nodes = cy.nodes().toArray();
  if (nodes.length === 0) return;

  const minDistSq = minDist * minDist;

  for (let iter = 0; iter < iterations; iter++) {
    let movedAny = false;

    for (let i = 0; i < nodes.length; i++) {
      const ni = nodes[i];
      const pi = ni.position();

      for (let j = i + 1; j < nodes.length; j++) {
        const nj = nodes[j];
        const pj = nj.position();

        let dx = pj.x - pi.x;
        let dy = pj.y - pi.y;
        let distSq = dx * dx + dy * dy;

        // Same position or NaN: give a random tiny nudge first
        if (!isFinite(distSq) || distSq === 0) {
          dx = Math.random() - 0.5;
          dy = Math.random() - 0.5;
          distSq = dx * dx + dy * dy;
        }

        if (distSq >= minDistSq) continue;

        const dist = Math.sqrt(distSq) || 1;
        const overlap = (minDist - dist) / 2; // split push between both nodes
        const nx = dx / dist;
        const ny = dy / dist;

        ni.position({
          x: pi.x - nx * overlap,
          y: pi.y - ny * overlap,
        });
        nj.position({
          x: pj.x + nx * overlap,
          y: pj.y + ny * overlap,
        });

        movedAny = true;
      }
    }

    if (!movedAny) break;
  }
}

/**
 * Run a COSE layout once and wait until it stops.
 */
function runCoseOnce(cy: CyCore, opts: LayoutOptions): Promise<void> {
  return new Promise<void>((resolve) => {
    const layout = cy.layout(opts);
    layout.on('layoutstop', () => resolve());
    layout.run();
  });
}

/**
 * Snapshot current node positions into a plain map.
 */
function snapshotPositions(cy: CyCore): NodePositionMap {
  const positions: NodePositionMap = {};
  cy.nodes().forEach((n) => {
    positions[n.id()] = { ...n.position() };
  });
  return positions;
}

/**
 * Run several COSE layouts and keep the fewest-crossings / best-spaced result.
 * Applies the winning positions with a 'preset' layout and then fits the view.
 */
async function runBestCoseLayout(
  cy: CyCore,
  nodeGapPx: number | undefined,
) {
  const attempts = Math.max(1, LAYOUT_RESTARTS);

  // Requested visual spacing in pixels
  const gap = Math.max(1, nodeGapPx ?? DEFAULT_NODE_GAP_PX);

  // Fixed internal baseline for tuning the forces
  const BASE_GAP_FOR_FORCES = 60;
  const gapScale = Math.max(0.25, Math.min(4, gap / BASE_GAP_FOR_FORCES));

  // Slightly bigger than gap so nodes really look separate
  const minNodeDistance = gap * 1.05;

  const base: LayoutOptions = {
    name: 'cose',
    randomize: true,
    animate: false,
    fit: false,

    componentSpacing: gap,
    idealEdgeLength: gap,
    nodeOverlap: Math.max(2, gap / 10),

    // Very strong repulsion; we still enforce a hard min distance afterwards.
    nodeRepulsion: 2048 * gapScale * gapScale * 8,
    gravity: 0.5,

    numIter: 1500,
    coolingFactor: 0.99,
    minTemp: 1.0,
  };

  let bestPositions: NodePositionMap | null = null;
  let bestCrossings = Number.POSITIVE_INFINITY;
  let bestSpacing = -Infinity;

  for (let i = 0; i < attempts; i++) {
    const opts = { ...base, randomize: i !== 0 };

    // 1) Let COSE do its thing
    await runCoseOnce(cy, opts);

    // 2) Force nodes to obey a hard min distance
    enforceMinNodeDistance(cy, minNodeDistance, 10);

    // 3) Evaluate this candidate
    const spacing = meanEdgeLength(cy);
    const crossings = countEdgeCrossings(cy);

    // We care about crossings, but among layouts with similar spacing.
    const spacingEpsilon = gap * 0.1;
    const spacingIsMuchBetter = spacing > bestSpacing + spacingEpsilon;
    const spacingIsSimilar = Math.abs(spacing - bestSpacing) <= spacingEpsilon;
    const crossingsAreBetter = crossings < bestCrossings;

    if (
      bestPositions === null ||
      spacingIsMuchBetter ||
      (spacingIsSimilar && crossingsAreBetter)
    ) {
      bestSpacing = spacing;
      bestCrossings = crossings;
      bestPositions = snapshotPositions(cy);
    }
  }

  if (bestPositions) {
    cy.layout({
      name: 'preset',
      positions: bestPositions,
    }).run();
  }

  cy.fit(undefined, VIEW_PADDING);
}

/**
 * === React component =========================================================
 */
interface Props {
  graph: GraphData | null;
  selectedIds: string[];
  onSelectionChange: (ids: string[]) => void;
  showEdgeEvidence: boolean;
  nodeGapPx: number;
}

/**
 * Renders Cytoscape graph, runs best-of-N COSE, highlights selection neighborhood.
 */
export default function NetworkView({
  graph,
  selectedIds,
  onSelectionChange,
  showEdgeEvidence,
  nodeGapPx,
}: Props) {
  const cyRef = useRef<CyCore | null>(null);

  // Build elements once per graph change
  const elements = useMemo(
    () => (graph ? [...graph.nodes, ...graph.edges] : []),
    [graph],
  );

  // Click handler -> single-select node
  useEffect(() => {
    const cy = cyRef.current;
    if (!cy) return;

    const onTap = (evt: cytoscape.EventObject) => {
      const id = (evt.target as cytoscape.NodeSingular)?.data('id');
      if (id) onSelectionChange([id]);
    };

    cy.on('tap', 'node', onTap);
    return () => {
      cy.removeListener('tap', 'node', onTap);
    };
  }, [onSelectionChange]);

  // Highlight selection neighborhood
  useEffect(() => {
    const cy = cyRef.current;
    if (!cy) return;

    // Reset all classes
    cy.elements().removeClass('faded highlighted');
    if (selectedIds.length === 0) return;

    const selected = cy.nodes().filter((n) => selectedIds.includes(n.id()));

    // If exactly 2 nodes are selected, only highlight those 2 + the edge(s) between them
    if (selectedIds.length === 2) {
      const [id1, id2] = selectedIds;

      const node1 = cy.getElementById(id1);
      const node2 = cy.getElementById(id2);

      // All edges directly between the two nodes (handles multi-edges too)
      const connectingEdges = node1.edgesWith(node2);

      const toHighlight = node1.union(node2).union(connectingEdges);

      cy.elements().difference(toHighlight).addClass('faded');
      toHighlight.addClass('highlighted');
    } else {
      // Default behavior: highlight full neighborhood of the selection
      const neighborhood = selected.closedNeighborhood();
      cy.elements().difference(neighborhood).addClass('faded');
      neighborhood.addClass('highlighted');
    }
  }, [selectedIds]);

  // Styles
  const stylesheet = useMemo(
    () =>
      [
        {
          selector: 'node',
          style: {
            label: 'data(label)',
            'font-size': 10,
            'text-valign': 'center',
            'text-halign': 'center',
            'text-wrap': 'wrap',
            'text-max-width': 80,
            width: 28,
            height: 28,
            // keep these as any to avoid style function generic constraints
            'background-color': (ele: any) =>
              ele.data('type') === 'protein' ? '#1976d2' : '#ef6c00',
            color: '#fff',
            'border-width': 2,
            'border-color': '#292929ff',
          },
        },
        {
          selector: 'edge',
          style: {
            'curve-style': 'bezier',
            width: 2,
            // Optional label with number of supporting evidence items (e.g. pmids)
            label: (ele: any) => {
              if (!showEdgeEvidence) return '';
              const evidence = ele.data('evidence') as string[] | undefined;
              if (!evidence || evidence.length === 0) return '';
              return `${evidence.length}`;
            },
            'font-size': 8,
            'text-background-opacity': showEdgeEvidence ? 0.8 : 0,
            'text-background-color': '#ffffff',
            'text-background-shape': 'roundrectangle',
            'text-padding': 2,
            'text-rotation': 'autorotate',
            // Color based solely on the relation type string
            'line-color': (ele: any) => getEdgeColor(ele.data('type')),
            'target-arrow-color': (ele: any) => getEdgeColor(ele.data('type')),
            // Treat relations as directed; all edges get an arrow now
            'target-arrow-shape': 'triangle',
          },
        },
        { selector: '.faded', style: { opacity: 0.2 } },
        {
          selector: '.highlighted',
          style: { 'border-color': '#000', 'border-width': 3 },
        },
      ] as any,
    [showEdgeEvidence],
  );

  // Run best-of-N layout when element count changes or spacing knob changes
  useEffect(() => {
    const cy = cyRef.current;
    if (!cy) return;
    if (cy.elements().length === 0) return;

    requestAnimationFrame(() => {
      void runBestCoseLayout(cy, nodeGapPx);
    });
  }, [elements.length, nodeGapPx]);

  return (
    <CytoscapeComponent
      className="cytoscape-container"
      elements={elements as any}
      stylesheet={stylesheet}
      style={{ width: '100%', height: '100%', backgroundColor: '#4b4b4bff' }}
      layout={{ name: 'preset' }} // positions are applied after best layout is chosen
      cy={(cy: CyCore) => {
        cyRef.current = cy;
      }}
    />
  );
}
