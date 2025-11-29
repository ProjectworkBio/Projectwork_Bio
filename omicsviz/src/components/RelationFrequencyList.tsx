import { useMemo } from 'react';
import {
  Box,
  Typography,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableRow,
  TableContainer,
} from '@mui/material';
import { GraphData } from '../types';

interface FrequencyRow {
  id: string;
  source: string;
  target: string;
  type?: string;
  count: number;
}

interface Props {
  graph: GraphData | null;
  // Optional callback: fired when the user clicks on a relation row
  onRelationClick?: (relation: { source: string; target: string; type?: string }) => void;
}

// Lists relations sorted by how many times they’re mentioned
// (number of evidence entries per edge).
export default function RelationFrequencyList({ graph, onRelationClick }: Props) {
  // Compute a sorted list of relation rows from the graph,
  // only when `graph` changes.
  const rows: FrequencyRow[] = useMemo(() => {
    if (!graph) return [];

    return graph.edges
      .map((e) => ({
        id: e.data.id,
        source: e.data.source,
        target: e.data.target,
        type: e.data.type,
        // count = how many evidence items this edge has
        count: e.data.evidence?.length ?? 0,
      }))
      // Ignore edges that have no evidence
      .filter((r) => r.count > 0)
      // Sort by highest count first, then alphabetically by source/target
      .sort((a, b) => {
        if (b.count !== a.count) return b.count - a.count;
        if (a.source !== b.source) return a.source.localeCompare(b.source);
        return a.target.localeCompare(b.target);
      });
  }, [graph]);

  // If there are no relations with evidence, show a simple placeholder
  if (rows.length === 0) {
    return (
      <Box sx={{ p: 1 }}>
        <Typography variant="body2" color="text.secondary">
          No relation evidence to show yet.
        </Typography>
      </Box>
    );
  }

  // Otherwise, render a scrollable table of relations and their counts
  return (
    <Box sx={{ mt: 1 }}>
      <TableContainer
        sx={{
          maxHeight: 332,
          overflow: 'auto',
        }}
      >
        <Table size="small" stickyHeader>
          <TableHead>
            <TableRow>
              <TableCell sx={{ background: '#989da6ff', fontWeight: 600 }}>
                Most mentioned
              </TableCell>
              <TableCell align="right" sx={{ background: '#989da6ff', fontWeight: 600 }}>
                N
              </TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {rows.map((row) => (
              <TableRow
                key={row.id}
                // Make rows hoverable/clickable only if a click handler was passed in
                hover={!!onRelationClick}
                onClick={
                  onRelationClick
                    ? () =>
                        onRelationClick({
                          source: row.source,
                          target: row.target,
                          type: row.type,
                        })
                    : undefined
                }
                sx={onRelationClick ? { cursor: 'pointer' } : undefined}
              >
                <TableCell>
                  <Typography variant="body2">
                    {row.source} → {row.target}
                  </Typography>
                  {row.type && (
                    <Typography variant="caption" color="text.secondary">
                      {row.type}
                    </Typography>
                  )}
                </TableCell>
                <TableCell align="right">{row.count}</TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>
    </Box>
  );
}
