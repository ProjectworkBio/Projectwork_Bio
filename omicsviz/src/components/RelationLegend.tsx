import { useMemo } from 'react';
import { Box, Typography, Table, TableBody, TableCell, TableHead, TableRow } from '@mui/material';
import { GraphData, Paper as PaperT } from '../types';
import { getEdgeColor } from './NetworkView';

interface Props {
  graph: GraphData | null;
  papers: PaperT[];
}

interface LegendRow {
  type: string;
  color: string;
}

// Shows a legend of relation types with their corresponding edge colors
export default function RelationLegend({ graph, papers }: Props) {
  // Build the list of relation types and their colors
  const rows: LegendRow[] = useMemo(() => {
    const types = new Set<string>();

    // Prefer relation types from the actual graph edges
    if (graph) {
      graph.edges.forEach((e) => {
        const t = e.data.type;
        if (t) types.add(t);
      });
    } else {
      // Fallback: collect relation types from the papers if graph is missing
      papers.forEach((p) =>
        p.relations.forEach((r) => {
          if (r.type) types.add(r.type);
        })
      );
    }

    // Turn the set into a sorted array and attach a color to each type
    return Array.from(types)
      .sort()
      .map((type) => ({
        type,
        color: getEdgeColor(type),
      }));
  }, [graph, papers]);

  // If there are no relations, show a simple message instead of the table
  if (rows.length === 0) {
    return (
      <Box sx={{ p: 1 }}>
        <Typography variant="body2" color="text.secondary">
          No relations to show yet.
        </Typography>
      </Box>
    );
  }

  // Otherwise, render a scrollable table with a color box and relation type
  return (
    <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      <Box sx={{ flex: 1, overflow: 'auto' }}>
        <Table size="small" stickyHeader>
          <TableHead>
            <TableRow>
              <TableCell sx={{ background: '#b8b9bbff' }}></TableCell>
              <TableCell sx={{ background: '#b8b9bbff', fontWeight: 600 }}>Type</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {rows.map((row) => (
              <TableRow key={row.type}>
                <TableCell sx={{ maxWidth: 0 }}>
                  <Box
                    sx={{
                      width: 16,
                      height: 16,
                      borderRadius: 0.5,
                      bgcolor: row.color,
                      border: '1px solid rgba(0,0,0,0.4)',
                    }}
                  />
                </TableCell>
                <TableCell>{row.type}</TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </Box>
    </Box>
  );
}
