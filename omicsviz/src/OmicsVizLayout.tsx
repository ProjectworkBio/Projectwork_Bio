/* Presentational layout for OmicsViz
 *
 * Responsibilities:
 *  - Render the top bar, network panel, and right-hand panel
 *  - Use props & callbacks from App for all state changes
 */

import React, { type ChangeEvent } from 'react';
import {
  AppBar,
  Toolbar,
  Typography,
  Box,
  Container,
  Grid,
  Paper,
  TextField,
  IconButton,
  Chip,
  Stack,
  Button,
  Divider,
  GlobalStyles,
} from '@mui/material';
import UploadFileIcon from '@mui/icons-material/UploadFile';
import RestartAltIcon from '@mui/icons-material/RestartAlt';

import { GraphData, Paper as PaperT } from './types';
import NetworkView from './components/NetworkView';
import PapersTable from './components/PapersTable';
import RelationLegend from './components/RelationLegend';
import RelationFrequencyList from './components/RelationFrequencyList';

interface OmicsVizLayoutProps {
  graph: GraphData | null;
  papers: PaperT[];
  relatedToSelection: PaperT[];
  selectionToDisplay: string[];

  query: string;
  onQueryChange: (value: string) => void;

  filterProteins: boolean;
  onToggleProteins: () => void;

  filterDiseases: boolean;
  onToggleDiseases: () => void;

  showEdgeEvidence: boolean;
  onToggleShowEdgeEvidence: () => void;

  nodeGapPx: number;
  onNodeGapPxChange: (value: number) => void;

  paperLimit: number;
  onPaperLimitChange: (value: number) => void;

  onReset: () => void;
  onUploadPapers: (evt: ChangeEvent<HTMLInputElement>) => void;

  onNetworkSelectionChange: (ids: string[]) => void;
  onRelationClick: (relation: { source: string; target: string; type?: string }) => void;
  onSelectPaper: (paper: PaperT) => void;
}

const OmicsVizLayout: React.FC<OmicsVizLayoutProps> = ({
  graph,
  papers,
  relatedToSelection,
  selectionToDisplay,
  query,
  onQueryChange,
  filterProteins,
  onToggleProteins,
  filterDiseases,
  onToggleDiseases,
  showEdgeEvidence,
  onToggleShowEdgeEvidence,
  nodeGapPx,
  onNodeGapPxChange,
  paperLimit,
  onPaperLimitChange,
  onReset,
  onUploadPapers,
  onNetworkSelectionChange,
  onRelationClick,
  onSelectPaper,
}) => {
  return (
    <>
      {/* Global background gradient for the whole page */}
      <GlobalStyles
        styles={{
          body: {
            backgroundImage: 'linear-gradient(to  top left, #41bedaff, #0C4A6E)',
            backgroundRepeat: 'no-repeat',
            backgroundAttachment: 'fixed',
            backgroundSize: 'cover',
          },
        }}
      />

      {/* Main app container: fills almost the full viewport height */}
      <Box sx={{ height: '97.5vh', display: 'flex', flexDirection: 'column' }}>
        {/* Top application bar: title + filters + search + upload */}
        <AppBar
          position="static"
          sx={{
            backgroundImage: 'linear-gradient(to right, #1CA7C5, #0C4A6E)',
          }}
        >
          <Toolbar>
            {/* App title */}
            <Typography variant="h6" sx={{ flexGrow: 1 }}>
              OmicsViz
            </Typography>

            {/* Toggle for showing/hiding edge evidence (relation counts) */}
            <Chip
              label={showEdgeEvidence ? 'Hide relation counts' : 'Show relation counts'}
              color={filterProteins ? 'primary' : 'default'}
              onClick={onToggleShowEdgeEvidence}
              aria-label="Toggle relation evidence labels"
              sx={{
                ...(filterProteins && {
                  bgcolor: '#22648bff',
                  color: '#fff',
                  '&:hover': { bgcolor: '#0C4A6E' },
                }),
              }}
            />

            {/* Numeric control for node spacing in the network layout */}
            <TextField
              size="small"
              type="number"
              label="Node gap (px)"
              value={nodeGapPx}
              onChange={(e) => {
                const raw = e.target.value;
                const v = Number(raw);
                if (Number.isNaN(v)) {
                  onNodeGapPxChange(0);
                  return;
                }
                // Prevent ridiculous values; keep it between 1 and 1000
                const clamped = Math.max(1, Math.min(1000, v));
                onNodeGapPxChange(clamped);
              }}
              sx={{
                maxWidth: 130,
                '& .MuiOutlinedInput-root.Mui-focused fieldset': {
                  borderColor: '#0C4A6E',
                },
                '& .MuiInputLabel-root.Mui-focused': {
                  color: '#000000ff',
                },
              }}
            />

            {/* Node type filters */}
            <Chip
              label="Proteins"
              color={filterProteins ? 'primary' : 'default'}
              onClick={onToggleProteins}
              aria-label="Toggle proteins"
              sx={{
                ...(filterProteins && {
                  bgcolor: '#22648bff',
                  color: '#fff',
                  '&:hover': { bgcolor: '#0C4A6E' },
                }),
              }}
            />
            <Chip
              label="Diseases"
              color={filterDiseases ? 'primary' : 'default'}
              onClick={onToggleDiseases}
              aria-label="Toggle diseases"
              sx={{
                // Fix: style based on filterDiseases (was filterProteins in original)
                ...(filterDiseases && {
                  bgcolor: '#22648bff',
                  color: '#fff',
                  '&:hover': { bgcolor: '#0C4A6E' },
                }),
              }}
            />

            {/* Search, paper limit, reset, upload */}
            <Stack direction="row" spacing={1} alignItems="center">
              {/* Free-text search over proteins/diseases */}
              <TextField
                size="small"
                placeholder="Search proteins/diseases…"
                value={query}
                onChange={(e) => onQueryChange(e.target.value)}
                sx={{ bgcolor: 'white', borderRadius: 1, minWidth: 270 }}
                inputProps={{ 'aria-label': 'Search proteins or diseases' }}
              />

              {/* How many papers to load (affects both default and uploaded data) */}
              <TextField
                size="small"
                type="number"
                label="Papers"
                value={paperLimit}
                onChange={(e) => {
                  const v = Number(e.target.value);
                  if (Number.isNaN(v)) return;
                  onPaperLimitChange(Math.max(1, v));
                }}
                sx={{
                  maxWidth: 130,
                  '& .MuiOutlinedInput-root.Mui-focused fieldset': {
                    borderColor: '#0C4A6E',
                  },
                  '& .MuiInputLabel-root.Mui-focused': {
                    color: '#000000ff',
                  },
                }}
              />

              {/* Reset all filters, selections, and layout settings */}
              <IconButton color="inherit" onClick={onReset} title="Reset filters">
                <RestartAltIcon />
              </IconButton>

              {/* Button to load a custom papers.json file */}
              <Button
                component="label"
                startIcon={<UploadFileIcon />}
                variant="outlined"
                color="inherit"
                size="small"
                sx={{ ml: 1 }}
              >
                Load papers
                <input
                  type="file"
                  accept="application/json"
                  hidden
                  onChange={onUploadPapers}
                />
              </Button>
            </Stack>
          </Toolbar>
        </AppBar>

        {/* Main content: network on the left, data panel on the right */}
        <Container maxWidth="xl" sx={{ flexGrow: 1, py: 2, minHeight: 0 }}>
          <Grid container spacing={2} sx={{ height: '100%' }}>
            {/* Left side: network view */}
            <Grid item xs={12} md={7} sx={{ height: { xs: 480, md: '100%' } }}>
              <Paper
                sx={{
                  height: '100%',
                  p: 1,
                  bgcolor: '#363636ff',
                  display: 'flex',
                  flexDirection: 'column',
                  gap: 1,
                }}
              >
                <Box sx={{ flex: 1, minHeight: 0 }}>
                  <NetworkView
                    graph={graph}
                    selectedIds={selectionToDisplay}
                    onSelectionChange={onNetworkSelectionChange}
                    showEdgeEvidence={showEdgeEvidence}
                    nodeGapPx={nodeGapPx}
                  />
                </Box>
              </Paper>
            </Grid>

            {/* Right side: legend + relation list + papers table */}
            <Grid
              item
              xs={12}
              md={5}
              sx={{
                display: 'flex',
                flexDirection: 'column',
                minHeight: 0,
              }}
            >
              <Paper
                sx={{
                  flex: 1,
                  minHeight: 0,
                  p: 1,
                  overflow: 'hidden',
                  display: 'flex',
                  flexDirection: 'column',
                  backgroundImage: 'linear-gradient(to bottom, #b8b9bbff, #6a7992ff)',
                }}
              >
                <Box
                  sx={{
                    flex: 1,
                    minHeight: 0,
                    display: 'flex',
                    flexDirection: 'column',
                    gap: 1,
                  }}
                >
                  {/* Right panel content: left column = legend + relations, right column = papers */}
                  <Box
                    sx={{
                      flex: 1,
                      minHeight: 0,
                      display: 'flex',
                      gap: 1,
                    }}
                  >
                    {/* LEFT column: legend (top) + top relations (bottom) */}
                    <Box
                      sx={{
                        flex: 1,
                        minWidth: 0,
                        minHeight: 0,
                        display: 'flex',
                        flexDirection: 'column',
                        gap: 1,
                      }}
                    >
                      {/* Relation legend (types, colors, etc.) */}
                      <Box sx={{ flexShrink: 0, maxHeight: 220, overflow: 'auto' }}>
                        <RelationLegend graph={graph} papers={papers} />
                      </Box>

                      <Divider sx={{ my: 0.5 }} />

                      {/* Scrollable list of most frequently mentioned relations */}
                      <Box sx={{ flex: 1, minHeight: 0, overflow: 'auto' }}>
                        <RelationFrequencyList
                          graph={graph}
                          onRelationClick={onRelationClick}
                        />
                      </Box>
                    </Box>

                    {/* RIGHT column: papers table */}
                    <Box
                      sx={{
                        flex: 1,
                        minWidth: 0,
                        minHeight: 0,
                        overflow: 'hidden', // let the DataGrid inside PapersTable scroll
                      }}
                    >
                      <PapersTable papers={relatedToSelection} onSelectPaper={onSelectPaper} />
                    </Box>
                  </Box>
                </Box>
              </Paper>
            </Grid>
          </Grid>
        </Container>
      </Box>
    </>
  );
};
export default OmicsVizLayout;