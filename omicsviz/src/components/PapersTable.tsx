import { useMemo, useCallback, useState, useEffect } from 'react';
import {
  Box,
  Link,
  TableContainer,
  Table,
  TableHead,
  TableRow,
  TableCell,
  TableBody,
  TablePagination,
  Typography,
} from '@mui/material';
import { Paper as PaperT } from '../types';

/*PapersTable
 *
 * Responsibilities
 *  - Render a paginated view of papers
 *  - On row click: send to the respective paper on pubmed
 */

interface Props {
  papers: PaperT[];
  onSelectPaper?: (paper: PaperT) => void;
}

// Local row model for the table (keeps table concerns separate from domain model)
interface RowModel {
  id: string; // pmid
  title: string;
  _hoverIds?: string[];
}

export default function PapersTable({ papers, onSelectPaper }: Props) {
  const [page, setPage] = useState(0);
  const [rowsPerPage, setRowsPerPage] = useState(5);

  // Build a lookup map for quick id->paper resolution on click
  const paperById = useMemo(
    () => new Map(papers.map((p) => [p.pmid, p])),
    [papers]
  );

  // Transform domain objects into table rows
  const rows: RowModel[] = useMemo(
    () =>
      papers.map((p) => ({
        id: p.pmid,
        title: p.title,
        _hoverIds: Array.from(
          new Set(p.relations.flatMap((r) => [r.source, r.target]))
        ),
      })),
    [papers]
  );

  // Reset page if papers change a lot (e.g., new search)
  useEffect(() => {
    setPage(0);
  }, [papers]);

  const handleRowClick = useCallback(
    (pmid: string) => {
      if (!onSelectPaper) return;
      const found = paperById.get(pmid);
      if (found) onSelectPaper(found);
    },
    [onSelectPaper, paperById]
  );

  const handleChangePage = (_: unknown, newPage: number) => {
    setPage(newPage);
  };

  const handleChangeRowsPerPage = (
    event: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>
  ) => {
    setRowsPerPage(parseInt(event.target.value, 10));
    setPage(0);
  };

  const pagedRows = useMemo(
    () =>
      rows.slice(page * rowsPerPage, page * rowsPerPage + rowsPerPage),
    [rows, page, rowsPerPage]
  );

  return (
    <Box sx={{ mt: 1 }}>
      <TableContainer
        sx={{
          maxHeight: 460,
          width: '100%',
          overflow: 'auto', // scroll inside this container
        }}
      >
        <Table size="small" stickyHeader>
          <TableHead>
            <TableRow>
              <TableCell
                sx={{
                  background: '#b8b9bbff',
                  fontWeight: 600,
                }}
              >
                Relation sentence
              </TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {pagedRows.map((row) => {
              const pmid = row.id;
              const url = `https://pubmed.ncbi.nlm.nih.gov/${pmid}/`;

              return (
                <TableRow
                  key={row.id}
                  hover={!!onSelectPaper}
                  onClick={() => handleRowClick(row.id)}
                  sx={onSelectPaper ? { cursor: 'pointer' } : undefined}
                >
                  <TableCell>
                    <Typography variant="body2" component="div">
                      <Link
                        href={url}
                        target="_blank"
                        rel="noopener noreferrer"
                        underline="hover"
                        onClick={(e) => e.stopPropagation()}
                        sx={{
                          display: 'block',
                          whiteSpace: 'normal', // allow wrapping
                          wordBreak: 'break-word', // break long tokens if needed
                          lineHeight: 1.3,
                          color: 'inherit',
                          textDecoration: 'none',
                        }}
                      >
                        {row.title}
                      </Link>
                    </Typography>
                  </TableCell>
                </TableRow>
              );
            })}
          </TableBody>
        </Table>
      </TableContainer>
      <TablePagination
        component="div"// this is the pagination at the bottom of the table
        count={rows.length}
        page={page}
        onPageChange={handleChangePage}
        rowsPerPage={rowsPerPage}
        onRowsPerPageChange={handleChangeRowsPerPage}
        rowsPerPageOptions={[5, 10, 25]}
        sx={{
          '& .MuiTablePagination-toolbar': {
            flexWrap: 'wrap',      // allow wrapping into 2 rows
          },
          // Row 1: label + select
          '& .MuiTablePagination-selectLabel, & .MuiTablePagination-select': {
            order: 1, marginTop: 1,
          },
          // Row 2: "x–y of z" (left) + page actions <  > (right)
          '& .MuiTablePagination-displayedRows, & .MuiTablePagination-actions': {
            order: 2, ml: 3.5, marginTop: 0,
          },
          '& .MuiTablePagination-displayedRows': {
            marginRight: 'auto',   // pushes actions to the far right(the <  > thing)
          },
        }}
      />
    </Box>
  );
}