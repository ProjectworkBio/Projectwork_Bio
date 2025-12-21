# OmicsViz

A React + Vite + TypeScript app for exploring protein/disease relations extracted from papers 
as an interactive Cytoscape network + linked papers table.

## Run (dev)
npm install
npm run dev

## Build / preview
npm run build
npm run preview

## Docker
docker build -t omicsviz .
docker run --rm -p 8080:80 omicsviz

## Data
- Default dataset: `public/proteindata/ppi_v3.json`
- You can upload your own `papers.json` (array of):
  `{ pmid: string, title: string, relations: [{ source: string, target: string, type: string }] }`
