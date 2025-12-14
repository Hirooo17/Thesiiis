# website_react

Modern React (JSX) + Tailwind frontend for your existing Flask backend in `website/app.py`.

## Run (Dev)

1) Start Flask backend (from your repo root):

- `python website/app.py`

It should run at `http://localhost:5000`.

2) Start React dev server (new terminal):

- `cd website_react`
- `npm install`
- `npm run dev`

Open `http://localhost:5173`.

### API base URL

Default dev behavior uses Vite proxy for `/api/*` to `http://localhost:5000`.

If you want direct calls instead, set:

- `VITE_API_BASE_URL=http://localhost:5000`

## Notes

- All features from the original Flask web UI are implemented: model load/upload, record/upload audio, SNR noise simulation, custom noise mixing, analysis results, mixed real/AI warning, all visualizations, and research references.
