import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// Dev proxy lets the React app call `/api/*` without CORS hassle.
// Set `VITE_API_BASE_URL` if you prefer direct calls instead of proxy.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: process.env.VITE_PROXY_TARGET || 'http://localhost:5000',
        changeOrigin: true,
      },
    },
  },
});
