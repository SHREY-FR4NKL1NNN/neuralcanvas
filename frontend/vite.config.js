import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Dev server on 5174 so it doesn't collide with LocalMind's 5173. The backend
// (FastAPI on 8001) whitelists this origin via CORS.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5174,
  },
})
