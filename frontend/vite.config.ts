import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import yaml from '@modyfi/vite-plugin-yaml';
import path from 'node:path';
import { cuLoader } from './scripts/cu-loader';

export default defineConfig({
  plugins: [
    react(),
    yaml(),
    cuLoader({ repoRoot: path.resolve(__dirname, '..') }),
  ],
  resolve: {
    alias: { '@': path.resolve(__dirname, 'src') },
  },
  server: { port: 5173, strictPort: false },
});
