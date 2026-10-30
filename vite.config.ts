import { defineConfig } from 'vite';
import vue from '@vitejs/plugin-vue';

export default defineConfig({
  // Relative base so the build works from any path (e.g. GitHub Pages).
  base: './',
  plugins: [vue()],
  // transformers.js ships prebuilt ESM with onnxruntime-web; pre-bundling it
  // only slows the dev server down.
  optimizeDeps: { exclude: ['@huggingface/transformers'] },
  build: { target: 'es2022', chunkSizeWarningLimit: 1000 },
});
