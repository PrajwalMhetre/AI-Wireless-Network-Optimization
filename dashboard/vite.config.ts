import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/api": process.env.AURORA_API_URL ?? "http://localhost:8000",
      "/optimize": process.env.AURORA_API_URL ?? "http://localhost:8000",
    },
  },
});
