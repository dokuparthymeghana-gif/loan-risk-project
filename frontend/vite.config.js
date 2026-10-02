import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Vite needs to know we are using React
export default defineConfig({
  plugins: [react()],
});
