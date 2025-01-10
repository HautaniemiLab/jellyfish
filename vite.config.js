import { defineConfig } from "vite";

const rollupOptions = {
  external: ["canvg", "html2canvas", "dompurify"],
  output: {
    globals: {
      canvg: "canvg",
      html2canvas: "html2canvas",
      dompurify: "dompurify",
    },
  },
};

export default defineConfig(({ command, mode }) => {
  if (command === "build" && mode === "lib") {
    return {
      build: {
        lib: {
          entry: "./src/gui/lib.ts",
          name: "jellyfish",
          fileName: (format) => `jellyfish.${format}.js`,
          formats: ["es", "umd"],
        },
        outDir: "dist/lib",
        cssCodeSplit: true,
        rollupOptions,
      },
    };
  }

  return {
    base: "",
    build: {
      sourcemap: true,
      outDir: "dist/app",
      rollupOptions,
    },
  };
});
