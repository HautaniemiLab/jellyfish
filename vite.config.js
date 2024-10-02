import { defineConfig } from "vite";

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
      },
    };
  }

  return {
    base: "",
    build: {
      sourcemap: true,
      outDir: "dist/app",
    },
  };
});
