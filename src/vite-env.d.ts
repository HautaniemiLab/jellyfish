/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_DATA_DIR: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
