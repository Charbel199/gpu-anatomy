import type { Plugin } from 'vite';
import fs from 'node:fs';
import path from 'node:path';
import { createHighlighter, type Highlighter } from 'shiki';

// Imports every .cu file under the parent repo's numbered folders and
// exposes them as `virtual:cu-files`. Shiki runs here so nothing ships at runtime.
export function cuLoader(opts: { repoRoot: string }): Plugin {
  const VIRTUAL_ID = 'virtual:cu-files';
  const RESOLVED_ID = '\0' + VIRTUAL_ID;
  let hl: Highlighter | null = null;

  async function getHighlighter() {
    if (!hl) {
      hl = await createHighlighter({
        themes: ['github-dark-default'],
        langs: ['cpp'],
      });
    }
    return hl;
  }

  return {
    name: 'cu-loader',
    resolveId(id) {
      return id === VIRTUAL_ID ? RESOLVED_ID : null;
    },
    async load(id) {
      if (id !== RESOLVED_ID) return null;
      const root = opts.repoRoot;
      const folders = fs
        .readdirSync(root, { withFileTypes: true })
        .filter((d) => d.isDirectory() && /^\d+[-_]/.test(d.name));

      const highlighter = await getHighlighter();
      const files: Array<{ path: string; folder: string; name: string; source: string; highlighted: string }> = [];

      for (const d of folders) {
        const dir = path.join(root, d.name);
        for (const name of fs.readdirSync(dir)) {
          if (!name.endsWith('.cu')) continue;
          const source = fs.readFileSync(path.join(dir, name), 'utf8');
          const highlighted = highlighter.codeToHtml(source, {
            lang: 'cpp',
            theme: 'github-dark-default',
          });
          files.push({ path: `${d.name}/${name}`, folder: d.name, name, source, highlighted });
        }
      }

      return `export default ${JSON.stringify(files)};`;
    },
    handleHotUpdate({ file, server }) {
      if (!file.endsWith('.cu')) return;
      const mod = server.moduleGraph.getModuleById(RESOLVED_ID);
      if (mod) server.moduleGraph.invalidateModule(mod);
      server.ws.send({ type: 'full-reload' });
    },
  };
}
