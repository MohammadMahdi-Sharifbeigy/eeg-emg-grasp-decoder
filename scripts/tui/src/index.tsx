#!/usr/bin/env node
/**
 * scripts/tui/src/index.tsx
 *
 * Entry point for the termcn TUI.
 * Spawns the App component which handles the menu and the Python child process.
 */

import { resolve } from "node:path";
import { render } from "ink";
import { App } from "./App.js";

const repoRoot = resolve(import.meta.dirname, "../../..");
const pythonBin = process.platform === "win32"
  ? resolve(repoRoot, "env/Scripts/python.exe")
  : resolve(repoRoot, "env/bin/python");
const pyScript = resolve(repoRoot, "scripts/train_transformer_json.py");

const { unmount } = render(
  <App 
    pythonBin={pythonBin} 
    pyScript={pyScript} 
    repoRoot={repoRoot} 
    onExit={(code) => { setTimeout(() => unmount(), 200); process.exit(code); }} 
  />
);
