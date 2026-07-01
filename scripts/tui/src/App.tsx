/**
 * scripts/tui/src/App.tsx
 *
 * Main termcn TUI application.
 * Manages configuration menu, spawns python, and renders training progress.
 */

import React, { useMemo, useState, useEffect, useRef } from "react";
import { Box, Text, Newline } from "ink";
import { spawn } from "node:child_process";
import { createInterface } from "node:readline";

import { Spinner } from "@/components/ui/spinner.js";
import { ProgressBar } from "@/components/ui/progress-bar.js";
import { Table } from "@/components/ui/table.js";
import { Badge } from "@/components/ui/badge.js";
import { StatusMessage } from "@/components/ui/status-message.js";
import { Alert } from "@/components/ui/alert.js";
import { Sparkline } from "@/components/ui/sparkline.js";
import { NumberInput } from "@/components/ui/number-input.js";
import { MultiSelect } from "@/components/ui/multi-select.js";
import { TextInput } from "@/components/ui/text-input.js";

// ── Message types from Python backend ─────────────────────────────────────────

export type TrainingMessage =
  | { type: "phase";   phase: string }
  | { type: "gpu";     device: string; name?: string; cuda_version?: string;
      capability?: string; vram_total_mb?: number; vram_alloc_mb?: number;
      amp?: boolean; torch_version: string }
  | { type: "config";  participants: number[]; batch_size: number;
      max_epochs: number; lr: number; n_cca: number;
      n_layers: number; n_heads: number; d_model: number; resume: boolean }
  | { type: "dataset"; split: string; windows: number; elapsed_s: number }
  | { type: "cca";     n_components: number; n_series: number;
      emg_mean: number[]; emg_std: number[] }
  | { type: "model";   arch: string; n_params: number }
  | { type: "batch";   epoch: number; batch: number;
      total_batches: number; loss: number }
  | { type: "epoch";   epoch: number; max_epochs: number;
      train_loss: number; val_loss: number; best_val: number;
      epochs_no_improve: number }
  | { type: "eval";    channels: string[]; rmse: number[];
      mae: number[]; r: number[];
      mean_rmse: number; mean_mae: number; mean_r: number }
  | { type: "done";    best_val: number; ckpt_dir: string; inference_ckpt: string }
  | { type: "error";   message: string };

// ── Helpers ───────────────────────────────────────────────────────────────────

const PHASES = ["setup", "datasets", "cca", "model", "training", "eval", "done"] as const;
type Phase = typeof PHASES[number];

function phaseLabel(phase: Phase): string {
  return {
    setup:    "§0  Initialising",
    datasets: "§1  Loading Datasets",
    cca:      "§2  CCA + EMG Stats",
    model:    "§3  Model",
    training: "§4  Training",
    eval:     "§5  Evaluation",
    done:     "Done",
  }[phase];
}

function rColor(r: number): string {
  if (r >= 0.7) return "green";
  if (r >= 0.4) return "yellow";
  return "red";
}

// ── Section header ─────────────────────────────────────────────────────────────
function SectionHeader({ label }: { label: string }) {
  return (
    <Box marginY={1}>
      <Text color="cyan" bold>{"─".repeat(4)} </Text>
      <Text bold>{label}</Text>
      <Text color="cyan"> {"─".repeat(40)}</Text>
    </Box>
  );
}

// ── Main App ──────────────────────────────────────────────────────────────────

export interface AppProps {
  pythonBin: string;
  pyScript: string;
  repoRoot: string;
  onExit: (code: number) => void;
}

export function App({ pythonBin, pyScript, repoRoot, onExit }: AppProps) {
  const [appState, setAppState] = useState<"menu" | "training">("menu");
  const [menuStep, setMenuStep] = useState<number>(0);

  // Form values
  const [participants, setParticipants] = useState<string[]>([]);
  const [epochs, setEpochs] = useState<number>(50);
  const [batchSize, setBatchSize] = useState<number>(32);
  const [lr, setLr] = useState<string>("0.001");

  // Training state
  const [messages, setMessages] = useState<TrainingMessage[]>([]);
  const queuedMessagesRef = useRef<TrainingMessage[]>([]);

  // Spawn python and flush messages when training starts
  useEffect(() => {
    if (appState !== "training") return;

    const args = [
      "--participants", ...(participants.length > 0 ? participants : ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"]),
      "--max-epochs", String(epochs),
      "--batch-size", String(batchSize),
      "--lr", lr,
    ];

    const child = spawn(pythonBin, [pyScript, ...args], {
      stdio: ["ignore", "pipe", "pipe"],
      cwd: repoRoot,
      env: {
        ...process.env,
        PYTHONIOENCODING: "utf-8",
        PYTHONUTF8: "1",
        PYTORCH_CUDA_ALLOC_CONF: process.env.PYTORCH_CUDA_ALLOC_CONF ?? "expandable_segments:True",
      },
    });

    const rl = createInterface({ input: child.stdout! });
    rl.on("line", (line) => {
      const trimmed = line.trim();
      if (!trimmed) return;
      try {
        const msg = JSON.parse(trimmed) as TrainingMessage;
        queuedMessagesRef.current.push(msg);
      } catch {}
    });

    child.stderr?.on("data", (d: Buffer) => {
      process.stderr.write(d);
    });

    child.on("close", (code) => {
      setTimeout(() => onExit(code ?? 0), 200);
    });

    // Throttle React state updates to 20 fps to prevent flickering
    const interval = setInterval(() => {
      if (queuedMessagesRef.current.length > 0) {
        setMessages((prev) => [...prev, ...queuedMessagesRef.current]);
        queuedMessagesRef.current = [];
      }
    }, 50);

    const handleSigint = () => child.kill("SIGINT");
    process.on("SIGINT", handleSigint);

    return () => {
      child.kill();
      clearInterval(interval);
      process.off("SIGINT", handleSigint);
    };
  }, [appState]);

  // Derive display state from messages
  const state = useMemo(() => {
    let currentPhase: Phase = "setup";
    let gpu: Extract<TrainingMessage, { type: "gpu" }> | null = null;
    let config: Extract<TrainingMessage, { type: "config" }> | null = null;
    const datasets: Record<string, { windows: number; elapsed_s: number }> = {};
    let cca: Extract<TrainingMessage, { type: "cca" }> | null = null;
    let model: Extract<TrainingMessage, { type: "model" }> | null = null;
    let lastBatch: Extract<TrainingMessage, { type: "batch" }> | null = null;
    const epochs: Extract<TrainingMessage, { type: "epoch" }>[] = [];
    let evalResult: Extract<TrainingMessage, { type: "eval" }> | null = null;
    let done: Extract<TrainingMessage, { type: "done" }> | null = null;
    let error: string | null = null;

    for (const msg of messages) {
      switch (msg.type) {
        case "phase":   currentPhase = msg.phase as Phase; break;
        case "gpu":     gpu = msg; break;
        case "config":  config = msg; break;
        case "dataset": datasets[msg.split] = msg; break;
        case "cca":     cca = msg; break;
        case "model":   model = msg; break;
        case "batch":   lastBatch = msg; break;
        case "epoch":   epochs.push(msg); break;
        case "eval":    evalResult = msg; break;
        case "done":    done = msg; currentPhase = "done"; break;
        case "error":   error = msg.message; break;
      }
    }
    return { currentPhase, gpu, config, datasets, cca, model,
             lastBatch, epochs, evalResult, done, error };
  }, [messages]);

  const { currentPhase, gpu, config, datasets, cca, model,
          lastBatch, epochsResult = state.epochs, evalResult, done, error } = state;

  const lastEpoch = epochsResult.at(-1);
  const trainLosses = epochsResult.map(e => e.train_loss);
  const valLosses   = epochsResult.map(e => e.val_loss);

  // ── Render Menu ─────────────────────────────────────────────────────────────
  if (appState === "menu") {
    return (
      <Box flexDirection="column" padding={1}>
        <Box marginBottom={1}>
          <Text color="cyanBright" bold>
            {"╔══════════════════════════════════════════════════╗\n"}
            {"║  EEG → EMG  ·  Transformer Training  ·  KG-GT  ║\n"}
            {"╚══════════════════════════════════════════════════╝"}
          </Text>
        </Box>
        <SectionHeader label="Configuration Menu" />

        <Box flexDirection="column" marginLeft={2} gap={1}>
          <Box flexDirection="column">
            <Text bold color={menuStep === 0 ? "white" : "gray"}>1. Select Participants (Space to toggle, Enter to confirm)</Text>
            {menuStep === 0 ? (
              <MultiSelect
                options={[1,2,3,4,5,6,7,8,9,10,11,12].map(p => ({ label: `Participant ${p}`, value: String(p) }))}
                value={participants}
                onChange={setParticipants}
                onSubmit={() => setMenuStep(1)}
                height={5}
              />
            ) : (
              <Text dimColor>Participants: {participants.length ? participants.join(", ") : "All (1-12)"}</Text>
            )}
          </Box>

          {menuStep >= 1 && (
            <Box flexDirection="column">
              <Text bold color={menuStep === 1 ? "white" : "gray"}>2. Max Epochs (Enter to confirm)</Text>
              {menuStep === 1 ? (
                <NumberInput
                  value={epochs}
                  onChange={setEpochs}
                  onSubmit={() => setMenuStep(2)}
                  min={1}
                />
              ) : (
                <Text dimColor>Max Epochs: {epochs}</Text>
              )}
            </Box>
          )}

          {menuStep >= 2 && (
            <Box flexDirection="column">
              <Text bold color={menuStep === 2 ? "white" : "gray"}>3. Batch Size (Enter to confirm)</Text>
              {menuStep === 2 ? (
                <NumberInput
                  value={batchSize}
                  onChange={setBatchSize}
                  onSubmit={() => setMenuStep(3)}
                  min={1}
                />
              ) : (
                <Text dimColor>Batch Size: {batchSize}</Text>
              )}
            </Box>
          )}

          {menuStep >= 3 && (
            <Box flexDirection="column">
              <Text bold color={menuStep === 3 ? "white" : "gray"}>4. Learning Rate (Enter to start training)</Text>
              {menuStep === 3 ? (
                <TextInput
                  value={lr}
                  onChange={setLr}
                  onSubmit={() => setAppState("training")}
                  validate={(v) => isNaN(parseFloat(v)) ? "Must be a valid number" : null}
                />
              ) : (
                <Text dimColor>Learning Rate: {lr}</Text>
              )}
            </Box>
          )}
        </Box>
      </Box>
    );
  }

  // ── Render Training Progress ────────────────────────────────────────────────
  return (
    <Box flexDirection="column" padding={1}>
      <Box marginBottom={1}>
        <Text color="cyanBright" bold>
          {"╔══════════════════════════════════════════════════╗\n"}
          {"║  EEG → EMG  ·  Transformer Training  ·  KG-GT  ║\n"}
          {"╚══════════════════════════════════════════════════╝"}
        </Text>
      </Box>

      {/* GPU Panel */}
      {gpu && (
        <>
          <SectionHeader label="GPU / Device Status" />
          <Box flexDirection="column" marginLeft={2} marginBottom={1}>
            <StatusMessage variant="info" label="device">{gpu.device}</StatusMessage>
            {gpu.name && <StatusMessage variant="success" label="GPU">{gpu.name}</StatusMessage>}
            {gpu.vram_total_mb !== undefined && (
              <StatusMessage variant="info" label="VRAM">
                {gpu.vram_alloc_mb?.toFixed(1)} / {gpu.vram_total_mb} MB
              </StatusMessage>
            )}
            <StatusMessage variant={gpu.amp ? "success" : "warning"} label="AMP">
              {gpu.amp ? "YES — Tensor cores active" : "NO — FP32 fallback"}
            </StatusMessage>
          </Box>
        </>
      )}

      {/* Config Panel */}
      {config && (
        <>
          <SectionHeader label="Config" />
          <Box flexDirection="column" marginLeft={2} marginBottom={1}>
            <Text><Text color="cyan">participants </Text><Text>{JSON.stringify(config.participants)}</Text></Text>
            <Text><Text color="cyan">transformer  </Text><Text>L={config.n_layers}  H={config.n_heads}  d={config.d_model}</Text></Text>
            <Text><Text color="cyan">batch/epochs </Text><Text>{config.batch_size} / {config.max_epochs}</Text></Text>
            <Text><Text color="cyan">lr           </Text><Text>{config.lr}</Text></Text>
          </Box>
        </>
      )}

      {/* Dataset Loading */}
      {(currentPhase === "datasets" || Object.keys(datasets).length > 0) && (
        <>
          <SectionHeader label="§1  Loading Datasets" />
          <Box flexDirection="column" marginLeft={2} marginBottom={1}>
            {(["train", "val", "test"] as const).map((split) => {
              const ds = datasets[split];
              return (
                <Box key={split}>
                  {ds ? (
                    <StatusMessage variant="success" label={split}>
                      {ds.windows.toLocaleString()} windows ({ds.elapsed_s}s)
                    </StatusMessage>
                  ) : (
                    <Box><Spinner type="dots" label={`Loading ${split}…`} /></Box>
                  )}
                </Box>
              );
            })}
          </Box>
        </>
      )}

      {/* CCA */}
      {cca && (
        <>
          <SectionHeader label="§2  CCA + EMG Statistics" />
          <Box flexDirection="column" marginLeft={2} marginBottom={1}>
            <StatusMessage variant="success" label="CCA fitted">
              n_components={cca.n_components}  series={cca.n_series}
            </StatusMessage>
          </Box>
        </>
      )}

      {/* Model */}
      {model && (
        <>
          <SectionHeader label="§3  Model" />
          <Box flexDirection="column" marginLeft={2} marginBottom={1}>
            <Badge variant="success">{model.n_params.toLocaleString()} trainable params</Badge>
          </Box>
        </>
      )}

      {/* Training */}
      {(currentPhase === "training" || epochsResult.length > 0) && (
        <>
          <SectionHeader label="§4  Training" />
          <Box flexDirection="column" marginLeft={2} marginBottom={1}>

            {/* Epoch progress */}
            {lastEpoch && config && (
              <Box marginBottom={1}>
                <ProgressBar
                  value={lastEpoch.epoch}
                  total={config.max_epochs}
                  width={40}
                  showPercent
                  label={`Epoch ${lastEpoch.epoch}/${config.max_epochs}`}
                  color="cyan"
                />
              </Box>
            )}

            {/* Batch progress */}
            {lastBatch && (
              <Box marginBottom={1}>
                <ProgressBar
                  value={lastBatch.batch}
                  total={lastBatch.total_batches}
                  width={40}
                  showPercent
                  label={`Batch  loss=${lastBatch.loss.toFixed(5)}`}
                  color="magenta"
                />
              </Box>
            )}

            {/* Waiting spinner */}
            {currentPhase === "training" && epochsResult.length === 0 && (
              <Spinner type="dots2" label="Waiting for first epoch…" color="yellow" />
            )}

            {/* Loss table */}
            {epochsResult.length > 0 && (
              <Box marginTop={1}>
                <Table
                  data={epochsResult.slice(-10).map(e => ({
                    epoch:      String(e.epoch),
                    train:      e.train_loss.toFixed(5),
                    val:        e.val_loss.toFixed(5),
                    best:       e.best_val.toFixed(5),
                    no_improve: String(e.epochs_no_improve),
                  }))}
                  columns={[
                    { key: "epoch",      header: "Epoch",     align: "right", width: 6 },
                    { key: "train",      header: "Train ↓",   align: "right", width: 10 },
                    { key: "val",        header: "Val ↓",     align: "right", width: 10 },
                    { key: "best",       header: "Best Val",  align: "right", width: 10 },
                    { key: "no_improve", header: "No-Improv", align: "right", width: 10 },
                  ]}
                  borderColor="cyan"
                />
              </Box>
            )}

            {/* Sparklines */}
            {trainLosses.length >= 2 && (
              <Box flexDirection="column" marginTop={1}>
                <Box>
                  <Text color="yellow">train  </Text>
                  <Sparkline values={trainLosses} color="yellow" />
                </Box>
                <Box>
                  <Text color="cyan">val    </Text>
                  <Sparkline values={valLosses} color="cyan" />
                </Box>
              </Box>
            )}
          </Box>
        </>
      )}

      {/* Evaluation */}
      {evalResult && (
        <>
          <SectionHeader label="§5  Evaluation (test split)" />
          <Box marginLeft={2} marginBottom={1}>
            <Table
              data={evalResult.channels.map((ch, i) => ({
                channel: ch,
                rmse:    evalResult!.rmse[i].toFixed(4),
                mae:     evalResult!.mae[i].toFixed(4),
                r:       evalResult!.r[i].toFixed(4),
              }))}
              columns={[
                { key: "channel", header: "Channel",   width: 22 },
                { key: "rmse",    header: "RMSE",      align: "right", width: 8 },
                { key: "mae",     header: "MAE",       align: "right", width: 8 },
                { key: "r",       header: "Pearson r", align: "right", width: 10 },
              ]}
              borderColor="yellow"
            />
          </Box>
        </>
      )}

      {/* Done */}
      {done && (
        <Alert variant="success">
          <Text bold>Training complete!</Text>
          <Newline />
          <Text>Best val loss: </Text>
          <Text bold color="green">{done.best_val.toFixed(6)}</Text>
          <Newline />
          <Text dim>Checkpoints : {done.ckpt_dir}</Text>
          <Newline />
          <Text dim>Inference   : {done.inference_ckpt}</Text>
        </Alert>
      )}

      {/* Error */}
      {error && (
        <Alert variant="error">
          <Text bold>Error: </Text>
          <Text>{error}</Text>
        </Alert>
      )}

      {/* Status Bar */}
      {!done && !error && (
        <Box marginTop={1}>
          <Text dim>Phase: </Text>
          <Badge variant="info">{phaseLabel(currentPhase)}</Badge>
        </Box>
      )}
    </Box>
  );
}
