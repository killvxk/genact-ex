# Feature Landscape: LLM Training Simulation

**Domain:** genact fake activity module - LLM training simulation
**Researched:** 2026-01-30
**Confidence:** HIGH (based on real-world training output analysis and project context)

## Table Stakes

Features users expect. Missing = simulation feels incomplete or fake.

### Initialization Phase Features

| Feature | Why Expected | Complexity | Output Example | Timing |
|---------|--------------|------------|----------------|--------|
| Distributed environment setup | Every multi-GPU training shows this | Low | `Initializing distributed training environment...` | 0.5-1s per line |
| NCCL backend initialization | Standard PyTorch distributed output | Low | `Setting up NCCL backend (64 GPUs across 8 nodes)` | 1-2s |
| Master node/port display | Required for distributed training | Low | `Master: node-0, Port: 29500` | 0.3s |
| Model loading message | Always shown when loading weights | Low | `Loading model: Llama-3-70B-Instruct...` | 1-3s |
| Parameter count display | Universal in training outputs | Low | `Model size: 70B parameters` | instant |
| Tokenizer loading | Standard in LLM training | Low | `Loading tokenizer: AutoTokenizer (vocab_size=128256)` | 0.5-1s |
| Dataset info | Training scripts show data stats | Low | `Dataset: RedPajama-v2 (1.2T tokens)` | 0.5s |
| CUDA/GPU detection | Real training always shows this | Low | `Found 64 NVIDIA A100-SXM4-80GB GPUs` | 1-2s |

### Training Phase Features

| Feature | Why Expected | Complexity | Output Example | Timing |
|---------|--------------|------------|----------------|--------|
| Epoch progress bar | Universal in training UIs | Med | `Epoch 1/3 [=================>    ] 75%` | Updates every 0.5-2s |
| Step counter | Always present | Low | `Step 1000/50000` | Per step |
| Training loss | THE core metric | Low | `Loss: 2.3456` | Per step |
| Learning rate display | Standard hyperparameter | Low | `LR: 1.0e-04` | Per step |
| GPU memory usage | Critical resource metric | Med | `GPU Mem: 78.2GB/80GB` | Per step or periodic |
| Tokens/samples per second | Speed metrics are ubiquitous | Med | `Speed: 125,432 tokens/s` | Per step |
| ETA/time remaining | Every progress bar has this | Med | `ETA: 2h 34m` | Updates with progress |
| Loss decreasing trend | Realistic training shows improvement | Med | Initial ~2.5, final ~0.8-1.2 | Over full run |

### Validation Phase Features

| Feature | Why Expected | Complexity | Output Example | Timing |
|---------|--------------|------------|----------------|--------|
| Validation loss | Always computed during eval | Low | `Val Loss: 2.1234` | End of eval |
| Perplexity (PPL) | Standard LLM metric | Low | `Val PPL: 8.37` | End of eval |
| Sample count | Shows dataset size | Low | `Evaluating on 10,000 samples...` | Start of eval |

### Checkpoint Phase Features

| Feature | Why Expected | Complexity | Output Example | Timing |
|---------|--------------|------------|----------------|--------|
| Checkpoint path | Always shows save location | Low | `Saving to checkpoints/epoch_1/` | Start of save |
| Save progress | Multi-shard models show this | Med | `Saving shards: [====] 64/64` | 2-5s total |
| Checkpoint complete message | Confirms save finished | Low | `Checkpoint saved in 45.2s` | End of save |

### Export Phase Features

| Feature | Why Expected | Complexity | Output Example | Timing |
|---------|--------------|------------|----------------|--------|
| Export start message | Signals training complete | Low | `Training complete! Exporting final model...` | instant |
| Format conversion | SafeTensors is standard now | Low | `Converting to SafeTensors format...` | 2-3s |
| Final model path | Shows where model saved | Low | `Model saved to: ./models/` | End |
| Training summary | Total time, final metrics | Low | `Total time: 4h 23m, Final loss: 0.82` | End |

## Differentiators

Features that set simulation apart. Not expected, but add convincing realism or entertainment.

### Visual Polish

| Feature | Value Proposition | Complexity | Output Example | Notes |
|---------|-------------------|------------|----------------|-------|
| Color-coded output | Matches real PyTorch/HF style | Low | Timestamps in magenta, phases in cyan | Use yansi crate |
| Multi-GPU status grid | Looks like real cluster monitoring | High | `[GPU 0-7: 98%, 72C] [GPU 8-15: 97%, 74C]` | Show 8 GPUs per line |
| GPU temperature display | Adds realism to GPU stats | Low | `72C` | Random 65-85C |
| Memory bar visualization | More visual than just numbers | Med | `[====75%====    ]` | Optional |
| Animated spinner for waits | Polished feel during long ops | Low | Loading... with spinner | During init/export |

### NCCL/Communication Features

| Feature | Value Proposition | Complexity | Output Example | Notes |
|---------|-------------------|------------|----------------|-------|
| AllReduce timing | Shows distributed overhead | Med | `[NCCL] AllReduce: 2.3ms` | Per sync operation |
| Broadcast timing | Additional comm metric | Low | `Broadcast: 0.8ms` | Occasional |
| Ring/Tree algorithm mention | Technical depth | Low | `AllReduce via Ring (64 ranks)` | Init phase only |
| Gradient sync messages | Shows DDP is working | Med | `Syncing gradients across 64 ranks...` | Periodic |

### Advanced Metrics

| Feature | Value Proposition | Complexity | Output Example | Notes |
|---------|-------------------|------------|----------------|-------|
| Gradient norm | Advanced training insight | Low | `Grad Norm: 1.234` | Per step |
| BLEU/ROUGE scores | Evaluation metrics for NLP | Med | `BLEU: 32.4, ROUGE-L: 0.456` | Validation only |
| Loss spike handling | Realistic training has issues | Med | `[WARN] Loss spike detected, reducing LR` | Rare event |
| Gradient accumulation steps | Shows memory optimization | Low | `Gradient Accum: 4 steps` | Init display |

### Humorous/Fun Elements (genact signature)

| Feature | Value Proposition | Complexity | Output Example | Notes |
|---------|-------------------|------------|----------------|-------|
| Funny model names | genact humor tradition | Low | `ButtGPT-420B`, `ChadLLM-Instruct` | Mix with real names |
| Absurd parameter counts | Over-the-top scale | Low | `69.420T parameters` | Occasionally |
| Satirical dataset names | Adds humor | Low | `StackOverflow-CopyPasta-2024` | Mix with real names |
| Fake error recovery | Drama without concern | Med | `[WARN] GPU 42 OOM, redistributing...` | Rare, always recovers |
| Silly checkpoint names | Humor in paths | Low | `checkpoints/big-chungus-epoch-3/` | When funny name selected |

### Realism Enhancements

| Feature | Value Proposition | Complexity | Output Example | Notes |
|---------|-------------------|------------|----------------|-------|
| Batch size display | Training hyperparameter | Low | `Batch: 2048 (global), 32 (per GPU)` | Init phase |
| Warmup steps mention | LR scheduling detail | Low | `Warmup: 2000 steps` | Init phase |
| Mixed precision indicator | Modern training uses this | Low | `Using BF16 mixed precision` | Init phase |
| DeepSpeed/FSDP mention | Training framework details | Low | `Strategy: FSDP (Full Shard)` | Init phase |
| Occasional NaN warning | Realistic edge case | Low | `[WARN] NaN detected in layer 42, skipping batch` | Very rare |

## Anti-Features

Features to explicitly NOT build. Would break immersion or violate genact principles.

| Anti-Feature | Why Avoid | What to Do Instead |
|--------------|-----------|-------------------|
| Actual network requests | genact must work offline | Simulate all data locally |
| Real GPU detection | Would expose actual hardware | Generate fake GPU specs randomly |
| Interactive prompts | genact is non-interactive | All output is autonomous |
| Actual file writes | genact shouldn't modify system | Only print "saved to..." messages |
| Excessively long runs | Should complete in reasonable time | Cap at 2-5 minutes max |
| Perfect metrics | Unrealistically smooth curves | Add noise to all metrics |
| Error states that hang | Should always "recover" | Any "error" resolves quickly |
| Real timestamps in future | Breaks immersion | Use current time via chrono |
| Consistent GPU counts | Would look scripted | Randomize 8-64 GPUs per run |
| Same model every time | Boring, detectable as fake | Random selection from pool |
| Too many warnings | Looks buggy, not realistic | Keep warnings rare (<5% of lines) |
| Progress bar going backwards | Would never happen in reality | Progress always increases |
| Instant completion | Suspicious, unrealistic | Minimum run time ~30 seconds |

## Feature Dependencies

```
Model Loading
     │
     ▼
Distributed Init ─────► NCCL Setup
     │
     ▼
Training Loop ◄────────────────────────┐
     │                                  │
     ├──► Step Metrics                  │
     │         │                        │
     │         ▼                        │
     ├──► GPU Status (optional)         │
     │                                  │
     └──► Progress Bar ─────────────────┤
                                        │
Validation Phase ◄──────────────────────┤
     │                                  │
     ▼                                  │
Checkpoint ─────────────────────────────┘
     │
     ▼ (after final epoch)
Export Phase
     │
     ▼
Summary
```

**Key Dependencies:**
- Progress bar requires step counter
- ETA requires progress bar and speed metrics
- GPU status display independent of metrics (can show either)
- Checkpoint depends on epoch completion
- Export depends on all training complete

## MVP Recommendation

For MVP, prioritize these table stakes (in order):

1. **Initialization phase** - Model/NCCL/GPU setup messages
2. **Training loop with progress bar** - Epoch/step counters, basic progress
3. **Training loss that decreases** - Core metric with noise
4. **Speed metrics** - Tokens/s or samples/s
5. **Validation phase** - Val loss/PPL after each epoch
6. **Checkpoint messages** - Simple "saving..." output
7. **Export phase** - Final summary

MVP differentiators (pick 2-3):
- Colored output (low effort, high impact)
- Multi-GPU status display (medium effort, high realism)
- One funny model name in the pool (low effort, genact signature)

Defer to post-MVP:
- **NCCL timing stats**: Adds complexity, diminishing returns
- **BLEU/ROUGE scores**: Only relevant for specific tasks
- **Gradient accumulation details**: Technical but not visually interesting
- **Advanced error/warning simulation**: Complexity vs value

## Timing/Pacing Considerations

### Phase Duration Targets

| Phase | Duration | Pacing Notes |
|-------|----------|--------------|
| Initialization | 5-10 seconds | Steady, deliberate pace |
| Training (per epoch) | 30-60 seconds | Fast step updates, slower at epoch boundaries |
| Validation | 5-10 seconds | Moderate pace, shows "thinking" |
| Checkpoint | 3-5 seconds | Progress bar with slight delays |
| Export | 5-10 seconds | Format conversion looks intensive |
| **Total** | **~2-4 minutes** | Feels substantial but not tedious |

### Update Frequency

| Element | Update Frequency | Rationale |
|---------|-----------------|-----------|
| Step counter | Every 0.3-1s | Fast but readable |
| Loss value | Every 1-3 steps | Changes feel meaningful |
| Progress bar | Every step | Smooth progress |
| GPU status | Every 5-10 steps | Not overwhelming |
| Speed metrics | Every 3-5 steps | Stable readings |

### Pacing Best Practices

1. **Variable delays**: Always use `rng.random_range(min..max)` for csleep
2. **Burst + pause pattern**: Several fast lines, then a brief pause
3. **Phase transitions**: Slightly longer pause between phases (1-2s)
4. **Progress acceleration**: Training feels faster toward end of epoch
5. **Exit points**: Check should_exit() every 5-10 lines minimum

## Output Format Examples

### Initialization (Table Stakes)

```
[2026-01-30 14:23:15] Initializing distributed training environment...
[2026-01-30 14:23:16] Found 64 NVIDIA A100-SXM4-80GB GPUs across 8 nodes
[2026-01-30 14:23:17] Setting up NCCL backend...
[2026-01-30 14:23:18] Master: node-0:29500 | World size: 64
[2026-01-30 14:23:19] Loading model: Llama-3-70B-Instruct
[2026-01-30 14:23:21] Model size: 70,553,706,496 parameters (sharded across 64 GPUs)
[2026-01-30 14:23:22] Loading tokenizer: LlamaTokenizer (vocab_size=128256)
[2026-01-30 14:23:23] Dataset: RedPajama-v2 (1.2T tokens, 2.1M documents)
[2026-01-30 14:23:24] Configuration: batch=2048, lr=1e-4, epochs=3
[2026-01-30 14:23:25] Using BF16 mixed precision training
[2026-01-30 14:23:25] Ready to train!
```

### Training Loop (Table Stakes)

```
Epoch 1/3 [==================>         ] 65%  Step 32500/50000
  Loss: 1.8234  |  LR: 9.8e-05  |  GPU Mem: 76.4/80.0 GB
  Speed: 142,532 tok/s  |  8,192 samples/s  |  ETA: 12m 34s
```

### Training Loop (With Differentiators)

```
Epoch 1/3 [==================>         ] 65%  Step 32500/50000
  Loss: 1.8234  |  LR: 9.8e-05  |  Grad Norm: 0.847  |  GPU Mem: 76.4/80.0 GB
  Speed: 142,532 tok/s  |  8,192 samples/s  |  ETA: 12m 34s
  [GPU 0-7:  98% 72C] [GPU 8-15:  97% 74C] [GPU 16-23: 96% 71C] [GPU 24-31: 98% 73C]
  [GPU 32-39: 97% 75C] [GPU 40-47: 96% 72C] [GPU 48-55: 98% 74C] [GPU 56-63: 97% 73C]
  [NCCL] AllReduce: 2.3ms | Broadcast: 0.8ms | Barrier: 0.1ms
```

### Validation (Table Stakes)

```
[Validation] Epoch 1 complete | Evaluating on 10,000 samples...
  Val Loss: 1.7123  |  Val PPL: 5.54  |  Val Acc: 0.912
```

### Checkpoint (Table Stakes)

```
[Checkpoint] Saving to checkpoints/llama-3-70b-epoch-1/
  Saving model shards: [====================] 64/64 complete
  Saving optimizer state: 312.4 GB
  Checkpoint saved in 42.3s
```

### Export (Table Stakes)

```
[Export] Training complete! Exporting final model...
  Converting to SafeTensors format...
  Export progress: [====================] 100%
  Model saved to: ./models/llama-3-70b-finetuned-v1/

================================================================================
  Training Summary
================================================================================
  Model:           Llama-3-70B-Instruct
  Total time:      4h 23m 15s
  Final loss:      0.8234
  Final PPL:       2.28
  Tokens trained:  1,234,567,890
  Checkpoints:     3 saved
================================================================================
```

### Funny Model Name Example (Differentiator)

```
[2026-01-30 14:23:19] Loading model: ButtGPT-420B-UltraInstruct-v69
[2026-01-30 14:23:21] Model size: 420,690,000,000 parameters (nice.)
```

## Sources

- **PyTorch Distributed Documentation:** [docs.pytorch.org/docs/stable/distributed.html](https://docs.pytorch.org/docs/stable/distributed.html) (HIGH confidence)
- **Hugging Face Trainer Documentation:** [huggingface.co/docs/transformers/trainer](https://huggingface.co/docs/transformers/en/trainer) (HIGH confidence)
- **NCCL Documentation:** [developer.nvidia.com/nccl](https://developer.nvidia.com/nccl) (HIGH confidence)
- **nvidia-smi output format:** [docs.nvidia.com/deploy/nvidia-smi](https://docs.nvidia.com/deploy/nvidia-smi/) (HIGH confidence)
- **DeepSpeed Communication Logging:** [deepspeed.ai/tutorials/comms-logging](https://www.deepspeed.ai/tutorials/comms-logging/) (HIGH confidence)
- **BLEU/ROUGE metrics:** [confident-ai.com/blog/llm-evaluation-metrics](https://www.confident-ai.com/blog/llm-evaluation-metrics-everything-you-need-for-llm-evaluation) (HIGH confidence)
- **Existing genact modules:** `F:\genact-ex\src\modules\*.rs` (HIGH confidence)

---

## Quality Gate Checklist

- [x] Categories are clear (table stakes vs differentiators vs anti-features)
- [x] Specific output examples for each feature
- [x] Timing/pacing considerations noted
- [x] MVP recommendation provided
- [x] Feature dependencies mapped
- [x] Anti-features explicitly listed with rationale
