//! Pretend to train a large language model
use async_trait::async_trait;
use chrono::Local;
use progress_string::BarBuilder;
use rand::rng;
use rand::seq::IndexedRandom;
use rand::Rng;
use yansi::Paint;

use crate::args::AppConfig;
use crate::data::{GPU_MODELS_LIST, LLM_DATASETS_LIST, LLM_MODELS_LIST};
use crate::io::{csleep, erase_line, newline, print};
use crate::modules::Module;

/// Log an INFO message with timestamp
async fn log_info(message: &str) {
    let ts = Local::now().format("%Y-%m-%d %H:%M:%S");
    print(format!("[{}] {} {}", ts, Paint::green("INFO").bold(), message)).await;
    newline().await;
}

/// Log an INFO message with timestamp and rank info
async fn log_info_rank(rank: u32, world_size: u32, message: &str) {
    let ts = Local::now().format("%Y-%m-%d %H:%M:%S");
    print(format!(
        "[{}] [Rank {}/{}] {} {}",
        ts,
        rank,
        world_size,
        Paint::green("INFO").bold(),
        message
    ))
    .await;
    newline().await;
}

/// Log a WARNING message with timestamp
#[allow(dead_code)]
async fn log_warning(message: &str) {
    let ts = Local::now().format("%Y-%m-%d %H:%M:%S");
    print(format!(
        "[{}] {} {}",
        ts,
        Paint::yellow("WARNING").bold(),
        message
    ))
    .await;
    newline().await;
}

/// Run the initialization phase of LLM training simulation
/// Displays model loading, tokenizer, dataset, NCCL init, and GPU detection
async fn run_initialization(appconfig: &AppConfig) {
    let mut rng = rng();

    // Configuration
    let num_gpus: u32 = 64;
    let num_nodes: u32 = 8;

    // Select random model, GPU, and dataset
    let model = LLM_MODELS_LIST
        .choose(&mut rng)
        .unwrap_or(&"GPT-Unknown");
    let gpu = GPU_MODELS_LIST.choose(&mut rng).unwrap_or(&"A100");
    let dataset = LLM_DATASETS_LIST
        .choose(&mut rng)
        .unwrap_or(&"CommonCrawl");

    // Generate random architecture details
    let params_b: f64 = rng.random_range(7.0..405.0);
    let num_layers: u32 = rng.random_range(32..128);
    let hidden_size: u32 = rng.random_range(4096..16384);
    let attention_heads: u32 = rng.random_range(32..128);
    let vocab_size: u32 = rng.random_range(32000..128000);
    let samples: u64 = rng.random_range(1_000_000..100_000_000);

    // ===================
    // INIT-01: Model Loading
    // ===================
    log_info(&format!("Loading model: {}", model)).await;
    csleep(200).await;

    if appconfig.should_exit() {
        return;
    }

    log_info(&format!("  Parameters: {:.1}B", params_b)).await;
    csleep(100).await;

    log_info(&format!(
        "  Architecture: {} layers, hidden_size={}, attention_heads={}",
        num_layers, hidden_size, attention_heads
    ))
    .await;
    csleep(100).await;

    if appconfig.should_exit() {
        return;
    }

    // Model loading progress bar
    let mut bar = BarBuilder::new()
        .total(100)
        .width(40)
        .full_char('=')
        .include_percent()
        .build();

    let load_start = std::time::Instant::now();

    for i in 0..=100 {
        bar.replace(i);
        let ts = Local::now().format("%Y-%m-%d %H:%M:%S");
        erase_line().await;
        print(format!(
            "[{}] {} Loading model weights... {}",
            ts,
            Paint::green("INFO").bold(),
            bar
        ))
        .await;

        csleep(rng.random_range(20..80)).await;

        if appconfig.should_exit() {
            newline().await;
            return;
        }
    }
    newline().await;

    let load_time = load_start.elapsed().as_secs_f64();
    log_info(&format!("Model loaded in {:.1}s", load_time)).await;
    csleep(300).await;

    if appconfig.should_exit() {
        return;
    }

    // ===================
    // INIT-02: Tokenizer Loading
    // ===================
    let tokenizer_name = if model.contains("Llama") || model.contains("llama") {
        "LlamaTokenizer"
    } else if model.contains("GPT") {
        "GPT2Tokenizer"
    } else {
        "SentencePieceTokenizer"
    };

    log_info(&format!("Loading tokenizer: {}", tokenizer_name)).await;
    csleep(200).await;

    log_info(&format!("  Vocab size: {}", vocab_size)).await;
    csleep(100).await;

    if appconfig.should_exit() {
        return;
    }

    // ===================
    // INIT-03: Dataset Loading
    // ===================
    log_info(&format!("Loading dataset: {}", dataset)).await;
    csleep(500).await;

    log_info(&format!(
        "  Dataset: {} samples, batch_size=2048, micro_batch=4",
        samples
    ))
    .await;
    csleep(300).await;

    if appconfig.should_exit() {
        return;
    }

    // ===================
    // INIT-04: Distributed Environment
    // ===================
    log_info("Initializing distributed environment...").await;
    csleep(500).await;

    log_info("NCCL version: 2.19.3").await;
    csleep(100).await;

    log_info(&format!(
        "World size: {} GPUs across {} nodes",
        num_gpus, num_nodes
    ))
    .await;
    csleep(200).await;

    if appconfig.should_exit() {
        return;
    }

    // ===================
    // INIT-05: GPU Detection
    // ===================
    // Show first 16 GPUs (2 nodes worth)
    for node in 0..2u32 {
        for local_gpu in 0..8u32 {
            let gpu_id = node * 8 + local_gpu;
            log_info_rank(
                gpu_id,
                num_gpus,
                &format!("Detected NVIDIA {} on node {}", gpu, node),
            )
            .await;
            csleep(rng.random_range(10..50)).await;

            if appconfig.should_exit() {
                return;
            }
        }
    }

    // Show remaining GPUs count
    log_info(&format!("... and {} more GPUs", num_gpus - 16)).await;
    csleep(500).await;

    if appconfig.should_exit() {
        return;
    }

    // NCCL AllReduce test
    log_info("Running NCCL AllReduce test...").await;
    csleep(1000).await;

    let allreduce_time: f64 = rng.random_range(5.0..50.0);
    log_info(&format!("AllReduce test passed ({:.2}ms)", allreduce_time)).await;
    csleep(300).await;

    newline().await;
    log_info("======== Initialization Complete ========").await;
    newline().await;
}

pub struct LlmTraining;

#[async_trait(?Send)]
impl Module for LlmTraining {
    fn name(&self) -> &'static str {
        "llm_training"
    }

    fn signature(&self) -> String {
        "python train.py --model llama-7b --gpus 64".to_string()
    }

    async fn run(&self, appconfig: &AppConfig) {
        let mut rng = rng();

        // Select random data
        let model = LLM_MODELS_LIST.choose(&mut rng).unwrap_or(&"GPT-Unknown");
        let gpu = GPU_MODELS_LIST.choose(&mut rng).unwrap_or(&"A100");
        let dataset = LLM_DATASETS_LIST.choose(&mut rng).unwrap_or(&"CommonCrawl");

        // Placeholder output - will be expanded in Phase 2
        print(format!("[INFO] Loading model: {model}")).await;
        newline().await;
        csleep(500).await;

        print(format!("[INFO] Detected 64x NVIDIA {gpu}")).await;
        newline().await;
        csleep(300).await;

        print(format!("[INFO] Dataset: {dataset}")).await;
        newline().await;
        csleep(300).await;

        // Simple loop with exit check
        for epoch in 1..=3 {
            print(format!("[TRAIN] Epoch {epoch}/3 - Training...")).await;
            newline().await;
            csleep(1000).await;

            if appconfig.should_exit() {
                return;
            }
        }

        print("[INFO] Training placeholder complete").await;
        newline().await;
    }
}
