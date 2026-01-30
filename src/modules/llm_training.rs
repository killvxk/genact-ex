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
