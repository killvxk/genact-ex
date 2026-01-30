//! Pretend to train a large language model
use async_trait::async_trait;
use rand::rng;
use rand::seq::IndexedRandom;

use crate::args::AppConfig;
use crate::data::{GPU_MODELS_LIST, LLM_DATASETS_LIST, LLM_MODELS_LIST};
use crate::io::{csleep, newline, print};
use crate::modules::Module;

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
