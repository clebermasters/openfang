//! Voice message processor for Telegram.
//!
//! Downloads voice messages from Telegram and transcribes them using Whisper.
//! Auto-cascades: Groq Whisper -> OpenAI Whisper.

use std::time::Duration;
use tracing::info;

const TELEGRAM_FILE_URL: &str = "https://api.telegram.org/file/bot";

pub struct VoiceProcessor {
    bot_token: String,
}

impl VoiceProcessor {
    pub fn new() -> Result<Self, String> {
        let bot_token = std::env::var("TELEGRAM_BOT_TOKEN")
            .map_err(|_| "TELEGRAM_BOT_TOKEN not set")?;
        Ok(Self { bot_token })
    }

    pub async fn process(&self, file_id: &str) -> Result<String, String> {
        let audio_bytes = self.download_audio(file_id).await?;
        self.transcribe(&audio_bytes).await
    }

    async fn download_audio(&self, file_id: &str) -> Result<Vec<u8>, String> {
        let get_file_url = format!(
            "https://api.telegram.org/bot{}/getFile?file_id={}",
            self.bot_token, file_id
        );

        info!(url = %get_file_url, "Calling getFile API");

        let client = reqwest::Client::new();
        let resp = client
            .get(&get_file_url)
            .timeout(Duration::from_secs(10))
            .send()
            .await
            .map_err(|e| format!("Failed to get file info: {}", e))?;

        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        
        info!(status = %status, body = %body, "getFile response");

        if !status.is_success() {
            return Err(format!("Telegram getFile failed ({}): {}", status, body));
        }

        let json: serde_json::Value = serde_json::from_str(&body)
            .map_err(|e| format!("Failed to parse getFile response: {}", e))?;

        let file_path = json["result"]["file_path"]
            .as_str()
            .ok_or("No file_path in getFile response")?;

        info!(file_path = %file_path, "Got file path from Telegram");

        let download_url = format!("{}{}/{}", TELEGRAM_FILE_URL, self.bot_token, file_path);
        info!(url = %download_url, "Downloading voice file from Telegram");

        let audio_resp = client
            .get(&download_url)
            .timeout(Duration::from_secs(60))
            .send()
            .await
            .map_err(|e| format!("Failed to download audio: {}", e))?;

        if !audio_resp.status().is_success() {
            let status = audio_resp.status();
            return Err(format!("Failed to download audio ({}): {}", status, audio_resp.text().await.unwrap_or_default()));
        }

        audio_resp
            .bytes()
            .await
            .map(|b| b.to_vec())
            .map_err(|e| format!("Failed to read audio bytes: {}", e))
    }

    async fn transcribe(&self, audio_bytes: &[u8]) -> Result<String, String> {
        let (api_url, api_key, provider, model) = self.detect_provider()?;

        let ext = "ogg";
        let filename = format!("voice.{}", ext);

        let file_part = reqwest::multipart::Part::bytes(audio_bytes.to_vec())
            .file_name(filename)
            .mime_str("audio/ogg")
            .map_err(|e| format!("Failed to set MIME type: {}", e))?;

        let form = reqwest::multipart::Form::new()
            .part("file", file_part)
            .text("model", model)
            .text("response_format", "text");

        let client = reqwest::Client::new();
        let resp = client
            .post(api_url)
            .bearer_auth(&api_key)
            .multipart(form)
            .timeout(Duration::from_secs(60))
            .send()
            .await
            .map_err(|e| format!("Transcription request failed: {}", e))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(format!("Transcription API error ({}): {}", status, body));
        }

        let text = resp
            .text()
            .await
            .map_err(|e| format!("Failed to read transcription response: {}", e))?;

        info!(provider, model, chars = text.len(), "Audio transcribed successfully");
        Ok(text)
    }

    fn detect_provider(&self) -> Result<(String, String, &'static str, &'static str), String> {
        if std::env::var("GROQ_API_KEY").is_ok() {
            info!("Using Groq for transcription");
            Ok((
                "https://api.groq.com/openai/v1/audio/transcriptions".to_string(),
                std::env::var("GROQ_API_KEY").unwrap(),
                "groq",
                "whisper-large-v3-turbo",
            ))
        } else if std::env::var("OPENAI_API_KEY").is_ok() {
            info!("Using OpenAI for transcription");
            Ok((
                "https://api.openai.com/v1/audio/transcriptions".to_string(),
                std::env::var("OPENAI_API_KEY").unwrap(),
                "openai",
                "whisper-1",
            ))
        } else {
            Err("No transcription provider available. Set GROQ_API_KEY or OPENAI_API_KEY".to_string())
        }
    }
}
