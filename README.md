# SRT Subtitle Translation Using LLM

This project demonstrates how to translate SRT (SubRip Subtitle) files using Large Language Models (LLMs), specifically Google's Gemini models via Vertex AI. It includes scripts for parallel processing to handle large subtitle files efficiently and ensures context is maintained across translation windows.

## Project Structure

- **`parallel_runner.py`**: The main script to run subtitle processing in parallel using multiple workers.
- **`subtitle_generator.py`**: Core logic for interacting with the LLM and parsing SRT blocks.
- **`requirements.txt`**: Python dependencies required for the project.
- **`reference_*.srt`**: Various reference and testing files (original, serial processed, and parallel processed outputs) used for validation.

## Prerequisites

Before running the application, ensure you have the following set up:

1.  **Google Cloud Platform (GCP) Account**: You need an active GCP project.
2.  **Vertex AI API**: Enable the [Vertex AI API](https://console.cloud.google.com/vertex-ai) for your project.
3.  **Google Cloud CLI (`gcloud`)**: Install the gcloud CLI. [Installation Guide](https://cloud.cloud.google.com/sdk/docs/install).

### Authentication
Authenticate your local environment with Google Cloud:

```bash
gcloud auth login
gcloud auth application-default login
```
Installation
Clone the repository:

```bash
git clone <repository_url>
cd sample_srt_translation_using_llm
```
Install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows, use .venv\Scripts\activate
pip install -r requirements.txt
```
Configuration
Create a .env file in the root directory of the project to store your environment variables. This is required for the LLM client to function correctly.

`.env` file content:

```
# Your Google Cloud Project ID
GOOGLE_CLOUD_PROJECT=your-gcp-project-id-here

# Flag to tell the library to use Vertex AI
GOOGLE_GENAI_USE_VERTEXAI=1
```
Usage
You can translate subtitle files using the parallel_runner.py script. This script splits the work among multiple workers for faster processing.

Basic Command:

```bash
python parallel_runner.py input_file.srt -l "Target Language"
```
Advanced Example:
Translate sample.srt to Kannada using 10 concurrent workers and a chunk size of 100 blocks:

```bash
python parallel_runner.py sample.srt -l Kannada --workers 10 --chunk-size 100
```
Arguments
`filename`: Path to the source .srt file.

`-l`, `--language`: Target language for translation (e.g., Hindi, Spanish, Kannada).

`--workers`: Number of parallel threads (default is usually set in script).

`--chunk-size`: Number of subtitle blocks per request.

Contributing
Feel free to fork the repository, make changes, and submit pull requests. For major changes, please open an issue first to discuss what you would like to change.

License
This project is open-sourced under the MIT License.
