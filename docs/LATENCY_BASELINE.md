# Latency Baseline

## What "Natural" Means

Natural spoken interaction is mostly governed by turn timing, not by full
answer completion time. The practical target is: after the user stops talking,
how soon do they hear the assistant produce a plausible first sound, and does
audio keep flowing without gaps?

Research and standards used to set the target:

- ITU-T G.114 / conversational audio practice: mouth-to-ear delay should stay
  around the low hundreds of milliseconds; delay above roughly 200 ms starts to
  degrade conversational quality in real-time voice systems.
- Chang et al., Interspeech 2022, "Turn-Taking Prediction for Natural
  Conversational Speech": true turn-taking can be predicted with 100 ms latency
  on a disfluency-heavy test set.
  https://arxiv.org/abs/2208.13321
- Udupa et al., 2025, "Streaming Endpointer for Spoken Dialogue": a streaming
  endpointing system reports 160 ms median latency and improves median response
  time by 1200 ms.
  https://arxiv.org/abs/2506.07081
- Jacoby et al., 2024, "Human Latency Conversational Turns for Spoken Avatar
  Systems": human-like dialogue requires understanding and response generation
  to begin before the speaker has fully completed the utterance.
  https://arxiv.org/abs/2404.16053

## Targets

Strict human-like target:

- Endpointing latency: <= 200 ms
- ASR latency: <= 150 ms for a short utterance
- Perceived first audio after endpoint/ASR: <= 700 ms
- Total user-stop-to-first-audio estimate: <= 1000 ms
- TTS RTF: <= 1.0
- No inter-chunk gap above 360 ms

Local RTX 4060 cascade target:

- Endpointing latency: <= 320 ms
- ASR latency: <= 400 ms
- Perceived first audio after ASR: <= 1200 ms
- Total user-stop-to-first-audio estimate: <= 1500 ms
- TTS first chunk median: <= 700 ms
- TTS RTF: <= 1.0
- No inter-chunk gap above 420 ms

The strict target needs a predictive/semantic endpointer and a faster LLM or a
native speech-to-speech model. The local target is the current deployable
baseline for this cascaded ASR -> LLM -> TTS stack.

## Current Baseline

Command:

```powershell
.\.venv\Scripts\python.exe scripts\benchmark_latency.py --profile local
```

Last passing run:

- Endpoint config: 320 ms
- ASR median: 206.2 ms on `voices/ref.wav`
- LLM first speakable chunk median: 1261.5 ms
- Perceived model first audio median: 640.0 ms via instant ack
- Estimated user-stop-to-first-audio: 1166.2 ms
- TTS first chunk median: 667.5 ms
- TTS RTF median: 0.6
- Max inter-chunk gap max: 211.0 ms

The main remaining gap to strict natural timing is not audio streaming anymore;
it is endpoint prediction plus the 2B Ollama model's roughly 1.2 s first
speakable chunk latency.
