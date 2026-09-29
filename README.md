# Codenames AI Arena

Four local AI players play one [Codenames](https://codenames.game/) game using `qwen3:14b` through Ollama. Playwright gives each player an isolated browser context. The arena creates the room, seats the four roles, starts the game, and records the match automatically.

The two spymasters can read the hidden card key. Operatives receive only the public board, revealed cards, clues, and their team's public clue history. The controller reads the page again after each guess and allows only the active player's browser to act.

## Requirements

- Python 3.12 or newer
- [Ollama](https://ollama.com/) with `qwen3:14b`
- Playwright Chromium
- A room on [codenames.game](https://codenames.game/)

## Setup (Windows PowerShell)

From the repository root:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python -m playwright install chromium
Copy-Item .env.example .env
ollama pull qwen3:14b
```

The default configuration in `.env.example` uses `http://localhost:11434`, `qwen3:14b`, and visible browser windows (`HEADLESS=false`). The `.env` file is ignored by Git.

## Run a game

```powershell
python -m codenames_ai.main arena
```

Before any browser opens, the arena checks every configured model. If the local Ollama server is not running, the arena starts `ollama serve` itself and stops it again at exit; a server that was already running is left alone. A remote `OLLAMA_BASE_URL` is never started locally. A missing model stops the run with the matching `ollama pull` command. Each distinct Ollama model is then warmed with one tiny request (`Ollama warm-up: qwen3:14b ... 8.4s`), so the first spymaster turn does not pay for loading it; a model that cannot run stops the run here. Every Ollama request asks the server to keep the model loaded for `OLLAMA_KEEP_ALIVE` (default `30m`). OpenAI and Anthropic roles are checked for their API key only, so startup spends no tokens.

The arena then opens a host window (`ArenaHost`, a spectator; change it with `--host-nickname`) that creates a new room. It opens four player windows that join these roles:

| Internal player | Room role |
| --- | --- |
| `BlueSpymasterAI` | Blue spymaster |
| `BlueOperativeAI` | Blue operative |
| `RedSpymasterAI` | Red spymaster |
| `RedOperativeAI` | Red operative |

When all four are seated, the host clicks **Start game** and the agents play until game over. Press Enter at the end to close the browser windows. The website may assign names such as `Player1`; the internal player names above identify the browser contexts and do not need to match website nicknames.

To use a room you created yourself, pass `--room "https://codenames.game/r/ROOM_CODE"` (or set `CODENAMES_ROOM_URL`). The arena then joins that room without a host window. Start the game in your own browser, then press **Enter** in the terminal. This mode needs visible windows, so keep `HEADLESS=false`.

## Match logs

Each match is recorded under `logs/` (ignored by Git) in the directory you run from:

```text
logs/2026-09-29T001530_radun-kapuf/
    match.json     room, players' providers and models, start/end times, winner, termination reason
    events.jsonl   one JSON object per event: clues, guesses, turn changes, agent decisions with
                   LLM call timings, errors, and the final result
```

Each guess also writes one `browser_action` event with the milliseconds spent in each browser stage (selecting the card, the confirmation, the reveal, and the observer catching up). Spymaster decisions list `generator_targets` (the words the clue was generated for), `expected_guesses` (what the operative likely finds with the chosen number), `predicted_ranking`, and `verification_ran` / `verification_passed`. If room creation fails, the log directory gets a `startup_failure` event and a `startup_failure.png` screenshot. A crash still leaves the events recorded so far, and `match.json` shows the error. Logs never contain API keys. If writing a log fails, the game continues and the error is reported once.

## Choosing models per role

Each role can use its own provider (`ollama`, `openai`, or `anthropic`) and model. Without any options every role uses Ollama with `OLLAMA_MODEL`. `--provider`/`--model` (or `LLM_PROVIDER`/`LLM_MODEL`) set the default for all four roles; the per-role flags override it:

```powershell
python -m codenames_ai.main arena `
  --blue-spymaster-provider anthropic --blue-spymaster-model "<claude-model>" `
  --blue-operative-provider ollama --blue-operative-model qwen3:14b `
  --red-spymaster-provider openai --red-spymaster-model "<openai-model>" `
  --red-operative-provider ollama --red-operative-model gemma3:12b
```

A role that overrides only its model keeps the default provider. A role that switches to OpenAI or Anthropic must also name its model. The same overrides are available as environment variables such as `BLUE_SPYMASTER_PROVIDER` and `BLUE_SPYMASTER_MODEL`, and CLI flags take precedence over them. OpenAI roles read `OPENAI_API_KEY` and Anthropic roles read `ANTHROPIC_API_KEY`. The arena prints each role's provider and model at startup and never prints keys.

## How decisions work

- The spymaster generates clue candidates from the friendly words, rejects illegal clues locally, and checks promising candidates against a color-blind ranking that approximates what the operative may choose. It considers the match position and the risk of enemy, neutral, and assassin cards before submitting one clue.
- The operative ranks unrevealed public words for the current clue. After each confirmed guess, the controller rereads the board before deciding whether to guess again or end the turn. The ranking for a clue can be reused within the same turn.
- Each team has separate public clue history. An unresolved older clue can support one bonus guess, subject to the site's `clue number + 1` legal limit. Spymaster intended targets remain private.

The agents can make semantic mistakes even when the browser actions and information boundaries work correctly. Model calls, especially spymaster turns, may take time on local hardware. A completed four-agent game has reached the site's game-over screen, but Codenames site changes can require selector updates.

## Checks and diagnostics

Run the offline test suite and lint checks:

```powershell
python -m pytest
python -m ruff check src tests
python -m ruff format --check src tests
```

`python -m codenames_ai.main --help` lists CLI options. If a live page stops matching the reader, `python -m codenames_ai.main inspect --room "ROOM_URL"` opens a one-page DOM inspector. In that inspector, Enter refreshes the summary, `s` saves snippets to the ignored `debug/` directory, and `q` exits. Review saved snippets before sharing them.

The offline evaluation cases can be listed with `python -m codenames_ai.evaluation --list`. Evaluation is experimental; use the test suite and a disposable live room to verify behavior before relying on a particular decision quality.
