# Codenames AI Arena

Four local AI players play one [Codenames](https://codenames.game/) game using `qwen3:14b` through Ollama. Playwright gives each player an isolated browser context. The arena joins the four roles automatically; you create the room and start the game from the host browser.

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

Start Ollama if it is not already running (`ollama serve`). The default configuration in `.env.example` uses `http://localhost:11434`, `qwen3:14b`, and visible browser windows (`HEADLESS=false`). You can put a room URL in `CODENAMES_ROOM_URL` in `.env` instead of passing `--room`. The `.env` file is ignored by Git.

## Run a game

Create a room in your own browser and keep it open. Then run:

```powershell
python -m codenames_ai.main --check-ollama
python -m codenames_ai.main arena --room "https://codenames.game/r/ROOM_CODE"
```

The arena opens four windows and joins these roles:

| Internal player | Room role |
| --- | --- |
| `BlueSpymasterAI` | Blue spymaster |
| `BlueOperativeAI` | Blue operative |
| `RedSpymasterAI` | Red spymaster |
| `RedOperativeAI` | Red operative |

Wait for `All four AI players are ready.` The website may assign names such as `Player1`; the internal player names above identify the browser contexts and do not need to match website nicknames. Click **Start Game** in the host browser, then press **Enter** in the terminal. The agents play until game over. Press Enter once more to close their browser windows.

The host browser stays under your control; the arena does not create a room or click Start Game. Arena mode requires visible windows, so keep `HEADLESS=false` and do not pass `--headless`.

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
