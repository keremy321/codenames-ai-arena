import json

import pytest
from pydantic import ValidationError

from codenames_ai.agents.operative import OperativeAgent
from codenames_ai.domain.enums import CardColor, Role, Team
from codenames_ai.domain.models import Card, Clue, PublicGameState, SpymasterGameState
from codenames_ai.llm.ollama import OllamaClient


def test_roles_and_validation() -> None:
    assert Role("operative") is Role.OPERATIVE
    assert Team("blue") is Team.BLUE
    with pytest.raises(ValidationError):
        Card(index=-1, word="MOON")
    with pytest.raises(ValidationError):
        Clue(word=" ", number=2)
    with pytest.raises(ValidationError):
        PublicGameState(team=Team.BLUE, role=Role.SPYMASTER, cards=[])


def test_public_state_sanitizes_and_is_immutable() -> None:
    cards = [
        Card(index=0, word="MOON", color=CardColor.ASSASSIN),
        Card(index=1, word="KING", color=CardColor.RED, revealed=True),
    ]
    state = PublicGameState(team=Team.BLUE, cards=cards)
    assert state.cards[0].color is None
    assert state.cards[1].color == CardColor.RED
    assert cards[0].color == CardColor.ASSASSIN
    cards.clear()
    assert len(state.cards) == 2
    with pytest.raises(ValidationError):
        state.cards[0].color = CardColor.BLUE
    assert "assassin" not in state.model_dump_json()


async def test_operative_prompt_revalidates_and_rejects_spymaster() -> None:
    hidden = Card(index=0, word="MOON", color=CardColor.ASSASSIN)
    async with OllamaClient() as llm:
        agent = OperativeAgent(Team.BLUE, llm)
        public = PublicGameState(team=Team.BLUE, cards=[]).model_copy(update={"cards": (hidden,)})
        assert json.loads(agent.prompt_state(public))["cards"][0]["color"] is None
        with pytest.raises(ValueError):
            agent.prompt_state(SpymasterGameState(team=Team.BLUE, cards=[hidden]))
        with pytest.raises(ValueError):
            agent.prompt_state(PublicGameState(team=Team.RED, cards=[]))
