"""Pure decision logic: tier parsing, operative policy, clue scoring, race model."""

import pytest

from codenames_ai.agents.association import (
    Fit,
    RankedWord,
    canonical_clue,
    parse_rankings,
    parse_single,
    ranking_schema,
    single_clue_request,
)
from codenames_ai.agents.clue_scoring import (
    Outcome,
    Side,
    action_numbers,
    assess_action,
    assess_actions,
    best_by_size,
    expected_value,
    outcomes,
    prune_dominated,
    select_clue,
    side_of,
    win_probability,
)
from codenames_ai.agents.guess_policy import decide_bonus, decide_current
from codenames_ai.agents.race import RaceModel, RacePosition, RaceState
from codenames_ai.domain.enums import CardColor, Team

F, E, N, A = Side.FRIENDLY, Side.ENEMY, Side.NEUTRAL, Side.ASSASSIN


def ranked(*items: tuple[str, str], clue: str | None = None) -> tuple[RankedWord, ...]:
    return tuple(RankedWord(word, Fit(fit), clue) for word, fit in items)


def board(ours: int, theirs: int, extra: dict[str, Side] | None = None) -> dict[str, Side]:
    """Sides for a position: friendly A.., enemy X.., plus named extras."""
    sides = {f"F{i}": F for i in range(ours)} | {f"X{i}": E for i in range(theirs)}
    return sides | (extra or {})


def race(sides: dict[str, Side]) -> RaceState:
    return RaceState.of(sum(s is F for s in sides.values()), sum(s is E for s in sides.values()))


# --- shared ranking format -------------------------------------------------------


def test_parse_rankings_keeps_valid_part_and_orders_strong_first() -> None:
    reply = {
        "rankings": [
            {"clue": "sea", "strong": ["WAVE", "nope", "WAVE"], "possible": ["BEACH", "WAVE"]},
            {"clue": "unknown", "strong": ["BEACH"], "possible": []},
            {"clue": "SEA", "strong": ["BEACH"], "possible": []},  # repeated clue ignored
            "garbage",
        ]
    }
    result = parse_rankings(reply, ["SEA"], ["WAVE", "BEACH"])
    assert result == {"SEA": ranked(("WAVE", "strong"), ("BEACH", "possible"), clue="SEA")}
    assert parse_rankings({}, ["SEA"], ["WAVE"]) == {}


def test_ranking_schema_caps_buckets_and_pins_clues() -> None:
    schema = ranking_schema(["A", "B"], ["W1", "W2", "W3", "W4", "W5"])
    items = schema["properties"]["rankings"]
    assert (items["minItems"], items["maxItems"]) == (2, 2)
    props = items["items"]["properties"]
    assert props["clue"]["enum"] == ["A", "B"]
    assert (props["strong"]["maxItems"], props["possible"]["maxItems"]) == (4, 3)


def test_single_request_is_canonical_and_deterministic() -> None:
    # qwen3 ranks "wave" and "WAVE" differently, so both sides must spell it the same.
    assert canonical_clue("  wave ") == "WAVE"
    a = single_clue_request("wave", ["SEA", "SAND"])
    b = single_clue_request("WAVE", ["SEA", "SAND"])
    assert a == b and a["temperature"] == 0.0
    assert a["schema"]["required"] == ["rankings", "best_guess"]
    reply = {"rankings": [{"clue": "WAVE", "strong": [], "possible": []}], "best_guess": "SAND"}
    assert parse_single(reply, "wave", ["SEA", "SAND"]) == ranked(("SAND", "weak"), clue="WAVE")


# --- operative policy ------------------------------------------------------------


def test_first_guess_is_required_even_when_weak() -> None:
    decision = decide_current(ranked(("X", "weak")), 1, first_guess_required=True)
    assert decision.word == "X"


@pytest.mark.parametrize(
    "strong,remaining,guess",
    [(1, 1, True), (2, 1, False), (2, 2, True), (3, 2, True), (4, 2, False), (4, 3, True)],
)
def test_continue_only_when_top_strong_is_probably_ours(
    strong: int, remaining: int, guess: bool
) -> None:
    ranking = ranked(*[(f"S{i}", "strong") for i in range(strong)], ("P", "possible"))
    decision = decide_current(ranking, remaining, first_guess_required=False)
    assert (decision.word == "S0") is guess
    assert decision.end_turn is not guess


def test_possible_words_never_justify_extra_guess() -> None:
    ranking = ranked(("P1", "possible"), ("P2", "possible"))
    assert decide_current(ranking, 2, first_guess_required=False).end_turn


def test_bonus_judges_each_old_clue_on_its_own_list() -> None:
    # Live bug: merging clue lists hid FLAG and DIAMOND under another clue, so BALL
    # looked isolated for SOUND. Each clue's own list must decide.
    rankings = {
        "SOUND": ranked(
            ("BALL", "strong"), ("FLAG", "strong"), ("DIAMOND", "strong"), clue="SOUND"
        ),
        "GEM": ranked(("DIAMOND", "strong"), ("FLAG", "possible"), clue="GEM"),
    }
    decision = decide_bonus(rankings, {"SOUND": 1, "GEM": 1})
    assert (decision.word, decision.clue) == ("DIAMOND", "GEM")
    assert decide_bonus({"SOUND": rankings["SOUND"]}, {"SOUND": 1}).end_turn
    assert "ambiguous" in decide_bonus({"SOUND": rankings["SOUND"]}, {"SOUND": 1}).reason
    assert decide_bonus({"SOUND": rankings["SOUND"]}, {"SOUND": 3}).word == "BALL"


def test_bonus_needs_a_strong_match_and_an_unresolved_count() -> None:
    weak = {"ANIMAL": ranked(("CAT", "possible"), clue="ANIMAL")}
    assert decide_bonus(weak, {"ANIMAL": 1}).end_turn
    strong = {"ANIMAL": ranked(("CAT", "strong"), clue="ANIMAL")}
    assert decide_bonus(strong, {"ANIMAL": 0}).end_turn
    assert decide_bonus(strong, {"ANIMAL": 1}).word == "CAT"


def test_side_of_is_relative_to_team() -> None:
    assert side_of(CardColor.RED, Team.RED) is F
    assert side_of(CardColor.BLUE, Team.RED) is E
    assert side_of(CardColor.ASSASSIN, Team.RED) is A
    with pytest.raises(ValueError):
        side_of(CardColor.UNKNOWN, Team.RED)


# --- race model --------------------------------------------------------------------


def test_race_model_basics() -> None:
    model = RaceModel()
    assert model.win_probability(0, 3, our_turn=False) == 1.0
    assert model.win_probability(3, 0, our_turn=True) == 0.0
    # Moving first is an advantage; fewer cards is an advantage.
    assert model.win_probability(4, 4, our_turn=True) > model.win_probability(4, 4, our_turn=False)
    assert model.win_probability(3, 5, our_turn=True) > model.win_probability(5, 3, our_turn=True)
    assert model.expected_turns(1) < model.expected_turns(2) < model.expected_turns(5)
    assert model.finish_chance(1) == pytest.approx(0.85)
    assert model.finish_chance(4) == pytest.approx(0.04)  # four-card turns exist
    assert len(model.progress) == 5
    with pytest.raises(ValueError):
        RaceModel(progress=(0.5, 0.4))


@pytest.mark.parametrize(
    "ours,theirs,pressure",
    [(9, 8, "even"), (4, 4, "even"), (5, 8, "ahead"), (6, 2, "behind"), (2, 1, "critical"),
     (1, 5, "last card")],
)  # fmt: skip
def test_race_pressure_labels(ours: int, theirs: int, pressure: str) -> None:
    assert RaceState.of(ours, theirs).pressure == pressure


def test_each_extra_card_is_worth_more_when_racing_a_near_winner() -> None:
    model = RaceModel()

    def gain(ours: int, theirs: int) -> float:  # win-probability value of a 2nd card
        one = model.win_probability(ours - 1, theirs, our_turn=False)
        two = model.win_probability(ours - 2, theirs, our_turn=False) if ours > 2 else 1.0
        return two - one

    assert gain(2, 1) > gain(4, 4) > gain(5, 8)


# --- clue scoring: actions are (clue, number) ------------------------------------------


def best(clue: str, ranking, sides, intended=("F0",), **kw):
    """Best action of a clue over its natural numbers (a helper, not agent logic)."""
    actions = assess_actions(clue, intended, ranking, sides, race(sides), **kw)
    return select_clue(actions)


def test_outcomes_are_a_distribution_and_ev_is_exact() -> None:
    sides = board(2, 1, {"N": N})
    ranking = ranked(("F0", "strong"), ("X0", "strong"))
    dist = outcomes(ranking, sides, 1)
    assert sum(o.probability for o in dist) == pytest.approx(1.0)
    # Two strong words, ours listed first: 1/(1+0.6) chance of ours.
    assert expected_value(ranking, sides, 1) == pytest.approx((1 - 0.6) / 1.6)
    two = ranked(("F0", "strong"), ("F1", "strong"))
    assert expected_value(two, sides, 2) == pytest.approx(2.0)
    # A verified ranking's order is trusted almost fully.
    verified = expected_value(ranking, sides, 1, verified=True)
    assert verified == pytest.approx((1 - 0.1) / 1.1)


def test_win_probability_prices_finishing_enemy_cards_and_assassin() -> None:
    state = RaceState.of(2, 1)
    assert win_probability([Outcome(1.0, 2, None)], state) == 1.0
    assert win_probability([Outcome(1.0, 0, E)], state) == 0.0  # their last card
    assert win_probability([Outcome(1.0, 1, A)], state) == 0.0
    safe_one = win_probability([Outcome(1.0, 1, None)], state)
    assert 0.0 < safe_one < 0.2


def test_assessment_never_changes_the_number() -> None:
    # Live bug: MOTION 3 was re-derived as MOTION 1 when its ranking changed.
    sides = board(8, 6)
    weak = ranked(("F0", "strong"), ("X0", "strong"))
    for number in (1, 2, 3, 4):
        assert assess_action("MOTION", number, ["F0"], weak, sides, race(sides)).number == number
    rescored = assess_actions(
        "MOTION", ["F0", "F1", "F2"], weak, sides, race(sides), numbers=[3], verified=True
    )
    assert [a.action for a in rescored] == [("MOTION", 3)]
    with pytest.raises(ValueError):
        assess_action("MOTION", 5, ["F0"], weak, sides, race(sides))
    with pytest.raises(ValueError):
        assess_action("MOTION", 3, ["F0"], weak, board(2, 6), race(board(2, 6)))


def test_action_numbers_cover_intended_and_listed_words_up_to_four() -> None:
    sides = board(6, 6, {"N": N})
    listed = ranked(("F0", "strong"), ("F1", "strong"), ("N", "strong"), ("F2", "strong"))
    assert action_numbers(3, listed, sides, race(sides)) == [1, 2, 3]
    five = ranked(*[(f"F{i}", "strong") for i in range(5)])
    assert action_numbers(1, five, sides, race(sides)) == [1, 2, 3, 4]
    assert action_numbers(4, five, board(3, 6), race(board(3, 6))) == [1, 2, 3]
    assert action_numbers(1, ranked(("N", "strong")), sides, race(sides)) == [1]


def test_bigger_and_smaller_versions_are_separate_actions() -> None:
    # VEHICLE live case: three friendly words listed, one strong neutral among them.
    sides = board(6, 6, {"STEEL": N})
    ranking = ranked(("F0", "strong"), ("F1", "strong"), ("STEEL", "strong"), ("F2", "strong"))
    actions = assess_actions("VEHICLE", ["F1", "F2", "F3"], ranking, sides, race(sides))
    assert [a.number for a in actions] == [1, 2, 3]
    one, two, three = actions
    # With 2 still needed after the first hit (3 strong words, one of them STEEL), the
    # operative keeps guessing for VEHICLE 3 but stops for VEHICLE 2 (1 needed of 3).
    # The two actions really differ; the race model, not a cap, picks between them.
    assert two.win_probability == pytest.approx(one.win_probability)
    assert three.win_probability > two.win_probability
    assert three.miss_chances[N] > two.miss_chances[N]
    assert select_clue(actions) is three


def test_verified_two_card_plan_beats_one_card_plan() -> None:
    sides = board(6, 6)
    pair = best("PAIR", ranked(("F0", "strong"), ("F1", "strong")), sides, ("F0", "F1"))
    solo = best("SOLO", ranked(("F2", "strong")), sides, ("F2",))
    assert (pair.number, solo.number) == (2, 1)
    assert pair.turns_to_finish < solo.turns_to_finish
    assert select_clue([solo, pair]) is pair


def test_perfect_single_does_not_beat_credible_pair() -> None:
    # 100% first guess vs 97%/1.9 hits: coverage wins in a normal race position.
    sides = board(6, 5, {"N": N})
    solo = best("SOLO", ranked(("F2", "strong")), sides, ("F2",))
    pair = best(
        "PAIR", ranked(("F0", "strong"), ("F1", "strong"), ("N", "possible")), sides, ("F0", "F1")
    )
    assert solo.p_first_friendly == 1.0 and pair.p_first_friendly < 1.0
    assert select_clue([solo, pair]) is pair


def test_match_point_prefers_a_risky_finish_over_a_certain_single() -> None:
    sides = board(2, 1, {"N": N})
    ranking = ranked(("F0", "strong"), ("F1", "strong"), ("N", "strong"))
    finish = assess_action("FINISH", 2, ["F0", "F1"], ranking, sides, race(sides))
    single = assess_action("ONE", 1, ["F0"], ranked(("F0", "strong")), sides, race(sides))
    assert finish.finish_chance < 0.8
    assert select_clue([single, finish]) is finish


def test_critical_four_vs_one_can_choose_clue_four() -> None:
    sides = board(4, 1, {"JAGUAR": N})
    four = ranked(
        ("F0", "strong"),
        ("F1", "strong"),
        ("F2", "strong"),
        ("F3", "strong"),
        ("JAGUAR", "possible"),
    )
    actions = assess_actions("FELINES", ["F0", "F1", "F2", "F3"], four, sides, race(sides))
    two = assess_action(
        "SAVANNA", 2, ["F0", "F1"], ranked(("F0", "strong"), ("F1", "strong")), sides, race(sides)
    )
    one = assess_action("STRIPES", 1, ["F2"], ranked(("F2", "strong")), sides, race(sides))
    choice = select_clue([one, two, *actions])
    assert choice.action == ("FELINES", 4)
    assert choice.finish_chance > 0.5 and one.win_probability < 0.1


def test_early_game_unsafe_four_loses_to_safe_single() -> None:
    sides = board(8, 8)
    unsafe = ranked(
        ("X0", "strong"), ("F0", "strong"), ("X1", "strong"), ("F1", "strong"), ("F2", "strong"),
        ("F3", "strong"), ("X2", "possible")
    )  # fmt: skip
    fruit = assess_action("FRUIT", 4, ["F0", "F1", "F2", "F3"], unsafe, sides, race(sides))
    safe = assess_action("VINEYARD", 1, ["F4"], ranked(("F4", "strong")), sides, race(sides))
    assert fruit.expected_hits < safe.expected_hits + 1  # coverage alone is not enough
    assert select_clue([fruit, safe]) is safe


def test_when_ahead_safety_can_beat_coverage() -> None:
    sides = board(5, 8)
    risky = best(
        "RISKY",
        ranked(
            ("X0", "strong"), ("F0", "strong"), ("X1", "strong"), ("F1", "strong"), ("F2", "strong")
        ),
        sides,
        ("F0", "F1", "F2"),
    )
    safe = best("SAFE", ranked(("F3", "strong")), sides, ("F3",))
    assert select_clue([risky, safe]) is safe


def test_failure_types_are_priced_assassin_over_enemy_over_neutral() -> None:
    # Same ranking shape; only the colour of the tempting second word differs.
    def win(side: Side, fit: str = "possible") -> float:
        # The same race (5 vs 5) every time: the enemy word is one of their 5 cards.
        word = "X0" if side is E else "W"
        sides = board(5, 5, {} if side is E else {word: side})
        ranking = ranked(("F0", "strong"), (word, fit))
        return assess_action("C", 1, ["F0"], ranking, sides, race(sides)).win_probability

    clean = assess_action("C", 1, ["F0"], ranked(("F0", "strong")), board(5, 5), race(board(5, 5)))
    neutral, enemy, assassin = win(N), win(E), win(A)
    assert clean.win_probability > neutral > enemy > assassin
    # The assassin gap is much larger than the enemy/neutral gap.
    assert (enemy - assassin) > 3 * (neutral - enemy)


def test_assassin_strong_is_vetoed_possible_is_a_severe_penalty() -> None:
    sides = board(3, 3, {"K": A})
    strong = ranked(("F0", "strong"), ("F1", "strong"), ("K", "strong"))
    vetoed = assess_action("RISK", 2, ["F0", "F1"], strong, sides, race(sides))
    assert vetoed.vetoed and select_clue([vetoed]) is None
    possible = ranked(("F0", "strong"), ("F1", "strong"), ("K", "possible"))
    priced = assess_action("RISK", 2, ["F0", "F1"], possible, sides, race(sides), verified=True)
    clean = assess_action(
        "SAFE", 2, ["F0", "F1"], ranked(("F0", "strong"), ("F1", "strong")), sides, race(sides),
        verified=True,
    )  # fmt: skip
    assert not priced.vetoed and priced.acceptable
    assert priced.miss_chances[A] > 0.05  # not discounted by its low list position
    assert priced.win_probability < clean.win_probability - 0.03


def test_neutral_and_enemy_competition_lower_win_probability() -> None:
    sides = board(4, 4, {"N": N})
    clean = assess_action("C", 1, ["F0"], ranked(("F0", "strong")), sides, race(sides))
    neutral = assess_action(
        "N", 1, ["F0"], ranked(("F0", "strong"), ("N", "strong")), sides, race(sides)
    )
    enemy = assess_action(
        "E", 1, ["F0"], ranked(("F0", "strong"), ("X0", "strong")), sides, race(sides)
    )
    assert clean.win_probability > neutral.win_probability > enemy.win_probability


def test_top_pick_must_be_ours() -> None:
    sides = board(3, 3, {"N": N})
    assessment = assess_action(
        "X", 1, ["F0"], ranked(("N", "strong"), ("F0", "strong")), sides, race(sides)
    )
    assert not assessment.acceptable and "top pick N" in assessment.reason


def test_continuing_is_a_race_decision_with_public_counts() -> None:
    ranking = ranked(("S0", "strong"), ("S1", "strong"))
    # Unknown counts: the fixed 60% rule stops on a coin flip.
    assert decide_current(ranking, 1, first_guess_required=False).end_turn
    # Match point (one card each): stopping hands the opponent the game, so press on.
    tied = RacePosition(ours=1, theirs=1, neutral=3, assassin_hidden=True)
    assert decide_current(ranking, 1, first_guess_required=False, position=tied).word == "S0"
    # Comfortably ahead: the same coin flip is not worth the risk.
    ahead = RacePosition(ours=2, theirs=6, neutral=6, assassin_hidden=True)
    assert decide_current(ranking, 1, first_guess_required=False, position=ahead).end_turn


def test_ties_prefer_the_smaller_number() -> None:
    # One listed friendly word: numbers 1-3 behave identically; no overstated number.
    sides = board(5, 5)
    actions = assess_actions("X", ["F0", "F1", "F2"], ranked(("F0", "strong")), sides, race(sides))
    assert [a.number for a in actions] == [1, 2, 3]
    assert select_clue(actions).number == 1


def test_best_by_size_reports_each_number() -> None:
    sides = board(4, 1)
    actions = assess_actions(
        "ALL", ["F0", "F1", "F2", "F3"], ranked(*[(f"F{i}", "strong") for i in range(4)]),
        sides, race(sides),
    )  # fmt: skip
    sizes = best_by_size(actions)
    assert list(sizes) == [1, 2, 3, 4]
    assert sizes[4].win_probability > sizes[1].win_probability


def test_empty_ranking_is_not_acceptable() -> None:
    sides = board(2, 2)
    assert not assess_action("X", 1, ["F0"], (), sides, race(sides)).acceptable


def test_least_bad_is_chosen_only_without_acceptable_and_never_vetoed() -> None:
    sides = board(3, 3, {"N": N, "K": A})
    bad = assess_action(
        "BAD", 1, ["F0"], ranked(("N", "strong"), ("F0", "possible")), sides, race(sides)
    )
    vetoed = assess_action(
        "VETO", 1, ["F0"], ranked(("F0", "strong"), ("K", "strong")), sides, race(sides)
    )
    assert select_clue([vetoed, bad]) is bad


def test_near_tie_goes_to_the_smaller_number() -> None:
    # MOTION 4 only beats MOTION 3 through a rare path (first pick = a possible word):
    # a rounding-level gain must not overstate the clue.
    sides = board(8, 6)
    ranking = ranked(("F0", "strong"), ("F1", "strong"), ("F2", "strong"), ("F3", "possible"))
    actions = assess_actions("MOTION", ["F0", "F1", "F2"], ranking, sides, race(sides))
    by_number = {a.number: a for a in actions}
    assert by_number[4].win_probability >= by_number[3].win_probability
    assert by_number[4].win_probability - by_number[3].win_probability < 0.005
    assert select_clue(actions).action == ("MOTION", 3)


@pytest.mark.parametrize("ours,theirs", [(8, 6), (6, 6), (4, 2), (3, 1)])
def test_single_never_beats_a_materially_better_multi_card_action(ours: int, theirs: int) -> None:
    sides = board(ours, theirs)
    single = assess_action("ONE", 1, ["F0"], ranked(("F0", "strong")), sides, race(sides))
    pair = assess_action(
        "TWO", 2, ["F1", "F2"], ranked(("F1", "strong"), ("F2", "strong")), sides, race(sides)
    )
    assert pair.win_probability > single.win_probability + 0.02
    assert select_clue([single, pair]) is pair


def test_prune_dominated_drops_redundant_bigger_numbers_only() -> None:
    sides = board(8, 6)
    ranking = ranked(("F0", "strong"), ("F1", "strong"), ("F2", "strong"), ("F3", "possible"))
    motion = assess_actions("MOTION", ["F0", "F1", "F2"], ranking, sides, race(sides))
    salad = assess_actions("SALAD", ["F4"], ranked(("F4", "strong")), sides, race(sides))
    kept = prune_dominated(motion + salad)
    assert {a.action for a in kept} == {("MOTION", 1), ("MOTION", 2), ("MOTION", 3), ("SALAD", 1)}
    assert all(a in motion + salad for a in kept)  # same objects, numbers untouched
    # Pruning never changes the choice: pruned actions lose the near-tie anyway.
    assert select_clue(kept) is select_clue(motion + salad)
