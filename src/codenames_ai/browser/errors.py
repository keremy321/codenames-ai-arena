"""Action errors stop the controller; ambiguous mutations are never retried."""


class BrowserIntegrationError(RuntimeError):
    """The current DOM cannot safely support the requested operation."""


class CardNotFoundError(BrowserIntegrationError):
    """No board card matched the requested word."""


class AmbiguousCardError(BrowserIntegrationError):
    """Multiple board cards matched the requested word."""


class AlreadyRevealedError(BrowserIntegrationError):
    """The requested card is already revealed."""


class GuessConfirmationNotFoundError(BrowserIntegrationError):
    """Selected card has no unique verified confirmation control."""


class GameStateTimeoutError(BrowserIntegrationError):
    """The expected DOM transition did not arrive before the deadline."""


class PlayerAssignmentError(BrowserIntegrationError):
    """The DOM does not match the expected local player assignment."""


class WrongPhaseError(BrowserIntegrationError):
    """This player cannot act in the current phase."""
