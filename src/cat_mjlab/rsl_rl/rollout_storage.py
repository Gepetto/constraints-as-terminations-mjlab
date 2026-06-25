import torch
from rsl_rl.storage.rollout_storage import RolloutStorage


class CaTRolloutStorage(RolloutStorage):
  class Transition(RolloutStorage.Transition):
    def __init__(self) -> None:
      super().__init__()

      """Constraint violations computed by CaT"""
      self.constraint_violations: torch.Tensor | None = None

  def __init__(self, *args, **kwargs) -> None:
    super().__init__(*args, **kwargs)
    self.constraint_violations = torch.zeros_like(self.dones)

  def add_transition(self, transition: Transition) -> None:
    # Check if the transition is valid
    if self.step >= self.num_transitions_per_env:
      raise OverflowError(
        "Rollout buffer overflow! You should call clear() before adding new transitions."
      )
    self.constraint_violations[self.step].copy_(
      transition.constraint_violations.view(-1, 1)
    )

    super().add_transition(transition)
