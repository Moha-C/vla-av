"""Checkpoint / runtime dimension self-check.

Run this standalone before wiring a trained checkpoint into any external
runtime/adapter, to confirm the policy's first-layer input dimension matches
the state vector you intend to feed it. A silent dimension mismatch is the
most common cause of a checkpoint producing meaningless output downstream.

Usage:
    python -m utils.checkpoint_check checkpoints/sdbs_final_model.pt --state-dim 55
"""


def check_checkpoint_compatibility(checkpoint_path, expected_state_dim,
                                   expected_action_dim):
    """Load a checkpoint and verify its first-layer input dimension matches
    ``expected_state_dim``. Prints a clear PASS/FAIL report and returns a bool.

    Does not raise — designed to be run standalone by anyone integrating this
    checkpoint elsewhere.
    """
    import torch
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    policy_state = ckpt.get('policy', ckpt)

    # The input layer is the first 2-D weight matrix; its second dimension is
    # the state (input) dimension. We scan rather than take the literal first
    # key because models can register 1-D parameters (e.g. a log-std vector)
    # ahead of the first Linear layer in the state_dict.
    first_key, first_weight, actual_input_dim = None, None, None
    for key, weight in policy_state.items():
        if hasattr(weight, 'dim') and weight.dim() == 2:
            first_key, first_weight = key, weight
            actual_input_dim = weight.shape[1]
            break

    print(f"Checkpoint: {checkpoint_path}")
    print(f"  First layer key: {first_key}")
    print(f"  First layer shape: "
          f"{tuple(first_weight.shape) if first_weight is not None else None}")
    print(f"  Expected input dim: {expected_state_dim}")

    if actual_input_dim == expected_state_dim:
        print(f"  RESULT: PASS - dimensions match")
        return True
    else:
        print(f"  RESULT: FAIL - checkpoint expects {actual_input_dim}, "
              f"you are feeding {expected_state_dim}")
        print(f"  This checkpoint will produce meaningless output "
              f"if used with a mismatched state vector.")
        return False


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint_path')
    parser.add_argument(
        '--state-dim', type=int, default=28,
        help="28 = original v1 layout (default, proven working). Use 48 with "
             "--enable-tier1, 55 with --enable-tier1 --enable-tier2 if training "
             "with expanded safety state.")
    parser.add_argument('--action-dim', type=int, default=4)
    args = parser.parse_args()
    check_checkpoint_compatibility(args.checkpoint_path, args.state_dim,
                                   args.action_dim)
