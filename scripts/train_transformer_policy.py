import argparse

from policyNet_variants import train_policy_variant


def parse_args():
    parser = argparse.ArgumentParser(description="Train Transformer-based policy network")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--saved-model", default="./model/transformer_policy")
    parser.add_argument("--fp-dim", type=int, default=2048)
    parser.add_argument("--patch-size", type=int, default=32)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dim-feedforward", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    return parser.parse_args()


def main():
    args = parse_args()
    train_policy_variant(
        model_type="transformer",
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        weight_decay=args.weight_decay,
        saved_model=args.saved_model,
        fp_dim=args.fp_dim,
        patch_size=args.patch_size,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
    )


if __name__ == "__main__":
    main()
